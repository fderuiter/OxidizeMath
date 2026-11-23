// [cite:modular_polynomials_review]
//! Side-panel controls, 3D viewport rendering, and 3D pulse brush picking for `SurfaceMorphogenesisTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    MorphogenesisKineticsKind, PulseBrushAction, SurfaceMorphogenesisTool, SurfaceOverlayMode,
    ViewportInteractionMode, MAX_SURFACE_MORPHO_RES, MIN_SURFACE_MORPHO_RES,
};
use crate::audio::mappings::surface_morphogenesis_frame;
use crate::audio::render_sonification_panel;
use crate::framework::playback::{PlaybackAction, PlaybackTransport};
use crate::framework::{Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use crate::tabs::geometry_topology::custom_surface::SurfaceType;
use crate::tabs::geometry_topology::export_utils::{MAX_WALL_THICKNESS, MIN_WALL_THICKNESS};
use eframe::egui;
use egui_plot::{LineStyle, MarkerShape, PlotPoints, Points};
use math_commons::math_kernel::colormap::mapped_gradient;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::differential_geometry::buckled_flow::SurfaceTopology;
use nalgebra::Point3;

const COLOR_BINS: usize = 24;

impl SurfaceMorphogenesisTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        self.ensure_engine();
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "surface_morphogenesis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Surface Morphogenesis");
            self.draw_transport_and_status(ui);
            ui.separator();
            self.draw_surface_controls(ui);
            ui.separator();
            self.draw_kinetics_and_buckling_controls(ui);
            ui.separator();
            self.draw_overlay_and_brush_controls(ui);
            ui.separator();
            egui::CollapsingHeader::new("View")
                .default_open(true)
                .show(ui, |ui| self.camera.ui(ui));
            ui.separator();
            let frame = self.current_sonification_frame();
            render_sonification_panel(ui, "SurfaceMorphogenesisTool", &frame, !self.is_playing);
        });
    }

    fn draw_transport_and_status(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "surface_morpho_transport",
            "Surface Morphogenesis",
            self.is_playing,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.is_playing = true,
                PlaybackAction::Pause => {
                    self.is_playing = false;
                    crate::audio::notify_playback_paused();
                }
                PlaybackAction::Step => {
                    self.is_playing = false;
                    let _ = self.step_once();
                }
                PlaybackAction::Reset => {
                    self.is_playing = false;
                    self.sanitize_and_recover();
                    self.reseed_counter = self.reseed_counter.wrapping_add(1);
                    self.reseed_active_engine(0.0);
                }
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
            }
        }
        if let Some(eng) = self.engine.as_ref() {
            ui.horizontal_wrapped(|ui| {
                ui.small(format!(
                    "Steps: {} | CFL dt: {:.4}",
                    eng.step_count, eng.effective_dt
                ));
                if eng.rollback_count > 0 {
                    let pal = crate::app::SemanticPalette::from_visuals(ui.visuals());
                    ui.colored_label(
                        pal.warning,
                        format!("FiniteStateGuard rollbacks: {}", eng.rollback_count),
                    );
                }
            });
        }
    }

    fn draw_surface_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Manifold Topology:");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut self.surface_type, SurfaceType::Sphere, "Sphere");
            ui.radio_value(&mut self.surface_type, SurfaceType::Torus, "Torus");
            ui.radio_value(
                &mut self.surface_type,
                SurfaceType::KleinBottle,
                "Klein Bottle",
            );
            ui.radio_value(
                &mut self.surface_type,
                SurfaceType::CustomHeightfield,
                "Custom z=f(x,y)",
            );
            ui.radio_value(
                &mut self.surface_type,
                SurfaceType::CustomParametric,
                "Custom r(u,v)",
            );
        });

        let open_default = matches!(
            self.surface_type,
            SurfaceType::CustomHeightfield | SurfaceType::CustomParametric
        );
        egui::CollapsingHeader::new("Geometry & Mesh")
            .default_open(open_default)
            .show(ui, |ui| self.draw_geometry_mesh_sliders(ui));
    }

    fn draw_geometry_mesh_sliders(&mut self, ui: &mut egui::Ui) {
        match self.surface_type {
            SurfaceType::Sphere => {
                ui.add(egui::Slider::new(&mut self.sphere_radius, 0.2..=5.0).text("Radius"));
            }
            SurfaceType::Torus => {
                ui.add(
                    egui::Slider::new(&mut self.torus_major_radius, 1.0..=5.0)
                        .text("Major Radius (R)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.torus_minor_radius, 0.15..=2.0)
                        .text("Minor Radius (r)"),
                );
            }
            SurfaceType::KleinBottle => {
                ui.add(egui::Slider::new(&mut self.klein_radius, 0.5..=5.0).text("Radius"));
            }
            SurfaceType::CustomHeightfield | SurfaceType::CustomParametric => {
                self.custom_sandbox.ui_for_type(ui, self.surface_type);
            }
        }
        ui.add(
            egui::Slider::new(
                &mut self.u_resolution,
                MIN_SURFACE_MORPHO_RES..=MAX_SURFACE_MORPHO_RES,
            )
            .text("Grid U"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.v_resolution,
                MIN_SURFACE_MORPHO_RES..=MAX_SURFACE_MORPHO_RES,
            )
            .text("Grid V"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.wall_thickness,
                MIN_WALL_THICKNESS..=MAX_WALL_THICKNESS,
            )
            .text("3D Export Wall Thickness"),
        );
    }

    fn draw_kinetics_and_buckling_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Reaction Kinetics:");
        let prev_kind = self.kinetics_kind;
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.kinetics_kind,
                MorphogenesisKineticsKind::Schnakenberg,
                MorphogenesisKineticsKind::Schnakenberg.label(),
            );
            ui.radio_value(
                &mut self.kinetics_kind,
                MorphogenesisKineticsKind::GrayScott,
                MorphogenesisKineticsKind::GrayScott.label(),
            );
        });
        if self.kinetics_kind != prev_kind {
            if self.kinetics_kind == MorphogenesisKineticsKind::GrayScott {
                self.d_u = 0.04;
                self.d_v = 0.02;
            } else {
                self.d_u = 0.05;
                self.d_v = 1.25;
            }
        }

        egui::CollapsingHeader::new("Kinetics & Buckling")
            .default_open(true)
            .show(ui, |ui| {
                match self.kinetics_kind {
                    MorphogenesisKineticsKind::Schnakenberg => {
                        ui.add(egui::Slider::new(&mut self.schnak_a, 0.02..=0.5).text("Source a"));
                        ui.add(egui::Slider::new(&mut self.schnak_b, 0.2..=2.0).text("Source b"));
                    }
                    MorphogenesisKineticsKind::GrayScott => {
                        ui.add(egui::Slider::new(&mut self.gs_feed, 0.01..=0.09).text("Feed F"));
                        ui.add(egui::Slider::new(&mut self.gs_kill, 0.03..=0.08).text("Kill k"));
                    }
                }
                ui.add(egui::Slider::new(&mut self.d_u, 0.005..=0.5).text("Diffusivity D_u"));
                ui.add(egui::Slider::new(&mut self.d_v, 0.005..=3.0).text("Diffusivity D_v"));
                ui.add(
                    egui::Slider::new(&mut self.buckling_gain, -1.0..=1.0).text("Buckling Gain γ"),
                );
                ui.add(
                    egui::Slider::new(&mut self.curvature_tension, 0.0..=0.4)
                        .text("Curvature Tension κ"),
                );
                ui.add(egui::Slider::new(&mut self.dt, 0.001..=0.05).text("Time Step dt"));
                ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=12).text("Substeps/Frame"));
            });
    }

    fn draw_overlay_and_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Surface Overlay:");
        for mode in [
            SurfaceOverlayMode::MorphogenConcentration,
            SurfaceOverlayMode::GaussianCurvature,
            SurfaceOverlayMode::MetricAnisotropy,
        ] {
            ui.radio_value(&mut self.overlay_mode, mode, mode.label());
        }

        ui.add_space(4.0);
        ui.label("3D Viewport Gesture:");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(
                &mut self.interaction_mode,
                ViewportInteractionMode::OrbitCamera,
                "Orbit Camera",
            );
            ui.radio_value(
                &mut self.interaction_mode,
                ViewportInteractionMode::PulseBrush,
                "Pulse Brush",
            );
        });
        if self.interaction_mode == ViewportInteractionMode::PulseBrush {
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(
                    &mut self.brush_action,
                    PulseBrushAction::ActivatorPulse,
                    "+u Pulse",
                );
                ui.radio_value(
                    &mut self.brush_action,
                    PulseBrushAction::InhibitorPulse,
                    "+v Pulse",
                );
                ui.radio_value(
                    &mut self.brush_action,
                    PulseBrushAction::ResetPatch,
                    "Clear",
                );
            });
            ui.add(egui::Slider::new(&mut self.brush_radius, 0.15..=1.8).text("Brush Radius"));
        }
        if ui.button("Seed Perturbation").clicked() {
            self.reseed_counter = self.reseed_counter.wrapping_add(1);
            let phase = self.reseed_counter as f64 * 1.17;
            self.reseed_active_engine(phase);
        }
    }

    fn reseed_active_engine(&mut self, phase: f64) {
        self.ensure_engine();
        if let Some(mut eng) = self.engine.take() {
            eng.reseed(self, phase);
            self.engine = Some(eng);
        }
    }

    pub(super) fn current_sonification_frame(&self) -> crate::audio::SonificationFrame {
        let Some(eng) = self.engine.as_ref() else {
            return crate::audio::SonificationFrame::silent();
        };
        let (var, mean, curv_shift, buckle_rms) = eng.sonification_metrics();
        surface_morphogenesis_frame(var, mean, curv_shift, buckle_rms)
    }

    pub(super) fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        self.ensure_engine();
        if self.is_playing {
            let n_steps = self.steps_per_frame.clamp(1, 16);
            for _ in 0..n_steps {
                if !self.step_once() {
                    self.is_playing = false;
                    break;
                }
            }
            parent_ui.ctx().request_repaint();
        }

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            self.populate_scene(palette.series_color(0));
            let helper = format!(
                "{CAMERA_3D_HELPER_TEXT} Overlay: {} (Shift+Click or Pulse Brush to paint).",
                self.overlay_mode.label()
            );
            let prev_cam = self.camera;
            let mut camera = self.camera;
            let mut painted_primary = false;
            Plot3DViewport::new("surface_morphogenesis_plot", 4.2)
                .helper_text(&helper)
                .show(ui, &mut camera, |plot_ui| {
                    self.scene_builder.finish(plot_ui);
                    painted_primary = self.sample_viewport_pulse_brush(plot_ui, palette.accent);
                });
            if painted_primary && self.interaction_mode == ViewportInteractionMode::PulseBrush {
                camera.pitch = prev_cam.pitch;
                camera.yaw = prev_cam.yaw;
                camera.pan_x = prev_cam.pan_x;
                camera.pan_y = prev_cam.pan_y;
            }
            self.camera = camera;
        });
    }

    fn sample_viewport_pulse_brush(
        &mut self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        reticle_color: egui::Color32,
    ) -> bool {
        let (primary_down, shift_held) = plot_ui
            .ctx()
            .input(|i| (i.pointer.primary_down(), i.modifiers.shift));
        let brush_enabled =
            self.interaction_mode == ViewportInteractionMode::PulseBrush || shift_held;
        if !brush_enabled {
            return false;
        }
        let Some(coord) = plot_ui.pointer_coordinate() else {
            return false;
        };
        let Some(hit_pt) = self.pick_frontmost_surface_vertex([coord.x, coord.y]) else {
            return false;
        };

        let proj = self.camera.project(&[hit_pt.x, hit_pt.y, hit_pt.z]);
        plot_ui.points(
            Points::new("Brush Target", PlotPoints::new(vec![proj]))
                .radius(7.0_f32)
                .shape(MarkerShape::Circle)
                .filled(false)
                .color(reticle_color)
                .allow_hover(false),
        );

        if primary_down && (plot_ui.response().hovered() || plot_ui.response().dragged()) {
            self.apply_brush_at_world_point(hit_pt);
            return true;
        }
        false
    }

    /// Picks the closest front-facing 3D surface vertex to projected 2D plot coordinates `plot_xy`.
    #[must_use]
    pub fn pick_frontmost_surface_vertex(&self, plot_xy: [f64; 2]) -> Option<Point3<f64>> {
        let eng = self.engine.as_ref()?;
        let pts = eng.flow.points();
        let mut best_front: Option<(f64, Point3<f64>)> = None;
        let mut nearest_any: Option<(f64, Point3<f64>)> = None;

        for &p in pts {
            let (proj, vz) = self.camera.project_with_depth(&[p.x, p.y, p.z]);
            let d2 = (proj[0] - plot_xy[0]).powi(2) + (proj[1] - plot_xy[1]).powi(2);
            if d2 <= 0.16 && best_front.is_none_or(|(best_vz, _)| vz > best_vz) {
                best_front = Some((vz, p));
            }
            if d2 <= 0.64 && nearest_any.is_none_or(|(best_d2, _)| d2 < best_d2) {
                nearest_any = Some((d2, p));
            }
        }
        best_front
            .map(|(_, p)| p)
            .or_else(|| nearest_any.map(|(_, p)| p))
    }

    /// Injects a 3D geodesic morphogen pulse centered at `world_pt`.
    pub fn apply_brush_at_world_point(&mut self, world_pt: Point3<f64>) {
        self.ensure_engine();
        let radius = self.brush_radius;
        let strength = self.brush_strength;
        let action = self.brush_action;
        if let Some(mut eng) = self.engine.take() {
            eng.apply_3d_pulse(world_pt, radius, strength, action, self);
            self.engine = Some(eng);
        }
    }

    pub(super) fn populate_scene(&mut self, series_color: egui::Color32) {
        let Some(eng) = self.engine.as_ref() else {
            return;
        };
        let pts = eng.flow.points();
        let scalars = eng.overlay_field(self.overlay_mode);
        let (min_s, max_s, lut) = build_overlay_lut(scalars);
        let (nu, nv) = (eng.nu, eng.nv);
        let wrap_u = matches!(
            eng.topology,
            SurfaceTopology::PeriodicTorus | SurfaceTopology::TwistedKlein
        );
        let wrap_v = matches!(
            eng.topology,
            SurfaceTopology::PeriodicTorus | SurfaceTopology::TwistedKlein
        );
        let wrap_sphere_u = eng.topology == SurfaceTopology::PolarSphere;

        self.scene_builder.begin(self.camera);
        for i in 0..nu {
            for j in 0..nv {
                let idx = flatten_2d_index(j, i, nv);
                let p0 = pts[idx];
                let s0 = scalars[idx];
                if j + 1 < nv || wrap_v {
                    let j_next = (j + 1) % nv;
                    let idx_v = flatten_2d_index(j_next, i, nv);
                    let col = lut[quantize_bin(0.5 * (s0 + scalars[idx_v]), min_s, max_s)];
                    self.scene_builder.add_styled_segment(
                        "Surface Morphogenesis",
                        [p0, pts[idx_v]],
                        egui::Stroke::new(1.35_f32, col),
                        LineStyle::Solid,
                        true,
                        false,
                    );
                }
                if i + 1 < nu || wrap_u || wrap_sphere_u {
                    let i_next = (i + 1) % nu;
                    let j_seam = if eng.topology == SurfaceTopology::TwistedKlein && i + 1 >= nu {
                        eng.flow.seam_forward_map()[j]
                    } else {
                        j
                    };
                    let idx_u = flatten_2d_index(j_seam, i_next, nv);
                    let col = lut[quantize_bin(0.5 * (s0 + scalars[idx_u]), min_s, max_s)];
                    self.scene_builder.add_styled_segment(
                        "Surface Morphogenesis",
                        [p0, pts[idx_u]],
                        egui::Stroke::new(1.35_f32, col),
                        LineStyle::Solid,
                        true,
                        false,
                    );
                }
            }
        }
        self.scene_builder
            .record_series("Surface Morphogenesis", pts, series_color);
        self.scene_builder.ensure_sorted();
    }
}

fn build_overlay_lut(values: &[f64]) -> (f64, f64, [egui::Color32; COLOR_BINS]) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        min_v = 0.0;
        max_v = 1.0;
    }
    let mut lut = [egui::Color32::WHITE; COLOR_BINS];
    for (idx, slot) in lut.iter_mut().enumerate() {
        let t = idx as f64 / (COLOR_BINS - 1) as f64;
        let rgb = mapped_gradient(t, 0.0, 1.0);
        *slot = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
    }
    (min_v, max_v, lut)
}

fn quantize_bin(val: f64, min_v: f64, max_v: f64) -> usize {
    let span = max_v - min_v;
    if !val.is_finite() || span.abs() < 1e-10 {
        return COLOR_BINS / 2;
    }
    let norm = ((val - min_v) / span).clamp(0.0, 1.0);
    ((norm * (COLOR_BINS - 1) as f64).round() as usize).min(COLOR_BINS - 1)
}
