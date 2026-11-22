#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, `Plot3DViewport` 3D rendering, 2D contour view, and canvas gestures.

use super::arena::{RacerKind, LR_MAX, LR_MIN};
use super::surfaces::{
    lerp_color, LandscapeSurface, SurfaceGrid, ARENA_3D_HALF_EXTENT, MAX_SCULPT_BUMPS,
    SCULPT_AMP_MAX, SCULPT_AMP_MIN, SCULPT_SIGMA_MAX, SCULPT_SIGMA_MIN,
};
use super::{
    CanvasInteractionMode, LandscapeViewMode, LossLandscapeTool, LOSS_MIN_SPAN,
    LOSS_PARAM_MAX_BOUND, LOSS_PARAM_MIN_BOUND,
};
use crate::app::SemanticPalette;
use crate::camera::{Camera3D, Plot3DViewport, CAMERA_3D_HELPER_TEXT};
use crate::framework::{
    bounded_drag_value, enforce_ordered_drag_range, InputMode, InteractionContext, InteractiveTool,
    PlaybackAction, PlaybackTransport,
};
use eframe::egui;
use egui_plot::{Line, MarkerShape, Plot, PlotUi, Points};
use math_explorer::pure_math::analysis::momentum_dynamics::{
    MAX_DAMPING, MAX_DT, MAX_MASS, MAX_TEMPERATURE, MIN_DAMPING, MIN_DT, MIN_MASS, MIN_TEMPERATURE,
};

impl LossLandscapeTool {
    pub(super) fn draw_ordered_bound_pair(
        ui: &mut egui::Ui,
        label: &str,
        min_v: &mut f64,
        max_v: &mut f64,
    ) {
        ui.label(label);
        let (mut min_changed, mut max_changed) = (false, false);
        ui.horizontal(|ui| {
            min_changed = ui
                .add(
                    bounded_drag_value(min_v, LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND, 0.1)
                        .prefix("Min: "),
                )
                .changed();
            max_changed = ui
                .add(
                    bounded_drag_value(max_v, LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND, 0.1)
                        .prefix("Max: "),
                )
                .changed();
        });
        enforce_ordered_drag_range(
            min_v,
            max_v,
            LOSS_PARAM_MIN_BOUND,
            LOSS_PARAM_MAX_BOUND,
            LOSS_MIN_SPAN,
            min_changed,
            max_changed,
        );
    }

    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "loss_landscape_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Loss Landscape Arena");
            Self::draw_ordered_bound_pair(ui, "Weight (w) Range", &mut self.w_min, &mut self.w_max);
            Self::draw_ordered_bound_pair(ui, "Bias (b) Range", &mut self.b_min, &mut self.b_max);
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("loss_arena_scroll")
                .show(ui, |ui| {
                    self.draw_transport_and_mode_ui(ui);
                    ui.separator();
                    self.draw_physics_and_sculpt_ui(ui);
                    ui.separator();
                    self.draw_racers_and_camera_ui(ui);
                });
        });
    }

    fn draw_transport_and_mode_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(act) =
            PlaybackTransport::in_thread("loss_arena_transport", self.name(), self.is_playing)
                .show(ui)
        {
            match act {
                PlaybackAction::Play => self.is_playing = true,
                PlaybackAction::Pause => self.is_playing = false,
                PlaybackAction::Step => self.step_simulation(),
                PlaybackAction::Reset => {
                    self.is_playing = false;
                    self.reset_arena();
                }
            }
        }
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view_mode, LandscapeViewMode::View3D, "3D Surface");
            ui.selectable_value(
                &mut self.view_mode,
                LandscapeViewMode::Contour2D,
                "2D Contours",
            );
        });
        let prev_surface = self.surface;
        egui::ComboBox::from_label("Surface")
            .selected_text(self.surface.label())
            .show_ui(ui, |ui| {
                for s in LandscapeSurface::ALL {
                    ui.selectable_value(&mut self.surface, s, s.label());
                }
            });
        if self.surface != prev_surface {
            self.apply_surface_preset(self.surface);
        }
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.interaction_mode, CanvasInteractionMode::Orbit, "Orbit");
            ui.selectable_value(
                &mut self.interaction_mode,
                CanvasInteractionMode::Slingshot,
                "Slingshot",
            );
            ui.selectable_value(
                &mut self.interaction_mode,
                CanvasInteractionMode::SculptBrush,
                "Sculpt",
            );
        });
    }

    fn draw_physics_and_sculpt_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.resolution, 10..=50).text("Grid"));
        ui.add(egui::Slider::new(&mut self.arena_config.mass, MIN_MASS..=MAX_MASS).text("Mass m"));
        ui.add(
            egui::Slider::new(&mut self.arena_config.damping, MIN_DAMPING..=MAX_DAMPING)
                .text("Damp γ"),
        );
        ui.add(
            egui::Slider::new(&mut self.arena_config.temperature, MIN_TEMPERATURE..=MAX_TEMPERATURE)
                .text("Temp kBT"),
        );
        ui.add(
            egui::Slider::new(&mut self.arena_config.dt, MIN_DT..=MAX_DT)
                .logarithmic(true)
                .text("Step dt"),
        );
        ui.add(
            egui::Slider::new(&mut self.arena_config.learning_rate, LR_MIN..=LR_MAX)
                .logarithmic(true)
                .text("LR η"),
        );
        ui.label(format!(
            "Sculpt Bumps: {}/{}",
            self.sculpt_bumps.len(),
            MAX_SCULPT_BUMPS
        ));
        ui.add(
            egui::Slider::new(&mut self.sculpt_amplitude, SCULPT_AMP_MIN..=SCULPT_AMP_MAX)
                .text("Amp A"),
        );
        ui.add(
            egui::Slider::new(&mut self.sculpt_sigma, SCULPT_SIGMA_MIN..=SCULPT_SIGMA_MAX)
                .text("Width σ"),
        );
        if !self.sculpt_bumps.is_empty() && ui.button("Clear Sculpted Bumps").clicked() {
            self.sculpt_bumps.clear();
            self.reset_arena();
        }
    }

    fn draw_racers_and_camera_ui(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(format!("Arena Step: {}", self.arena.step_count));
        for (idx, kind) in RacerKind::ALL.iter().copied().enumerate() {
            if !kind.is_available_in_build() {
                continue;
            }
            let tr = &self.arena.racers[idx];
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.arena_config.enabled_racers[idx], "");
                ui.colored_label(
                    kind.color(&palette),
                    format!(
                        "{}: L={:.3} H={:.3}",
                        kind.label(),
                        tr.potential_energy,
                        tr.potential_energy + tr.kinetic_energy
                    ),
                );
            });
        }
        egui::CollapsingHeader::new("Camera 3D")
            .default_open(false)
            .show(ui, |ui| self.camera.ui(ui));
    }

    pub(super) fn handle_viewport_gesture(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if self.interaction_mode == CanvasInteractionMode::Orbit {
            (self.slingshot_anchor, self.slingshot_current, self.last_brush_wb) =
                (None, None, None);
            return;
        }
        let pointer_pos = response.interact_pointer_pos();
        if self.interaction_mode == CanvasInteractionMode::SculptBrush {
            if response.dragged_by(egui::PointerButton::Primary) || response.clicked() {
                let ctx = InteractionContext {
                    pointer_pos,
                    delta: response.drag_delta(),
                    is_dragging: response.dragged(),
                    is_clicked: response.clicked(),
                    response,
                    input_mode: InputMode::Mouse,
                    multi_touch: ui.input(|i| i.multi_touch()),
                    keys_down: ui.input(|i| i.keys_down.clone()),
                    modifiers: ui.input(|i| i.modifiers),
                };
                self.on_brush(&ctx);
            } else {
                self.last_brush_wb = None;
            }
            return;
        }
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = pointer_pos {
                let wb = self.pointer_to_wb(response.rect, pos);
                (self.slingshot_anchor, self.slingshot_current) = (Some(wb), Some(wb));
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = pointer_pos {
                self.slingshot_current = Some(self.pointer_to_wb(response.rect, pos));
            }
        } else if response.drag_stopped_by(egui::PointerButton::Primary) {
            if let (Some(a), Some(c)) = (self.slingshot_anchor.take(), self.slingshot_current.take())
            {
                let vel = [(a[0] - c[0]) * 2.5, (a[1] - c[1]) * 2.5];
                let bounds = self.domain_bounds();
                self.arena.launch_slingshot(
                    &mut self.arena_config,
                    a,
                    vel,
                    self.surface,
                    &self.sculpt_bumps,
                    bounds,
                );
            }
        }
    }

    pub(super) fn show_3d_arena(&mut self, ui: &mut egui::Ui, grid: &SurfaceGrid) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let suppress_orbit = self.interaction_mode != CanvasInteractionMode::Orbit;
        let saved_camera = self.camera;
        let helper = match self.interaction_mode {
            CanvasInteractionMode::Orbit => CAMERA_3D_HELPER_TEXT,
            CanvasInteractionMode::Slingshot => {
                "Slingshot: drag on surface to pull back and launch active particles (orbit locked)."
            }
            CanvasInteractionMode::SculptBrush => {
                "Sculpt: drag to raise barriers (+A) or Alt-drag to carve wells (-A) (orbit locked)."
            }
        };

        let mut cam = self.camera;
        let plot_resp = Plot3DViewport::new("loss_landscape_3d_plot", ARENA_3D_HALF_EXTENT)
            .helper_text(helper)
            .show(ui, &mut cam, |plot_ui| {
                let pid = plot_ui.plot_id();
                plot_ui.ctx().data_mut(|d| {
                    d.insert_temp(pid.with("dragging_callout"), suppress_orbit);
                });
                for (t, line_pts) in grid.wireframe_polylines(&saved_camera) {
                    let color = lerp_color(palette.axis_z, palette.axis_x, t);
                    plot_ui.line(Line::new("Surface Mesh", line_pts).color(color).width(1.1_f32));
                }
                for &[mw, mb] in self.surface.global_minima() {
                    let mloss = self.surface.evaluate(&self.sculpt_bumps, mw, mb);
                    let p2 = saved_camera.project(&grid.world_point(mw, mb, mloss, 0.03));
                    plot_ui.points(
                        Points::new("Global Minimum", vec![p2])
                            .color(palette.success)
                            .radius(5.5_f32)
                            .shape(MarkerShape::Diamond),
                    );
                }
                self.draw_3d_racers(plot_ui, grid, &saved_camera, &palette);
                if let (Some(a), Some(c)) = (self.slingshot_anchor, self.slingshot_current) {
                    let la = self.surface.evaluate(&self.sculpt_bumps, a[0], a[1]);
                    let lc = self.surface.evaluate(&self.sculpt_bumps, c[0], c[1]);
                    let pa = saved_camera.project(&grid.world_point(a[0], a[1], la, 0.05));
                    let pc = saved_camera.project(&grid.world_point(c[0], c[1], lc, 0.05));
                    plot_ui.line(
                        Line::new("Slingshot Vector", vec![pa, pc])
                            .color(palette.warning)
                            .width(2.5_f32),
                    );
                }
            });
        if !suppress_orbit {
            self.camera = cam;
        }
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }

    pub(super) fn show_2d_contours(&mut self, ui: &mut egui::Ui, grid: &SurfaceGrid) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let points = grid.heatmap_points(palette.axis_z, palette.axis_x);
        let contours = grid.contour_segments(8);
        let allow_pan = self.interaction_mode == CanvasInteractionMode::Orbit;

        let plot_resp = Plot::new("loss_landscape_plot")
            .x_axis_label("Weight (w)")
            .y_axis_label("Bias (b)")
            .data_aspect(1.0)
            .allow_drag(allow_pan)
            .show(ui, |plot_ui| {
                for point in points {
                    plot_ui.points(point);
                }
                for (iso, segs) in contours {
                    let c = lerp_color(palette.axis_z, palette.axis_x, iso);
                    for seg in segs {
                        plot_ui.line(
                            Line::new("Iso-Contour", seg.to_vec())
                                .color(c)
                                .width(1.2_f32),
                        );
                    }
                }
                for &[mw, mb] in self.surface.global_minima() {
                    plot_ui.points(
                        Points::new("Global Minima (True Solution)", vec![[mw, mb]])
                            .color(palette.success)
                            .radius(6.0_f32)
                            .shape(MarkerShape::Circle),
                    );
                }
                self.draw_2d_racers(plot_ui, &palette);
            });
        self.handle_viewport_gesture(ui, &plot_resp.response);
    }

    fn draw_3d_racers(
        &self,
        plot_ui: &mut PlotUi<'_>,
        grid: &SurfaceGrid,
        cam: &Camera3D,
        palette: &SemanticPalette,
    ) {
        for (idx, racer) in self.arena.racers.iter().enumerate() {
            if !self.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build() {
                continue;
            }
            let color = racer.kind.color(palette);
            if racer.trail.len() >= 2 {
                let pts: Vec<[f64; 2]> = racer
                    .trail
                    .iter()
                    .map(|[w, b]| {
                        let l = self.surface.evaluate(&self.sculpt_bumps, w, b);
                        cam.project(&grid.world_point(w, b, l, 0.035))
                    })
                    .collect();
                plot_ui.line(Line::new(racer.kind.label(), pts).color(color).width(2.2_f32));
            }
            let head = cam.project(&grid.world_point(
                racer.phase.q.x,
                racer.phase.q.y,
                racer.potential_energy,
                0.045,
            ));
            plot_ui.points(
                Points::new(racer.kind.label(), vec![head])
                    .color(color)
                    .radius(5.0_f32),
            );
        }
    }

    fn draw_2d_racers(&self, plot_ui: &mut PlotUi<'_>, palette: &SemanticPalette) {
        for (idx, racer) in self.arena.racers.iter().enumerate() {
            if !self.arena_config.enabled_racers[idx] || !racer.kind.is_available_in_build() {
                continue;
            }
            let color = racer.kind.color(palette);
            if racer.trail.len() >= 2 {
                let trail: Vec<[f64; 2]> = racer.trail.iter().collect();
                plot_ui.line(Line::new(racer.kind.label(), trail).color(color).width(2.2_f32));
            }
            plot_ui.points(
                Points::new(racer.kind.label(), vec![[racer.phase.q.x, racer.phase.q.y]])
                    .color(color)
                    .radius(5.0_f32),
            );
        }
    }
}
