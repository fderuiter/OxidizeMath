#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, WCAG 2D optical bench canvas, and Fresnel/caustic plots for [`CausticsRayBenchTool`].

// [cite:optics]

use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, Vec2};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use math_explorer::physics::optics::{
    brewster_angle, coffee_cup_cusp_x, critical_angle, fresnel_polarized, nephroid_caustic_point,
    GlassMaterial, PolarizationMode,
};

use super::presets::{OpticsColormap, RayBenchSceneChoice, RaySecondaryPlot};
use super::ray_bench_tool::{CausticsRayBenchTool, BENCH_EXTENT};
use crate::app::{palette::DARK_PALETTE, SemanticPalette};
use crate::framework::bounded_drag_value;
use crate::reflective_ui::render_copyable_metric;

impl CausticsRayBenchTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "optics_ray_bench_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Optical Bench Controls");
                ui.separator();
                let mut dirty = self.draw_scene_and_medium_selectors(ui);
                ui.separator();
                dirty |= self.draw_beam_and_element_drags(ui);
                ui.separator();
                dirty |= self.draw_visual_toggles(ui);
                if dirty {
                    self.revision = self.revision.wrapping_add(1).max(1);
                    self.recompute();
                }
                ui.separator();
                self.draw_optical_readouts(ui);
            });
        });
    }

    fn draw_scene_and_medium_selectors(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.label("Optical Bench Scene:");
        let prev_scene = self.state.scene;
        egui::ComboBox::from_id_salt("optics_bench_scene")
            .selected_text(self.state.scene.label())
            .show_ui(ui, |ui| {
                for sc in [
                    RayBenchSceneChoice::CoffeeCupNephroid,
                    RayBenchSceneChoice::SphericalBiconvexLens,
                    RayBenchSceneChoice::LuneburgMetamaterial,
                    RayBenchSceneChoice::MaxwellFisheye,
                    RayBenchSceneChoice::EatonRetroreflector,
                    RayBenchSceneChoice::ParabolicGrinFiber,
                    RayBenchSceneChoice::DispersivePrism,
                    RayBenchSceneChoice::PlanarFresnelInterface,
                ] {
                    ui.selectable_value(&mut self.state.scene, sc, sc.label());
                }
            });
        dirty |= self.state.scene != prev_scene;

        ui.label("Glass Dispersion Catalog:");
        let prev_glass = self.state.glass;
        egui::ComboBox::from_id_salt("optics_glass_material")
            .selected_text(self.state.glass.label())
            .show_ui(ui, |ui| {
                for g in [
                    GlassMaterial::Bk7Crown,
                    GlassMaterial::Sf11Flint,
                    GlassMaterial::FusedSilica,
                    GlassMaterial::Diamond,
                    GlassMaterial::Water,
                ] {
                    ui.selectable_value(&mut self.state.glass, g, g.label());
                }
            });
        dirty |= self.state.glass != prev_glass;

        ui.label("Polarization State:");
        let prev_pol = self.state.polarization;
        egui::ComboBox::from_id_salt("optics_polarization")
            .selected_text(self.state.polarization.label())
            .show_ui(ui, |ui| {
                for p in [
                    PolarizationMode::Unpolarized,
                    PolarizationMode::SPolarized,
                    PolarizationMode::PPolarized,
                ] {
                    ui.selectable_value(&mut self.state.polarization, p, p.label());
                }
            });
        dirty | (self.state.polarization != prev_pol)
    }

    fn draw_beam_and_element_drags(&mut self, ui: &mut egui::Ui) -> bool {
        self.draw_beam_launch_drags(ui) | self.draw_element_geometry_drags(ui)
    }

    fn draw_beam_launch_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.horizontal(|ui| {
            ui.label("Rays:");
            dirty |= ui
                .add(egui::DragValue::new(&mut self.state.ray_count).range(3..=61))
                .changed();
            ui.label("Aperture:");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.beam_half_width,
                    0.02..=2.2,
                    0.02,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Source (x, y):");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.source_x,
                    -2.5..=-0.3,
                    0.02,
                ))
                .changed();
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.source_y,
                    -1.6..=1.6,
                    0.02,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Angle θ₀ (rad):");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.launch_angle_rad,
                    -1.2..=1.2,
                    0.01,
                ))
                .changed();
            ui.label("λ (nm):");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.wavelength_nm,
                    380.0..=720.0,
                    2.0,
                ))
                .changed();
        });
        dirty
    }

    fn draw_element_geometry_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.horizontal(|ui| {
            ui.label("Scale R:");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.element_scale,
                    0.4..=2.4,
                    0.02,
                ))
                .changed();
            ui.label("Detector x:");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.detector_x,
                    -2.2..=2.4,
                    0.02,
                ))
                .changed();
        });
        if self.state.scene == RayBenchSceneChoice::DispersivePrism {
            ui.horizontal(|ui| {
                ui.label("Prism Apex (°):");
                dirty |= ui
                    .add(bounded_drag_value(
                        &mut self.state.prism_apex_deg,
                        25.0..=75.0,
                        0.5,
                    ))
                    .changed();
            });
        }
        if self.state.scene == RayBenchSceneChoice::PlanarFresnelInterface {
            ui.horizontal(|ui| {
                ui.label("Transmitted n₂:");
                dirty |= ui
                    .add(bounded_drag_value(
                        &mut self.state.interface_n2,
                        1.0..=2.5,
                        0.01,
                    ))
                    .changed();
            });
        }
        dirty
    }

    fn draw_visual_toggles(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        dirty |= ui
            .checkbox(
                &mut self.state.polychromatic,
                "Polychromatic Dispersion Fan",
            )
            .changed();
        dirty |= ui
            .checkbox(
                &mut self.state.show_caustic_envelope,
                "Show Caustic Envelope",
            )
            .changed();
        ui.label("Colormap:");
        let prev_cm = self.state.colormap;
        egui::ComboBox::from_id_salt("optics_bench_colormap")
            .selected_text(self.state.colormap.label())
            .show_ui(ui, |ui| {
                for cm in [
                    OpticsColormap::PerceptualThermal,
                    OpticsColormap::SpectralDispersion,
                    OpticsColormap::WcagHighContrast,
                ] {
                    ui.selectable_value(&mut self.state.colormap, cm, cm.label());
                }
            });
        dirty |= self.state.colormap != prev_cm;

        ui.label("Secondary Plot:");
        let prev_sp = self.state.secondary_plot;
        egui::ComboBox::from_id_salt("optics_secondary_plot")
            .selected_text(self.state.secondary_plot.label())
            .show_ui(ui, |ui| {
                for sp in [
                    RaySecondaryPlot::FresnelPolarization,
                    RaySecondaryPlot::DetectorCausticHistogram,
                ] {
                    ui.selectable_value(&mut self.state.secondary_plot, sp, sp.label());
                }
            });
        dirty | (self.state.secondary_plot != prev_sp)
    }

    fn draw_optical_readouts(&self, ui: &mut egui::Ui) {
        let (n1, n2) = self.interface_indices();
        let tb = brewster_angle(n1, n2).to_degrees();
        render_copyable_metric(ui, "Brewster θ_B", &format!("{tb:.2}°"));
        if let Some(tc) = critical_angle(n1, n2) {
            render_copyable_metric(ui, "Critical θ_c", &format!("{:.2}°", tc.to_degrees()));
        }
        render_copyable_metric(ui, "Refractive n(λ)", &format!("{n2:.4}"));
        render_copyable_metric(
            ui,
            "Caustic Points",
            &format!("{}", self.caustics.points().len()),
        );
    }

    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("2D Geometric Ray Bench & Caustic Envelope");
            if ui.available_width() > 640.0 {
                ui.columns(2, |cols| {
                    self.draw_bench_canvas(&mut cols[0]);
                    self.draw_secondary_plot(&mut cols[1]);
                });
            } else {
                self.draw_bench_canvas(ui);
                ui.add_space(6.0);
                self.draw_secondary_plot(ui);
            }
        });
    }

    fn draw_bench_canvas(&mut self, ui: &mut egui::Ui) {
        let side = ui
            .available_width()
            .min(ui.available_height() - 28.0)
            .clamp(180.0, 540.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        let rect = response.rect;
        let summary = format!(
            "Scene {}, {} traced rays, {} caustic points, launch {:.2} rad",
            self.state.scene.label(),
            self.workspace.rays().len(),
            self.caustics.points().len(),
            self.state.launch_angle_rad
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "2D Optical Ray Bench & Caustics Canvas",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        self.handle_bench_input(ui, &response, rect);

        painter.rect_filled(rect, 4.0_f32, Color32::BLACK);
        self.paint_bench_geometry(&painter, rect);
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small(
            "Drag to steer source • Arrows adjust beam y & angle • Enter toggles polychromatic fan",
        );
    }

    fn handle_bench_input(&mut self, ui: &mut egui::Ui, response: &egui::Response, rect: Rect) {
        let ctx = ui.ctx().clone();
        let mut changed = false;
        if response.has_focus() {
            let (left, right, down, up, enter) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                )
            });
            if left || right || down || up || enter {
                ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            }
            let d_ang = (if right { 0.03 } else { 0.0 }) - (if left { 0.03 } else { 0.0 });
            let d_y = (if up { 0.05 } else { 0.0 }) - (if down { 0.05 } else { 0.0 });
            if d_ang != 0.0 || d_y != 0.0 {
                self.state.launch_angle_rad =
                    (self.state.launch_angle_rad + d_ang).clamp(-1.2, 1.2);
                self.state.source_y = (self.state.source_y + d_y).clamp(-1.6, 1.6);
                changed = true;
            }
            if enter {
                self.state.polychromatic = !self.state.polychromatic;
                changed = true;
            }
        }
        if (response.clicked() || response.dragged()) && rect.width() > 1.0 {
            response.request_focus();
            if let Some(pos) = response.interact_pointer_pos() {
                let wy = BENCH_EXTENT * (1.0 - 2.0 * ((pos.y - rect.top()) / rect.height()) as f64);
                self.state.source_y = wy.clamp(-1.6, 1.6);
                changed = true;
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1).max(1);
            self.recompute();
        }
    }

    fn paint_bench_geometry(&self, painter: &egui::Painter, rect: Rect) {
        let to_scr = |wx: f64, wy: f64| -> Pos2 {
            let nx = ((wx + BENCH_EXTENT) / (2.0 * BENCH_EXTENT)) as f32;
            let ny = ((BENCH_EXTENT - wy) / (2.0 * BENCH_EXTENT)) as f32;
            Pos2::new(
                rect.left() + nx * rect.width(),
                rect.top() + ny * rect.height(),
            )
        };
        self.for_each_element_segment(|a, b| {
            painter.line_segment(
                [to_scr(a[0], a[1]), to_scr(b[0], b[1])],
                Stroke::new(1.6_f32, DARK_PALETTE.node_stroke),
            );
        });
        painter.line_segment(
            [
                to_scr(self.state.detector_x, -2.4),
                to_scr(self.state.detector_x, 2.4),
            ],
            Stroke::new(1.3_f32, DARK_PALETTE.success),
        );
        for (idx, meta) in self.workspace.rays().iter().enumerate() {
            let col = self.state.colormap.ray_stroke_color(meta.wavelength_nm);
            for seg in self.workspace.ray_vertices(idx).windows(2) {
                painter.line_segment(
                    [to_scr(seg[0][0], seg[0][1]), to_scr(seg[1][0], seg[1][1])],
                    Stroke::new(
                        1.25_f32,
                        col.gamma_multiply((0.35 + 0.65 * meta.power) as f32),
                    ),
                );
            }
        }
        if self.state.show_caustic_envelope {
            for cp in self.caustics.points() {
                painter.circle_filled(
                    to_scr(cp.position[0], cp.position[1]),
                    2.4_f32,
                    DARK_PALETTE.warning,
                );
            }
            if self.state.scene == RayBenchSceneChoice::CoffeeCupNephroid {
                let cusp = coffee_cup_cusp_x(self.state.element_scale);
                painter.circle_stroke(
                    to_scr(cusp, 0.0),
                    4.5_f32,
                    Stroke::new(1.6_f32, DARK_PALETTE.accent),
                );
            }
        }
    }

    fn draw_secondary_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        match self.state.secondary_plot {
            RaySecondaryPlot::FresnelPolarization => self.draw_fresnel_plot(ui, palette),
            RaySecondaryPlot::DetectorCausticHistogram => self.draw_histogram_plot(ui, palette),
        }
    }

    fn draw_fresnel_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        let (n1, n2) = self.interface_indices();
        ui.label(format!(
            "Polarized Fresnel Reflectance (n₁ = {n1:.3}, n₂ = {n2:.3})"
        ));
        let sample_fresnel = |pick_s: bool| -> PlotPoints {
            (0..=180)
                .map(|k| {
                    let deg = 0.5 * (k as f64).min(179.5);
                    let f = fresnel_polarized(n1, n2, deg.to_radians());
                    [
                        deg,
                        if pick_s {
                            f.reflectance_s
                        } else {
                            f.reflectance_p
                        },
                    ]
                })
                .collect()
        };
        let rs_pts = sample_fresnel(true);
        let rp_pts = sample_fresnel(false);
        let tb_deg = brewster_angle(n1, n2).to_degrees();
        let tc_deg = critical_angle(n1, n2).map(f64::to_degrees);
        Plot::new("optics_fresnel_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("R_s (TE)", rs_pts)
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                plot_ui.line(
                    Line::new("R_p (TM)", rp_pts)
                        .color(palette.series_color(1))
                        .width(2.0_f32),
                );
                plot_ui.vline(VLine::new("Brewster θ_B", tb_deg).color(palette.warning));
                if let Some(tc) = tc_deg {
                    plot_ui.vline(VLine::new("Critical θ_c", tc).color(palette.accent));
                }
            });
    }

    fn draw_histogram_plot(&self, ui: &mut egui::Ui, palette: SemanticPalette) {
        ui.label(format!(
            "Detector Intensity Histogram I(y) at x = {:.2}",
            self.state.detector_x
        ));
        let pts: PlotPoints = self
            .caustics
            .detector_bins()
            .iter()
            .enumerate()
            .map(|(i, &v)| [self.caustics.bin_center_y(i), v])
            .collect();
        Plot::new("optics_caustic_hist_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Detector I(y)", pts)
                        .color(palette.series_color(2))
                        .width(2.0_f32),
                );
                if self.state.scene == RayBenchSceneChoice::CoffeeCupNephroid {
                    let nephroid: PlotPoints = (-30..=30)
                        .map(|k| {
                            nephroid_caustic_point(self.state.element_scale, (k as f64) * 0.045)
                        })
                        .collect();
                    plot_ui.line(Line::new("Analytic Nephroid", nephroid).color(palette.warning));
                }
            });
    }
}
