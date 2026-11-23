#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, 2D far-field screen canvas, and calibrated profile plot for [`FraunhoferDiffractionTool`].

// [cite:optics]

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, TextureOptions, Vec2};
use egui_plot::{Line, Plot, PlotPoints, Points, VLine};
use math_explorer::physics::optics::{CriticalPointKind, AIRY_FIRST_ZERO_J1};

use super::diffraction_tool::{FraunhoferDiffractionTool, SCREEN_SIDE};
use super::presets::{ApertureChoice, OpticsColormap};
use crate::app::SemanticPalette;
use crate::framework::bounded_drag_value;
use crate::reflective_ui::render_copyable_metric;

impl FraunhoferDiffractionTool {
    pub(super) fn draw_controls_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "optics_diffraction_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Diffraction Controls");
                ui.separator();
                let mut dirty = self.draw_aperture_and_palette_selectors(ui);
                ui.separator();
                dirty |= self.draw_geometry_drags(ui);
                ui.separator();
                dirty |= ui
                    .checkbox(
                        &mut self.state.log_intensity_view,
                        "Log Contrast Screen Boost",
                    )
                    .changed();
                dirty |= ui
                    .checkbox(
                        &mut self.state.show_critical_markers,
                        "Show Critical-Point Markers",
                    )
                    .changed();
                if dirty {
                    self.revision = self.revision.wrapping_add(1).max(1);
                    self.recompute();
                }
                ui.separator();
                self.draw_diffraction_readouts(ui);
            });
        });
    }

    fn draw_aperture_and_palette_selectors(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.label("Aperture Geometry:");
        let prev_ap = self.state.aperture;
        egui::ComboBox::from_id_salt("fraunhofer_aperture")
            .selected_text(self.state.aperture.label())
            .show_ui(ui, |ui| {
                for ap in [
                    ApertureChoice::SingleSlit,
                    ApertureChoice::MultiSlitGrating,
                    ApertureChoice::CircularAiry,
                ] {
                    ui.selectable_value(&mut self.state.aperture, ap, ap.label());
                }
            });
        dirty |= self.state.aperture != prev_ap;

        ui.label("Screen Colormap:");
        let prev_cm = self.state.colormap;
        egui::ComboBox::from_id_salt("fraunhofer_colormap")
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
        dirty | (self.state.colormap != prev_cm)
    }

    fn draw_geometry_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        ui.horizontal(|ui| {
            ui.label("Wavelength λ (nm):");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.wavelength_nm,
                    380.0..=720.0,
                    2.0,
                ))
                .changed();
        });
        dirty |= self.draw_aperture_specific_drags(ui);
        ui.horizontal(|ui| {
            ui.label("Screen L (m):");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.screen_distance_m,
                    0.1..=5.0,
                    0.05,
                ))
                .changed();
            ui.label("Span θ (rad):");
            dirty |= ui
                .add(bounded_drag_value(
                    &mut self.state.half_angle_rad,
                    0.01..=0.60,
                    0.005,
                ))
                .changed();
        });
        dirty
    }

    fn draw_aperture_specific_drags(&mut self, ui: &mut egui::Ui) -> bool {
        let mut dirty = false;
        match self.state.aperture {
            ApertureChoice::SingleSlit => {
                ui.horizontal(|ui| {
                    ui.label("Slit Width a (μm):");
                    dirty |= ui
                        .add(bounded_drag_value(
                            &mut self.state.slit_width_um,
                            0.5..=40.0,
                            0.2,
                        ))
                        .changed();
                });
            }
            ApertureChoice::MultiSlitGrating => {
                ui.horizontal(|ui| {
                    ui.label("Width a (μm):");
                    dirty |= ui
                        .add(bounded_drag_value(
                            &mut self.state.slit_width_um,
                            0.5..=40.0,
                            0.2,
                        ))
                        .changed();
                    ui.label("Pitch d (μm):");
                    dirty |= ui
                        .add(bounded_drag_value(
                            &mut self.state.slit_pitch_um,
                            1.0..=80.0,
                            0.2,
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Slits N:");
                    dirty |= ui
                        .add(egui::DragValue::new(&mut self.state.slit_count).range(2..=16))
                        .changed();
                });
            }
            ApertureChoice::CircularAiry => {
                ui.horizontal(|ui| {
                    ui.label("Diameter D (μm):");
                    dirty |= ui
                        .add(bounded_drag_value(
                            &mut self.state.airy_diameter_um,
                            1.0..=60.0,
                            0.5,
                        ))
                        .changed();
                });
            }
        }
        dirty
    }

    fn draw_diffraction_readouts(&self, ui: &mut egui::Ui) {
        let lam_um = self.state.wavelength_nm * 1.0e-3;
        match self.state.aperture {
            ApertureChoice::CircularAiry => {
                let sin_r1 = (AIRY_FIRST_ZERO_J1 * lam_um)
                    / (std::f64::consts::PI * self.state.airy_diameter_um);
                let r1_mm =
                    self.state.screen_distance_m * 1_000.0 * sin_r1.clamp(0.0, 0.99).asin().tan();
                render_copyable_metric(ui, "Airy 1st Dark Ring", &format!("{r1_mm:.3} mm"));
            }
            ApertureChoice::MultiSlitGrating => {
                let resolvance = self.state.slit_count;
                render_copyable_metric(
                    ui,
                    "Chromatic Resolvance R (m=1)",
                    &format!("{resolvance}"),
                );
            }
            ApertureChoice::SingleSlit => {
                let sin_m1 = (lam_um / self.state.slit_width_um).clamp(0.0, 0.99);
                let y1_mm = self.state.screen_distance_m * 1_000.0 * sin_m1.asin().tan();
                render_copyable_metric(ui, "1st Dark Minimum", &format!("±{y1_mm:.3} mm"));
            }
        }
        render_copyable_metric(
            ui,
            "Critical Markers",
            &format!("{}", self.profile.markers().len()),
        );
    }

    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Fraunhofer Far-Field Diffraction Screen & Cross-Section");
            if ui.available_width() > 640.0 {
                ui.columns(2, |cols| {
                    self.draw_screen_canvas(&mut cols[0]);
                    self.draw_profile_plot(&mut cols[1]);
                });
            } else {
                self.draw_screen_canvas(ui);
                ui.add_space(6.0);
                self.draw_profile_plot(ui);
            }
        });
    }

    fn upload_screen_texture(&mut self, ctx: &egui::Context) {
        let img = ColorImage {
            size: [SCREEN_SIDE, SCREEN_SIDE],
            pixels: self.screen_pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(img, TextureOptions::LINEAR);
        } else {
            self.texture =
                Some(ctx.load_texture("optics_fraunhofer_screen", img, TextureOptions::LINEAR));
        }
    }

    fn draw_screen_canvas(&mut self, ui: &mut egui::Ui) {
        self.upload_screen_texture(ui.ctx());
        let side = ui
            .available_width()
            .min(ui.available_height() - 28.0)
            .clamp(180.0, 520.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        let summary = format!(
            "{} at {:.0} nm, {} critical markers",
            self.state.aperture.label(),
            self.state.wavelength_nm,
            self.profile.markers().len()
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Fraunhofer 2D Far-Field Diffraction Screen",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        if response.clicked() {
            response.request_focus();
        }
        self.focused_canvas = response.has_focus().then_some(response.id);
        self.handle_screen_keyboard(ui, &response);

        if let Some(tex) = &self.texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), response.rect, uv, Color32::WHITE);
        }
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small("Arrows scrub wavelength λ & angular span • Enter toggles log contrast boost");
    }

    fn handle_screen_keyboard(&mut self, ui: &mut egui::Ui, response: &egui::Response) {
        if !response.has_focus() {
            return;
        }
        let ctx = ui.ctx().clone();
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
            let d_wl = (if right { 5.0 } else { 0.0 }) - (if left { 5.0 } else { 0.0 });
            let d_ang = (if up { 0.01 } else { 0.0 }) - (if down { 0.01 } else { 0.0 });
            self.state.wavelength_nm = (self.state.wavelength_nm + d_wl).clamp(380.0, 720.0);
            self.state.half_angle_rad = (self.state.half_angle_rad + d_ang).clamp(0.01, 0.60);
            if enter {
                self.state.log_intensity_view = !self.state.log_intensity_view;
            }
            self.revision = self.revision.wrapping_add(1).max(1);
            self.recompute();
        }
    }

    fn draw_profile_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.label("Normalized Far-Field Irradiance I(y) / I₀ vs Screen Position y (mm)");
        let line_pts: PlotPoints = self
            .profile
            .screen_y_mm()
            .iter()
            .copied()
            .zip(self.profile.intensities().iter().copied())
            .map(|(y, val)| [y, val])
            .collect();
        Plot::new("optics_fraunhofer_profile_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("I(y) / I₀", line_pts)
                        .color(palette.series_color(0))
                        .width(2.0_f32),
                );
                if self.state.show_critical_markers {
                    let maxima: Vec<[f64; 2]> = self
                        .profile
                        .markers()
                        .iter()
                        .filter(|m| m.kind == CriticalPointKind::PrincipalMaximum)
                        .map(|m| [m.screen_y_mm, m.intensity])
                        .collect();
                    if !maxima.is_empty() {
                        plot_ui.points(
                            Points::new("Principal Maxima", maxima)
                                .color(palette.success)
                                .radius(4.0_f32),
                        );
                    }
                    for m in self.profile.markers() {
                        if m.kind != CriticalPointKind::PrincipalMaximum && m.order.abs() <= 2 {
                            plot_ui.vline(
                                VLine::new(format!("Zero m={}", m.order), m.screen_y_mm)
                                    .color(palette.warning),
                            );
                        }
                    }
                }
            });
    }
}
