#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:advanced_linear_algebra]
// [cite:graph_parameters_rust]

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2};
use egui_plot::{ExportPrimitive, Line, Plot, PlotPoints, Points, VLine};
use math_explorer::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast, mapped_gradient,
};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::kdv::{wrap_periodic_coord, KdvEquationMode};

use super::*;
use crate::app::{palette::DARK_PALETTE, SemanticPalette};

impl SolitonCollisionTool {
    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("KdV Soliton & 1D Cubic NLS Studio");
            let avail = ui.available_size();
            if avail.x > 640.0 {
                ui.columns(2, |cols| {
                    self.draw_waterfall_canvas(&mut cols[0]);
                    self.draw_waveform_plot(&mut cols[1]);
                });
            } else {
                self.draw_waterfall_canvas(ui);
                ui.add_space(6.0);
                self.draw_waveform_plot(ui);
            }
        });
    }

    fn draw_waterfall_canvas(&mut self, ui: &mut egui::Ui) {
        self.upload_waterfall_texture(ui.ctx());
        ui.label("Space-Time Waterfall u(x, t)");
        let side = ui
            .available_width()
            .min(ui.available_height() - 36.0)
            .clamp(180.0, 520.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        let rect = response.rect;

        let summary = format!(
            "Preset {}, mode {:?}, nonlinear {}, t = {:.2}, peaks ({:.2}, {:.2})",
            self.state.preset.label(),
            self.state.mode,
            self.state.nonlinear_enabled,
            self.solver.time(),
            self.top_peaks.0.amplitude,
            self.top_peaks.1.amplitude,
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "1D KdV Soliton Space-Time Waterfall Visualization",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);

        self.handle_canvas_input(ui, &response, rect);

        if let Some(tex) = &self.texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), rect, uv, Color32::WHITE);
        }
        let palette = if ui.visuals().dark_mode {
            SemanticPalette::from_visuals(ui.visuals())
        } else {
            DARK_PALETTE
        };
        let norm_x = ((self.state.pulse_x0 / self.state.domain_length) + 0.5).clamp(0.0, 1.0);
        let px = rect.left() + (norm_x as f32) * rect.width();
        painter.line_segment(
            [Pos2::new(px, rect.top()), Pos2::new(px, rect.bottom())],
            Stroke::new(1.5_f32, palette.accent),
        );
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small("Click canvas or press Enter to inject sech² pulse • Left/Right steer pulse x0");
    }

    fn handle_canvas_input(&mut self, ui: &mut egui::Ui, response: &egui::Response, rect: Rect) {
        let ctx = ui.ctx().clone();
        let half_l = (0.5 * self.state.domain_length - 0.5).max(2.0);
        if response.has_focus() {
            let (left, right, pulse) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                )
            });
            if left || right || pulse {
                ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            }
            let step_x = 0.025 * self.state.domain_length;
            let dx = (if right { step_x } else { 0.0 }) - (if left { step_x } else { 0.0 });
            if dx != 0.0 {
                self.state.pulse_x0 = (self.state.pulse_x0 + dx).clamp(-half_l, half_l);
                self.state_rev = self.state_rev.wrapping_add(1);
                crate::accessibility::announce_live_status(
                    &ctx,
                    format!("Pulse cursor moved to x = {:.2}", self.state.pulse_x0),
                );
            }
            if pulse {
                self.solver.inject_soliton_pulse(
                    self.state.pulse_x0,
                    self.state.pulse_speed_c,
                    self.state.mode,
                );
                self.reanchor_invariant_baseline();
                self.push_waterfall_row();
                self.refresh_diagnostics();
                crate::accessibility::announce_live_status(&ctx, "Injected soliton pulse");
            }
        }

        if (response.clicked() || response.dragged()) && rect.width() > 1.0 {
            if let Some(pos) = response.interact_pointer_pos() {
                let nx = (((pos.x - rect.left()) / rect.width()) as f64).clamp(0.02, 0.98);
                self.state.pulse_x0 =
                    ((nx - 0.5) * self.state.domain_length).clamp(-half_l, half_l);
                if response.clicked() {
                    self.solver.inject_soliton_pulse(
                        self.state.pulse_x0,
                        self.state.pulse_speed_c,
                        self.state.mode,
                    );
                    self.reanchor_invariant_baseline();
                    self.push_waterfall_row();
                    self.refresh_diagnostics();
                }
                self.state_rev = self.state_rev.wrapping_add(1);
            }
        }
    }

    fn draw_waveform_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let mode_label = match (self.state.mode, self.state.nonlinear_enabled) {
            (KdvEquationMode::KdvSoliton, true) => "KdV Soliton Wave u(x, t)",
            (KdvEquationMode::KdvSoliton, false) => "Linear Airy Dispersion u(x, t)",
            (KdvEquationMode::CubicNls1D, true) => "1D Cubic NLS Envelope |ψ(x, t)| & Re(ψ)",
            (KdvEquationMode::CubicNls1D, false) => "Linear 1D Schrödinger Envelope |ψ(x, t)|",
        };
        ui.label(mode_label);

        let xs = self.solver.x_coords();
        let field = self.solver.field();
        let live_pts: PlotPoints = xs
            .iter()
            .zip(field.iter())
            .map(|(&x, &u)| [x, if u.is_finite() { u } else { 0.0 }])
            .collect();

        Plot::new("kdv_soliton_waveform_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                if self.state.show_initial_overlay {
                    let init_pts: PlotPoints = xs
                        .iter()
                        .zip(self.solver.initial_field().iter())
                        .map(|(&x, &u0)| [x, if u0.is_finite() { u0 } else { 0.0 }])
                        .collect();
                    plot_ui.line(
                        Line::new("Initial u(x, 0)", init_pts)
                            .color(palette.node_stroke.gamma_multiply(0.55))
                            .width(1.3_f32),
                    );
                }
                plot_ui.line(
                    Line::new("Live Wave u(x, t)", live_pts)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                if self.state.mode == KdvEquationMode::CubicNls1D {
                    let re_pts: PlotPoints = xs
                        .iter()
                        .zip(self.solver.psi_field().iter())
                        .map(|(&x, z)| [x, if z.re.is_finite() { z.re } else { 0.0 }])
                        .collect();
                    plot_ui.line(
                        Line::new("Carrier Re ψ(x, t)", re_pts)
                            .color(palette.series_color(2))
                            .width(1.3_f32),
                    );
                }
                self.overlay_phase_shift_markers(plot_ui, palette);
            });
    }

    fn overlay_phase_shift_markers(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: SemanticPalette,
    ) {
        let (p1, p2) = self.top_peaks;
        plot_ui.points(
            Points::new(
                "Tracked Pulse Peaks",
                vec![[p1.x, p1.amplitude], [p2.x, p2.amplitude]],
            )
            .color(palette.success)
            .radius(4.5_f32),
        );
        if self.state.show_ballistic_markers && self.state.mode == KdvEquationMode::KdvSoliton {
            let t = self.solver.time();
            let len = self.solver.length();
            let b_fast =
                wrap_periodic_coord(self.state.fast_x0 + self.state.fast_speed_c1 * t, len);
            let b_slow =
                wrap_periodic_coord(self.state.slow_x0 + self.state.slow_speed_c2 * t, len);
            plot_ui.vline(VLine::new("Unshifted Fast x1 + c1 t", b_fast).color(palette.warning));
            plot_ui.vline(VLine::new("Unshifted Slow x2 + c2 t", b_slow).color(palette.info));
        }
    }

    fn upload_waterfall_texture(&mut self, ctx: &egui::Context) {
        populate_waterfall_pixels(
            &self.waterfall_ring,
            self.waterfall_head,
            self.solver.field(),
            self.state.colormap,
            &mut self.rgba_pixels,
        );
        let image = ColorImage {
            size: [WATERFALL_WIDTH, WATERFALL_ROWS],
            pixels: self.rgba_pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(image, TextureOptions::LINEAR);
        } else {
            self.texture =
                Some(ctx.load_texture("kdv_waterfall_tex", image, TextureOptions::LINEAR));
        }
    }

    pub(super) fn build_export_color_image(&self) -> ColorImage {
        let mut pixels = vec![Color32::BLACK; WATERFALL_WIDTH * WATERFALL_ROWS];
        populate_waterfall_pixels(
            &self.waterfall_ring,
            self.waterfall_head,
            self.solver.field(),
            self.state.colormap,
            &mut pixels,
        );
        ColorImage {
            size: [WATERFALL_WIDTH, WATERFALL_ROWS],
            pixels,
            ..Default::default()
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let xs = self.solver.x_coords();
        let live: Vec<[f64; 2]> = xs
            .iter()
            .zip(self.solver.field().iter())
            .map(|(&x, &u)| [x, if u.is_finite() { u } else { 0.0 }])
            .collect();
        let init: Vec<[f64; 2]> = xs
            .iter()
            .zip(self.solver.initial_field().iter())
            .map(|(&x, &u0)| [x, if u0.is_finite() { u0 } else { 0.0 }])
            .collect();
        vec![
            ExportPrimitive::Polyline {
                name: "Initial Wave u(x, 0)".into(),
                points: init.into(),
                color: DARK_PALETTE.node_stroke,
                stroke_width: 1.2,
                closed: false,
                fill: None,
            },
            ExportPrimitive::Polyline {
                name: "Live Wave u(x, t)".into(),
                points: live.into(),
                color: DARK_PALETTE.info,
                stroke_width: 2.0,
                closed: false,
                fill: None,
            },
        ]
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let xs = self.solver.x_coords();
        let wave_now: Vec<[f64; 2]> = xs
            .iter()
            .zip(self.solver.field().iter())
            .map(|(&x, &u)| [x, if u.is_finite() { u } else { 0.0 }])
            .collect();
        let wave_init: Vec<[f64; 2]> = xs
            .iter()
            .zip(self.solver.initial_field().iter())
            .map(|(&x, &u0)| [x, if u0.is_finite() { u0 } else { 0.0 }])
            .collect();
        let drift_hist: Vec<[f64; 2]> = self
            .telemetry
            .iter()
            .map(|pt| [pt.time, pt.max_drift])
            .collect();
        let peak_hist: Vec<[f64; 2]> = self
            .telemetry
            .iter()
            .map(|pt| [pt.time, pt.peak_ratio])
            .collect();
        vec![
            ("wave_profile_u_x_t".into(), wave_now),
            ("initial_profile_u_x_0".into(), wave_init),
            ("max_relative_invariant_drift".into(), drift_hist),
            ("normalized_peak_amplitude".into(), peak_hist),
        ]
    }
}

fn populate_waterfall_pixels(
    ring: &[f64],
    head: usize,
    live_field: &[f64],
    colormap: SolitonColormap,
    out_pixels: &mut [Color32],
) {
    let mut max_abs = 1.0e-6_f64;
    for &v in ring.iter().chain(live_field.iter()) {
        if v.is_finite() && v.abs() > max_abs {
            max_abs = v.abs();
        }
    }
    let filled_rows = head.clamp(1, WATERFALL_ROWS);
    for img_row in 0..WATERFALL_ROWS {
        let history_age = (img_row * filled_rows) / WATERFALL_ROWS;
        let ring_row = (head + WATERFALL_ROWS - 1 - history_age) % WATERFALL_ROWS;
        for col in 0..WATERFALL_WIDTH {
            let val = ring[flatten_2d_index(col, ring_row, WATERFALL_WIDTH)];
            let rgb = match colormap {
                SolitonColormap::PerceptualThermal => {
                    let t = (0.15 + 0.85 * (val.max(0.0) / max_abs)).clamp(0.0, 1.0);
                    heatmap_color(t)
                }
                SolitonColormap::BipolarWave => mapped_gradient(val, -max_abs, max_abs),
                SolitonColormap::WcagHighContrast => {
                    let t = (0.5 + 0.5 * (val / max_abs)).clamp(0.0, 1.0);
                    ensure_dark_canvas_contrast(heatmap_color_high_contrast(t))
                }
            };
            let dst = flatten_2d_index(col, img_row, WATERFALL_WIDTH);
            out_pixels[dst] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
    }
}
