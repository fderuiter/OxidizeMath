#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:quantum_mechanics]

use std::f64::consts::PI;

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2};
use egui_plot::{ExportPrimitive, Line, Plot, PlotPoints};
use math_explorer::math_kernel::colormap::{
    cyclic_cosine_palette_high_contrast, ensure_dark_canvas_contrast, heatmap_color,
    heatmap_color_high_contrast,
};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::physics::quantum::gross_pitaevskii::GrossPitaevskiiSolver2D;

use super::*;
use crate::app::{palette::DARK_PALETTE, SemanticPalette};

impl SuperfluidVortexTool {
    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Superfluid Condensate & Quantized Vortex Studio");
            let avail = ui.available_size();
            if avail.x > 640.0 {
                ui.columns(2, |cols| {
                    self.draw_condensate_canvas(&mut cols[0]);
                    self.draw_cross_section_plot(&mut cols[1]);
                });
            } else {
                self.draw_condensate_canvas(ui);
                ui.add_space(6.0);
                self.draw_cross_section_plot(ui);
            }
        });
    }

    fn draw_condensate_canvas(&mut self, ui: &mut egui::Ui) {
        self.upload_condensate_texture(ui.ctx());
        ui.label(self.state.view_mode.label());
        let side = ui
            .available_width()
            .min(ui.available_height() - 36.0)
            .clamp(180.0, 540.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        self.focused_canvas = response.has_focus().then_some(response.id);
        let rect = response.rect;

        let summary = format!(
            "Preset {}, view {}, +1 vortices {}, -1 vortices {}, net winding Q = {:+}",
            self.state.preset.label(),
            self.state.view_mode.label(),
            self.charge_summary.positive_count,
            self.charge_summary.negative_count,
            self.charge_summary.net_charge,
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "2D Superfluid Condensate & Quantized Vortex Canvas",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);

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
        self.paint_vortex_and_cursor_overlays(&painter, rect, palette);
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small(
            "Click to stamp +2π vortex • Shift+Click for -2π • Arrows move cursor • Enter stamps",
        );
    }

    fn handle_canvas_input(&mut self, ui: &mut egui::Ui, response: &egui::Response, rect: Rect) {
        let ctx = ui.ctx().clone();
        if response.has_focus() {
            let (left, right, down, up, stamp) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                )
            });
            if left || right || down || up || stamp {
                ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            }
            let dx = (if right { 0.35 } else { 0.0 }) - (if left { 0.35 } else { 0.0 });
            let dy = (if up { 0.35 } else { 0.0 }) - (if down { 0.35 } else { 0.0 });
            if dx != 0.0 || dy != 0.0 {
                self.state.cursor_x = (self.state.cursor_x + dx).clamp(-CURSOR_BOUND, CURSOR_BOUND);
                self.state.cursor_y = (self.state.cursor_y + dy).clamp(-CURSOR_BOUND, CURSOR_BOUND);
                self.state_rev = self.state_rev.wrapping_add(1);
                crate::accessibility::announce_live_status(
                    &ctx,
                    format!(
                        "Vortex cursor moved to ({:.2}, {:.2})",
                        self.state.cursor_x, self.state.cursor_y
                    ),
                );
            }
            if stamp {
                let q = self.state.stamp_charge;
                self.stamp_vortex_at(self.state.cursor_x, self.state.cursor_y, q);
                self.state_rev = self.state_rev.wrapping_add(1);
                crate::accessibility::announce_live_status(
                    &ctx,
                    format!("Imprinted charge {q:+} superfluid vortex"),
                );
            }
        }

        if response.clicked() && rect.width() > 1.0 && rect.height() > 1.0 {
            if let Some(pos) = response.interact_pointer_pos() {
                let nx = (((pos.x - rect.left()) / rect.width()) as f64).clamp(0.05, 0.95);
                let ny = ((1.0 - (pos.y - rect.top()) / rect.height()) as f64).clamp(0.05, 0.95);
                let wx = ((nx - 0.5) * DOMAIN_SPAN).clamp(-CURSOR_BOUND, CURSOR_BOUND);
                let wy = ((ny - 0.5) * DOMAIN_SPAN).clamp(-CURSOR_BOUND, CURSOR_BOUND);
                let shift = ctx.input(|i| i.modifiers.shift);
                let q = if shift { -1 } else { self.state.stamp_charge };
                self.state.cursor_x = wx;
                self.state.cursor_y = wy;
                self.stamp_vortex_at(wx, wy, q);
                self.state_rev = self.state_rev.wrapping_add(1);
            }
        }
    }

    fn paint_vortex_and_cursor_overlays(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        palette: SemanticPalette,
    ) {
        let world_to_screen = |wx: f64, wy: f64| -> Pos2 {
            let nx = ((wx / DOMAIN_SPAN) + 0.5).clamp(0.0, 1.0);
            let ny = ((wy / DOMAIN_SPAN) + 0.5).clamp(0.0, 1.0);
            Pos2::new(
                rect.left() + (nx as f32) * rect.width(),
                rect.bottom() - (ny as f32) * rect.height(),
            )
        };

        if self.state.show_vortex_markers {
            for core in self.solver.vortices() {
                let center = world_to_screen(core.x, core.y);
                let color = if core.charge > 0 {
                    palette.success
                } else {
                    palette.warning
                };
                painter.circle_stroke(center, 6.0_f32, Stroke::new(2.0_f32, color));
                painter.line_segment(
                    [
                        Pos2::new(center.x - 3.5, center.y),
                        Pos2::new(center.x + 3.5, center.y),
                    ],
                    Stroke::new(1.8_f32, color),
                );
                if core.charge > 0 {
                    painter.line_segment(
                        [
                            Pos2::new(center.x, center.y - 3.5),
                            Pos2::new(center.x, center.y + 3.5),
                        ],
                        Stroke::new(1.8_f32, color),
                    );
                }
            }
        }

        let cur = world_to_screen(self.state.cursor_x, self.state.cursor_y);
        painter.circle_stroke(cur, 4.5_f32, Stroke::new(1.4_f32, palette.accent));
    }

    fn draw_cross_section_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.label("Midplane Condensate Density |ψ(x, 0)|² & Phase arg ψ(x, 0)");

        let mid_y = GRID_DIM / 2;
        let xs = self.solver.x_coords();
        let rho = self.solver.density();
        let ph = self.solver.phase();
        let max_rho = rho.iter().copied().fold(1.0e-9_f64, f64::max);

        let rho_pts: PlotPoints = (0..GRID_DIM)
            .map(|x| {
                let v = rho[flatten_2d_index(x, mid_y, GRID_DIM)];
                [xs[x], if v.is_finite() { v } else { 0.0 }]
            })
            .collect();
        let phase_pts: PlotPoints = (0..GRID_DIM)
            .map(|x| {
                let idx = flatten_2d_index(x, mid_y, GRID_DIM);
                let scaled = (ph[idx] / PI) * 0.5 * max_rho + 0.5 * max_rho;
                [xs[x], if scaled.is_finite() { scaled } else { 0.0 }]
            })
            .collect();

        Plot::new("superfluid_gpe_cross_section")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Density |ψ(x, 0)|²", rho_pts)
                        .color(palette.series_color(0))
                        .width(2.2_f32),
                );
                plot_ui.line(
                    Line::new("Scaled Phase θ(x, 0)", phase_pts)
                        .color(palette.series_color(2))
                        .width(1.4_f32),
                );
            });
    }

    fn upload_condensate_texture(&mut self, ctx: &egui::Context) {
        populate_condensate_pixels(&self.solver, self.state.view_mode, &mut self.rgba_pixels);
        let image = ColorImage {
            size: [GRID_DIM, GRID_DIM],
            pixels: self.rgba_pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(image, TextureOptions::LINEAR);
        } else {
            self.texture =
                Some(ctx.load_texture("superfluid_gpe_tex", image, TextureOptions::LINEAR));
        }
    }

    pub(super) fn build_export_color_image(&self) -> ColorImage {
        let mut pixels = vec![Color32::BLACK; GRID_DIM * GRID_DIM];
        populate_condensate_pixels(&self.solver, self.state.view_mode, &mut pixels);
        ColorImage {
            size: [GRID_DIM, GRID_DIM],
            pixels,
            ..Default::default()
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mid_y = GRID_DIM / 2;
        let xs = self.solver.x_coords();
        let rho = self.solver.density();
        let slice_pts: Vec<[f64; 2]> = (0..GRID_DIM)
            .map(|x| {
                let v = rho[flatten_2d_index(x, mid_y, GRID_DIM)];
                [xs[x], if v.is_finite() { v } else { 0.0 }]
            })
            .collect();
        let mut prims = vec![ExportPrimitive::Polyline {
            name: "Midplane Density |psi(x, 0)|^2".into(),
            points: slice_pts.into(),
            color: DARK_PALETTE.info,
            stroke_width: 2.0,
            closed: false,
            fill: None,
        }];

        let pos_cores: Vec<[f64; 2]> = self
            .solver
            .vortices()
            .iter()
            .filter(|v| v.charge > 0 && v.x.is_finite() && v.y.is_finite())
            .map(|v| [v.x, v.y])
            .collect();
        if !pos_cores.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Positive +2pi Vortices".into(),
                points: pos_cores.into(),
                color: DARK_PALETTE.success,
                radius: 3.5,
            });
        }
        let neg_cores: Vec<[f64; 2]> = self
            .solver
            .vortices()
            .iter()
            .filter(|v| v.charge < 0 && v.x.is_finite() && v.y.is_finite())
            .map(|v| [v.x, v.y])
            .collect();
        if !neg_cores.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Negative -2pi Antivortices".into(),
                points: neg_cores.into(),
                color: DARK_PALETTE.warning,
                radius: 3.5,
            });
        }
        prims
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mid_y = GRID_DIM / 2;
        let xs = self.solver.x_coords();
        let rho = self.solver.density();
        let ph = self.solver.phase();
        let density_slice: Vec<[f64; 2]> = (0..GRID_DIM)
            .map(|x| {
                let v = rho[flatten_2d_index(x, mid_y, GRID_DIM)];
                [xs[x], if v.is_finite() { v } else { 0.0 }]
            })
            .collect();
        let phase_slice: Vec<[f64; 2]> = (0..GRID_DIM)
            .map(|x| {
                let v = ph[flatten_2d_index(x, mid_y, GRID_DIM)];
                [xs[x], if v.is_finite() { v } else { 0.0 }]
            })
            .collect();
        let cores_xy: Vec<[f64; 2]> = self.solver.vortices().iter().map(|c| [c.x, c.y]).collect();
        let norm_err_hist: Vec<[f64; 2]> = self
            .telemetry
            .iter()
            .map(|pt| [pt.time, pt.norm_error])
            .collect();
        let energy_hist: Vec<[f64; 2]> = self
            .telemetry
            .iter()
            .map(|pt| [pt.time, pt.energy])
            .collect();
        vec![
            ("centerline_density_x".into(), density_slice),
            ("centerline_phase_x".into(), phase_slice),
            ("detected_vortex_cores_xy".into(), cores_xy),
            ("norm_conservation_error".into(), norm_err_hist),
            ("gpe_energy_history".into(), energy_hist),
        ]
    }
}

fn populate_condensate_pixels(
    solver: &GrossPitaevskiiSolver2D,
    view_mode: CondensateViewMode,
    out_pixels: &mut [Color32],
) {
    let w = solver.width();
    let h = solver.height();
    let rho = solver.density();
    let phase = solver.phase();
    let mut max_rho = 1.0e-12_f64;
    for &r in rho {
        if r.is_finite() && r > max_rho {
            max_rho = r;
        }
    }

    for grid_y in 0..h {
        let img_row = (h - 1) - grid_y;
        for grid_x in 0..w {
            let src = flatten_2d_index(grid_x, grid_y, w);
            let dst = flatten_2d_index(grid_x, img_row, w);
            let r_norm = (rho[src] / max_rho).clamp(0.0, 1.0);
            if r_norm < 1.5e-3 {
                out_pixels[dst] = DARK_PALETTE.surface_fill;
                continue;
            }
            let rgb = match view_mode {
                CondensateViewMode::SuperfluidDensity => heatmap_color(r_norm.sqrt()),
                CondensateViewMode::CyclicPhase => {
                    let cyc_arg = (phase[src] + PI) / 0.15;
                    cyclic_cosine_palette_high_contrast(cyc_arg)
                }
                CondensateViewMode::WcagHighContrastDensity => {
                    ensure_dark_canvas_contrast(heatmap_color_high_contrast(r_norm.sqrt()))
                }
            };
            out_pixels[dst] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
    }
}
