#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 2D lattice crystal renderer, AccessKit keyboard steering, log-log `FitModel::PowerLaw` plot,
//! and raster/vector/CSV export builders for `AbelianSandpileStudio`.

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2};
use egui_plot::{ExportPrimitive, Line, Plot, PlotPoints, Points};
use math_explorer::fit::FitModel;
use math_explorer::math_kernel::colormap::{
    cyclic_cosine_palette_high_contrast, ensure_dark_canvas_contrast, heatmap_color,
    heatmap_color_high_contrast,
};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::physics::stat_mech::sandpile::{
    AvalancheObservable, SandpileGrid, CRITICAL_HEIGHT, MAX_STABLE_HEIGHT,
};

use super::{AbelianSandpileStudio, SandpileColormap, SandpileViewMode};
use crate::app::{palette::DARK_PALETTE, SemanticPalette};

impl AbelianSandpileStudio {
    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Abelian Sandpile Crystal & 1/f Avalanche Scaling");
            let avail = ui.available_size();
            if avail.x > 620.0 {
                ui.columns(2, |cols| {
                    self.draw_sandpile_crystal_canvas(&mut cols[0]);
                    self.draw_power_law_scaling_plot(&mut cols[1]);
                });
            } else {
                self.draw_sandpile_crystal_canvas(ui);
                ui.add_space(6.0);
                self.draw_power_law_scaling_plot(ui);
            }
        });
    }

    fn draw_sandpile_crystal_canvas(&mut self, ui: &mut egui::Ui) {
        self.upload_crystal_texture(ui.ctx());
        let side = ui
            .available_width()
            .min(ui.available_height() - 32.0)
            .clamp(180.0, 560.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        let rect = response.rect;

        let summary = format!(
            "{}×{} lattice, preset {}, cursor ({}, {}), avalanches {}, recurrent {}",
            self.state.grid_size,
            self.state.grid_size,
            self.state.preset.label(),
            self.state.cursor_x,
            self.state.cursor_y,
            self.grid.telemetry().total_avalanches(),
            self.burning_result.is_recurrent
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "2D Abelian Sandpile Lattice & Recurrent Crystal",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);

        self.handle_canvas_interactions(ui, &response, rect);

        if let Some(tex) = &self.texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), rect, uv, Color32::WHITE);
        }
        let overlay_palette = if ui.visuals().dark_mode {
            SemanticPalette::from_visuals(ui.visuals())
        } else {
            DARK_PALETTE
        };
        self.paint_cursor_and_burning_overlays(&painter, rect, overlay_palette);
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small("Click/Drag to drop grains • Arrows move cursor • Enter drops brush");
    }

    fn handle_canvas_interactions(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        rect: Rect,
    ) {
        let ctx = ui.ctx().clone();
        let n = self.state.grid_size;
        if response.has_focus() {
            let (left, right, down, up, drop_key) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                )
            });
            if left || right || down || up || drop_key {
                ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            }
            let moved = self.step_cursor_by_keys(left, right, down, up);
            if moved {
                self.state_rev = self.state_rev.wrapping_add(1);
                crate::accessibility::announce_live_status(
                    &ctx,
                    format!(
                        "Sandpile cursor at ({}, {})",
                        self.state.cursor_x, self.state.cursor_y
                    ),
                );
            }
            if drop_key {
                self.apply_brush_at(self.state.cursor_x, self.state.cursor_y);
                crate::accessibility::announce_live_status(&ctx, "Dropped grains onto lattice");
            }
        }

        if (response.clicked() || response.dragged()) && rect.width() > 1.0 && rect.height() > 1.0 {
            if let Some(pos) = response.interact_pointer_pos() {
                let fx = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 0.9999);
                let fy = (1.0 - (pos.y - rect.top()) / rect.height()).clamp(0.0, 0.9999);
                let cx = ((fx * (n as f32)) as usize).min(n - 1);
                let cy = ((fy * (n as f32)) as usize).min(n - 1);
                self.state.cursor_x = cx;
                self.state.cursor_y = cy;
                self.apply_brush_at(cx, cy);
                self.state_rev = self.state_rev.wrapping_add(1);
            }
        }
    }

    fn step_cursor_by_keys(&mut self, left: bool, right: bool, down: bool, up: bool) -> bool {
        let max_c = self.state.grid_size.saturating_sub(1);
        let prev = (self.state.cursor_x, self.state.cursor_y);
        if left {
            self.state.cursor_x = self.state.cursor_x.saturating_sub(1);
        }
        if right {
            self.state.cursor_x = (self.state.cursor_x + 1).min(max_c);
        }
        if down {
            self.state.cursor_y = self.state.cursor_y.saturating_sub(1);
        }
        if up {
            self.state.cursor_y = (self.state.cursor_y + 1).min(max_c);
        }
        (self.state.cursor_x, self.state.cursor_y) != prev
    }

    fn paint_cursor_and_burning_overlays(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        palette: SemanticPalette,
    ) {
        let n = self.state.grid_size.max(1) as f32;
        let cell_w = rect.width() / n;
        let cell_h = rect.height() / n;

        if self.state.show_burning_overlay && self.burning_result.max_wave_depth > 0 {
            let max_d = f32::from(self.burning_result.max_wave_depth.max(1));
            let depths = self.grid.burn_depths();
            for y in 0..self.state.grid_size {
                for x in 0..self.state.grid_size {
                    let d = depths[flatten_2d_index(x, y, self.state.grid_size)];
                    if d == self.burning_result.max_wave_depth {
                        let cx = rect.left() + (x as f32 + 0.5) * cell_w;
                        let cy = rect.bottom() - (y as f32 + 0.5) * cell_h;
                        let radius = (cell_w.min(cell_h) * 0.28).clamp(1.2, 4.5);
                        painter.circle_stroke(
                            Pos2::new(cx, cy),
                            radius,
                            Stroke::new(1.1 + 0.3 * (f32::from(d) / max_d), palette.warning),
                        );
                    }
                }
            }
        }

        let rx = rect.left() + (self.state.cursor_x as f32 + 0.5) * cell_w;
        let ry = rect.bottom() - (self.state.cursor_y as f32 + 0.5) * cell_h;
        let marker_r = (cell_w.min(cell_h) * 0.45).clamp(3.0, 9.0);
        painter.circle_stroke(
            Pos2::new(rx, ry),
            marker_r,
            Stroke::new(1.7_f32, palette.accent),
        );
    }

    fn draw_power_law_scaling_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let obs_label = match self.state.observable {
            AvalancheObservable::Size => "Avalanche Size S (Topples)",
            AvalancheObservable::Area => "Avalanche Area A (Distinct Sites)",
            AvalancheObservable::Duration => "Avalanche Duration T (Wavefronts)",
        };
        ui.label(format!(
            "Log-Log Survival Scaling: log10 P(X ≥ x) vs log10({obs_label})"
        ));
        if let Some(fit) = &self.latest_fit {
            let coeffs = fit.coefficients();
            let tau = 1.0 - coeffs[1];
            ui.small(format!(
                "FitModel::PowerLaw  P(X≥x) = {:.3}·x^({:.3})  •  PDF τ ≈ {:.3}  •  R² = {:.4}",
                coeffs[0], coeffs[1], tau, fit.r_squared
            ));
        }

        let empirical_pts: PlotPoints = self.survival_points[..self.survival_len]
            .iter()
            .filter(|p| p[0] > 0.0 && p[1] > 0.0)
            .map(|p| [p[0].log10(), p[1].log10()])
            .collect();

        Plot::new("sandpile_power_law_plot")
            .allow_boxed_zoom(false)
            .x_axis_label("log10(x)")
            .y_axis_label("log10 P(X ≥ x)")
            .show(ui, |plot_ui| {
                plot_ui.points(
                    Points::new("Empirical CCDF", empirical_pts)
                        .color(palette.series_color(0))
                        .radius(3.8_f32),
                );
                if let Some(fit) = &self.latest_fit {
                    let fit_pts =
                        build_log_log_fit_points(fit, &self.survival_points[..self.survival_len]);
                    plot_ui.line(
                        Line::new("FitModel::PowerLaw", PlotPoints::from(fit_pts))
                            .color(palette.series_color(1))
                            .width(2.2_f32),
                    );
                }
            });
    }

    fn upload_crystal_texture(&mut self, ctx: &egui::Context) {
        populate_sandpile_pixels(
            &self.grid,
            &self.burning_result.max_wave_depth,
            self.state.view_mode,
            self.state.colormap,
            &mut self.rgba_pixels,
        );
        let n = self.state.grid_size;
        let image = ColorImage {
            size: [n, n],
            pixels: self.rgba_pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(image, TextureOptions::NEAREST);
        } else {
            self.texture =
                Some(ctx.load_texture("abelian_sandpile_tex", image, TextureOptions::NEAREST));
        }
    }

    pub(super) fn build_export_color_image(&self) -> ColorImage {
        let n = self.state.grid_size;
        let mut pixels = vec![Color32::BLACK; n * n];
        populate_sandpile_pixels(
            &self.grid,
            &self.burning_result.max_wave_depth,
            self.state.view_mode,
            self.state.colormap,
            &mut pixels,
        );
        ColorImage {
            size: [n, n],
            pixels,
            ..Default::default()
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut prims = Vec::new();
        let emp: Vec<[f64; 2]> = self.survival_points[..self.survival_len]
            .iter()
            .filter(|p| p[0] > 0.0 && p[1] > 0.0)
            .map(|p| [p[0].log10(), p[1].log10()])
            .collect();
        if !emp.is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Empirical Avalanche CCDF (log10)".into(),
                points: emp,
                color: DARK_PALETTE.series_color(0),
                radius: 2.5,
            });
        }
        if let Some(fit) = &self.latest_fit {
            let line_pts =
                build_log_log_fit_points(fit, &self.survival_points[..self.survival_len]);
            if line_pts.len() >= 2 {
                prims.push(ExportPrimitive::Polyline {
                    name: "FitModel::PowerLaw".into(),
                    points: line_pts,
                    color: DARK_PALETTE.series_color(1),
                    stroke_width: 2.0,
                    closed: false,
                    fill: None,
                });
            }
        }
        if prims.is_empty() {
            prims.push(ExportPrimitive::Polyline {
                name: "Critical Reference".into(),
                points: vec![[0.0, 0.0], [1.5, -0.35]],
                color: DARK_PALETTE.accent,
                stroke_width: 1.8,
                closed: false,
                fill: None,
            });
        }
        prims
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let ccdf: Vec<[f64; 2]> = self.survival_points[..self.survival_len].to_vec();
        let fit_series: Vec<[f64; 2]> = if let Some(fit) = &self.latest_fit {
            self.survival_points[..self.survival_len]
                .iter()
                .map(|p| [p[0], fit.eval(p[0])])
                .filter(|q| q[1].is_finite())
                .collect()
        } else {
            Vec::new()
        };
        let n = self.state.grid_size;
        let mid_y = n / 2;
        let heights = self.grid.heights();
        let profile: Vec<[f64; 2]> = (0..n)
            .map(|x| [x as f64, f64::from(heights[flatten_2d_index(x, mid_y, n)])])
            .collect();

        vec![
            ("avalanche_survival_ccdf".into(), ccdf),
            ("power_law_fit_curve".into(), fit_series),
            ("centerline_grain_heights".into(), profile),
        ]
    }
}

fn build_log_log_fit_points(fit: &math_explorer::fit::CurveFit, pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if fit.fit_model != FitModel::PowerLaw || pts.len() < 2 {
        return Vec::new();
    }
    let x_min = pts[0][0].max(1.0);
    let x_max = pts[pts.len() - 1][0].max(x_min + 1.0);
    let log_min = x_min.log10();
    let log_max = x_max.log10();
    (0..32)
        .filter_map(|i| {
            let t = (i as f64) / 31.0;
            let lx = log_min + t * (log_max - log_min);
            let x = 10.0_f64.powf(lx);
            let y = fit.eval(x);
            (y.is_finite() && y > 0.0).then(|| [lx, y.log10()])
        })
        .collect()
}

fn populate_sandpile_pixels(
    grid: &SandpileGrid,
    max_burn_depth: &u16,
    view_mode: SandpileViewMode,
    colormap: SandpileColormap,
    out_pixels: &mut [Color32],
) {
    let w = grid.width();
    let h = grid.height();
    let heights = grid.heights();
    let depths = grid.burn_depths();
    let max_d = f64::from((*max_burn_depth).max(1));

    for grid_y in 0..h {
        let img_row = (h - 1) - grid_y;
        for grid_x in 0..w {
            let src_idx = flatten_2d_index(grid_x, grid_y, w);
            let dst_idx = flatten_2d_index(grid_x, img_row, w);
            let z = heights[src_idx];
            let t = match view_mode {
                SandpileViewMode::GrainHeights => {
                    if z >= CRITICAL_HEIGHT {
                        1.0
                    } else {
                        f64::from(z) / f64::from(MAX_STABLE_HEIGHT + 1)
                    }
                }
                SandpileViewMode::DharBurningWaves => {
                    (f64::from(depths[src_idx]) / max_d).clamp(0.0, 1.0)
                }
            };
            let rgb = match colormap {
                SandpileColormap::CriticalThermal => heatmap_color(t),
                SandpileColormap::ApollonianHighContrast => {
                    if z == 0 && view_mode == SandpileViewMode::GrainHeights {
                        let base = heatmap_color_high_contrast(0.05);
                        [base[0] / 2, base[1] / 2, base[2] / 2]
                    } else {
                        ensure_dark_canvas_contrast(heatmap_color_high_contrast(t))
                    }
                }
                SandpileColormap::BurningWavePhase => cyclic_cosine_palette_high_contrast(t),
            };
            out_pixels[dst_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        }
    }
}
