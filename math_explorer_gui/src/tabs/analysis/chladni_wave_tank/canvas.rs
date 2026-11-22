#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

// [cite:graph_parameters_rust]

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2};
use egui_plot::{ExportPrimitive, Line, Plot, PlotPoints, VLine};
use math_explorer::math_kernel::colormap::{
    ensure_dark_canvas_contrast, heatmap_color, heatmap_color_high_contrast, mapped_gradient,
};
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::analysis::pde::wave_2d::young_double_slit_fringe_spacing;

use super::*;

impl ChladniWaveTankTool {
    pub(super) fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("2D Chladni Plate & Wave Tank");
            let avail = ui.available_size();
            if avail.x > 620.0 {
                ui.columns(2, |cols| {
                    self.draw_interactive_tank_canvas(&mut cols[0]);
                    self.draw_secondary_analysis_plot(&mut cols[1]);
                });
            } else {
                self.draw_interactive_tank_canvas(ui);
                ui.add_space(6.0);
                self.draw_secondary_analysis_plot(ui);
            }
        });
    }

    fn draw_interactive_tank_canvas(&mut self, ui: &mut egui::Ui) {
        self.upload_wave_texture(ui.ctx());
        let side = ui.available_width().min(ui.available_height() - 28.0).clamp(180.0, 560.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click_and_drag());
        let rect = response.rect;

        let summary = format!(
            "Preset {}, medium {:?}, driver ({:.2}, {:.2}) at {:.2} Hz, dominant |k| {:.2}",
            self.state.preset.label(),
            self.state.medium,
            self.state.driver_x,
            self.state.driver_y,
            self.state.driver_frequency,
            self.dominant_peak.k_radial
        );
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "2D Chladni Plate & Wave Tank Visualization",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);

        self.handle_canvas_input(ui, &response, rect);

        if let Some(tex) = &self.texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), rect, uv, Color32::WHITE);
        }
        self.paint_canvas_overlays(&painter, rect);
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small("Drag to steer driver • Shift+Click to pin clamp • Arrows move • Enter pulses");
    }

    fn handle_canvas_input(&mut self, ui: &mut egui::Ui, response: &egui::Response, rect: Rect) {
        let ctx = ui.ctx().clone();
        if response.has_focus() {
            let (left, right, down, up, pulse) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Enter),
                )
            });
            let dx = (if right { 0.02 } else { 0.0 }) - (if left { 0.02 } else { 0.0 });
            let dy = (if up { 0.02 } else { 0.0 }) - (if down { 0.02 } else { 0.0 });
            if left || right || down || up || pulse {
                ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
            }
            if dx != 0.0 || dy != 0.0 {
                self.state.driver_x = (self.state.driver_x + dx).clamp(0.02, 0.98);
                self.state.driver_y = (self.state.driver_y + dy).clamp(0.02, 0.98);
                self.state_rev = self.state_rev.wrapping_add(1);
                crate::accessibility::announce_live_status(
                    &ctx,
                    format!(
                        "Driver moved to ({:.2}, {:.2})",
                        self.state.driver_x, self.state.driver_y
                    ),
                );
            }
            if pulse {
                self.solver.inject_gaussian_pulse(
                    self.state.driver_x,
                    self.state.driver_y,
                    0.045,
                    1.0,
                );
                self.refresh_analysis_outputs();
                crate::accessibility::announce_live_status(&ctx, "Injected Gaussian wave pulse");
            }
        }

        if (response.clicked() || response.dragged()) && rect.width() > 1.0 && rect.height() > 1.0 {
            if let Some(pos) = response.interact_pointer_pos() {
                let nx = (((pos.x - rect.left()) / rect.width()) as f64).clamp(0.02, 0.98);
                let ny = ((1.0 - (pos.y - rect.top()) / rect.height()) as f64).clamp(0.02, 0.98);
                let shift = ctx.input(|i| i.modifiers.shift);
                if shift && response.clicked() && self.custom_clamps.len() < 16 {
                    self.custom_clamps.push([nx, ny]);
                    self.rebuild_masks();
                } else {
                    self.state.driver_x = nx;
                    self.state.driver_y = ny;
                    if response.clicked() {
                        self.solver.inject_gaussian_pulse(nx, ny, 0.04, 0.65);
                    }
                }
                self.state_rev = self.state_rev.wrapping_add(1);
            }
        }
    }

    fn paint_canvas_overlays(&self, painter: &egui::Painter, rect: Rect) {
        let to_screen = |nx: f64, ny: f64| -> Pos2 {
            Pos2::new(
                rect.left() + (nx as f32) * rect.width(),
                rect.bottom() - (ny as f32) * rect.height(),
            )
        };
        let inv_grid = 1.0 / (GRID_SIZE as f64);
        let cell_to_screen =
            |cx: f64, cy: f64| -> Pos2 { to_screen((cx + 0.5) * inv_grid, (cy + 0.5) * inv_grid) };

        if self.state.show_nodal_contours {
            let stroke = Stroke::new(
                1.35_f32,
                Color32::from_rgba_unmultiplied(245, 250, 255, 220),
            );
            for &[p0, p1] in self.contours.segments() {
                painter.line_segment(
                    [cell_to_screen(p0[0], p0[1]), cell_to_screen(p1[0], p1[1])],
                    stroke,
                );
            }
        }

        if self.state.show_sand_particles {
            let sand_color = Color32::from_rgba_unmultiplied(250, 224, 130, 225);
            for (&px, &py) in self
                .sand
                .positions_x()
                .iter()
                .zip(self.sand.positions_y().iter())
            {
                painter.circle_filled(cell_to_screen(px, py), 1.3_f32, sand_color);
            }
        }

        if self.state.barrier_enabled || self.state.lens_enabled {
            let det_nx = ((self.state.detector_x as f64) + 0.5) * inv_grid;
            painter.line_segment(
                [to_screen(det_nx, 0.02), to_screen(det_nx, 0.98)],
                Stroke::new(1.3_f32, Color32::from_rgb(90, 230, 190)),
            );
        }

        let drv_pos = to_screen(self.state.driver_x, self.state.driver_y);
        painter.circle_stroke(
            drv_pos,
            5.0_f32,
            Stroke::new(1.8_f32, Color32::from_rgb(255, 110, 90)),
        );
    }

    fn draw_secondary_analysis_plot(&self, ui: &mut egui::Ui) {
        match self.state.secondary_plot {
            SecondaryPlotMode::DetectorCrossSection => self.draw_detector_cross_section_plot(ui),
            SecondaryPlotMode::RadialFftSpectrum => self.draw_radial_fft_plot(ui),
        }
    }

    fn draw_detector_cross_section_plot(&self, ui: &mut egui::Ui) {
        ui.label(format!(
            "Detector Intensity Cross-Section I(y) at x = {}",
            self.state.detector_x
        ));
        let peak_i = self
            .detector_buf
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0_f64, f64::max);
        let pts: PlotPoints = self
            .detector_buf
            .iter()
            .enumerate()
            .map(|(y, &val)| [y as f64, if val.is_finite() { val } else { 0.0 }])
            .collect();
        Plot::new("chladni_detector_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Detector I(y)", pts)
                        .color(Color32::from_rgb(90, 230, 190))
                        .width(2.0_f32),
                );
                self.overlay_young_reference(plot_ui, peak_i);
            });
    }

    fn overlay_young_reference(&self, plot_ui: &mut egui_plot::PlotUi<'_>, peak_i: f64) {
        if !self.state.barrier_enabled || self.state.wave_speed <= 0.0 {
            return;
        }
        let lam = self.state.wave_speed / self.state.driver_frequency.max(1.0e-3);
        let d_scr = (self.state.detector_x.saturating_sub(self.state.barrier_x)).max(1) as f64;
        let dy = young_double_slit_fringe_spacing(lam, self.state.slit_separation, d_scr);
        let cy = ((GRID_SIZE - 1) as f64) * 0.5;
        let marker_c = Color32::from_rgb(255, 190, 80);
        plot_ui.vline(VLine::new("Young +1", cy + dy).color(marker_c));
        plot_ui.vline(VLine::new("Young -1", cy - dy).color(marker_c));
        if self.state.slit_count == 2 && peak_i > 1.0e-12 {
            let half_d = 0.5 * self.state.slit_separation;
            let young_pts: PlotPoints = (0..GRID_SIZE)
                .map(|y| {
                    let yrel = (y as f64) - cy;
                    let r1 = (d_scr * d_scr + (yrel - half_d) * (yrel - half_d)).sqrt();
                    let r2 = (d_scr * d_scr + (yrel + half_d) * (yrel + half_d)).sqrt();
                    let delta = std::f64::consts::PI * (r2 - r1) / lam;
                    let env = (-0.5 * (yrel / (0.38 * GRID_SIZE as f64)).powi(2)).exp();
                    [y as f64, peak_i * env * delta.cos().powi(2)]
                })
                .collect();
            plot_ui.line(
                Line::new("Young Analytic I(y)", young_pts)
                    .color(marker_c)
                    .width(1.4_f32),
            );
        }
    }

    fn draw_radial_fft_plot(&self, ui: &mut egui::Ui) {
        ui.label("2D FFT Azimuthal Radial Spectrum S(|k|)");
        let pts: PlotPoints = self
            .radial_spectrum
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, &val)| [k as f64, if val.is_finite() { val } else { 0.0 }])
            .collect();
        Plot::new("chladni_fft_plot")
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Radial |F(k)|", pts)
                        .color(Color32::from_rgb(130, 185, 255))
                        .width(2.0_f32),
                );
                if self.dominant_peak.k_radial > 0.0 {
                    plot_ui.vline(
                        VLine::new("Dominant |k|", self.dominant_peak.k_radial)
                            .color(Color32::from_rgb(255, 140, 90)),
                    );
                }
            });
    }

    fn upload_wave_texture(&mut self, ctx: &egui::Context) {
        populate_field_pixels(&self.solver, self.state.colormap, &mut self.rgba_pixels);
        let image = ColorImage {
            size: [GRID_SIZE, GRID_SIZE],
            pixels: self.rgba_pixels.clone(),
            ..Default::default()
        };
        if let Some(tex) = &mut self.texture {
            tex.set(image, TextureOptions::LINEAR);
        } else {
            self.texture = Some(ctx.load_texture(
                "chladni_wave_tank_tex",
                image,
                TextureOptions::LINEAR,
            ));
        }
    }

    pub(super) fn build_export_color_image(&self) -> ColorImage {
        let mut pixels = vec![Color32::BLACK; GRID_SIZE * GRID_SIZE];
        populate_field_pixels(&self.solver, self.state.colormap, &mut pixels);
        ColorImage {
            size: [GRID_SIZE, GRID_SIZE],
            pixels,
            ..Default::default()
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut prims = Vec::new();
        if self.state.show_nodal_contours {
            for (i, &[p0, p1]) in self.contours.segments().iter().take(1_500).enumerate() {
                if p0[0].is_finite() && p0[1].is_finite() && p1[0].is_finite() && p1[1].is_finite()
                {
                    prims.push(ExportPrimitive::Polyline {
                        name: if i == 0 {
                            "Nodal Contours".into()
                        } else {
                            String::new()
                        },
                        points: vec![[p0[0] + 0.5, p0[1] + 0.5], [p1[0] + 0.5, p1[1] + 0.5]],
                        color: Color32::from_rgb(245, 250, 255),
                        stroke_width: 1.2,
                        closed: false,
                        fill: None,
                    });
                }
            }
        }
        if self.state.show_sand_particles {
            let pts: Vec<[f64; 2]> = self
                .sand
                .positions_x()
                .iter()
                .zip(self.sand.positions_y().iter())
                .step_by(2)
                .filter(|(&px, &py)| px.is_finite() && py.is_finite())
                .map(|(&px, &py)| [px + 0.5, py + 0.5])
                .collect();
            if !pts.is_empty() {
                prims.push(ExportPrimitive::Points {
                    name: "Chladni Sand".into(),
                    points: pts,
                    color: Color32::from_rgb(250, 224, 130),
                    radius: 1.4,
                });
            }
        }
        if prims.is_empty() {
            let det_x = (self.state.detector_x as f64) + 0.5;
            prims.push(ExportPrimitive::Polyline {
                name: "Detector Screen".into(),
                points: vec![[det_x, 2.0], [det_x, (GRID_SIZE - 2) as f64]],
                color: Color32::from_rgb(90, 230, 190),
                stroke_width: 1.5,
                closed: false,
                fill: None,
            });
        }
        prims
    }

    pub(super) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let det_series: Vec<[f64; 2]> = self
            .detector_buf
            .iter()
            .enumerate()
            .map(|(y, &v)| [y as f64, if v.is_finite() { v } else { 0.0 }])
            .collect();
        let fft_series: Vec<[f64; 2]> = self
            .radial_spectrum
            .iter()
            .enumerate()
            .map(|(k, &v)| [k as f64, if v.is_finite() { v } else { 0.0 }])
            .collect();
        let mid_y = GRID_SIZE / 2;
        let disp = self.solver.displacement();
        let wave_slice: Vec<[f64; 2]> = (0..GRID_SIZE)
            .map(|x| {
                let v = disp[flatten_2d_index(x, mid_y, GRID_SIZE)];
                [x as f64, if v.is_finite() { v } else { 0.0 }]
            })
            .collect();

        vec![
            ("detector_intensity".into(), det_series),
            ("radial_fft_spectrum".into(), fft_series),
            ("centerline_wave_profile".into(), wave_slice),
        ]
    }
}

fn populate_field_pixels(
    solver: &Wave2DSolver,
    colormap: WaveColormap,
    out_pixels: &mut [Color32],
) {
    let w = solver.width();
    let h = solver.height();
    let disp = solver.displacement();
    let energy = solver.energy_field();

    let mut max_abs_u = 1.0e-6_f64;
    let mut max_e = 1.0e-9_f64;
    for i in 0..disp.len() {
        let au = disp[i].abs();
        if au.is_finite() && au > max_abs_u {
            max_abs_u = au;
        }
        if energy[i].is_finite() && energy[i] > max_e {
            max_e = energy[i];
        }
    }

    for grid_y in 0..h {
        let img_row = (h - 1) - grid_y;
        for grid_x in 0..w {
            let src_idx = flatten_2d_index(grid_x, grid_y, w);
            let dst_idx = flatten_2d_index(grid_x, img_row, w);
            if solver.wall_mask[src_idx] <= 0.5 {
                out_pixels[dst_idx] = Color32::from_rgb(58, 66, 82);
                continue;
            }
            let rgb = match colormap {
                WaveColormap::PerceptualThermal => {
                    let t = (energy[src_idx] / max_e).sqrt().clamp(0.0, 1.0);
                    heatmap_color(t)
                }
                WaveColormap::BipolarPhase => {
                    mapped_gradient(disp[src_idx], -max_abs_u, max_abs_u)
                }
                WaveColormap::WcagHighContrast => {
                    let t = (0.5 + 0.5 * (disp[src_idx] / max_abs_u)).clamp(0.0, 1.0);
                    ensure_dark_canvas_contrast(heatmap_color_high_contrast(t))
                }
            };
            let speed_scale = solver.wave_speed_mask[src_idx];
            if (speed_scale - 1.0).abs() > 1.0e-3 {
                out_pixels[dst_idx] = Color32::from_rgb(
                    rgb[0].saturating_add(18),
                    rgb[1].saturating_add(34),
                    rgb[2].saturating_add(52),
                );
            } else {
                out_pixels[dst_idx] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
    }
}
