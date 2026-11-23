#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 2D raster colormap viewport, accessibility interactions, diagnostics plot, and export datasets.

// [cite:graph_parameters_rust]

use std::f64::consts::PI;

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureOptions, Vec2};
use egui_plot::{HLine, Line, Plot, PlotPoints, VLine};
use math_explorer::math_kernel::colormap::{
    cyclic_cosine_palette, cyclic_cosine_palette_high_contrast, heatmap_color,
    heatmap_color_high_contrast, mapped_gradient,
};
use math_explorer::math_kernel::types::flatten_2d_index;

use super::{
    CARPET_STEPS, DIFF_GRID, FourierOpticsWorkbenchTool, GS_GRID, OpticsDisplayMode, OpticsMode,
    REAL_SCALAR_TWIN_FLOOR, WorkbenchCache,
};
use crate::app::{SemanticPalette, palette::DARK_PALETTE};
use crate::framework::scoped_panel_id;

impl FourierOpticsWorkbenchTool {
    pub(super) fn draw_central_canvas_and_plots(&mut self, parent_ui: &mut egui::Ui) {
        let bottom_id = scoped_panel_id(parent_ui, "optics_workbench_bottom");
        egui::TopBottomPanel::bottom(bottom_id)
            .exact_height(190.0)
            .show_inside(parent_ui, |ui| {
                let palette = SemanticPalette::from_visuals(ui.visuals());
                self.draw_diagnostic_plot(ui, &palette);
            });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let title = format!(
                "{} — {}",
                self.state.mode.label(),
                self.state.display.label()
            );
            ui.strong(title);
            self.draw_raster_viewport(ui);
        });
    }

    fn draw_raster_viewport(&mut self, ui: &mut egui::Ui) {
        self.sync_viewport_texture(ui.ctx());
        let avail = ui.available_size();
        let side = avail.x.min(avail.y - 24.0).clamp(160.0, 540.0);
        let size = if self.state.mode == OpticsMode::HelmholtzDiffraction
            && self.state.display == OpticsDisplayMode::AxialCarpetXZ
        {
            Vec2::new((side * 1.25).min(avail.x.max(160.0)), side)
        } else {
            Vec2::splat(side)
        };

        let (response, painter) = ui.allocate_painter(size, Sense::click_and_drag());
        let rect = response.rect;
        let summary = self.accessibility_summary();
        crate::accessibility::register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Fourier Optics Wavefield & Hologram Canvas",
            Some(&summary),
        );
        crate::accessibility::lock_canvas_arrow_keys(ui, &response);
        self.focused_canvas = response.has_focus().then_some(response.id);

        self.handle_viewport_interactions(ui, &response, rect);

        if let Some(tex) = &self.texture {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            painter.image(tex.id(), rect, uv, Color32::WHITE);
        }

        let overlay_palette = if ui.visuals().dark_mode
            && self.state.display != OpticsDisplayMode::SlmPhaseMask8Bit
        {
            SemanticPalette::from_visuals(ui.visuals())
        } else {
            DARK_PALETTE
        };
        self.paint_viewport_overlays(&painter, rect, &overlay_palette);
        crate::accessibility::paint_canvas_focus_ring(ui, &response);
        ui.small("Drag horizontally to scrub propagation z • Arrow keys adjust z / GS iterations");
    }

    fn accessibility_summary(&self) -> String {
        let guard = self.cache.borrow();
        let Some(c) = guard.as_ref() else {
            return "Fourier optics canvas ready".to_string();
        };
        match self.state.mode {
            OpticsMode::HelmholtzDiffraction => format!(
                "{}, z = {:.2} mm, Fresnel N_F = {:.2}, center intensity {:.3}",
                self.state.aperture.label(),
                self.state.propagation_mm,
                c.fresnel_nf,
                c.center_intensity
            ),
            OpticsMode::GerchbergSaxtonHolography => format!(
                "Gerchberg-Saxton {} iterations, NRMSE {:.3}, efficiency {:.1}%, twin leakage {:.1}%",
                c.gs_metrics.iteration,
                c.gs_metrics.target_nrmse,
                c.gs_metrics.diffraction_efficiency * 100.0,
                c.gs_metrics.twin_leakage * 100.0
            ),
        }
    }

    fn handle_viewport_interactions(
        &mut self,
        ui: &mut egui::Ui,
        response: &egui::Response,
        rect: Rect,
    ) {
        let ctx = ui.ctx().clone();
        let mut changed = false;
        if response.dragged() || response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let u = ((pos.x - rect.min.x) / rect.width().max(1.0_f32)).clamp(0.0_f32, 1.0_f32);
                if self.state.mode == OpticsMode::HelmholtzDiffraction {
                    self.state.propagation_mm =
                        (f64::from(u) * self.state.carpet_max_mm).clamp(0.1, 150.0);
                    changed = true;
                }
            }
        }
        if response.has_focus() {
            let (left, right, up, down) = ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
                )
            });
            if left || right || up || down {
                ui.memory_mut(|m| m.move_focus(egui::FocusDirection::None));
                let step_dir = (if right || up { 1.0 } else { 0.0 })
                    - (if left || down { 1.0 } else { 0.0 });
                if self.state.mode == OpticsMode::HelmholtzDiffraction {
                    self.state.propagation_mm =
                        (self.state.propagation_mm + step_dir * 0.5).clamp(-150.0, 150.0);
                } else {
                    let next = (self.state.gs_iterations as isize + step_dir as isize).clamp(1, 100);
                    self.state.gs_iterations = next as usize;
                }
                changed = true;
            }
        }
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn paint_viewport_overlays(&self, painter: &egui::Painter, rect: Rect, pal: &SemanticPalette) {
        painter.rect_stroke(
            rect,
            0.0_f32,
            Stroke::new(1.0_f32, pal.info),
            egui::StrokeKind::Inside,
        );
        if self.state.mode == OpticsMode::HelmholtzDiffraction
            && self.state.display == OpticsDisplayMode::AxialCarpetXZ
        {
            let frac = (self.state.propagation_mm / self.state.carpet_max_mm.max(1.0e-6))
                .clamp(0.0, 1.0) as f32;
            let x_line = rect.lerp_inside(Vec2::new(frac, 0.5_f32)).x;
            painter.line_segment(
                [Pos2::new(x_line, rect.min.y), Pos2::new(x_line, rect.max.y)],
                Stroke::new(1.8_f32, pal.accent),
            );
        } else {
            let c = rect.center();
            painter.line_segment(
                [Pos2::new(c.x - 6.0_f32, c.y), Pos2::new(c.x + 6.0_f32, c.y)],
                Stroke::new(1.2_f32, pal.accent),
            );
            painter.line_segment(
                [Pos2::new(c.x, c.y - 6.0_f32), Pos2::new(c.x, c.y + 6.0_f32)],
                Stroke::new(1.2_f32, pal.accent),
            );
        }
    }

    fn sync_viewport_texture(&mut self, ctx: &egui::Context) {
        self.ensure_evaluated();
        let key = self.revision ^ ((u64::from(ctx.style().visuals.dark_mode)) << 62);
        if self.texture.is_none() || self.texture_key != key {
            let img = self.build_active_raster_image();
            self.texture = Some(ctx.load_texture(
                "fourier_optics_viewport",
                img,
                TextureOptions::NEAREST,
            ));
            self.texture_key = key;
        }
    }

    pub(crate) fn build_active_raster_image(&self) -> ColorImage {
        self.ensure_evaluated();
        let guard = self.cache.borrow();
        let Some(cache) = guard.as_ref() else {
            return ColorImage::filled([DIFF_GRID, DIFF_GRID], Color32::BLACK);
        };
        match self.state.mode {
            OpticsMode::HelmholtzDiffraction => self.rasterize_diffraction_layer(cache),
            OpticsMode::GerchbergSaxtonHolography => self.rasterize_holography_layer(cache),
        }
    }

    fn rasterize_diffraction_layer(&self, cache: &WorkbenchCache) -> ColorImage {
        let hc = self.state.high_contrast_colormap;
        match self.state.display {
            OpticsDisplayMode::IntensityXY => {
                Self::rasterize_intensity(&cache.diff_intensity, DIFF_GRID, DIFF_GRID, cache.max_intensity, hc)
            }
            OpticsDisplayMode::WrappedPhaseXY => {
                Self::rasterize_phase(&cache.diff_phase, DIFF_GRID, DIFF_GRID, hc)
            }
            OpticsDisplayMode::AxialCarpetXZ => Self::rasterize_axial_carpet(cache, hc),
            OpticsDisplayMode::SlmPhaseMask8Bit => {
                Self::rasterize_quantized_phase(&cache.diff_phase, DIFF_GRID, DIFF_GRID)
            }
        }
    }

    fn rasterize_holography_layer(&self, cache: &WorkbenchCache) -> ColorImage {
        let hc = self.state.high_contrast_colormap;
        match self.state.display {
            OpticsDisplayMode::IntensityXY => Self::rasterize_intensity(
                &cache.gs_replay_intensity,
                GS_GRID,
                GS_GRID,
                cache.max_intensity,
                hc,
            ),
            OpticsDisplayMode::WrappedPhaseXY => {
                Self::rasterize_phase(cache.gs_solver.slm_phase(), GS_GRID, GS_GRID, hc)
            }
            OpticsDisplayMode::AxialCarpetXZ => {
                let peak = cache
                    .gs_solver
                    .target_amplitude()
                    .iter()
                    .copied()
                    .fold(1.0e-12_f64, f64::max);
                Self::rasterize_intensity(cache.gs_solver.target_amplitude(), GS_GRID, GS_GRID, peak, hc)
            }
            OpticsDisplayMode::SlmPhaseMask8Bit => {
                let pixels = cache
                    .gs_slm_u8
                    .iter()
                    .map(|&v| Color32::from_rgb(v, v, v))
                    .collect();
                ColorImage::new([GS_GRID, GS_GRID], pixels)
            }
        }
    }

    fn rasterize_intensity(
        field: &[f64],
        width: usize,
        height: usize,
        peak: f64,
        high_contrast: bool,
    ) -> ColorImage {
        let inv_peak = 1.0 / peak.max(1.0e-12);
        let mut pixels = Vec::with_capacity(width * height);
        for &val in field {
            let t = (val * inv_peak).clamp(0.0, 1.0).sqrt();
            let rgb = if high_contrast {
                heatmap_color_high_contrast(t)
            } else {
                heatmap_color(t)
            };
            pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }
        ColorImage::new([width, height], pixels)
    }

    fn rasterize_phase(
        phase: &[f64],
        width: usize,
        height: usize,
        high_contrast: bool,
    ) -> ColorImage {
        let mut pixels = Vec::with_capacity(width * height);
        for &phi in phase {
            let rgb = if high_contrast {
                cyclic_cosine_palette_high_contrast((phi + PI) * 6.5)
            } else {
                mapped_gradient(phi, -PI, PI)
            };
            pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }
        ColorImage::new([width, height], pixels)
    }

    fn rasterize_quantized_phase(phase: &[f64], width: usize, height: usize) -> ColorImage {
        let inv_two_pi = 1.0 / (2.0 * PI);
        let mut pixels = Vec::with_capacity(width * height);
        for &phi in phase {
            let u = ((phi + PI) * inv_two_pi).clamp(0.0, 1.0);
            let level = (u * 255.0).round() as u8;
            pixels.push(Color32::from_rgb(level, level, level));
        }
        ColorImage::new([width, height], pixels)
    }

    fn rasterize_axial_carpet(cache: &WorkbenchCache, high_contrast: bool) -> ColorImage {
        let peak = cache
            .axial_carpet
            .iter()
            .copied()
            .fold(1.0e-12_f64, f64::max);
        let inv_peak = 1.0 / peak;
        // Render horizontal axis = z_idx (CARPET_STEPS), vertical axis = x (DIFF_GRID)
        let mut pixels = vec![Color32::BLACK; CARPET_STEPS * DIFF_GRID];
        for z_idx in 0..CARPET_STEPS {
            for x in 0..DIFF_GRID {
                let val = cache.axial_carpet[flatten_2d_index(x, z_idx, DIFF_GRID)];
                let t = (val * inv_peak).clamp(0.0, 1.0).sqrt();
                let rgb = if high_contrast {
                    heatmap_color_high_contrast(t)
                } else {
                    cyclic_cosine_palette(t * 28.0)
                };
                pixels[flatten_2d_index(z_idx, x, CARPET_STEPS)] =
                    Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
        ColorImage::new([CARPET_STEPS, DIFF_GRID], pixels)
    }

    fn draw_diagnostic_plot(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        self.ensure_evaluated();
        let guard = self.cache.borrow();
        let Some(cache) = guard.as_ref() else {
            return;
        };
        let plot_id = scoped_panel_id(ui, "optics_diagnostics_plot");
        match self.state.mode {
            OpticsMode::HelmholtzDiffraction => {
                ui.strong("Central Transverse Cross-Section I(x, 0, z) vs Aperture |U(x, 0, 0)|^2");
                let mid_y = DIFF_GRID / 2;
                let pitch = self.state.pixel_pitch_um;
                let mut prof_z = Vec::with_capacity(DIFF_GRID);
                let mut prof_0 = Vec::with_capacity(DIFF_GRID);
                for x in 0..DIFF_GRID {
                    let x_um = ((x as isize) - ((DIFF_GRID / 2) as isize)) as f64 * pitch;
                    let idx = flatten_2d_index(x, mid_y, DIFF_GRID);
                    prof_z.push([x_um, cache.diff_intensity[idx]]);
                    prof_0.push([x_um, cache.aperture_field[idx].norm_sqr()]);
                }
                Plot::new(plot_id)
                    .height(152.0)
                    .allow_boxed_zoom(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(
                            Line::new("Aperture |U(x,0,0)|^2", PlotPoints::new(prof_0))
                                .color(palette.series_color(3)),
                        );
                        plot_ui.line(
                            Line::new("Propagated I(x,0,z)", PlotPoints::new(prof_z))
                                .color(palette.series_color(0)),
                        );
                        plot_ui.vline(VLine::new("Optical Axis x = 0", 0.0).color(palette.accent));
                    });
            }
            OpticsMode::GerchbergSaxtonHolography => {
                ui.strong("Gerchberg-Saxton Target NRMSE Convergence vs Real-Scalar Twin Limit");
                let curve = cache.gs_convergence.clone();
                Plot::new(plot_id)
                    .height(152.0)
                    .allow_boxed_zoom(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(
                            Line::new("GS Phase Hologram NRMSE", PlotPoints::new(curve))
                                .color(palette.series_color(0)),
                        );
                        plot_ui.hline(
                            HLine::new("Real-Scalar Twin Limit (0.707)", REAL_SCALAR_TWIN_FLOOR)
                                .color(palette.warning),
                        );
                        plot_ui.hline(
                            HLine::new("Measured Real-Scalar Baseline", cache.real_scalar_metrics.target_nrmse)
                                .color(palette.series_color(2)),
                        );
                    });
            }
        }
    }

    pub(crate) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.ensure_evaluated();
        let guard = self.cache.borrow();
        let Some(cache) = guard.as_ref() else {
            return Vec::new();
        };
        match self.state.mode {
            OpticsMode::HelmholtzDiffraction => {
                let mid_y = DIFF_GRID / 2;
                let pitch = self.state.pixel_pitch_um;
                let mut intensity_slice = Vec::with_capacity(DIFF_GRID);
                let mut phase_slice = Vec::with_capacity(DIFF_GRID);
                for x in 0..DIFF_GRID {
                    let x_um = ((x as isize) - ((DIFF_GRID / 2) as isize)) as f64 * pitch;
                    let idx = flatten_2d_index(x, mid_y, DIFF_GRID);
                    intensity_slice.push([x_um, cache.diff_intensity[idx]]);
                    phase_slice.push([x_um, cache.diff_phase[idx]]);
                }
                vec![
                    ("Transverse Intensity I(x_um, 0, z)".to_string(), intensity_slice),
                    ("Wrapped Phase phi(x_um, 0, z)".to_string(), phase_slice),
                ]
            }
            OpticsMode::GerchbergSaxtonHolography => {
                let mut slm_hist = Vec::with_capacity(GS_GRID);
                for x in 0..GS_GRID {
                    let idx = flatten_2d_index(x, GS_GRID / 2, GS_GRID);
                    slm_hist.push([x as f64, f64::from(cache.gs_slm_u8[idx])]);
                }
                vec![
                    ("GS Target NRMSE vs Iteration".to_string(), cache.gs_convergence.clone()),
                    ("Central Row 8-Bit SLM Phase [0..255]".to_string(), slm_hist),
                ]
            }
        }
    }
}
