#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Side-panel gravitational lensing controls, caustic overlays, and file export dialogs for
//! [`super::GaussianSplattingTool`].
// [cite:gaussian_splatting]
// [cite:general_relativity_geodesics]

use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, PlotPoints, Points};

use crate::web_io::{spawn_save_binary_file_dialog, SaveAsDialogConfig};

use super::scene::{build_circle_ring, SplatLensMode};
use super::GaussianSplattingTool;

const PNG_SAVE_CFG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "PNG Image",
    extensions: &["png"],
    fallback_filename: "lensed_gaussian_splats.png",
    write_error_prefix: "Failed to write PNG",
};

const SVG_SAVE_CFG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "SVG Vector Figure",
    extensions: &["svg"],
    fallback_filename: "lensed_gaussian_splats.svg",
    write_error_prefix: "Failed to write SVG",
};

const CSV_SAVE_CFG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Dataset",
    extensions: &["csv"],
    fallback_filename: "lensed_gaussian_splats.csv",
    write_error_prefix: "Failed to write CSV",
};

const PLY_SAVE_CFG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "INRIA 3DGS PLY",
    extensions: &["ply"],
    fallback_filename: "gaussian_splats_3d.ply",
    write_error_prefix: "Failed to write PLY",
};

impl GaussianSplattingTool {
    pub(super) fn poll_export_status(&mut self) {
        let done = self.export_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(res) = done {
            match res {
                Ok(path) => self.export_status = Some((format!("Saved {}", path.display()), false)),
                Err(err) => self.export_status = Some((err, true)),
            }
            self.export_rx = None;
        }
    }

    pub(super) fn draw_lensing_controls(&mut self, ui: &mut egui::Ui) {
        self.poll_export_status();
        egui::CollapsingHeader::new("Gravitational Lensing & Beaming")
            .default_open(true)
            .show(ui, |ui| {
                let prev = self.lens;
                egui::ComboBox::from_label("Lens Model")
                    .selected_text(self.lens.mode.label())
                    .show_ui(ui, |ui| {
                        for mode in SplatLensMode::ALL {
                            ui.selectable_value(&mut self.lens.mode, mode, mode.label());
                        }
                    });
                if self.lens.mode != SplatLensMode::Off {
                    self.draw_active_lens_sliders(ui);
                }
                if self.lens != prev {
                    self.lens.clamp_in_place();
                    self.invalidate_render_cache();
                }
                self.draw_lens_telemetry_and_exports(ui);
            });
    }

    fn draw_active_lens_sliders(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.lens.drag_lens_on_canvas,
            "Drag Lens on Canvas (instead of Orbit)",
        );
        ui.checkbox(
            &mut self.lens.show_caustic_overlay,
            "Show Einstein / NFW vs. Burkert Caustics",
        );
        ui.add(egui::Slider::new(&mut self.lens.center[0], -2.2..=2.2).text("Lens X"));
        ui.add(egui::Slider::new(&mut self.lens.center[1], -2.2..=2.2).text("Lens Y"));
        if self.lens.mode == SplatLensMode::Schwarzschild {
            ui.add(
                egui::Slider::new(&mut self.lens.einstein_radius, 0.10..=1.80)
                    .text("Einstein Radius θ_E"),
            );
            ui.add(
                egui::Slider::new(&mut self.lens.viewport_rs, 0.04..=0.65)
                    .text("Horizon Scale R_s (Shadow 1.5 R_s)"),
            );
        } else {
            ui.add(
                egui::Slider::new(&mut self.lens.halo_log10_mass, 9.0..=13.0)
                    .text("log10(M_200 / M_☉)"),
            );
            ui.add(
                egui::Slider::new(&mut self.lens.halo_scale_radius, 0.15..=2.0)
                    .text("Scale / Core Radius"),
            );
            ui.add(
                egui::Slider::new(&mut self.lens.log10_sigma_crit, 7.0..=9.5)
                    .text("log10(Σ_crit)"),
            );
        }
        ui.add(
            egui::Slider::new(&mut self.lens.peak_orbital_beta, 0.0..=0.92)
                .text("Peak Orbital v / c"),
        );
        ui.add(
            egui::Slider::new(&mut self.lens.magnetic_u_b, 0.0..=10.0)
                .text("Coronal U_B (J/m³)"),
        );
        if ui.small_button("Center Lens").clicked() {
            self.lens.center = [0.0, 0.0];
        }
    }

    fn draw_lens_telemetry_and_exports(&mut self, ui: &mut egui::Ui) {
        let t = &self.lens_telemetry;
        if self.lens.mode != SplatLensMode::Off {
            ui.small(format!(
                "Lensed 2D images: {} | 1.5 R_s shadow masked: {}",
                t.emitted_images, t.shadow_masked_rays
            ));
            ui.small(format!(
                "Peak beaming: {:.2}× | Γ^r_00(6R_s): {:.2e} m⁻¹",
                t.max_radiance_boost, t.christoffel_gamma_r00
            ));
            if let (Some(rn), Some(rb)) = (t.nfw_critical_radius, t.burkert_critical_radius) {
                ui.small(format!("Critical radii — NFW: {rn:.2}, Burkert: {rb:.2}"));
            }
        }
        ui.separator();
        ui.label("Export Radiance & Scene:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("PNG").clicked() {
                self.trigger_save_dialog(PNG_SAVE_CFG, self.export_lensed_png_bytes());
            }
            if ui.button("SVG").clicked() {
                self.trigger_save_dialog(
                    SVG_SAVE_CFG,
                    self.export_lensed_svg_string().map(String::into_bytes),
                );
            }
            if ui.button("CSV").clicked() {
                self.trigger_save_dialog(
                    CSV_SAVE_CFG,
                    self.export_lensed_csv_string().map(String::into_bytes),
                );
            }
            if ui.button("PLY").clicked() {
                self.trigger_save_dialog(PLY_SAVE_CFG, self.serialize_splats_ply_binary());
            }
        });
        if let Some((msg, is_err)) = &self.export_status {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let col = if *is_err {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(col, msg);
        }
    }

    fn trigger_save_dialog(
        &mut self,
        cfg: SaveAsDialogConfig,
        payload: Result<Vec<u8>, String>,
    ) {
        match payload {
            Ok(bytes) => {
                let (tx, rx) = std::sync::mpsc::channel();
                self.export_rx = Some(rx);
                spawn_save_binary_file_dialog(tx, cfg, cfg.fallback_filename, bytes);
            }
            Err(err) => self.export_status = Some((err, true)),
        }
    }

    pub(super) fn draw_caustic_plot_overlays(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        if self.lens.mode == SplatLensMode::Off || !self.lens.show_caustic_overlay {
            return;
        }
        let c = self.lens.center;
        let t = &self.lens_telemetry;
        if self.lens.mode == SplatLensMode::Schwarzschild {
            plot_ui.line(
                Line::new(
                    "Einstein Ring θ_E",
                    PlotPoints::from(build_circle_ring(c, self.lens.einstein_radius).to_vec()),
                )
                .color(palette.warning)
                .width(1.75_f32),
            );
            plot_ui.line(
                Line::new(
                    "Photon Sphere Shadow (1.5 R_s)",
                    PlotPoints::from(build_circle_ring(c, t.shadow_radius).to_vec()),
                )
                .color(palette.error)
                .style(LineStyle::Dashed { length: 6.0 })
                .width(1.6_f32),
            );
        }
        if let Some(rn) = t.nfw_critical_radius {
            plot_ui.line(
                Line::new(
                    "NFW Cusp Caustic/Critical",
                    PlotPoints::from(build_circle_ring(c, rn).to_vec()),
                )
                .color(palette.accent)
                .style(LineStyle::Dotted { spacing: 5.0 })
                .width(1.5_f32),
            );
        }
        if let Some(rb) = t.burkert_critical_radius {
            plot_ui.line(
                Line::new(
                    "Burkert Core Caustic/Critical",
                    PlotPoints::from(build_circle_ring(c, rb).to_vec()),
                )
                .color(palette.success)
                .style(LineStyle::Dashed { length: 5.0 })
                .width(1.5_f32),
            );
        }
        plot_ui.points(
            Points::new("Lens Center", vec![c])
                .color(palette.warning)
                .shape(MarkerShape::Cross)
                .radius(6.0_f32),
        );
    }
}
