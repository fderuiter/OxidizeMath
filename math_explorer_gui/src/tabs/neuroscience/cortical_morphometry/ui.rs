#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel morphometry controls, singular GLM status banner, live telemetry, and 3D viewport.

use super::{CorticalMorphometryTool, CorticalOverlayMode, SurfaceDisplayMode};
use crate::accessibility::AccessibleHoverText;
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction, render_recovery_banner};
use crate::framework::{CAMERA_3D_HELPER_TEXT, Plot3DViewport};
use eframe::egui;

impl CorticalMorphometryTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "cortical_morphometry_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Cortical Morphometry");
            ui.separator();
            self.draw_surface_and_overlay_controls(ui);
            ui.separator();
            self.draw_deformable_controls(ui);
            self.draw_mrf_and_glm_controls(ui);
            self.draw_resolution_and_view_controls(ui);
            ui.separator();
            self.draw_telemetry_readout(ui);
        });
    }

    fn draw_surface_and_overlay_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Scalar Heatmap Overlay:");
        egui::ComboBox::from_id_salt("cortical_overlay_combo")
            .selected_text(self.overlay_mode.label())
            .show_ui(ui, |ui| {
                for mode in CorticalOverlayMode::ALL {
                    ui.selectable_value(&mut self.overlay_mode, mode, mode.label());
                }
            });

        ui.add_space(4.0);
        ui.label("Cortical Boundary Surface:");
        egui::ComboBox::from_id_salt("cortical_surface_combo")
            .selected_text(self.surface_mode.label())
            .show_ui(ui, |ui| {
                for mode in SurfaceDisplayMode::ALL {
                    ui.selectable_value(&mut self.surface_mode, mode, mode.label());
                }
            });

        ui.add_space(4.0);
        ui.add(egui::Slider::new(&mut self.inflation, 0.0..=1.0).text("Cortical Inflation"))
            .accessible_hover_text(
                "Continuously unfolds buried sulci (0.0 folded cortex to 1.0 smooth ellipsoid)",
            );
        ui.add(
            egui::Slider::new(&mut self.sulcal_depth_mm, 0.2..=4.0)
                .text("Sulcal Fold Amplitude (mm)"),
        );
        ui.add(
            egui::Slider::new(&mut self.base_thickness_mm, 1.2..=4.0)
                .text("Baseline Ribbon Thickness (mm)"),
        );
    }

    fn draw_deformable_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Deformable Surface Physics")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(egui::Slider::new(&mut self.alpha, 0.0..=2.0).text("Membrane Tension (α)"));
                ui.add(egui::Slider::new(&mut self.beta, 0.0..=2.0).text("Laplacian Bending (β)"));
                ui.add(
                    egui::Slider::new(&mut self.sigma, 0.2..=3.0)
                        .text("MRI Smoothing Kernel σ (mm)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.learning_rate, 0.001..=0.2)
                        .text("Evolution Step Rate (η)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.evolution_steps, 0..=25).text("Deformable Steps"),
                );
            });
    }

    fn draw_mrf_and_glm_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Potts MRF & Cohort GLM")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.potts_beta, 0.0..=3.0)
                        .text("Potts MRF Coupling (β MRF)"),
                );
                ui.add(
                    egui::Slider::new(&mut self.mri_noise_level, 0.0..=30.0)
                        .text("T1 MRI Noise Amplitude"),
                );
                ui.add(
                    egui::Slider::new(&mut self.atrophy_effect_mm, 0.0..=1.5)
                        .text("Focal Atrophy Thinning (mm)"),
                );
                ui.checkbox(
                    &mut self.collinear_glm_design,
                    "Collinear GLM Predictors (Singular XᵀX)",
                )
                .accessible_hover_text(
                    "Tests graceful non-panicking handling when the cohort design matrix is rank-deficient",
                );
            });
    }

    fn draw_resolution_and_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Mesh Resolution & Camera", |ui| {
            self.u_resolution = self.u_resolution.clamp(12, 48);
            self.v_resolution = self.v_resolution.clamp(12, 36);
            ui.add(egui::Slider::new(&mut self.u_resolution, 12..=48).text("Azimuthal Rings (U)"));
            ui.add(egui::Slider::new(&mut self.v_resolution, 12..=36).text("Meridional Rings (V)"));
            self.camera.ui(ui);
        });
    }

    fn draw_telemetry_readout(&mut self, ui: &mut egui::Ui) {
        self.ensure_evaluated();
        ui.label(egui::RichText::new("Morphometry Readout").strong());
        ui.label(format!(
            "Mean Cortical Thickness: {:.2} mm",
            self.cache.mean_thickness_mm
        ));
        ui.label(format!(
            "Total Surface Energy: {:.1} (E_int {:.1}, E_ext {:.1})",
            self.cache.total_energy, self.cache.internal_energy_val, self.cache.external_energy_val
        ));
        ui.label(format!(
            "Segmentation Accuracy: MRF {:.1}% vs ML {:.1}%",
            self.cache.mrf_accuracy * 100.0,
            self.cache.ml_accuracy * 100.0
        ));
        ui.label(format!(
            "Peak Cohort |t-statistic|: {:.2}",
            self.cache.peak_abs_t_stat
        ));
    }

    pub(super) fn draw_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.diverged()
                && let Some(action) = render_recovery_banner(ui, self.diagnostic_report())
            {
                match action {
                    RecoveryAction::AutoScaleStepSize => self.auto_scale_step_size(),
                    RecoveryAction::ClampParameters => {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    RecoveryAction::ResetParameters => self.reset_parameters(),
                }
            }
            self.populate_scene(ui.visuals().dark_mode);
            if let Some(ref warn_msg) = self.cache.glm_warning {
                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                egui::Frame::new()
                    .fill(palette.error_banner_bg)
                    .stroke(egui::Stroke::new(1.25_f32, palette.warning))
                    .corner_radius(egui::CornerRadius::same(5))
                    .inner_margin(egui::Margin::same(8))
                    .show(ui, |ui| {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(format!("⚠️ {warn_msg}"))
                                    .color(palette.warning)
                                    .strong(),
                            )
                            .sense(egui::Sense::focusable_noninteractive()),
                        );
                    });
                ui.add_space(4.0);
            }

            let helper = format!(
                "{CAMERA_3D_HELPER_TEXT} Overlay: {} (Inflation {:.0}%). Blue = Low, Red = High.",
                self.overlay_mode.label(),
                self.inflation * 100.0
            );
            let mut camera = self.camera;
            Plot3DViewport::new("cortical_morphometry_plot", 38.0)
                .helper_text(&helper)
                .show(ui, &mut camera, |plot_ui| {
                    self.scene_builder.finish(plot_ui);
                });
            self.camera = camera;
        });
    }
}
