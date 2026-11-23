//! Interactive radiation therapy dose calculation & 4-beam IMRT inverse planning tool.

use super::imrt::{
    DoseState, ImrtWorkspace, BASELINE_IMRT_ANGLES_DEG, BASELINE_IMRT_WEIGHTS, CHALLENGE_IMRT_BETA,
    CHALLENGE_IMRT_OAR_LIMIT_GY, CHALLENGE_IMRT_OAR_SPARING, CHALLENGE_IMRT_PRESCRIPTION_GY,
    IMRT_BEAM_COUNT, IMRT_GRID_DIM, IMRT_OAR_CENTER, IMRT_OAR_RADIUS_CM, IMRT_PHANTOM_SIZE_CM,
    IMRT_TUMOR_CENTER, IMRT_TUMOR_RADIUS_CM,
};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::SemanticPalette;
use crate::framework::{
    CalloutManager, ChallengeEvaluation, ExportCapabilities, InteractiveTool, PaneSidePanel,
};
use eframe::egui;
use egui::{Color32, ColorImage, TextureOptions};
use egui_plot::{Plot, PlotImage, PlotPoint, Polygon};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::primitives::PositiveFloat;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};

/// Interactive radiation therapy tool supporting point-source PSF convolution and 4-beam IMRT planning.
pub struct DoseCalculationTool {
    kernel: ExponentialKernel,
    width: usize,
    height: usize,
    /// Whether 4-beam coplanar IMRT mode is active.
    pub imrt_mode: bool,
    /// Coplanar gantry beam angles (degrees) in `[0, 360]`.
    pub beam_angles_deg: [f64; IMRT_BEAM_COUNT],
    /// Coplanar beam weights in `[0, 10]`.
    pub beam_weights: [f64; IMRT_BEAM_COUNT],
    /// Target tumor prescription `D_Rx` (Gy).
    pub prescription_gy: f64,
    /// Organ-at-risk maximum dose limit `D_lim` (Gy).
    pub oar_limit_gy: f64,
    /// Active inverse challenge ID, if any.
    pub active_challenge: Option<&'static str>,
    /// Preallocated zero-allocation 4-beam IMRT workspace.
    pub imrt_workspace: ImrtWorkspace,
    /// Automated and manual plot callouts.
    pub callout_manager: CalloutManager,
    high_contrast: bool,
    texture: Option<egui::TextureHandle>,
}

impl Default for DoseCalculationTool {
    fn default() -> Self {
        let mut tool = Self {
            kernel: ExponentialKernel::new(1.0, 1.0),
            width: 200,
            height: 200,
            imrt_mode: false,
            beam_angles_deg: BASELINE_IMRT_ANGLES_DEG,
            beam_weights: BASELINE_IMRT_WEIGHTS,
            prescription_gy: CHALLENGE_IMRT_PRESCRIPTION_GY,
            oar_limit_gy: CHALLENGE_IMRT_OAR_LIMIT_GY,
            active_challenge: None,
            imrt_workspace: ImrtWorkspace::default(),
            callout_manager: CalloutManager::new(),
            high_contrast: false,
            texture: None,
        };
        tool.refresh_model();
        tool
    }
}

impl DoseCalculationTool {
    /// Re-evaluates the IMRT workspace (when active), updates callouts, and invalidates the texture.
    pub fn refresh_model(&mut self) {
        if self.active_challenge == Some(CHALLENGE_IMRT_OAR_SPARING) {
            self.imrt_mode = true;
            self.kernel.beta = CHALLENGE_IMRT_BETA;
            self.width = IMRT_GRID_DIM;
            self.height = IMRT_GRID_DIM;
            self.prescription_gy = CHALLENGE_IMRT_PRESCRIPTION_GY;
            self.oar_limit_gy = CHALLENGE_IMRT_OAR_LIMIT_GY;
        }
        if self.imrt_mode {
            self.imrt_workspace.evaluate_in_place(
                &self.beam_angles_deg,
                &self.beam_weights,
                self.kernel.beta,
                self.prescription_gy,
                self.oar_limit_gy,
            );
            self.imrt_workspace.sync_callouts(
                self.prescription_gy,
                self.oar_limit_gy,
                &mut self.callout_manager,
            );
        } else {
            let d_ref = self.kernel.value_at(2.0).unwrap_or(0.0);
            let src_lbl = format!(
                "Source A={:.1}, β={:.2}",
                self.kernel.amplitude, self.kernel.beta
            );
            self.callout_manager
                .set_auto_callout("psf_source", [0.0, 0.0], src_lbl, [28.0, -22.0]);
            let ref_lbl = format!("K(2cm) = {:.3}", d_ref);
            self.callout_manager
                .set_auto_callout("psf_r2", [2.0, 0.0], ref_lbl, [28.0, 20.0]);
        }
        self.texture = None;
    }

    /// Applies one projected gradient-descent step on the 4 beam weights via `optimization::update_weights`.
    pub fn apply_gradient_weight_hint(&mut self) {
        self.beam_weights = self.imrt_workspace.one_step_weight_hint(
            &self.beam_weights,
            self.prescription_gy,
            self.oar_limit_gy,
        );
        self.refresh_model();
    }

    pub(crate) fn compute_dose_image_for_contrast(&self, high_contrast: bool) -> ColorImage {
        if self.imrt_mode {
            return self.compute_imrt_image(high_contrast);
        }
        let width = self.width.max(1);
        let height = self.height.max(1);
        let mut pixels = vec![Color32::BLACK; width * height];
        let kernel = self.kernel;
        let dx = IMRT_PHANTOM_SIZE_CM / width as f64;
        let dy = IMRT_PHANTOM_SIZE_CM / height as f64;
        let max_dose = kernel.value_at(0.1).unwrap_or(10.0);

        for y in 0..height {
            let py = (y as f64 - height as f64 / 2.0) * dy;
            for x in 0..width {
                let px = (x as f64 - width as f64 / 2.0) * dx;
                let r = (px * px + py * py).sqrt();
                let dose = if r < 1e-3 {
                    max_dose
                } else {
                    kernel.value_at(r).unwrap_or(0.0)
                };
                let intensity = (dose / max_dose).clamp(0.0, 1.0);
                let rgb = if high_contrast {
                    heatmap_color_high_contrast(intensity)
                } else {
                    heatmap_color(intensity)
                };
                pixels[flatten_2d_index(x, y, width)] = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
        ColorImage::new([width, height], pixels)
    }

    fn compute_imrt_image(&self, high_contrast: bool) -> ColorImage {
        let mut pixels = vec![Color32::BLACK; IMRT_GRID_DIM * IMRT_GRID_DIM];
        let scale_gy = (self.prescription_gy * 1.25).max(10.0);
        for y in 0..IMRT_GRID_DIM {
            let img_y = IMRT_GRID_DIM - 1 - y;
            for x in 0..IMRT_GRID_DIM {
                let dose = self.imrt_workspace.dose_vec[flatten_2d_index(x, y, IMRT_GRID_DIM)];
                let intensity = (dose / scale_gy).clamp(0.0, 1.0);
                let rgb = if high_contrast {
                    heatmap_color_high_contrast(intensity)
                } else {
                    heatmap_color(intensity)
                };
                pixels[flatten_2d_index(x, img_y, IMRT_GRID_DIM)] =
                    Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
        ColorImage::new([IMRT_GRID_DIM, IMRT_GRID_DIM], pixels)
    }

    fn recompute_texture(&mut self, ctx: &egui::Context) {
        let image = self.compute_dose_image_for_contrast(self.high_contrast);
        let filter = if self.imrt_mode {
            TextureOptions::NEAREST
        } else {
            TextureOptions::LINEAR
        };
        self.texture = Some(ctx.load_texture("dose_field", image, filter));
    }

    fn draw_imrt_beam_group(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.group(|ui| {
            ui.strong("4-Beam Coplanar IMRT Plan");
            for b in 0..IMRT_BEAM_COUNT {
                let angle_lbl = format!("Beam {} Angle θ (deg)", b + 1);
                let angle_slider = egui::Slider::new(&mut self.beam_angles_deg[b], 0.0..=360.0)
                    .step_by(1.0)
                    .text(angle_lbl);
                changed |= ui.add(angle_slider).changed();
                let weight_lbl = format!("Beam {} Weight w", b + 1);
                let weight_slider = egui::Slider::new(&mut self.beam_weights[b], 0.0..=6.0)
                    .step_by(0.05)
                    .text(weight_lbl);
                changed |= ui.add(weight_slider).changed();
            }
            let hint_tip = "Take one projected gradient descent step on the 4 beam weights via optimization::update_weights";
            if ui
                .button("✨ 1-Step Weight Hint (update_weights)")
                .accessible_hover_text(hint_tip)
                .clicked()
            {
                self.apply_gradient_weight_hint();
                let s = &self.imrt_workspace.summary;
                let msg = format!(
                    "Applied 1 update_weights step: cost J = {:.1}, OAR max = {:.1} Gy",
                    s.cost_j, s.oar_max_gy.0
                );
                announce_live_status(ui.ctx(), msg);
            }
        });
        changed
    }

    fn draw_imrt_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = self.draw_imrt_beam_group(ui);
        ui.add_enabled_ui(self.active_challenge.is_none(), |ui| {
            let beta_s =
                egui::Slider::new(&mut self.kernel.beta, 0.01..=1.0).text("Attenuation β (1/cm)");
            changed |= ui.add(beta_s).changed();
            let rx_s =
                egui::Slider::new(&mut self.prescription_gy, 20.0..=80.0).text("PTV D_Rx (Gy)");
            changed |= ui.add(rx_s).changed();
            let lim_s =
                egui::Slider::new(&mut self.oar_limit_gy, 5.0..=50.0).text("OAR D_lim (Gy)");
            changed |= ui.add(lim_s).changed();
        });
        let s = &self.imrt_workspace.summary;
        let p = SemanticPalette::from_visuals(ui.visuals());
        let color = if s.passed { p.success } else { p.warning };
        let summary = format!(
            "Cost J = {:.1} (J_base = {:.1}) | PTV = {:.1} Gy | OAR max = {:.1}/{:.0} Gy",
            s.cost_j, s.baseline_cost_j, s.tumor_mean_gy, s.oar_max_gy.0, self.oar_limit_gy
        );
        ui.colored_label(color, summary);
        changed
    }

    fn draw_contour_overlays(&self, palette: &SemanticPalette, plot_ui: &mut egui_plot::PlotUi) {
        if !self.imrt_mode {
            return;
        }
        let ring = |c: [f64; 2], r: f64| -> Vec<[f64; 2]> {
            (0..32)
                .map(|k| {
                    let a = std::f64::consts::TAU * (k as f64) / 32.0;
                    [c[0] + r * a.cos(), c[1] + r * a.sin()]
                })
                .collect()
        };
        let ptv = Polygon::new(
            "PTV Target (D_Rx)",
            ring(IMRT_TUMOR_CENTER, IMRT_TUMOR_RADIUS_CM),
        )
        .fill_color(Color32::TRANSPARENT)
        .stroke(egui::Stroke::new(2.0_f32, palette.success));
        let oar = Polygon::new("OAR (D_lim)", ring(IMRT_OAR_CENTER, IMRT_OAR_RADIUS_CM))
            .fill_color(Color32::TRANSPARENT)
            .stroke(egui::Stroke::new(2.0_f32, palette.error));
        plot_ui.polygon(ptv);
        plot_ui.polygon(oar);
    }
}

impl InteractiveTool for DoseCalculationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Dose Calculation"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::raster_only()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        Some(self.compute_dose_image_for_contrast(self.high_contrast))
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.kernel.amplitude.to_bits().hash(&mut h);
        self.kernel.beta.to_bits().hash(&mut h);
        self.width.hash(&mut h);
        self.height.hash(&mut h);
        self.imrt_mode.hash(&mut h);
        for b in 0..IMRT_BEAM_COUNT {
            self.beam_angles_deg[b].to_bits().hash(&mut h);
            self.beam_weights[b].to_bits().hash(&mut h);
        }
        self.prescription_gy.to_bits().hash(&mut h);
        self.oar_limit_gy.to_bits().hash(&mut h);
        self.active_challenge.hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let state = DoseState {
            amplitude: PositiveFloat::new(self.kernel.amplitude).ok()?,
            beta: PositiveFloat::new(self.kernel.beta).ok()?,
            width: self.width,
            height: self.height,
            imrt_mode: self.imrt_mode,
            beam_angles_deg: self.imrt_mode.then_some(self.beam_angles_deg),
            beam_weights: self.imrt_mode.then_some(self.beam_weights),
            prescription_gy: self.imrt_mode.then_some(self.prescription_gy),
            oar_limit_gy: self.imrt_mode.then_some(self.oar_limit_gy),
        };
        serde_json::to_string(&state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: DoseState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid DoseCalculation state: {e}"))?;
        loaded.validate()?;
        self.kernel = ExponentialKernel::new(loaded.amplitude.value(), loaded.beta.value());
        self.width = loaded.width;
        self.height = loaded.height;
        self.imrt_mode = loaded.imrt_mode;
        self.beam_angles_deg = loaded.beam_angles_deg.unwrap_or(BASELINE_IMRT_ANGLES_DEG);
        self.beam_weights = loaded.beam_weights.unwrap_or(BASELINE_IMRT_WEIGHTS);
        self.prescription_gy = loaded
            .prescription_gy
            .unwrap_or(CHALLENGE_IMRT_PRESCRIPTION_GY);
        self.oar_limit_gy = loaded.oar_limit_gy.unwrap_or(CHALLENGE_IMRT_OAR_LIMIT_GY);
        self.refresh_model();
        Ok(())
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.refresh_model();
                Ok(())
            }
            Some(CHALLENGE_IMRT_OAR_SPARING) => {
                self.active_challenge = Some(CHALLENGE_IMRT_OAR_SPARING);
                self.imrt_mode = true;
                self.refresh_model();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for DoseCalculationTool"
            )),
        }
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge
            .map(|_| self.imrt_workspace.challenge_evaluation(self.oar_limit_gy))
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        let is_high_contrast = crate::app::ThemeMode::from_visuals(parent_ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        let mut changed = false;
        if self.high_contrast != is_high_contrast {
            self.high_contrast = is_high_contrast;
            changed = true;
        }

        let panel_id = crate::framework::scoped_panel_id(parent_ui, "dose_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Dose Parameters");
            ui.add_enabled_ui(self.active_challenge.is_none(), |ui| {
                changed |= ui
                    .checkbox(&mut self.imrt_mode, "4-Beam Coplanar IMRT Mode")
                    .changed();
            });
            ui.separator();
            if self.imrt_mode {
                changed |= self.draw_imrt_controls(ui);
            } else {
                let amp = egui::Slider::new(&mut self.kernel.amplitude, 0.1..=10.0)
                    .text("Source Amplitude");
                let beta =
                    egui::Slider::new(&mut self.kernel.beta, 0.1..=5.0).text("Attenuation (Beta)");
                changed |= ui.add(amp).changed() | ui.add(beta).changed();
                ui.separator();
                ui.label("Kernel: Exponential K(r) = (A / r²) · exp(-βr)");
            }
        });

        if changed {
            self.refresh_model();
        }
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.texture.is_none() {
                self.recompute_texture(&ctx);
            }
            let palette = SemanticPalette::from_visuals(ui.visuals());
            if let Some(texture) = self.texture.clone() {
                Plot::new("dose_heatmap")
                    .data_aspect(1.0)
                    .view_aspect(1.0)
                    .show_grid(false)
                    .show_axes([true, true])
                    .show(ui, |plot_ui| {
                        plot_ui.image(PlotImage::new(
                            "dose_field",
                            texture.id(),
                            PlotPoint::new(0.0, 0.0),
                            [IMRT_PHANTOM_SIZE_CM as f32, IMRT_PHANTOM_SIZE_CM as f32],
                        ));
                        self.draw_contour_overlays(&palette, plot_ui);
                        self.callout_manager.render_on_plot(plot_ui);
                    });
            }
        });
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "DoseCalculationTool",
        display_name: "Dose Calculation",
        domain: "medical",
        tags: &[],
        build: || Box::new(DoseCalculationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for DoseCalculationTool {
    fn theory_description(&self) -> String {
        format!(
            "Radiation therapy dose distribution computed via point spread function (PSF) kernel convolution: $D(\\vec{{r}}) = \\int \\Phi(\\vec{{r}}') K(\\vec{{r}} - \\vec{{r}}') d^3\\vec{{r}}'$, using an exponential kernel $K(r) = \\frac{{A}}{{r^2}} e^{{-\\beta r}}$ with amplitude $A = {:.2}$ and attenuation coefficient $\\beta = {:.2}$.",
            self.kernel.amplitude, self.kernel.beta
        )
    }
    fn phonetic_description(&self) -> String {
        "Radiation therapy dose calculation and inverse IMRT optimization".into()
    }
    fn theory_citation(&self) -> String {
        "The Physics of Radiation Therapy (Khan, 2014); Convolution/superposition dose calculation (Mackie et al., 1985)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([
            (
                "amplitude".into(),
                "Kernel primary energy fluence scaling factor A".into(),
            ),
            (
                "beta".into(),
                "Radial dose attenuation and scattering decay coefficient beta".into(),
            ),
        ])
    }
}

const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Point-Source Dose Kernel","domain":"medical","timestamp":"2026-11-16T00:00:00Z"},"config":{"amplitude":2.5,"beta":0.6,"width":200,"height":200}}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "dose_calculation_starter",
        title: "Point-Source Dose Kernel",
        description: "K(r) = (A/r²)e^(-βr): 2D radial point-spread dose convolution heatmap.",
        category: "Dosimetry",
        icon: "🎯",
        domain: "medical",
        tool_name: Some("DoseCalculationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
