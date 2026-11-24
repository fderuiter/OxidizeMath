#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Far-field Fraunhofer single-slit, multi-slit grating, and circular Airy diffraction tool.

// [cite:optics]

use eframe::egui::{self, Color32, ColorImage};
use egui_plot::ExportPrimitive;
use math_explorer::math_kernel::types::flatten_2d_index;
use math_explorer::physics::optics::{
    fraunhofer_intensity_at_angle, sample_fraunhofer_profile, DiffractionAperture,
    FraunhoferConfig, FraunhoferProfileBuffer, AIRY_FIRST_ZERO_J1,
};
use serde::{Deserialize, Serialize};

use super::presets::{clamp_finite, ApertureChoice, OpticsColormap};
use crate::app::palette::DARK_PALETTE;
use crate::framework::{ExportCapabilities, FiniteStateGuard, InteractiveTool, ToolStateBounds};

pub(super) const PROFILE_SAMPLES: usize = 241;
pub(super) const SCREEN_SIDE: usize = 160;

/// Serializable parameter configuration for [`FraunhoferDiffractionTool`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FraunhoferDiffractionState {
    /// Aperture geometry (single slit, $N$-slit grating, or circular Airy disk).
    pub aperture: ApertureChoice,
    /// WCAG-compliant irradiance colormap.
    pub colormap: OpticsColormap,
    /// Vacuum wavelength $\lambda$ in nanometers (`380.0..=720.0`).
    pub wavelength_nm: f64,
    /// Slit opening width $a$ in micrometers (`0.5..=40.0`).
    pub slit_width_um: f64,
    /// Grating slit-to-slit pitch $d$ in micrometers (`1.0..=80.0`).
    pub slit_pitch_um: f64,
    /// Number of coherent grating slits $N$ (`2..=16`).
    pub slit_count: usize,
    /// Circular Airy aperture diameter $D$ in micrometers (`1.0..=60.0`).
    pub airy_diameter_um: f64,
    /// Observation screen distance $L$ in meters (`0.1..=5.0`).
    pub screen_distance_m: f64,
    /// Sampled angular half-span $\theta_{\max}$ in radians (`0.01..=0.60`).
    pub half_angle_rad: f64,
    /// Applies logarithmic dynamic-range boost to reveal outer secondary rings/sidelobes.
    pub log_intensity_view: bool,
    /// Overlays analytical dark minima / principal maxima markers on the profile plot.
    pub show_critical_markers: bool,
}

impl Default for FraunhoferDiffractionState {
    fn default() -> Self {
        Self {
            aperture: ApertureChoice::CircularAiry,
            colormap: OpticsColormap::PerceptualThermal,
            wavelength_nm: 532.0,
            slit_width_um: 3.0,
            slit_pitch_um: 9.0,
            slit_count: 5,
            airy_diameter_um: 14.0,
            screen_distance_m: 1.0,
            half_angle_rad: 0.16,
            log_intensity_view: true,
            show_critical_markers: true,
        }
    }
}

/// Interactive Fraunhofer far-field diffraction pattern and critical-point profiler.
pub struct FraunhoferDiffractionTool {
    pub(crate) state: FraunhoferDiffractionState,
    pub(crate) profile: FraunhoferProfileBuffer,
    pub(super) screen_pixels: Vec<Color32>,
    pub(super) texture: Option<egui::TextureHandle>,
    pub(super) revision: u64,
}

impl Default for FraunhoferDiffractionTool {
    fn default() -> Self {
        let mut tool = Self {
            state: FraunhoferDiffractionState::default(),
            profile: FraunhoferProfileBuffer::new(PROFILE_SAMPLES, 36),
            screen_pixels: vec![Color32::BLACK; SCREEN_SIDE * SCREEN_SIDE],
            texture: None,
            revision: 1,
        };
        tool.recompute();
        tool
    }
}

impl FraunhoferDiffractionTool {
    pub(crate) fn fraunhofer_config(&self) -> FraunhoferConfig {
        let s = &self.state;
        FraunhoferConfig {
            aperture: s.aperture.build_aperture(
                s.slit_width_um,
                s.slit_pitch_um,
                s.slit_count,
                s.airy_diameter_um,
            ),
            wavelength_nm: s.wavelength_nm,
            screen_distance_m: s.screen_distance_m,
            half_angle_rad: s.half_angle_rad,
        }
    }

    pub(crate) fn recompute(&mut self) {
        let cfg = self.fraunhofer_config();
        sample_fraunhofer_profile(&cfg, PROFILE_SAMPLES, &mut self.profile);
        self.render_screen_pixels(&cfg);
    }

    fn render_screen_pixels(&mut self, cfg: &FraunhoferConfig) {
        let half = ((SCREEN_SIDE - 1) as f64) * 0.5;
        let boost = self.state.log_intensity_view;
        for row in 0..SCREEN_SIDE {
            let uy = ((row as f64) - half) / half;
            for col in 0..SCREEN_SIDE {
                let ux = ((col as f64) - half) / half;
                let raw_i = match cfg.aperture {
                    DiffractionAperture::CircularAiry { .. } => {
                        let r_theta = (ux * ux + uy * uy).sqrt() * cfg.half_angle_rad;
                        fraunhofer_intensity_at_angle(
                            cfg.aperture,
                            cfg.wavelength_nm,
                            r_theta.min(1.4),
                        )
                    }
                    _ => {
                        let tx = ux * cfg.half_angle_rad;
                        let ix = fraunhofer_intensity_at_angle(cfg.aperture, cfg.wavelength_nm, tx);
                        let vert_env = (-2.2 * uy * uy).exp();
                        ix * vert_env
                    }
                };
                let mapped = if boost {
                    (1.0 + 40.0 * raw_i).ln() / 41.0_f64.ln()
                } else {
                    raw_i
                };
                self.screen_pixels[flatten_2d_index(col, row, SCREEN_SIDE)] = self
                    .state
                    .colormap
                    .sample_irradiance(mapped, cfg.wavelength_nm);
            }
        }
    }

    pub(super) fn is_valid_state(s: &FraunhoferDiffractionState) -> bool {
        s.wavelength_nm.is_finite()
            && (380.0..=720.0).contains(&s.wavelength_nm)
            && s.slit_width_um.is_finite()
            && (0.5..=40.0).contains(&s.slit_width_um)
            && s.slit_pitch_um.is_finite()
            && (1.0..=80.0).contains(&s.slit_pitch_um)
            && (2..=16).contains(&s.slit_count)
            && s.airy_diameter_um.is_finite()
            && (1.0..=60.0).contains(&s.airy_diameter_um)
            && s.screen_distance_m.is_finite()
            && (0.1..=5.0).contains(&s.screen_distance_m)
            && s.half_angle_rad.is_finite()
            && (0.01..=0.60).contains(&s.half_angle_rad)
    }

    fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let curve: Vec<[f64; 2]> = self
            .profile
            .screen_y_mm()
            .iter()
            .copied()
            .zip(self.profile.intensities().iter().copied())
            .map(|(y, val)| [y, val])
            .collect();
        let mut prims = vec![ExportPrimitive::Polyline {
            name: "Fraunhofer I(y)/I0".into(),
            points: curve.into(),
            color: DARK_PALETTE.accent,
            stroke_width: 1.8,
            closed: false,
            fill: None,
        }];
        if self.state.show_critical_markers && !self.profile.markers().is_empty() {
            prims.push(ExportPrimitive::Points {
                name: "Critical Points".into(),
                points: self
                    .profile
                    .markers()
                    .iter()
                    .map(|m| [m.screen_y_mm, m.intensity])
                    .collect(),
                color: DARK_PALETTE.warning,
                radius: 2.5,
            });
        }
        prims
    }

    fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let screen_profile: Vec<[f64; 2]> = self
            .profile
            .screen_y_mm()
            .iter()
            .copied()
            .zip(self.profile.intensities().iter().copied())
            .map(|(y, val)| [y, val])
            .collect();
        let angular_profile: Vec<[f64; 2]> = self
            .profile
            .angles_rad()
            .iter()
            .copied()
            .zip(self.profile.intensities().iter().copied())
            .map(|(ang, val)| [ang, val])
            .collect();
        let mut out = vec![
            ("fraunhofer_screen_mm_intensity".into(), screen_profile),
            ("fraunhofer_angle_rad_intensity".into(), angular_profile),
        ];
        if !self.profile.markers().is_empty() {
            out.push((
                "critical_point_markers".into(),
                self.profile
                    .markers()
                    .iter()
                    .map(|m| [m.screen_y_mm, m.intensity])
                    .collect(),
            ));
        }
        out
    }
}

impl FiniteStateGuard for FraunhoferDiffractionTool {
    fn is_finite_state(&self) -> bool {
        Self::is_valid_state(&self.state)
    }

    fn auto_scale_step_size(&mut self) {
        self.state.half_angle_rad = (self.state.half_angle_rad * 0.85).clamp(0.01, 0.60);
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn clamp_parameters(&mut self) {
        let s = &mut self.state;
        s.wavelength_nm = clamp_finite(s.wavelength_nm, 380.0, 720.0, 532.0);
        s.slit_width_um = clamp_finite(s.slit_width_um, 0.5, 40.0, 3.0);
        s.slit_pitch_um = clamp_finite(s.slit_pitch_um, 1.0, 80.0, 9.0).max(s.slit_width_um);
        s.slit_count = s.slit_count.clamp(2, 16);
        s.airy_diameter_um = clamp_finite(s.airy_diameter_um, 1.0, 60.0, 14.0);
        s.screen_distance_m = clamp_finite(s.screen_distance_m, 0.1, 5.0, 1.0);
        s.half_angle_rad = clamp_finite(s.half_angle_rad, 0.01, 0.60, 0.16);
        self.revision = self.revision.wrapping_add(1).max(1);
        self.recompute();
    }
}

impl InteractiveTool for FraunhoferDiffractionTool {
    fn name(&self) -> &'static str {
        "Fraunhofer Diffraction Lab"
    }
    fn domain(&self) -> &'static str {
        "optics"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 4_096,
            max_depth: 5,
            max_array_len: 16,
        }
    }
    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }
    fn auto_scale_step_size(&mut self) -> bool {
        FiniteStateGuard::auto_scale_step_size(self);
        true
    }
    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: FraunhoferDiffractionState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid FraunhoferDiffractionTool state JSON: {e}"))?;
        if !Self::is_valid_state(&parsed) {
            return Err("FraunhoferDiffractionTool parameters out of valid physical bounds".into());
        }
        self.state = parsed;
        self.revision = self.revision.wrapping_add(1).max(1);
        self.recompute();
        Ok(())
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_raster_image(&self) -> Option<ColorImage> {
        Some(ColorImage {
            size: [SCREEN_SIDE, SCREEN_SIDE],
            pixels: self.screen_pixels.clone(),
            ..Default::default()
        })
    }
    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        self.build_export_primitives()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }
    fn step_simulation(&mut self) {
        self.recompute();
    }
    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if !self.has_finite_parameters() {
            FiniteStateGuard::clamp_parameters(self);
        }
        self.draw_controls_panel(parent_ui);
        self.draw_central_workspace(parent_ui);
    }
}

impl scientific_metadata::theory::TheoryDescribable for FraunhoferDiffractionTool {
    fn theory_description(&self) -> String {
        let airy_coeff = AIRY_FIRST_ZERO_J1 / std::f64::consts::PI;
        format!(
            "Far-field Fraunhofer Fourier diffraction ({}, $\\lambda = {:.0}\\,\\text{{nm}}$), resolving single-slit $\\operatorname{{sinc}}^2\\beta$, $N$-slit interference $(\\frac{{\\sin N\\gamma}}{{N\\sin\\gamma}})^2$, and circular Airy $(2J_1(x)/x)^2$ first dark ring $\\sin\\theta_1 \\approx {airy_coeff:.4}\\,\\lambda/D$.",
            self.state.aperture.label(),
            self.state.wavelength_nm
        )
    }
    fn phonetic_description(&self) -> String {
        "Far field Fraunhofer single slit, multi slit grating, and circular Airy diffraction analyzer".into()
    }
    fn theory_citation(&self) -> String {
        "Born & Wolf (1999), Principles of Optics (born1999principles)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
