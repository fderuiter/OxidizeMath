//! Serializable `DoseState` with optional RLE PTV/OAR contour masks, `URL_JSON_BOUNDS`
//! compliance, and `imrt_oar_sparing` inverse challenge clamping.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BASELINE_IMRT_ANGLES_DEG, BASELINE_IMRT_WEIGHTS, CHALLENGE_IMRT_BETA,
    CHALLENGE_IMRT_OAR_LIMIT_GY, CHALLENGE_IMRT_OAR_SPARING, CHALLENGE_IMRT_PRESCRIPTION_GY,
    IMRT_BEAM_COUNT, IMRT_GRID_DIM, IMRT_VOXEL_COUNT,
};
use crate::framework::{decode_rle_mask, ChallengeMetadata, ToolStateBounds};

/// Summary dosimetric observables from a 4-beam IMRT evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImrtSummary {
    /// Quadratic objective cost `J` from `optimization::calculate_cost`.
    pub cost_j: f64,
    /// Calibrated baseline cost `J_base`.
    pub baseline_cost_j: f64,
    /// Mean tumor (`PTV`) dose (Gy).
    pub tumor_mean_gy: f64,
    /// Minimum and maximum tumor (`PTV`) dose `(D_min, D_max)` (Gy).
    pub tumor_min_max_gy: (f64, f64),
    /// Maximum dose `D_oar_max` (Gy) across all `organ_indices` voxels and its `(x, y)` cm point.
    pub oar_max_gy: (f64, [f64; 2]),
    /// Global hotspot dose (Gy) and its `(x, y)` cm coordinate.
    pub global_max_gy: (f64, [f64; 2]),
    /// Score `S_imrt = clamp(100 * (1 - J / J_base), 0, 100)`.
    pub score: f64,
    /// Whether `D_oar_max <= D_lim` and `J <= 0.15 * J_base`.
    pub passed: bool,
}

impl ImrtSummary {
    /// Formats a programmatic `AccessKit` summary and color/contour key for the dose canvas.
    #[must_use]
    pub fn format_dose_a11y_summary(
        &self,
        tool: &crate::tabs::medical::dose::DoseCalculationTool,
    ) -> String {
        if tool.imrt_mode {
            let b = tool.selected_beam;
            format!(
                "IMRT phantom ({}) — Beam {} at {:.0} deg, PTV mean {:.1} Gy (Rx {:.1} Gy), OAR max {:.1} Gy (limit {:.1} Gy); Color/shape key: blue-cyan-yellow-red dose intensity ramp; PTV target green contour; OAR organ-at-risk red contour; 4 coplanar gantry rays and circular handles",
                tool.display_mode.label(), b + 1, tool.beam_angles_deg[b], self.tumor_mean_gy, tool.prescription_gy, self.oar_max_gy.0, tool.oar_limit_gy
            )
        } else {
            format!(
                "Point-source exponential dose kernel A={:.1}, beta={:.2}; Color key: blue-cyan-yellow-red dose intensity ramp from low peripheral dose to central isocenter peak",
                tool.amplitude, tool.beta
            )
        }
    }
}

/// Selects which dose matrix is rendered on the `64 x 64` IMRT phantom heatmap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DoseDisplayMode {
    /// Planned static 4-beam reference dose distribution.
    #[default]
    StaticPlan,
    /// Cumulative 4D motion-compensated dose delivered strictly during `BeamActive`.
    Gated4d,
    /// Cumulative 4D uncompensated dose delivered across all respiratory phases.
    Ungated4d,
}

impl DoseDisplayMode {
    /// Human-readable UI label for the active heatmap display mode.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::StaticPlan => "Static Plan",
            Self::Gated4d => "Gated 4D",
            Self::Ungated4d => "Ungated 4D",
        }
    }
}

/// Controls whether arrow keys on the focused plot canvas steer the brush reticle or gantry beams.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CanvasKeyTarget {
    /// Arrow keys move the `SpatialBrushKit` contour reticle; `Space`/`Enter` stamps.
    #[default]
    Brush,
    /// Arrow keys nudge the selected coplanar gantry angle `±1°` (`Shift ±5°`).
    Gantry,
}

/// Serializable state for [`crate::tabs::medical::dose::DoseCalculationTool`] preserving
/// strict byte-for-byte backward preset compatibility when optional fields are unset.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoseState {
    /// Point-source amplitude `A`.
    pub amplitude: f64,
    /// Radial attenuation coefficient `beta`.
    pub beta: f64,
    /// Raster width in pixels.
    pub width: usize,
    /// Raster height in pixels.
    pub height: usize,
    /// Whether 4-beam IMRT mode is enabled.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub imrt_mode: bool,
    /// Coplanar gantry beam angles (deg).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beam_angles_deg: Option<[f64; IMRT_BEAM_COUNT]>,
    /// Coplanar beam intensity weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beam_weights: Option<[f64; IMRT_BEAM_COUNT]>,
    /// Tumor target prescription `D_Rx` (Gy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prescription_gy: Option<f64>,
    /// Organ-at-risk maximum dose limit `D_lim` (Gy).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oar_limit_gy: Option<f64>,
    /// Whether 4D respiratory-gated delivery mode is enabled.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gating_4d: bool,
    /// LINAC pulse gun current `I_b` in milliamps (`[0, 400]`) feeding `beam_loading_energy`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beam_current_ma: Option<f64>,
    /// Optional custom sculpted PTV mask encoded via `encode_rle_mask` (`[val, run]` pairs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ptv_rle: Option<Vec<[u32; 2]>>,
    /// Optional custom sculpted OAR mask encoded via `encode_rle_mask` (`[val, run]` pairs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oar_rle: Option<Vec<[u32; 2]>>,
}

impl DoseState {
    /// Strict `< 4 KiB` JSON state budget matching `URL_JSON_BOUNDS` capacity.
    pub const STATE_BOUNDS: ToolStateBounds = ToolStateBounds::new(4_096, 8, 2_048);

    /// Validates finiteness, physical parameter ranges, and RLE contour lengths.
    pub fn validate(&self) -> Result<(), String> {
        if !self.amplitude.is_finite() || !(0.1..=10.0).contains(&self.amplitude) {
            return Err(format!("amplitude out of [0.1, 10.0]: {}", self.amplitude));
        }
        let min_beta = if self.imrt_mode { 0.01 } else { 0.1 };
        if !self.beta.is_finite() || !(min_beta..=5.0).contains(&self.beta) {
            return Err(format!("beta out of [{min_beta}, 5.0]: {}", self.beta));
        }
        if !(10..=1024).contains(&self.width) || !(10..=1024).contains(&self.height) {
            return Err("width and height must be in [10, 1024]".to_string());
        }
        self.validate_imrt_scalars()?;
        self.validate_rle_masks()
    }

    fn validate_imrt_scalars(&self) -> Result<(), String> {
        if let Some(angles) = &self.beam_angles_deg {
            if !angles
                .iter()
                .all(|a| a.is_finite() && (0.0..=360.0).contains(a))
            {
                return Err("beam_angles_deg must be finite in [0, 360]".to_string());
            }
        }
        if let Some(weights) = &self.beam_weights {
            if !weights
                .iter()
                .all(|w| w.is_finite() && (0.0..=10.0).contains(w))
            {
                return Err("beam_weights must be finite in [0, 10]".to_string());
            }
        }
        if let Some(rx) = self.prescription_gy {
            if !rx.is_finite() || !(1.0..=120.0).contains(&rx) {
                return Err("prescription_gy out of [1, 120]".to_string());
            }
        }
        if let Some(lim) = self.oar_limit_gy {
            if !lim.is_finite() || !(1.0..=100.0).contains(&lim) {
                return Err("oar_limit_gy out of [1, 100]".to_string());
            }
        }
        if let Some(curr) = self.beam_current_ma {
            if !curr.is_finite() || !(0.0..=400.0).contains(&curr) {
                return Err("beam_current_ma out of [0, 400]".to_string());
            }
        }
        Ok(())
    }

    fn validate_rle_masks(&self) -> Result<(), String> {
        if let Some(runs) = &self.ptv_rle {
            let decoded = decode_rle_mask(runs, IMRT_VOXEL_COUNT)?;
            if decoded.iter().any(|&v| v > 1) {
                return Err("ptv_rle contains non-binary cell value".to_string());
            }
        }
        if let Some(runs) = &self.oar_rle {
            let decoded = decode_rle_mask(runs, IMRT_VOXEL_COUNT)?;
            if decoded.iter().any(|&v| v > 1) {
                return Err("oar_rle contains non-binary cell value".to_string());
            }
        }
        Ok(())
    }
}

fn generate_imrt_challenge_baseline() -> String {
    serde_json::to_string(&DoseState {
        amplitude: 2.5,
        beta: CHALLENGE_IMRT_BETA,
        width: IMRT_GRID_DIM,
        height: IMRT_GRID_DIM,
        imrt_mode: true,
        beam_angles_deg: Some(BASELINE_IMRT_ANGLES_DEG),
        beam_weights: Some(BASELINE_IMRT_WEIGHTS),
        prescription_gy: Some(CHALLENGE_IMRT_PRESCRIPTION_GY),
        oar_limit_gy: Some(CHALLENGE_IMRT_OAR_LIMIT_GY),
        gating_4d: false,
        beam_current_ma: None,
        ptv_rle: None,
        oar_rle: None,
    })
    .unwrap_or_default()
}

fn clamp_imrt_challenge_state(raw_json: &str) -> Result<String, String> {
    let mut state: DoseState =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid IMRT challenge JSON: {e}"))?;
    state.amplitude = state.amplitude.clamp(0.1, 10.0);
    state.imrt_mode = true;
    state.beta = CHALLENGE_IMRT_BETA;
    state.width = IMRT_GRID_DIM;
    state.height = IMRT_GRID_DIM;
    state.prescription_gy = Some(CHALLENGE_IMRT_PRESCRIPTION_GY);
    state.oar_limit_gy = Some(CHALLENGE_IMRT_OAR_LIMIT_GY);
    state.gating_4d = false;
    state.beam_current_ma = None;
    state.ptv_rle = None;
    state.oar_rle = None;
    let mut angles = state.beam_angles_deg.unwrap_or(BASELINE_IMRT_ANGLES_DEG);
    for a in &mut angles {
        *a = a.clamp(0.0, 360.0);
    }
    let mut weights = state.beam_weights.unwrap_or(BASELINE_IMRT_WEIGHTS);
    for w in &mut weights {
        *w = w.clamp(0.0, 10.0);
    }
    state.beam_angles_deg = Some(angles);
    state.beam_weights = Some(weights);
    state.validate()?;
    serde_json::to_string(&state).map_err(|e| format!("IMRT serialize error: {e}"))
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_IMRT_OAR_SPARING,
        title: "4-Beam IMRT OAR Sparing",
        description: "Tune 4 coplanar ExponentialKernel beam gantry angles and weights on a 64×64 phantom to deliver D_Rx = 60 Gy to the central tumor while keeping OAR max dose D_oar,max ≤ 26 Gy and cutting quadratic cost to J ≤ 0.15 · J_base.",
        category: "Dosimetry & Radiotherapy",
        icon: "🎯",
        domain: "medical",
        tool_name: "DoseCalculationTool",
        locked_inputs: "Grid = 64×64, β = 0.04 cm⁻¹, PTV r ≤ 1.8 cm (D_Rx = 60 Gy), OAR @ (4.5, 0) cm (D_lim = 26 Gy)",
        tunable_budget: "4 Beam Angles θ_b ∈ [0°, 360°] and 4 Beam Weights w_b ∈ [0, 10]",
        target_summary: "D_oar,max ≤ 26.0 Gy and J ≤ 0.15 · J_base",
        baseline_state_generator: generate_imrt_challenge_baseline,
        clamp_locked_state: clamp_imrt_challenge_state,
    }
}

// [cite:favorite_child]
