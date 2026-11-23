//! Preallocated 4-beam IMRT inverse planning workspace (`ExponentialKernel`, `calculate_cost`, `update_weights`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{CalloutManager, ChallengeEvaluation, ChallengeMetadata};
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::primitives::PositiveFloat;
use math_explorer::nalgebra::DVector;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};
use math_explorer::physics::medical::optimization::{calculate_cost, update_weights};

/// IMRT phantom grid dimension (`64 x 64 = 4,096` voxels).
pub const IMRT_GRID_DIM: usize = 64;
/// Total IMRT phantom voxel count.
pub const IMRT_VOXEL_COUNT: usize = IMRT_GRID_DIM * IMRT_GRID_DIM;
/// Physical width and height of the phantom (`[-10, 10] x [-10, 10]` cm).
pub const IMRT_PHANTOM_SIZE_CM: f64 = 20.0;
/// Number of coplanar IMRT photon beams.
pub const IMRT_BEAM_COUNT: usize = 4;

/// Source-axis distance (cm) from virtual gantry source to `(0, 0)` isocenter.
const IMRT_SAD_CM: f64 = 14.0;
/// Collimated lateral beam half-width `sigma_lat` (cm).
const IMRT_BEAM_SIGMA_CM: f64 = 1.85;
/// Reference dose (Gy) delivered at `(0, 0)` by a single beam of weight `1.0`.
const IMRT_ISOCENTER_GY_PER_WEIGHT: f64 = 10.0;

/// Tumor target (`PTV`) center `(0.0, 0.0)` cm and radius `1.8` cm.
pub const IMRT_TUMOR_CENTER: [f64; 2] = [0.0, 0.0];
/// Tumor target radius (cm).
pub const IMRT_TUMOR_RADIUS_CM: f64 = 1.8;
/// Organ-at-risk (`OAR`) center `(4.5, 0.0)` cm and radius `1.4` cm.
pub const IMRT_OAR_CENTER: [f64; 2] = [4.5, 0.0];
/// Organ-at-risk radius (cm).
pub const IMRT_OAR_RADIUS_CM: f64 = 1.4;

/// Canonical challenge ID for the IMRT OAR-sparing inverse planning challenge.
pub const CHALLENGE_IMRT_OAR_SPARING: &str = "imrt_oar_sparing";
/// Locked attenuation coefficient `beta` (1/cm) for `imrt_oar_sparing`.
pub const CHALLENGE_IMRT_BETA: f64 = 0.04;
/// Locked tumor prescription dose `D_Rx` (Gy) for `imrt_oar_sparing`.
pub const CHALLENGE_IMRT_PRESCRIPTION_GY: f64 = 60.0;
/// Locked organ-at-risk dose limit `D_lim` (Gy) for `imrt_oar_sparing`.
pub const CHALLENGE_IMRT_OAR_LIMIT_GY: f64 = 26.0;
/// Quadratic tumor weight `alpha` in `calculate_cost`.
pub const IMRT_COST_ALPHA: f64 = 1.0;
/// Quadratic OAR overdose weight `beta` in `calculate_cost`.
pub const IMRT_COST_BETA_OAR: f64 = 3.0;

/// Unsolved baseline gantry angles (deg) for `imrt_oar_sparing`.
pub const BASELINE_IMRT_ANGLES_DEG: [f64; IMRT_BEAM_COUNT] = [0.0, 90.0, 180.0, 270.0];
/// Unsolved baseline beam weights for `imrt_oar_sparing`.
pub const BASELINE_IMRT_WEIGHTS: [f64; IMRT_BEAM_COUNT] = [3.6, 0.6, 1.4, 0.6];

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

/// Preallocated zero-heap-allocation 4-beam IMRT workspace.
pub struct ImrtWorkspace {
    /// Precomputed fixed voxel indices inside the tumor target (`PTV`).
    pub tumor_indices: Vec<usize>,
    /// Precomputed fixed voxel indices inside the organ at risk (`OAR`).
    pub organ_indices: Vec<usize>,
    /// Unit-weight directional `ExponentialKernel` dose columns (`4 x 4096`).
    pub unit_beams: Box<[[f64; IMRT_VOXEL_COUNT]; IMRT_BEAM_COUNT]>,
    /// Preallocated total voxel dose vector passed to `calculate_cost`.
    pub dose_vec: DVector<f64>,
    /// Calibrated unsolved baseline objective cost `J_base`.
    pub baseline_cost_j: f64,
    /// Latest dosimetric evaluation summary.
    pub summary: ImrtSummary,
}

fn cached_imrt_baseline(tumor_indices: &[usize], organ_indices: &[usize]) -> f64 {
    static BASE_J: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *BASE_J.get_or_init(|| {
        let mut ws = ImrtWorkspace {
            tumor_indices: tumor_indices.to_vec(),
            organ_indices: organ_indices.to_vec(),
            unit_beams: Box::new([[0.0; IMRT_VOXEL_COUNT]; IMRT_BEAM_COUNT]),
            dose_vec: DVector::zeros(IMRT_VOXEL_COUNT),
            baseline_cost_j: 1.0,
            summary: ImrtSummary::default(),
        };
        let base = ws.evaluate_in_place(
            &BASELINE_IMRT_ANGLES_DEG,
            &BASELINE_IMRT_WEIGHTS,
            CHALLENGE_IMRT_BETA,
            CHALLENGE_IMRT_PRESCRIPTION_GY,
            CHALLENGE_IMRT_OAR_LIMIT_GY,
        );
        base.cost_j.max(1e-6)
    })
}

impl Default for ImrtWorkspace {
    fn default() -> Self {
        let mut tumor_indices = Vec::with_capacity(128);
        let mut organ_indices = Vec::with_capacity(128);
        for y in 0..IMRT_GRID_DIM {
            let py = voxel_coord_cm(y);
            for x in 0..IMRT_GRID_DIM {
                let px = voxel_coord_cm(x);
                let idx = flatten_2d_index(x, y, IMRT_GRID_DIM);
                let dt2 = (px - IMRT_TUMOR_CENTER[0]).powi(2) + (py - IMRT_TUMOR_CENTER[1]).powi(2);
                if dt2 <= IMRT_TUMOR_RADIUS_CM * IMRT_TUMOR_RADIUS_CM {
                    tumor_indices.push(idx);
                }
                let do2 = (px - IMRT_OAR_CENTER[0]).powi(2) + (py - IMRT_OAR_CENTER[1]).powi(2);
                if do2 <= IMRT_OAR_RADIUS_CM * IMRT_OAR_RADIUS_CM {
                    organ_indices.push(idx);
                }
            }
        }
        let baseline_cost_j = cached_imrt_baseline(&tumor_indices, &organ_indices);
        Self {
            tumor_indices,
            organ_indices,
            unit_beams: Box::new([[0.0; IMRT_VOXEL_COUNT]; IMRT_BEAM_COUNT]),
            dose_vec: DVector::zeros(IMRT_VOXEL_COUNT),
            baseline_cost_j,
            summary: ImrtSummary {
                cost_j: baseline_cost_j,
                baseline_cost_j,
                ..ImrtSummary::default()
            },
        }
    }
}

/// Converts a `0..64` grid index to physical phantom coordinates in `[-10.0, 10.0]` cm.
#[inline]
#[must_use]
pub fn voxel_coord_cm(index: usize) -> f64 {
    let step = IMRT_PHANTOM_SIZE_CM / IMRT_GRID_DIM as f64;
    (index as f64 + 0.5) * step - 0.5 * IMRT_PHANTOM_SIZE_CM
}

impl ImrtWorkspace {
    /// Evaluates all 4 directional `ExponentialKernel` beams and `optimization::calculate_cost`
    /// in place with zero dynamic heap allocations.
    pub fn evaluate_in_place(
        &mut self,
        angles_deg: &[f64; IMRT_BEAM_COUNT],
        weights: &[f64; IMRT_BEAM_COUNT],
        beta: f64,
        prescription_gy: f64,
        oar_limit_gy: f64,
    ) -> ImrtSummary {
        let clean_beta = beta.clamp(0.01, 5.0);
        let amp = IMRT_ISOCENTER_GY_PER_WEIGHT
            * IMRT_SAD_CM
            * IMRT_SAD_CM
            * (clean_beta * IMRT_SAD_CM).exp();
        let kernel = ExponentialKernel::new(amp, clean_beta);

        self.dose_vec.fill(0.0);
        for b in 0..IMRT_BEAM_COUNT {
            let theta = angles_deg[b].to_radians();
            let (sin_t, cos_t) = theta.sin_cos();
            let src_x = IMRT_SAD_CM * cos_t;
            let src_y = IMRT_SAD_CM * sin_t;
            let w_b = weights[b].max(0.0);
            for y in 0..IMRT_GRID_DIM {
                let py = voxel_coord_cm(y);
                for x in 0..IMRT_GRID_DIM {
                    let px = voxel_coord_cm(x);
                    let r_src = ((px - src_x).hypot(py - src_y)).max(2.0);
                    let depth_term = kernel.value_at(r_src).unwrap_or(0.0);
                    let d_perp = -px * sin_t + py * cos_t;
                    let lat_term = (-0.5 * (d_perp / IMRT_BEAM_SIGMA_CM).powi(2)).exp();
                    let unit_val = depth_term * lat_term;
                    let idx = flatten_2d_index(x, y, IMRT_GRID_DIM);
                    self.unit_beams[b][idx] = unit_val;
                    self.dose_vec[idx] += w_b * unit_val;
                }
            }
        }

        let cost_j = calculate_cost(
            &self.dose_vec,
            &self.tumor_indices,
            &self.organ_indices,
            prescription_gy,
            oar_limit_gy,
            IMRT_COST_ALPHA,
            IMRT_COST_BETA_OAR,
        );
        let summary = self.summarize(cost_j, oar_limit_gy);
        self.summary = summary;
        summary
    }

    fn summarize(&self, cost_j: f64, oar_limit_gy: f64) -> ImrtSummary {
        let mut tumor_sum = 0.0;
        let mut tumor_min = f64::INFINITY;
        let mut tumor_max = 0.0_f64;
        for &idx in &self.tumor_indices {
            let d = self.dose_vec[idx];
            tumor_sum += d;
            tumor_min = tumor_min.min(d);
            tumor_max = tumor_max.max(d);
        }
        let tumor_mean_gy = if self.tumor_indices.is_empty() {
            0.0
        } else {
            tumor_sum / self.tumor_indices.len() as f64
        };

        let mut oar_max = (0.0_f64, IMRT_OAR_CENTER);
        for &idx in &self.organ_indices {
            let d = self.dose_vec[idx];
            if d > oar_max.0 {
                let x = idx % IMRT_GRID_DIM;
                let y = idx / IMRT_GRID_DIM;
                oar_max = (d, [voxel_coord_cm(x), voxel_coord_cm(y)]);
            }
        }

        let mut global_max = (0.0_f64, [0.0_f64, 0.0_f64]);
        for y in 0..IMRT_GRID_DIM {
            for x in 0..IMRT_GRID_DIM {
                let d = self.dose_vec[flatten_2d_index(x, y, IMRT_GRID_DIM)];
                if d > global_max.0 {
                    global_max = (d, [voxel_coord_cm(x), voxel_coord_cm(y)]);
                }
            }
        }

        let j_base = self.baseline_cost_j.max(1e-6);
        let score = (100.0 * (1.0 - cost_j / j_base)).clamp(0.0, 100.0);
        let passed = oar_max.0 <= oar_limit_gy && cost_j <= 0.15 * j_base;

        ImrtSummary {
            cost_j,
            baseline_cost_j: j_base,
            tumor_mean_gy,
            tumor_min_max_gy: (tumor_min, tumor_max),
            oar_max_gy: oar_max,
            global_max_gy: global_max,
            score,
            passed,
        }
    }

    /// Computes one projected gradient-descent weight update via `optimization::update_weights`
    /// at the current beam angles.
    #[must_use]
    pub fn one_step_weight_hint(
        &self,
        weights: &[f64; IMRT_BEAM_COUNT],
        prescription_gy: f64,
        oar_limit_gy: f64,
    ) -> [f64; IMRT_BEAM_COUNT] {
        let mut grad = [0.0_f64; IMRT_BEAM_COUNT];
        let mut diag_hess = 1e-6_f64;

        for (b, grad_slot) in grad.iter_mut().enumerate() {
            let beam_k = &self.unit_beams[b];
            let mut g_b = 0.0;
            let mut h_b = 0.0;
            for &idx in &self.tumor_indices {
                let k_bi = beam_k[idx];
                g_b += 2.0 * IMRT_COST_ALPHA * (self.dose_vec[idx] - prescription_gy) * k_bi;
                h_b += 2.0 * IMRT_COST_ALPHA * k_bi * k_bi;
            }
            for &idx in &self.organ_indices {
                let d_i = self.dose_vec[idx];
                let k_bi = beam_k[idx];
                if d_i > oar_limit_gy {
                    g_b += 2.0 * IMRT_COST_BETA_OAR * (d_i - oar_limit_gy) * k_bi;
                }
                h_b += 2.0 * IMRT_COST_BETA_OAR * k_bi * k_bi;
            }
            *grad_slot = g_b;
            diag_hess = diag_hess.max(h_b);
        }

        let w_vec = DVector::from_column_slice(weights);
        let g_vec = DVector::from_column_slice(&grad);
        let eta = 0.65 / diag_hess;
        let updated = update_weights(&w_vec, &g_vec, eta);
        let mut out = [0.0; IMRT_BEAM_COUNT];
        for b in 0..IMRT_BEAM_COUNT {
            out[b] = updated[b].clamp(0.0, 10.0);
        }
        out
    }

    /// Synchronizes `CalloutManager` overlays for the tumor target and OAR peak dose.
    pub fn sync_callouts(
        &self,
        prescription_gy: f64,
        oar_limit_gy: f64,
        callouts: &mut CalloutManager,
    ) {
        let s = &self.summary;
        callouts.set_auto_callout(
            "imrt_tumor",
            IMRT_TUMOR_CENTER,
            format!("PTV {:.1}/{:.0} Gy", s.tumor_mean_gy, prescription_gy),
            [-42.0_f32, -28.0_f32],
        );
        callouts.set_auto_callout(
            "imrt_oar",
            s.oar_max_gy.1,
            format!("OAR max {:.1}/{:.0} Gy", s.oar_max_gy.0, oar_limit_gy),
            [36.0_f32, 24.0_f32],
        );
    }

    /// Formats a [`ChallengeEvaluation`] for `imrt_oar_sparing`.
    #[must_use]
    pub fn challenge_evaluation(&self, oar_limit_gy: f64) -> ChallengeEvaluation {
        let s = &self.summary;
        let j_ratio = s.cost_j / s.baseline_cost_j.max(1e-6);
        ChallengeEvaluation::new(
            s.score,
            s.passed,
            "J / J_base",
            j_ratio,
            format!(
                "J = {:.1} ({:.1}% of J_base {:.1}, need ≤ 15%), D_oar,max = {:.1}/{:.1} Gy, PTV mean = {:.1} Gy",
                s.cost_j,
                j_ratio * 100.0,
                s.baseline_cost_j,
                s.oar_max_gy.0,
                oar_limit_gy,
                s.tumor_mean_gy
            ),
        )
    }
}

/// Serializable state for [`super::dose::DoseCalculationTool`] preserving strict forward preset compatibility.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DoseState {
    /// Point-source amplitude `A`.
    pub amplitude: PositiveFloat,
    /// Radial attenuation coefficient `beta`.
    pub beta: PositiveFloat,
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
}

impl DoseState {
    /// Validates finiteness and parameter ranges for both point-source and IMRT modes.
    pub fn validate(&self) -> Result<(), String> {
        let amplitude = self.amplitude.value();
        let beta = self.beta.value();
        if !(0.1..=10.0).contains(&amplitude) {
            return Err(format!("amplitude out of [0.1, 10.0]: {amplitude}"));
        }
        let min_beta = if self.imrt_mode { 0.01 } else { 0.1 };
        if !(min_beta..=5.0).contains(&beta) {
            return Err(format!("beta out of [{min_beta}, 5.0]: {beta}"));
        }
        if !(10..=1024).contains(&self.width) || !(10..=1024).contains(&self.height) {
            return Err("width and height must be in [10, 1024]".to_string());
        }
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
        Ok(())
    }
}

fn generate_imrt_challenge_baseline() -> String {
    serde_json::to_string(&DoseState {
        amplitude: PositiveFloat::new(2.5).expect("valid baseline amplitude"),
        beta: PositiveFloat::new(CHALLENGE_IMRT_BETA).expect("valid baseline beta"),
        width: IMRT_GRID_DIM,
        height: IMRT_GRID_DIM,
        imrt_mode: true,
        beam_angles_deg: Some(BASELINE_IMRT_ANGLES_DEG),
        beam_weights: Some(BASELINE_IMRT_WEIGHTS),
        prescription_gy: Some(CHALLENGE_IMRT_PRESCRIPTION_GY),
        oar_limit_gy: Some(CHALLENGE_IMRT_OAR_LIMIT_GY),
    })
    .unwrap_or_default()
}

fn clamp_imrt_challenge_state(raw_json: &str) -> Result<String, String> {
    let mut state: DoseState =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid IMRT challenge JSON: {e}"))?;
    state.amplitude =
        PositiveFloat::new(state.amplitude.value().clamp(0.1, 10.0)).map_err(|e| e.to_string())?;
    state.imrt_mode = true;
    state.beta = PositiveFloat::new(CHALLENGE_IMRT_BETA).map_err(|e| e.to_string())?;
    state.width = IMRT_GRID_DIM;
    state.height = IMRT_GRID_DIM;
    state.prescription_gy = Some(CHALLENGE_IMRT_PRESCRIPTION_GY);
    state.oar_limit_gy = Some(CHALLENGE_IMRT_OAR_LIMIT_GY);
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
