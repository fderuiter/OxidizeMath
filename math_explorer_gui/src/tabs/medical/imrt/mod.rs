//! Preallocated 4-beam IMRT inverse planning and 4D respiratory-gated dosimetry workspace.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod contours;
pub mod motion;
pub mod state;

pub use state::{DoseState, ImrtSummary};

use crate::framework::{BrushStroke, CalloutManager, ChallengeEvaluation};
use crate::tabs::medical::link_bus::GatingTelemetry;
use contours::{
    RleMaskRuns, compute_isodose_dice, decode_bool_mask_rle, encode_bool_mask_rle,
    mask_centroid_cm, paint_contour_stroke, rebuild_indices_from_masks, reset_circular_contours,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::nalgebra::DVector;
use math_explorer::physics::medical::dose::kernel::{DoseKernel, ExponentialKernel};
use math_explorer::physics::medical::optimization::{calculate_cost, update_weights};
use motion::{DoseAccumBuffers, Imrt4dTracker, accumulate_4d_step_in_place};

/// IMRT phantom grid dimension (`64 x 64 = 4,096` voxels).
pub const IMRT_GRID_DIM: usize = 64;
/// Total IMRT phantom voxel count.
pub const IMRT_VOXEL_COUNT: usize = IMRT_GRID_DIM * IMRT_GRID_DIM;
/// Physical width and height of the phantom (`[-10, 10] x [-10, 10]` cm).
pub const IMRT_PHANTOM_SIZE_CM: f64 = 20.0;
/// Number of coplanar IMRT photon beams.
pub const IMRT_BEAM_COUNT: usize = 4;

const IMRT_SAD_CM: f64 = 14.0;
const IMRT_BEAM_SIGMA_CM: f64 = 1.85;
const IMRT_ISOCENTER_GY_PER_WEIGHT: f64 = 10.0;

/// Tumor target (`PTV`) center `(0.0, 0.0)` cm.
pub const IMRT_TUMOR_CENTER: [f64; 2] = [0.0, 0.0];
/// Tumor target radius (cm).
pub const IMRT_TUMOR_RADIUS_CM: f64 = 1.8;
/// Organ-at-risk (`OAR`) center `(4.5, 0.0)` cm.
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
/// Coplanar OAR-sparing gantry angles (deg) steering beams clear of the `+x` OAR.
pub const OAR_SPARING_4D_ANGLES_DEG: [f64; IMRT_BEAM_COUNT] = [65.0, 115.0, 245.0, 295.0];
/// Balanced beam weights delivering `~60 Gy` PTV coverage while keeping static OAR `< 21 Gy`.
pub const OAR_SPARING_4D_WEIGHTS: [f64; IMRT_BEAM_COUNT] = [1.60, 1.60, 1.60, 1.60];

/// Preallocated zero-heap-allocation 4-beam IMRT and 4D motion-accumulated workspace.
pub struct ImrtWorkspace {
    /// Precomputed voxel indices inside the tumor target (`PTV`).
    pub tumor_indices: Vec<usize>,
    /// Precomputed voxel indices inside the organ at risk (`OAR`).
    pub organ_indices: Vec<usize>,
    /// Boolean mask of PTV voxels (`64 x 64`).
    pub ptv_mask: Box<[bool; IMRT_VOXEL_COUNT]>,
    /// Boolean mask of OAR voxels (`64 x 64`).
    pub oar_mask: Box<[bool; IMRT_VOXEL_COUNT]>,
    /// Whether custom brush-sculpted contours differ from the default circular phantom.
    pub custom_contours: bool,
    isodose_scratch: Box<[bool; IMRT_VOXEL_COUNT]>,
    /// Unit-weight directional `ExponentialKernel` dose columns (`4 x 4096`).
    pub unit_beams: Box<[[f64; IMRT_VOXEL_COUNT]; IMRT_BEAM_COUNT]>,
    /// Preallocated static planned voxel dose vector passed to `calculate_cost`.
    pub dose_vec: DVector<f64>,
    /// Preallocated 4D gated cumulative voxel dose vector (`BeamActive` steps only).
    pub gated_dose_vec: DVector<f64>,
    /// Preallocated 4D ungated cumulative voxel dose vector (all active respiratory phases).
    pub ungated_dose_vec: DVector<f64>,
    /// Cumulative 4D motion, `beam_loading_energy`, and Dice tracking summary.
    pub motion: Imrt4dTracker,
    /// Calibrated unsolved baseline objective cost `J_base`.
    pub baseline_cost_j: f64,
    /// Latest static dosimetric evaluation summary.
    pub summary: ImrtSummary,
}

static BASE_J: std::sync::OnceLock<f64> = std::sync::OnceLock::new();

impl Default for ImrtWorkspace {
    fn default() -> Self {
        let mut tumor_indices = Vec::with_capacity(IMRT_VOXEL_COUNT);
        let mut organ_indices = Vec::with_capacity(IMRT_VOXEL_COUNT);
        let mut ptv_mask = Box::new([false; IMRT_VOXEL_COUNT]);
        let mut oar_mask = Box::new([false; IMRT_VOXEL_COUNT]);
        reset_circular_contours(
            &mut ptv_mask,
            &mut oar_mask,
            &mut tumor_indices,
            &mut organ_indices,
        );
        let baseline_cost_j = BASE_J.get().copied().unwrap_or(0.0);
        Self {
            tumor_indices,
            organ_indices,
            ptv_mask,
            oar_mask,
            custom_contours: false,
            isodose_scratch: Box::new([false; IMRT_VOXEL_COUNT]),
            unit_beams: Box::new([[0.0; IMRT_VOXEL_COUNT]; IMRT_BEAM_COUNT]),
            dose_vec: DVector::zeros(IMRT_VOXEL_COUNT),
            gated_dose_vec: DVector::zeros(IMRT_VOXEL_COUNT),
            ungated_dose_vec: DVector::zeros(IMRT_VOXEL_COUNT),
            motion: Imrt4dTracker::default(),
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
    /// Resets PTV and OAR masks back to the canonical circular phantom without heap allocations.
    pub fn reset_contours_to_canonical(&mut self) {
        if !self.custom_contours {
            return;
        }
        reset_circular_contours(
            &mut self.ptv_mask,
            &mut self.oar_mask,
            &mut self.tumor_indices,
            &mut self.organ_indices,
        );
        self.custom_contours = false;
        self.reset_4d_dose();
    }

    /// Paints a `SpatialBrushKit` stroke onto PTV/OAR masks and resets stale 4D accumulators.
    pub fn apply_brush_stroke(&mut self, stroke: &BrushStroke) -> bool {
        let changed = paint_contour_stroke(
            &mut self.ptv_mask,
            &mut self.oar_mask,
            &mut self.tumor_indices,
            &mut self.organ_indices,
            stroke,
        );
        if changed {
            self.custom_contours = true;
            self.reset_4d_dose();
        }
        changed
    }

    /// Encodes custom PTV and OAR masks as compact RLE pairs when `custom_contours` is active.
    #[must_use]
    pub fn encode_custom_rle_masks(&self) -> (Option<RleMaskRuns>, Option<RleMaskRuns>) {
        if !self.custom_contours {
            return (None, None);
        }
        (
            Some(encode_bool_mask_rle(&self.ptv_mask)),
            Some(encode_bool_mask_rle(&self.oar_mask)),
        )
    }

    /// Restores optional RLE PTV/OAR contour masks from [`DoseState`].
    pub fn restore_rle_masks(
        &mut self,
        ptv_rle: Option<&[[u32; 2]]>,
        oar_rle: Option<&[[u32; 2]]>,
    ) -> Result<(), String> {
        if ptv_rle.is_none() && oar_rle.is_none() {
            self.custom_contours = true;
            self.reset_contours_to_canonical();
            return Ok(());
        }
        reset_circular_contours(
            &mut self.ptv_mask,
            &mut self.oar_mask,
            &mut self.tumor_indices,
            &mut self.organ_indices,
        );
        if let Some(runs) = ptv_rle {
            decode_bool_mask_rle(runs, &mut self.ptv_mask)?;
        }
        if let Some(runs) = oar_rle {
            decode_bool_mask_rle(runs, &mut self.oar_mask)?;
            for idx in 0..IMRT_VOXEL_COUNT {
                if self.ptv_mask[idx] {
                    self.oar_mask[idx] = false;
                }
            }
        }
        rebuild_indices_from_masks(
            &self.ptv_mask,
            &self.oar_mask,
            &mut self.tumor_indices,
            &mut self.organ_indices,
        );
        self.custom_contours = true;
        self.reset_4d_dose();
        Ok(())
    }

    /// Clears accumulated 4D gated and ungated dose vectors in place with zero heap allocations.
    pub fn reset_4d_dose(&mut self) {
        self.gated_dose_vec.fill(0.0);
        self.ungated_dose_vec.fill(0.0);
        let (sd, em, fs) = (
            self.motion.static_dice,
            self.motion.beam_energy_mev,
            self.motion.fluence_scale,
        );
        self.motion = Imrt4dTracker {
            static_dice: sd,
            beam_energy_mev: em,
            fluence_scale: fs,
            ..Imrt4dTracker::default()
        };
    }

    /// Steps 4D respiratory tissue-shifted dose accumulation in place with zero heap allocations.
    pub fn step_4d_delivery_in_place(
        &mut self,
        telemetry: &GatingTelemetry,
        prescription_gy: f64,
        beam_current_ma: f64,
    ) {
        accumulate_4d_step_in_place(
            &mut self.motion,
            DoseAccumBuffers {
                static_dose: self.dose_vec.as_slice(),
                gated_dose: &mut self.gated_dose_vec,
                ungated_dose: &mut self.ungated_dose_vec,
                ptv_mask: &self.ptv_mask,
                tumor_indices: &self.tumor_indices,
                organ_indices: &self.organ_indices,
                isodose_scratch: &mut self.isodose_scratch,
            },
            telemetry,
            prescription_gy,
            beam_current_ma,
        );
    }

    fn populate_beams_and_cost(
        &mut self,
        angles_deg: &[f64; IMRT_BEAM_COUNT],
        weights: &[f64; IMRT_BEAM_COUNT],
        beta: f64,
        prescription_gy: f64,
        oar_limit_gy: f64,
    ) -> f64 {
        let clean_beta = beta.clamp(0.01, 5.0);
        let amp = IMRT_ISOCENTER_GY_PER_WEIGHT
            * IMRT_SAD_CM
            * IMRT_SAD_CM
            * (clean_beta * IMRT_SAD_CM).exp();
        let kernel = ExponentialKernel::new(amp, clean_beta);
        self.dose_vec.fill(0.0);
        for b in 0..IMRT_BEAM_COUNT {
            let (sin_t, cos_t) = angles_deg[b].to_radians().sin_cos();
            let (src_x, src_y, w_b) = (
                IMRT_SAD_CM * cos_t,
                IMRT_SAD_CM * sin_t,
                weights[b].max(0.0),
            );
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
        calculate_cost(
            &self.dose_vec,
            &self.tumor_indices,
            &self.organ_indices,
            prescription_gy,
            oar_limit_gy,
            IMRT_COST_ALPHA,
            IMRT_COST_BETA_OAR,
        )
    }

    /// Evaluates all 4 directional `ExponentialKernel` beams, `optimization::calculate_cost`,
    /// and static `dice_similarity_coefficient` in place with zero dynamic heap allocations.
    pub fn evaluate_in_place(
        &mut self,
        angles_deg: &[f64; IMRT_BEAM_COUNT],
        weights: &[f64; IMRT_BEAM_COUNT],
        beta: f64,
        prescription_gy: f64,
        oar_limit_gy: f64,
    ) -> ImrtSummary {
        if self.baseline_cost_j <= 0.0 {
            self.baseline_cost_j = *BASE_J.get_or_init(|| {
                let mut ref_ws = ImrtWorkspace {
                    baseline_cost_j: 1.0,
                    ..ImrtWorkspace::default()
                };
                ref_ws
                    .populate_beams_and_cost(
                        &BASELINE_IMRT_ANGLES_DEG,
                        &BASELINE_IMRT_WEIGHTS,
                        CHALLENGE_IMRT_BETA,
                        CHALLENGE_IMRT_PRESCRIPTION_GY,
                        CHALLENGE_IMRT_OAR_LIMIT_GY,
                    )
                    .max(1e-6)
            });
        }
        let cost_j =
            self.populate_beams_and_cost(angles_deg, weights, beta, prescription_gy, oar_limit_gy);
        self.motion.static_dice = compute_isodose_dice(
            self.dose_vec.as_slice(),
            &self.ptv_mask,
            &mut self.isodose_scratch,
            prescription_gy,
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
            tumor_min = 0.0;
            0.0
        } else {
            tumor_sum / self.tumor_indices.len() as f64
        };
        let mut oar_max = (0.0_f64, IMRT_OAR_CENTER);
        for &idx in &self.organ_indices {
            let d = self.dose_vec[idx];
            if d > oar_max.0 {
                oar_max = (
                    d,
                    [
                        voxel_coord_cm(idx % IMRT_GRID_DIM),
                        voxel_coord_cm(idx / IMRT_GRID_DIM),
                    ],
                );
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
        ImrtSummary {
            cost_j,
            baseline_cost_j: j_base,
            tumor_mean_gy,
            tumor_min_max_gy: (tumor_min, tumor_max),
            oar_max_gy: oar_max,
            global_max_gy: global_max,
            score,
            passed: oar_max.0 <= oar_limit_gy && cost_j <= 0.15 * j_base,
        }
    }

    /// Computes one projected gradient-descent weight update via `optimization::update_weights`.
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
            let (mut g_b, mut h_b) = (0.0, 0.0);
            for &idx in &self.tumor_indices {
                let k_bi = beam_k[idx];
                g_b += 2.0 * IMRT_COST_ALPHA * (self.dose_vec[idx] - prescription_gy) * k_bi;
                h_b += 2.0 * IMRT_COST_ALPHA * k_bi * k_bi;
            }
            for &idx in &self.organ_indices {
                let (d_i, k_bi) = (self.dose_vec[idx], beam_k[idx]);
                if d_i > oar_limit_gy {
                    g_b += 2.0 * IMRT_COST_BETA_OAR * (d_i - oar_limit_gy) * k_bi;
                }
                h_b += 2.0 * IMRT_COST_BETA_OAR * k_bi * k_bi;
            }
            *grad_slot = g_b;
            diag_hess = diag_hess.max(h_b);
        }
        let updated = update_weights(
            &DVector::from_column_slice(weights),
            &DVector::from_column_slice(&grad),
            0.65 / diag_hess,
        );
        std::array::from_fn(|b| updated[b].clamp(0.0, 10.0))
    }

    /// Synchronizes `CalloutManager` overlays for the tumor target and OAR peak dose.
    pub fn sync_callouts(
        &self,
        prescription_gy: f64,
        oar_limit_gy: f64,
        callouts: &mut CalloutManager,
    ) {
        let s = &self.summary;
        let ptv_center = mask_centroid_cm(&self.tumor_indices, IMRT_TUMOR_CENTER);
        callouts.set_auto_callout(
            "imrt_tumor",
            ptv_center,
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

// [cite:favorite_child]
