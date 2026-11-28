//! Zero-allocation 4D respiratory tissue-shift dosimetry, `beam_loading_energy` scaling,
//! and single-pane standalone Kalman gating stepper.

use super::contours::compute_isodose_dice;
use super::{
    voxel_coord_cm, IMRT_GRID_DIM, IMRT_OAR_CENTER, IMRT_PHANTOM_SIZE_CM, IMRT_VOXEL_COUNT,
};
use crate::tabs::medical::link_bus::GatingTelemetry;
use crate::tabs::medical::radar_gating::challenge::fast_cv_from_tracking_filter;
use crate::tabs::medical::radar_gating::pipeline::{HISTORY_LEN, SIM_DT};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::engine_diffeq::FastCvKalman;
use math_explorer::nalgebra::DVector;
use math_explorer::physics::medical::accelerator::beam_loading_energy;
use math_explorer::physics::medical::evaluation::{
    pearson_correlation, root_mean_square_error, spatial_accuracy_error,
};
use math_explorer::physics::medical::optical_motion::simulation::cosine_respiratory_curve;
use math_explorer::physics::medical::radar_gating::{
    BeamState, ConstantVelocityModel, GatingLogic, TrackingFilter,
};
use std::f64::consts::PI;

/// Nominal LINAC pulse gun current in milliamps (`5.925 MeV` loaded energy).
pub const NOMINAL_BEAM_CURRENT_MA: f64 = 50.0;
/// End-exhale planning CT reference chest-wall position in millimeters.
pub const PLANNING_EXHALE_REF_MM: f64 = -2.5;
/// Internal thoracic/diaphragmatic lateral tissue-shift ratio (`cm` per `mm` surface excursion).
pub const INTERNAL_MOTION_GAIN_CM_PER_MM: f64 = 0.24;
/// Nominal radar slant-range reference in millimeters for single-pane `spatial_accuracy_error`.
const STANDALONE_RANGE_MM: f64 = 700.0;

/// Computes verified `accelerator::beam_loading_energy` (MeV) and TG-51 calibrated fluence scale.
#[must_use]
pub fn linac_energy_and_fluence_scale(beam_current_ma: f64) -> (f64, f64) {
    let clamped_ma = if beam_current_ma.is_finite() {
        beam_current_ma.clamp(0.0, 400.0)
    } else {
        NOMINAL_BEAM_CURRENT_MA
    };
    let e_mev = beam_loading_energy(clamped_ma);
    let e_nominal = beam_loading_energy(NOMINAL_BEAM_CURRENT_MA).max(1.0e-6);
    let scale = (e_mev / e_nominal).clamp(0.20, 1.25);
    (e_mev, scale)
}

/// Cumulative 4D gated vs. ungated dosimetric and tracking metrics.
#[derive(Clone, Copy, Debug)]
pub struct Imrt4dTracker {
    /// Active 4D steps delivered while `BeamState == BeamActive`.
    pub gated_steps: u64,
    /// Active 4D steps delivered in ungated mode (all non-`Standby` / non-`SafetyInterlock` steps).
    pub ungated_steps: u64,
    /// Total 4D motion steps processed since reset.
    pub total_4d_steps: u64,
    /// Latest internal tissue shift `delta_x` in centimeters relative to planning exhale.
    pub latest_shift_cm: f64,
    /// Loaded LINAC beam energy in MeV from `accelerator::beam_loading_energy`.
    pub beam_energy_mev: f64,
    /// Fluence scaling ratio `E_load(I_b) / E_load(0)` applied to active beams.
    pub fluence_scale: f64,
    /// Cumulative gated mean PTV dose (Gy).
    pub gated_ptv_mean_gy: f64,
    /// Cumulative gated OAR peak dose (Gy) and its `(x, y)` cm coordinate.
    pub gated_oar_max_gy: (f64, [f64; 2]),
    /// Cumulative ungated mean PTV dose (Gy).
    pub ungated_ptv_mean_gy: f64,
    /// Cumulative ungated OAR peak dose (Gy) and its `(x, y)` cm coordinate.
    pub ungated_oar_max_gy: (f64, [f64; 2]),
    /// Static plan `dice_similarity_coefficient` (`95% D_Rx` isodose vs. PTV mask).
    pub static_dice: f64,
    /// Cumulative gated 4D `dice_similarity_coefficient` (`95% D_Rx` isodose vs. PTV mask).
    pub gated_dice: f64,
    /// Cumulative ungated 4D `dice_similarity_coefficient` (`95% D_Rx` isodose vs. PTV mask).
    pub ungated_dice: f64,
    /// Latest verified `evaluation::pearson_correlation` from 4D telemetry.
    pub eval_pearson_r: f64,
    /// Latest verified `evaluation::root_mean_square_error` (mm) from 4D telemetry.
    pub eval_rmse_mm: f64,
    /// Latest verified `evaluation::spatial_accuracy_error` (%) from 4D telemetry.
    pub eval_spatial_err_pct: f64,
}

impl Default for Imrt4dTracker {
    fn default() -> Self {
        let (beam_energy_mev, fluence_scale) =
            linac_energy_and_fluence_scale(NOMINAL_BEAM_CURRENT_MA);
        Self {
            gated_steps: 0,
            ungated_steps: 0,
            total_4d_steps: 0,
            latest_shift_cm: 0.0,
            beam_energy_mev,
            fluence_scale,
            gated_ptv_mean_gy: 0.0,
            gated_oar_max_gy: (0.0, IMRT_OAR_CENTER),
            ungated_ptv_mean_gy: 0.0,
            ungated_oar_max_gy: (0.0, IMRT_OAR_CENTER),
            static_dice: 0.0,
            gated_dice: 0.0,
            ungated_dice: 0.0,
            eval_pearson_r: 1.0,
            eval_rmse_mm: 0.0,
            eval_spatial_err_pct: 0.0,
        }
    }
}

/// Bilinearly samples a `64 x 64` room-fixed dose grid at `(x_cm, y_cm)` with zero allocations.
#[inline]
#[must_use]
pub fn sample_static_dose_bilinear(static_dose: &[f64], x_cm: f64, y_cm: f64) -> f64 {
    let step = IMRT_PHANTOM_SIZE_CM / IMRT_GRID_DIM as f64;
    let max_f = (IMRT_GRID_DIM - 1) as f64;
    let fx = (((x_cm + 0.5 * IMRT_PHANTOM_SIZE_CM) / step) - 0.5).clamp(0.0, max_f);
    let fy = (((y_cm + 0.5 * IMRT_PHANTOM_SIZE_CM) / step) - 0.5).clamp(0.0, max_f);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(IMRT_GRID_DIM - 1);
    let y1 = (y0 + 1).min(IMRT_GRID_DIM - 1);
    let tx = fx - (x0 as f64);
    let ty = fy - (y0 as f64);

    let d00 = static_dose[flatten_2d_index(x0, y0, IMRT_GRID_DIM)];
    let d10 = static_dose[flatten_2d_index(x1, y0, IMRT_GRID_DIM)];
    let d01 = static_dose[flatten_2d_index(x0, y1, IMRT_GRID_DIM)];
    let d11 = static_dose[flatten_2d_index(x1, y1, IMRT_GRID_DIM)];
    let bottom = d00 + tx * (d10 - d00);
    let top = d01 + tx * (d11 - d01);
    bottom + ty * (top - bottom)
}

/// Bundles mutable workspace buffers updated on each 4D respiratory step.
pub struct DoseAccumBuffers<'a> {
    /// Room-fixed static planned dose vector.
    pub static_dose: &'a [f64],
    /// Cumulative gated 4D dose vector updated on `BeamActive`.
    pub gated_dose: &'a mut DVector<f64>,
    /// Cumulative ungated 4D dose vector updated when not in `Standby` or `SafetyInterlock`.
    pub ungated_dose: &'a mut DVector<f64>,
    /// Preallocated boolean PTV mask.
    pub ptv_mask: &'a [bool; IMRT_VOXEL_COUNT],
    /// Preallocated PTV index list.
    pub tumor_indices: &'a [usize],
    /// Preallocated OAR index list.
    pub organ_indices: &'a [usize],
    /// Scratch boolean isodose buffer for zero-allocation Dice computation.
    pub isodose_scratch: &'a mut [bool; IMRT_VOXEL_COUNT],
}

/// Accumulates one 4D respiratory step into `gated_dose` and `ungated_dose` with zero heap allocations.
pub fn accumulate_4d_step_in_place(
    tracker: &mut Imrt4dTracker,
    bufs: DoseAccumBuffers<'_>,
    telemetry: &GatingTelemetry,
    prescription_gy: f64,
    beam_current_ma: f64,
) {
    let (e_mev, scale) = linac_energy_and_fluence_scale(beam_current_ma);
    tracker.beam_energy_mev = e_mev;
    tracker.fluence_scale = scale;
    tracker.eval_pearson_r = telemetry.pearson_r;
    tracker.eval_rmse_mm = telemetry.rmse_mm;
    tracker.eval_spatial_err_pct = telemetry.spatial_err_pct;

    if !telemetry.step_event
        || matches!(
            telemetry.beam_state,
            BeamState::Standby | BeamState::SafetyInterlock
        )
        || !telemetry.true_disp_mm.is_finite()
    {
        return;
    }

    tracker.total_4d_steps = tracker.total_4d_steps.saturating_add(1);
    let shift_cm = ((telemetry.true_disp_mm - PLANNING_EXHALE_REF_MM)
        * INTERNAL_MOTION_GAIN_CM_PER_MM)
        .clamp(-6.0, 6.0);
    tracker.latest_shift_cm = shift_cm;

    tracker.ungated_steps = tracker.ungated_steps.saturating_add(1);
    let inv_u = 1.0 / (tracker.ungated_steps as f64);
    let beam_on = telemetry.beam_state == BeamState::BeamActive;
    let inv_g = if beam_on {
        tracker.gated_steps = tracker.gated_steps.saturating_add(1);
        1.0 / (tracker.gated_steps as f64)
    } else {
        0.0
    };

    step_shifted_voxels(
        bufs.static_dose,
        bufs.gated_dose.as_mut_slice(),
        bufs.ungated_dose.as_mut_slice(),
        shift_cm,
        scale,
        beam_on,
        inv_g,
        inv_u,
    );
    refresh_4d_tracker_summaries(tracker, bufs, prescription_gy, beam_on);
}

fn refresh_4d_tracker_summaries(
    tracker: &mut Imrt4dTracker,
    bufs: DoseAccumBuffers<'_>,
    prescription_gy: f64,
    beam_on: bool,
) {
    let (u_mean, u_oar) = summarize_dose_slice(
        bufs.ungated_dose.as_slice(),
        bufs.tumor_indices,
        bufs.organ_indices,
    );
    tracker.ungated_ptv_mean_gy = u_mean;
    tracker.ungated_oar_max_gy = u_oar;
    tracker.ungated_dice = compute_isodose_dice(
        bufs.ungated_dose.as_slice(),
        bufs.ptv_mask,
        bufs.isodose_scratch,
        prescription_gy,
    );
    if beam_on {
        let (g_mean, g_oar) = summarize_dose_slice(
            bufs.gated_dose.as_slice(),
            bufs.tumor_indices,
            bufs.organ_indices,
        );
        tracker.gated_ptv_mean_gy = g_mean;
        tracker.gated_oar_max_gy = g_oar;
        tracker.gated_dice = compute_isodose_dice(
            bufs.gated_dose.as_slice(),
            bufs.ptv_mask,
            bufs.isodose_scratch,
            prescription_gy,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn step_shifted_voxels(
    static_dose: &[f64],
    gated: &mut [f64],
    ungated: &mut [f64],
    shift_cm: f64,
    fluence_scale: f64,
    beam_on: bool,
    inv_g: f64,
    inv_u: f64,
) {
    for y in 0..IMRT_GRID_DIM {
        let y_room = voxel_coord_cm(y);
        for x in 0..IMRT_GRID_DIM {
            let x_room = voxel_coord_cm(x) - shift_cm;
            let inst = fluence_scale * sample_static_dose_bilinear(static_dose, x_room, y_room);
            let idx = flatten_2d_index(x, y, IMRT_GRID_DIM);
            ungated[idx] += (inst - ungated[idx]) * inv_u;
            if beam_on {
                gated[idx] += (inst - gated[idx]) * inv_g;
            }
        }
    }
}

fn summarize_dose_slice(
    dose: &[f64],
    tumor_indices: &[usize],
    organ_indices: &[usize],
) -> (f64, (f64, [f64; 2])) {
    let mut t_sum = 0.0;
    for &idx in tumor_indices {
        t_sum += dose[idx];
    }
    let t_mean = if tumor_indices.is_empty() {
        0.0
    } else {
        t_sum / (tumor_indices.len() as f64)
    };
    let mut oar_max = (0.0_f64, IMRT_OAR_CENTER);
    for &idx in organ_indices {
        let d = dose[idx];
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
    (t_mean, oar_max)
}

/// Embedded single-pane 4D respiratory Kalman + Schmitt-gating generator for `DoseCalculationTool`
/// when `RadarGatingTool` is not actively streaming on `link_bus`.
pub struct Standalone4dStepper {
    kalman: FastCvKalman,
    /// Verified `GatingLogic` state machine.
    pub gating: GatingLogic,
    true_ring_mm: Box<[f64; HISTORY_LEN]>,
    pred_ring_mm: Box<[f64; HISTORY_LEN]>,
    /// Current standalone simulation time in seconds.
    pub sim_time_s: f64,
    cough_timer_s: f64,
    nan_fault_active: bool,
    total_frames: u64,
    active_frames: u64,
    /// Latest synthesized `GatingTelemetry` snapshot.
    pub latest: GatingTelemetry,
}

impl Default for Standalone4dStepper {
    fn default() -> Self {
        let mut s = Self {
            kalman: new_standalone_kalman(),
            gating: new_standalone_gating(),
            true_ring_mm: Box::new([0.0; HISTORY_LEN]),
            pred_ring_mm: Box::new([0.0; HISTORY_LEN]),
            sim_time_s: 0.0,
            cough_timer_s: 0.0,
            nan_fault_active: false,
            total_frames: 0,
            active_frames: 0,
            latest: GatingTelemetry::empty(),
        };
        s.warm_history();
        s
    }
}

impl Standalone4dStepper {
    /// Resets the standalone 4D respiratory generator and re-arms gating.
    pub fn reset(&mut self) {
        self.kalman = new_standalone_kalman();
        self.gating = new_standalone_gating();
        self.sim_time_s = 0.0;
        self.cough_timer_s = 0.0;
        self.nan_fault_active = false;
        self.total_frames = 0;
        self.active_frames = 0;
        self.warm_history();
    }

    fn warm_history(&mut self) {
        for slot in 0..HISTORY_LEN {
            let t = -((HISTORY_LEN - slot) as f64) * SIM_DT;
            let z_m = cosine_respiratory_curve(t, 0.005, 12.0);
            self.kalman.predict();
            self.kalman.update(z_m);
            self.true_ring_mm[slot] = z_m * 1.0e3;
            self.pred_ring_mm[slot] = self.kalman.state[0] * 1.0e3;
        }
        self.latest = GatingTelemetry {
            beam_state: self.gating.state(),
            true_disp_mm: self.true_ring_mm[HISTORY_LEN - 1],
            measured_disp_mm: self.pred_ring_mm[HISTORY_LEN - 1],
            predicted_disp_mm: self.pred_ring_mm[HISTORY_LEN - 1],
            pearson_r: 1.0,
            ..GatingTelemetry::empty()
        };
    }

    /// Injects a `+10 mm` thoracic cough pulse lasting `0.45 s` (`18` steps at `40 Hz`).
    pub fn inject_cough(&mut self) {
        self.cough_timer_s = 0.45;
    }

    /// Injects a non-finite sensor fault latching `BeamState::SafetyInterlock`.
    pub fn inject_nan_fault(&mut self) {
        self.nan_fault_active = true;
        let _ = self.gating.evaluate(f64::NAN, self.kalman.state[1]);
        self.latest.beam_state = self.gating.state();
        self.latest.step_event = false;
    }

    /// Clears `SafetyInterlock` back to `Standby` and re-arms when `rearm` is true.
    pub fn reset_interlock(&mut self, rearm: bool) -> BeamState {
        self.nan_fault_active = false;
        let _ = self.gating.reset_interlock();
        if rearm {
            let _ = self.gating.arm();
        }
        self.latest.beam_state = self.gating.state();
        self.latest.step_event = false;
        self.latest.beam_state
    }

    /// Steps one `25 ms` respiratory frame with zero heap allocations.
    pub fn step_zero_alloc(&mut self) -> GatingTelemetry {
        self.sim_time_s += SIM_DT;
        let base_m = cosine_respiratory_curve(self.sim_time_s, 0.005, 12.0);
        let cough_m = if self.cough_timer_s > 0.0 {
            let p = 1.0 - (self.cough_timer_s / 0.45).clamp(0.0, 1.0);
            self.cough_timer_s = (self.cough_timer_s - SIM_DT).max(0.0);
            0.010 * (PI * p).sin().powi(2)
        } else {
            0.0
        };
        let true_m = base_m + cough_m;
        self.kalman.predict();
        self.kalman.update(true_m);

        let eval_pos = if self.nan_fault_active {
            f64::NAN
        } else {
            self.kalman.state[0]
        };
        let beam_state = self.gating.evaluate(eval_pos, self.kalman.state[1]);

        self.total_frames = self.total_frames.saturating_add(1);
        if beam_state == BeamState::BeamActive {
            self.active_frames = self.active_frames.saturating_add(1);
        }
        let true_mm = true_m * 1.0e3;
        let pos_mm = self.kalman.state[0] * 1.0e3;
        let vel_mm_s = self.kalman.state[1] * 1.0e3;
        let pred_mm = pos_mm + vel_mm_s * 0.050;
        shift_push(&mut self.true_ring_mm, true_mm);
        shift_push(&mut self.pred_ring_mm, pos_mm);

        let r = pearson_correlation(&*self.pred_ring_mm, &*self.true_ring_mm).unwrap_or(0.0);
        let rmse = root_mean_square_error(&*self.pred_ring_mm, &*self.true_ring_mm).unwrap_or(0.0);
        let err =
            spatial_accuracy_error(STANDALONE_RANGE_MM + pos_mm, STANDALONE_RANGE_MM + true_mm);
        let duty = 100.0 * (self.active_frames as f64) / (self.total_frames.max(1) as f64);

        let snap = GatingTelemetry {
            seq: self.total_frames,
            sim_time_s: self.sim_time_s,
            true_disp_mm: true_mm,
            measured_disp_mm: pos_mm,
            predicted_disp_mm: pred_mm,
            velocity_mm_s: vel_mm_s,
            duty_cycle_pct: duty,
            beam_state,
            pearson_r: r,
            rmse_mm: rmse,
            spatial_err_pct: if err.is_finite() { err } else { 0.0 },
            step_event: true,
        };
        self.latest = snap;
        snap
    }
}

fn new_standalone_kalman() -> FastCvKalman {
    let model = ConstantVelocityModel::new(3.0e-4, 2.0e-5);
    let tf = TrackingFilter::new(0.0, SIM_DT, model);
    fast_cv_from_tracking_filter(&tf, SIM_DT, 3.0e-4, 2.0e-5)
}

fn new_standalone_gating() -> GatingLogic {
    let mut g = GatingLogic::new(-1.5e-3, 0.5e-3, 0.050);
    let _ = g.arm();
    g
}

fn shift_push(buf: &mut [f64; HISTORY_LEN], val: f64) {
    buf.copy_within(1..HISTORY_LEN, 0);
    buf[HISTORY_LEN - 1] = val;
}

// [cite:mmwave_radiotherapy_setup]
