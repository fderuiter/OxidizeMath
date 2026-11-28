//! Closed-loop ADME pharmacokinetic twin (`BatemanModel` / `SuperpositionModel`) coupled to `FastEkf`.
//!
//! Pairs adaptive oral dosing against a fixed-schedule `SuperpositionModel` baseline while
//! inferring latent patient clearance $k_e$ from noisy plasma concentration measurements.

use super::{BatemanModel, PKParameters, PharmacokineticModel, SuperpositionModel};
use crate::applied::algorithms::kalman::{
    FastEkf, PkKalmanModel, pk_rhs_and_jac, rk4_step_with_jacobian,
};
use crate::applied::engineering::{PlantStepOutput, positive_control_fraction, quantize_bin};
use crate::error::PharmacokineticsError;
use oxidize_core::rng::{OxidizeRng, sample_standard_normal_f64};

const PK_BIOAVAIL: f64 = 0.85;
const PK_KA: f64 = 1.10;
const PK_VOLUME_L: f64 = 25.0;
const PK_TRUE_KE: f64 = 0.13;
const PK_NOMINAL_KE: f64 = 0.28;
const PK_FIXED_DOSE_MG: f64 = 105.0;
const PK_FIXED_INTERVAL_H: f64 = 6.0;
const PK_SCHEDULE_DOSES: usize = 32;

/// Configuration governing one closed-loop ADME pharmacokinetic step or twin reset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PkClosedLoopConfig {
    /// Fixed clinical integration step $\Delta t$ in hours.
    pub dt: f64,
    /// Continuous-time EKF process noise variance scale $Q$.
    pub process_noise: f64,
    /// Dimensionless plasma concentration sensor noise scale.
    pub sensor_noise: f64,
    /// Dimensionless oral dosing authority multiplier (`1.0` = `42 mg` per step).
    pub actuation_budget: f64,
    /// Minimum Effective Concentration (`MEC`, mg/L).
    pub mec: f64,
    /// Minimum Toxic Concentration (`MTC`, mg/L).
    pub mtc: f64,
}

impl Default for PkClosedLoopConfig {
    fn default() -> Self {
        Self {
            dt: 0.5,
            process_noise: 0.02,
            sensor_noise: 0.20,
            actuation_budget: 1.0,
            mec: 1.8,
            mtc: 4.5,
        }
    }
}

/// Nonlinear ADME pharmacokinetic twin pairing adaptive oral dosing against [`SuperpositionModel`].
///
/// # Examples
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::pharmacokinetics::{PkClosedLoopConfig, PkClosedLoopTwin};
/// use oxidize_core::rng::OxidizeRng;
///
/// let cfg = PkClosedLoopConfig::default();
/// let mut twin = PkClosedLoopTwin::try_new(cfg).unwrap();
/// let mut rng = OxidizeRng::new(11);
/// let out = twin.step(0.5, cfg, &mut rng);
/// assert!(out.controlled_primary >= 0.0 && out.ekf_primary.is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct PkClosedLoopTwin {
    /// True controlled 3-state ADME vector `[A_g (mg), C_p (mg/L), k_e (1/h)]`.
    pub controlled_state: [f64; 3],
    /// Fixed-schedule open-loop [`SuperpositionModel`] baseline.
    pub open_loop_superposition: SuperpositionModel<BatemanModel>,
    /// Stack-resident 3D Extended Kalman Filter estimating `[A_g, C_p, k_e]`.
    pub ekf: FastEkf<3>,
    /// Elapsed clinical time in hours.
    pub time: f64,
}

impl PkClosedLoopTwin {
    /// Constructs the ADME twin, [`SuperpositionModel`] baseline, and validated [`FastEkf`].
    #[verified_engine::verified]
    pub fn try_new(cfg: PkClosedLoopConfig) -> Result<Self, PharmacokineticsError> {
        let params = PKParameters::new(
            PK_BIOAVAIL,
            PK_FIXED_DOSE_MG,
            PK_KA,
            PK_TRUE_KE,
            PK_VOLUME_L,
        )?;
        let single_dose_model = BatemanModel::new(params);
        let dose_times: Vec<f64> = (0..PK_SCHEDULE_DOSES)
            .map(|k| k as f64 * PK_FIXED_INTERVAL_H)
            .collect();
        let open_loop_superposition = SuperpositionModel::new(single_dose_model, dose_times);

        let model = PkKalmanModel {
            ka: PK_KA,
            volume: PK_VOLUME_L,
            bioavail: PK_BIOAVAIL,
            dose_bolus_mg: 0.0,
            q_var: cfg.process_noise,
            r_var: pk_r_var(cfg.sensor_noise),
        };
        let ekf = FastEkf::try_from_validated_builder(
            model,
            cfg.dt,
            [0.0, 0.0, PK_NOMINAL_KE],
            [16.0, 0.25, 0.018],
        )
        .map_err(|e| PharmacokineticsError::InvalidParameter(e.to_string()))?;
        Ok(Self {
            controlled_state: [0.0, 0.0, PK_TRUE_KE],
            open_loop_superposition,
            ekf,
            time: 0.0,
        })
    }

    /// Resets the controlled ADME plant and EKF prior in place without heap allocations.
    #[verified_engine::verified]
    pub fn reset_in_place(&mut self, ke_jitter: f64) {
        let ke = (PK_TRUE_KE + ke_jitter).clamp(0.08, 0.35);
        self.controlled_state = [0.0, 0.0, ke];
        self.ekf
            .reset_in_place([0.0, 0.0, PK_NOMINAL_KE], [16.0, 0.25, 0.018]);
        self.time = 0.0;
        debug_assert!(self.time == 0.0);
    }

    /// Extracts normalized 4D features and a `0..64` discretized bin from EKF `[A_g, C_p, k_e]`.
    #[must_use]
    #[verified_engine::verified]
    pub fn features_and_bin(&self, mec: f64, mtc: f64) -> ([f64; 4], usize) {
        let ag_hat = self.ekf.state[0].max(0.0);
        let cp_hat = self.ekf.state[1].max(0.0);
        let ke_hat = self.ekf.state[2].clamp(0.04, 0.60);
        let mid = 0.5 * (mec + mtc);
        let half_w = (0.5 * (mtc - mec)).max(0.25);
        let c_proj = cp_hat + 0.82 * (ag_hat / PK_VOLUME_L);
        let phi = [
            ((mid - c_proj) / half_w).clamp(-2.0, 2.0) * 0.60,
            ((mid - cp_hat) / half_w).clamp(-2.0, 2.0) * 0.40,
            ((ke_hat - 0.20) / 0.12).clamp(-1.0, 1.0),
            1.0,
        ];
        let span = (mtc - mec).max(0.5);
        let proj_bin = quantize_bin((c_proj - mec) / span, -0.3, 1.4, 8);
        let cp_bin = quantize_bin((cp_hat - mec) / span, -0.3, 1.4, 8);
        let bin = proj_bin * 8 + cp_bin;
        debug_assert!(bin < 64);
        (phi, bin)
    }

    /// Steps the controlled ADME plant, [`SuperpositionModel`] open-loop twin, and [`FastEkf`] with zero allocations.
    #[verified_engine::verified]
    pub fn step(
        &mut self,
        raw_action: f64,
        cfg: PkClosedLoopConfig,
        rng: &mut OxidizeRng,
    ) -> PlantStepOutput {
        let d_max = 42.0 * cfg.actuation_budget.max(0.0);
        let dose_mg = (positive_control_fraction(raw_action) * d_max).clamp(0.0, d_max);

        self.controlled_state[0] += PK_BIOAVAIL * dose_mg;
        let (next_true, _) = rk4_step_with_jacobian(&self.controlled_state, cfg.dt, |s| {
            pk_rhs_and_jac(s, PK_KA, PK_VOLUME_L)
        });
        self.controlled_state = [next_true[0].max(0.0), next_true[1].max(0.0), next_true[2]];
        self.time += cfg.dt;

        self.ekf.state[0] = (self.ekf.state[0] + PK_BIOAVAIL * dose_mg).max(0.0);
        let r_var = pk_r_var(cfg.sensor_noise);
        let q_diag = (PkKalmanModel {
            ka: PK_KA,
            volume: PK_VOLUME_L,
            bioavail: PK_BIOAVAIL,
            dose_bolus_mg: dose_mg,
            q_var: cfg.process_noise,
            r_var,
        })
        .q_diag(cfg.dt);
        self.ekf
            .predict_rk4(cfg.dt, &q_diag, |s| pk_rhs_and_jac(s, PK_KA, PK_VOLUME_L));

        let cp_true = self.controlled_state[1];
        let noise_std = 0.28 * cfg.sensor_noise.max(0.0);
        let z_meas = (cp_true + noise_std * sample_standard_normal_f64(rng)).max(0.0);
        self.ekf.update_unit_obs(1, z_meas, r_var);
        self.ekf.state[0] = self.ekf.state[0].max(0.0);
        self.ekf.state[1] = self.ekf.state[1].max(0.0);
        self.ekf.state[2] = self.ekf.state[2].clamp(0.03, 0.70);

        let open_cp = self.open_loop_superposition.concentration(self.time);
        let (reward, viol) = compute_pk_reward_and_violation(
            cp_true,
            self.controlled_state[0],
            dose_mg,
            d_max,
            cfg.mec,
            cfg.mtc,
        );
        debug_assert!(reward.is_finite() && viol >= 0.0);

        PlantStepOutput {
            time: self.time,
            uncontrolled_primary: open_cp,
            controlled_primary: cp_true,
            noisy_measurement: z_meas,
            ekf_primary: self.ekf.state[1],
            ekf_primary_std: self.ekf.std_dev(1),
            true_latent: self.controlled_state[2] * 10.0,
            ekf_latent: self.ekf.state[2] * 10.0,
            ekf_latent_std: self.ekf.std_dev(2) * 10.0,
            actuation: dose_mg,
            innovation: self.ekf.last_innovation,
            reward,
            tracking_metric: viol,
        }
    }
}

#[verified_engine::verified]
fn pk_r_var(sensor_noise: f64) -> f64 {
    let s = 0.28 * sensor_noise.max(0.0);
    let var = s * s;
    debug_assert!(var >= 0.0);
    var
}

#[verified_engine::verified]
fn compute_pk_reward_and_violation(
    cp: f64,
    ag: f64,
    dose_mg: f64,
    d_max: f64,
    mec: f64,
    mtc: f64,
) -> (f64, f64) {
    let mid = 0.5 * (mec + mtc);
    let half_w = (0.5 * (mtc - mec)).max(0.25);
    let proj_over = (cp + 0.80 * (ag / PK_VOLUME_L) - mtc).max(0.0);
    let dose_norm = if d_max > 1.0e-6 { dose_mg / d_max } else { 0.0 };
    let (base_r, viol) = if cp > mtc {
        let over = cp - mtc;
        (-4.0 * over * over - 0.8 * over, over)
    } else if cp < mec {
        let under = mec - cp;
        (-2.0 * under * under - 0.4 * under, under)
    } else {
        let centered = (cp - mid) / half_w;
        (1.5 - 0.5 * centered * centered, 0.0)
    };
    let reward = base_r - 0.6 * proj_over * proj_over - 0.04 * dose_norm * dose_norm;
    debug_assert!(viol >= 0.0 && reward.is_finite());
    (reward, viol)
}

// [cite:pharmacokinetics]

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_pk_closed_loop_twin_zero_alloc_step() {
        let cfg = PkClosedLoopConfig::default();
        let mut twin = PkClosedLoopTwin::try_new(cfg).expect("valid default PK config");
        let mut rng = OxidizeRng::new(99);

        {
            let _guard = crate::VERIFIED_ENGINE_TEST_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            VerifiedEngine::run_verified(|| {
                twin.reset_in_place(0.01);
                for _ in 0..10 {
                    let (phi, bin) = twin.features_and_bin(cfg.mec, cfg.mtc);
                    assert!(bin < 64);
                    let out = twin.step(phi[0], cfg, &mut rng);
                    assert!(out.ekf_primary.is_finite());
                }
            });
        }
    }
}
