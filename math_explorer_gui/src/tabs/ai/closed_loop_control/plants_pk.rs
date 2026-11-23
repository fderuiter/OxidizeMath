//! Nonlinear ADME pharmacokinetic twin (`BatemanModel` / `SuperpositionModel`) coupled to `FastEkf<3>`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ekf::{FastEkf, PkKalmanModel};
use super::plants::{
    positive_control_fraction, quantize_bin, sample_standard_normal, PlantStepOutput,
};
use super::state::{ClosedLoopState, PlantKind};
use math_explorer::applied::pharmacokinetics::{
    BatemanCompartmentDynamics, BatemanModel, PKParameters, PharmacokineticModel,
    SuperpositionModel,
};
use math_explorer::pure_math::analysis::ode::rk4_step_dual_jacobian;
use math_explorer::rng::OxidizeRng;

const PK_BIOAVAIL: f64 = 0.85;
const PK_KA: f64 = 1.10;
const PK_VOLUME_L: f64 = 25.0;
const PK_TRUE_KE: f64 = 0.13;
const PK_NOMINAL_KE: f64 = 0.28;
const PK_FIXED_DOSE_MG: f64 = 105.0;
const PK_FIXED_INTERVAL_H: f64 = 6.0;
const PK_SCHEDULE_DOSES: usize = 32;

/// Nonlinear ADME pharmacokinetic twin pairing adaptive oral dosing against `SuperpositionModel<BatemanModel>`.
#[derive(Debug, Clone)]
pub struct PkPlantTwin {
    /// True controlled 3-state ADME vector `[A_g (mg), C_p (mg/L), k_e (1/h)]`.
    pub controlled_state: [f64; 3],
    /// Fixed-schedule open-loop `SuperpositionModel<BatemanModel>` baseline.
    pub open_loop_superposition: SuperpositionModel<BatemanModel>,
    /// Stack-resident 3D Extended Kalman Filter estimating `[A_g, C_p, k_e]`.
    pub ekf: FastEkf<3>,
    /// Elapsed clinical time in hours.
    pub time: f64,
}

impl PkPlantTwin {
    /// Constructs the ADME twin, `SuperpositionModel<BatemanModel>` baseline, and validated `FastEkf<3>`.
    pub fn new(cfg: &ClosedLoopState) -> Self {
        let dt = PlantKind::Pharmacokinetics.dt();
        let params = PKParameters::new(
            PK_BIOAVAIL,
            PK_FIXED_DOSE_MG,
            PK_KA,
            PK_TRUE_KE,
            PK_VOLUME_L,
        )
        .expect("valid PKParameters");
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
        let ekf = FastEkf::from_validated_builder(
            model,
            dt,
            [0.0, 0.0, PK_NOMINAL_KE],
            [16.0, 0.25, 0.018],
        );
        Self {
            controlled_state: [0.0, 0.0, PK_TRUE_KE],
            open_loop_superposition,
            ekf,
            time: 0.0,
        }
    }

    /// Resets the controlled ADME plant and EKF prior in place without heap allocations.
    pub fn reset_in_place(&mut self, ke_jitter: f64) {
        let ke = (PK_TRUE_KE + ke_jitter).clamp(0.08, 0.35);
        self.controlled_state = [0.0, 0.0, ke];
        self.ekf
            .reset_in_place([0.0, 0.0, PK_NOMINAL_KE], [16.0, 0.25, 0.018]);
        self.time = 0.0;
    }

    /// Extracts normalized 4D features and `0..64` discrete bin from EKF `[A_g, C_p, k_e]`.
    pub fn features_and_bin(&self, cfg: &ClosedLoopState) -> ([f64; 4], usize) {
        let ag_hat = self.ekf.state[0].max(0.0);
        let cp_hat = self.ekf.state[1].max(0.0);
        let ke_hat = self.ekf.state[2].clamp(0.04, 0.60);
        let mid = 0.5 * (cfg.pk_mec + cfg.pk_mtc);
        let half_w = (0.5 * (cfg.pk_mtc - cfg.pk_mec)).max(0.25);
        let c_proj = cp_hat + 0.82 * (ag_hat / PK_VOLUME_L);
        let phi = [
            ((mid - c_proj) / half_w).clamp(-2.0, 2.0) * 0.60,
            ((mid - cp_hat) / half_w).clamp(-2.0, 2.0) * 0.40,
            ((ke_hat - 0.20) / 0.12).clamp(-1.0, 1.0),
            1.0,
        ];
        let span = (cfg.pk_mtc - cfg.pk_mec).max(0.5);
        let proj_bin = quantize_bin((c_proj - cfg.pk_mec) / span, -0.3, 1.4, 8);
        let cp_bin = quantize_bin((cp_hat - cfg.pk_mec) / span, -0.3, 1.4, 8);
        (phi, proj_bin * 8 + cp_bin)
    }

    /// Steps the controlled ADME plant, `SuperpositionModel<BatemanModel>` open-loop twin, and `FastEkf<3>`.
    pub fn step(
        &mut self,
        raw_action: f64,
        cfg: &ClosedLoopState,
        rng: &mut OxidizeRng,
    ) -> PlantStepOutput {
        let dt = PlantKind::Pharmacokinetics.dt();
        let d_max = 42.0 * cfg.actuation_budget.max(0.0);
        let dose_mg = (positive_control_fraction(raw_action) * d_max).clamp(0.0, d_max);

        let pk_dyn = BatemanCompartmentDynamics::new(PK_KA, PK_VOLUME_L);
        self.controlled_state[0] += PK_BIOAVAIL * dose_mg;
        let (next_true, _) = rk4_step_dual_jacobian(&pk_dyn, 0.0, &self.controlled_state, dt);
        self.controlled_state = [next_true[0].max(0.0), next_true[1].max(0.0), next_true[2]];
        self.time += dt;

        self.ekf.state[0] = (self.ekf.state[0] + PK_BIOAVAIL * dose_mg).max(0.0);
        let q_diag = (PkKalmanModel {
            ka: PK_KA,
            volume: PK_VOLUME_L,
            bioavail: PK_BIOAVAIL,
            dose_bolus_mg: dose_mg,
            q_var: cfg.process_noise,
            r_var: pk_r_var(cfg.sensor_noise),
        })
        .q_diag(dt);
        self.ekf.predict_ode_rk4(&pk_dyn, dt, &q_diag);

        let cp_true = self.controlled_state[1];
        let noise_std = 0.28 * cfg.sensor_noise.max(0.0);
        let z_meas = (cp_true + noise_std * sample_standard_normal(rng)).max(0.0);
        self.ekf
            .update_unit_obs(1, z_meas, pk_r_var(cfg.sensor_noise));
        self.ekf.state[0] = self.ekf.state[0].max(0.0);
        self.ekf.state[1] = self.ekf.state[1].max(0.0);
        self.ekf.state[2] = self.ekf.state[2].clamp(0.03, 0.70);

        let open_cp = self.open_loop_superposition.concentration(self.time);
        let (reward, viol) = compute_pk_reward_and_violation(
            cp_true,
            self.controlled_state[0],
            dose_mg,
            d_max,
            cfg.pk_mec,
            cfg.pk_mtc,
        );

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

fn pk_r_var(sensor_noise: f64) -> f64 {
    let s = 0.28 * sensor_noise.max(0.0);
    s * s
}

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
    (
        base_r - 0.6 * proj_over * proj_over - 0.04 * dose_norm * dose_norm,
        viol,
    )
}

// [cite:grpo]
