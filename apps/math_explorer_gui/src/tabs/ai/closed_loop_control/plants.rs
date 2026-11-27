//! Thin twin-plant adapters (`LorenzSystem`, `SEIRModel`, `PkClosedLoopTwin`) over `domain_applied`.

use super::state::{ClosedLoopState, PlantKind};
use math_explorer::applied::algorithms::kalman::{
    seir_rhs_and_jac, FastEkf, LorenzKalmanModel, SeirKalmanModel,
};
#[cfg(test)]
use math_explorer::applied::engineering::nearest_upo_index;
use math_explorer::applied::engineering::{
    compute_ogy_rayleigh_thrust, lorenz_sensor_variance, observe_and_score_upo_step,
    precompute_viswanath_upo, upo_features_and_bin, UpoControlConfig, LORENZ_BETA, LORENZ_RHO0,
    LORENZ_SIGMA,
};
pub use math_explorer::applied::engineering::{
    positive_control_fraction, quantize_bin, PlantStepOutput, LORENZ_UPO_STEPS,
};
use math_explorer::applied::pharmacokinetics::{PkClosedLoopConfig, PkClosedLoopTwin};
use math_explorer::epidemiology::compartmental::{SEIRModel, SEIRState};
use math_explorer::nalgebra::Vector3;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
use math_explorer::rng::{sample_standard_normal_f64, OxidizeRng};
use std::ops::{Deref, DerefMut};

const SEIR_POP_N: f64 = 1000.0;
const SEIR_BETA0: f64 = 0.78;
const SEIR_SIGMA: f64 = 0.25;
const SEIR_GAMMA: f64 = 0.12;

/// Generates a deterministic standard normal sample $\mathcal{N}(0, 1)$ via `OxidizeRng`.
pub fn sample_standard_normal(rng: &mut OxidizeRng) -> f64 {
    sample_standard_normal_f64(rng)
}

/// Chaotic `LorenzSystem` twin delegating UPO reference and OGY/EKF stepping to `domain_applied`.
#[derive(Debug, Clone)]
pub struct LorenzPlantTwin {
    /// Controlled `LorenzSystem` receiving OGY Rayleigh thrusts $\rho_0 + \Delta\rho(t)$.
    pub controlled: LorenzSystem,
    /// Identically initialized open-loop `LorenzSystem` at $\rho_0 = 28$.
    pub uncontrolled: LorenzSystem,
    solver_ctrl: RungeKutta4<Vector3<f64>>,
    solver_open: RungeKutta4<Vector3<f64>>,
    /// Precomputed 3D coordinates along the unstable periodic orbit.
    pub upo_points: [[f64; 3]; LORENZ_UPO_STEPS],
    upo_vel: [[f64; 3]; LORENZ_UPO_STEPS],
    /// Stack-resident 3D Extended Kalman Filter validated via `KalmanFilter::builder`.
    pub ekf: FastEkf<3>,
    /// Elapsed simulation time.
    pub time: f64,
}

impl LorenzPlantTwin {
    /// Constructs the Lorenz twin and validates `LorenzKalmanModel` via `KalmanFilter::builder`.
    pub fn new(cfg: &ClosedLoopState) -> Self {
        let dt = PlantKind::LorenzChaos.dt();
        let (upo_points, upo_vel) = precompute_viswanath_upo(dt);
        let init = Vector3::new(
            upo_points[0][0] + 0.55,
            upo_points[0][1] - 0.75,
            upo_points[0][2] + 0.45,
        );
        let controlled = LorenzBuilder::new()
            .sigma(LORENZ_SIGMA)
            .rho(LORENZ_RHO0)
            .beta(LORENZ_BETA)
            .build(LorenzState::new(init.x, init.y, init.z));
        let uncontrolled = controlled.clone();
        let model = LorenzKalmanModel {
            sigma: LORENZ_SIGMA,
            rho_eff: LORENZ_RHO0,
            beta: LORENZ_BETA,
            q_var: cfg.process_noise,
            r_var: lorenz_sensor_variance(cfg.sensor_noise),
        };
        let ekf = FastEkf::from_validated_builder(
            model,
            dt,
            [init.x - 0.8, init.y + 1.6, init.z - 1.4],
            [2.5, 8.0, 8.0],
        );
        Self {
            controlled,
            uncontrolled,
            solver_ctrl: RungeKutta4::new(&init),
            solver_open: RungeKutta4::new(&init),
            upo_points,
            upo_vel,
            ekf,
            time: 0.0,
        }
    }

    /// Resets both twins and the EKF in place without heap allocations.
    pub fn reset_in_place(&mut self, jitter: [f64; 3]) {
        let init = Vector3::new(
            self.upo_points[0][0] + 0.55 + jitter[0],
            self.upo_points[0][1] - 0.75 + jitter[1],
            self.upo_points[0][2] + 0.45 + jitter[2],
        );
        self.controlled.rho = LORENZ_RHO0;
        self.controlled.state.vec = init;
        self.uncontrolled.rho = LORENZ_RHO0;
        self.uncontrolled.state.vec = init;
        self.ekf
            .reset_in_place([init.x - 0.8, init.y + 1.6, init.z - 1.4], [2.5, 8.0, 8.0]);
        self.time = 0.0;
    }

    /// Projects a 3D state onto the nearest velocity-consistent phase index along the UPO.
    #[cfg(test)]
    pub fn nearest_upo_index(&self, pos: &[f64; 3]) -> (usize, f64) {
        nearest_upo_index(&self.upo_points, &self.upo_vel, pos)
    }

    /// Extracts normalized 4D policy features and `0..64` discrete bin from the EKF estimate.
    pub fn features_and_bin(&self) -> ([f64; 4], usize) {
        upo_features_and_bin(&self.ekf.state, &self.upo_points, &self.upo_vel)
    }

    /// Steps controlled and uncontrolled `LorenzSystem` twins plus `FastEkf<3>` with zero allocations.
    pub fn step(
        &mut self,
        raw_action: f64,
        cfg: &ClosedLoopState,
        rng: &mut OxidizeRng,
    ) -> PlantStepOutput {
        let dt = PlantKind::LorenzChaos.dt();
        let (delta_rho, u_max) = compute_ogy_rayleigh_thrust(raw_action, cfg.actuation_budget);
        self.controlled.rho = LORENZ_RHO0 + delta_rho;
        self.controlled.step_with(&mut self.solver_ctrl, dt);
        self.uncontrolled.step_with(&mut self.solver_open, dt);
        self.time += dt;
        observe_and_score_upo_step(
            &mut self.ekf,
            &self.upo_points,
            &self.upo_vel,
            self.controlled.state.vec,
            self.uncontrolled.state.vec,
            self.time,
            delta_rho,
            u_max,
            UpoControlConfig {
                dt,
                actuation_budget: cfg.actuation_budget,
                process_noise: cfg.process_noise,
                sensor_noise: cfg.sensor_noise,
            },
            rng,
        )
    }
}

/// Thin `ClosedLoopState` adapter around `domain_applied::applied::pharmacokinetics::PkClosedLoopTwin`.
#[derive(Debug, Clone)]
pub struct PkPlantTwin(pub PkClosedLoopTwin);

impl Deref for PkPlantTwin {
    type Target = PkClosedLoopTwin;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PkPlantTwin {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl PkPlantTwin {
    /// Constructs the ADME twin via `PkClosedLoopTwin::try_new`.
    pub fn new(cfg: &ClosedLoopState) -> Self {
        Self(
            PkClosedLoopTwin::try_new(pk_cfg(cfg))
                .expect("valid PKClosedLoopTwin parameters from ClosedLoopState"),
        )
    }

    /// Extracts normalized 4D features and `0..64` discrete bin from EKF `[A_g, C_p, k_e]`.
    pub fn features_and_bin(&self, cfg: &ClosedLoopState) -> ([f64; 4], usize) {
        self.0.features_and_bin(cfg.pk_mec, cfg.pk_mtc)
    }

    /// Steps the ADME twin via `PkClosedLoopTwin::step`.
    pub fn step(
        &mut self,
        raw_action: f64,
        cfg: &ClosedLoopState,
        rng: &mut OxidizeRng,
    ) -> PlantStepOutput {
        self.0.step(raw_action, pk_cfg(cfg), rng)
    }
}

fn pk_cfg(cfg: &ClosedLoopState) -> PkClosedLoopConfig {
    PkClosedLoopConfig {
        dt: PlantKind::Pharmacokinetics.dt(),
        process_noise: cfg.process_noise,
        sensor_noise: cfg.sensor_noise,
        actuation_budget: cfg.actuation_budget,
        mec: cfg.pk_mec,
        mtc: cfg.pk_mtc,
    }
}

/// Compartmental `SEIRModel` controlled vs uncontrolled epidemic twin with 4D EKF latent-`Exposed` observer.
#[derive(Debug, Clone)]
pub struct SeirPlantTwin {
    /// Controlled `SEIRModel` with throttled $\beta_{\text{eff}}(t) = \beta_0(1 - u_t)$.
    pub controlled: SEIRModel<RungeKutta4<SEIRState>>,
    /// Identically initialized open-loop `SEIRModel` at baseline $\beta_0$.
    pub uncontrolled: SEIRModel<RungeKutta4<SEIRState>>,
    /// Stack-resident 4D Extended Kalman Filter validated via `KalmanFilter::builder`.
    pub ekf: FastEkf<4>,
    /// Elapsed simulation time in days.
    pub time: f64,
}

impl SeirPlantTwin {
    /// Constructs the `SEIRModel` twin and validates `SeirKalmanModel` via `KalmanFilter::builder`.
    pub fn new(cfg: &ClosedLoopState) -> Self {
        let dt = PlantKind::SeirEpidemic.dt();
        let mut controlled = SEIRModel::new(SEIR_POP_N, 12.0, SEIR_BETA0, SEIR_SIGMA, SEIR_GAMMA)
            .expect("valid SEIR parameters");
        set_seir_state(controlled.get_state_mut(), 35.0, 12.0);
        let uncontrolled = controlled.clone();
        let model = SeirKalmanModel {
            pop_n: SEIR_POP_N,
            beta_eff: SEIR_BETA0,
            sigma: SEIR_SIGMA,
            gamma: SEIR_GAMMA,
            q_var: cfg.process_noise,
            r_var: seir_r_var(cfg.sensor_noise),
        };
        let ekf = FastEkf::from_validated_builder(
            model,
            dt,
            [975.0, 8.0, 17.0, 0.0],
            [400.0, 350.0, 80.0, 25.0],
        );
        Self {
            controlled,
            uncontrolled,
            ekf,
            time: 0.0,
        }
    }

    /// Resets both `SEIRModel` instances and EKF in place without heap allocations.
    pub fn reset_in_place(&mut self, extra_exposed: f64) {
        let e0 = (35.0 + extra_exposed).clamp(5.0, 120.0);
        set_seir_state(self.controlled.get_state_mut(), e0, 12.0);
        set_seir_state(self.uncontrolled.get_state_mut(), e0, 12.0);
        self.controlled.dynamics.beta = SEIR_BETA0;
        self.uncontrolled.dynamics.beta = SEIR_BETA0;
        self.ekf
            .reset_in_place([975.0, 8.0, 17.0, 0.0], [400.0, 350.0, 80.0, 25.0]);
        self.time = 0.0;
    }

    /// Extracts normalized 4D features and `0..64` discrete bin from EKF `Exposed` and `Infectious` estimates.
    pub fn features_and_bin(&self) -> ([f64; 4], usize) {
        let s_hat = self.ekf.state[0].clamp(0.0, SEIR_POP_N);
        let e_hat = self.ekf.state[1].clamp(0.0, SEIR_POP_N);
        let i_hat = self.ekf.state[2].clamp(0.0, SEIR_POP_N);
        let phi = [
            ((e_hat - 18.0) / 45.0).clamp(-1.0, 1.5),
            ((i_hat - 15.0) / 45.0).clamp(-1.0, 1.5),
            ((s_hat - 400.0) / 600.0).clamp(-1.0, 1.0),
            1.0,
        ];
        let e_bin = quantize_bin(e_hat, 0.0, 140.0, 8);
        let i_bin = quantize_bin(i_hat, 0.0, 140.0, 8);
        (phi, e_bin * 8 + i_bin)
    }

    /// Steps controlled and uncontrolled `SEIRModel` twins plus `FastEkf<4>` with zero allocations.
    pub fn step(
        &mut self,
        raw_action: f64,
        cfg: &ClosedLoopState,
        rng: &mut OxidizeRng,
    ) -> PlantStepOutput {
        let dt = PlantKind::SeirEpidemic.dt();
        let u_max = (0.72 * cfg.actuation_budget.max(0.0)).clamp(0.0, 0.94);
        let u_throttle = (positive_control_fraction(raw_action) * u_max).clamp(0.0, u_max);
        let beta_eff = SEIR_BETA0 * (1.0 - u_throttle);
        self.controlled.dynamics.beta = beta_eff;
        self.controlled.step(dt);
        self.uncontrolled.step(dt);
        self.time += dt;

        let r_var = seir_r_var(cfg.sensor_noise);
        let q_diag = (SeirKalmanModel {
            pop_n: SEIR_POP_N,
            beta_eff,
            sigma: SEIR_SIGMA,
            gamma: SEIR_GAMMA,
            q_var: cfg.process_noise,
            r_var,
        })
        .q_diag(dt);
        self.ekf.predict_rk4(dt, &q_diag, |s| {
            seir_rhs_and_jac(s, SEIR_POP_N, beta_eff, SEIR_SIGMA, SEIR_GAMMA)
        });

        let c_st = *self.controlled.get_state();
        let noise_std = 14.0 * cfg.sensor_noise.max(0.0);
        let z_meas = (c_st.i + noise_std * sample_standard_normal(rng)).max(0.0);
        self.ekf.update_unit_obs(2, z_meas, r_var);
        for slot in &mut self.ekf.state {
            *slot = slot.clamp(0.0, SEIR_POP_N);
        }

        let burden = (c_st.i + 0.55 * c_st.e) / SEIR_POP_N;
        let reward = 1.2 - 10.0 * burden - 0.30 * u_throttle * u_throttle;

        PlantStepOutput {
            time: self.time,
            uncontrolled_primary: self.uncontrolled.get_state().i,
            controlled_primary: c_st.i,
            noisy_measurement: z_meas,
            ekf_primary: self.ekf.state[2],
            ekf_primary_std: self.ekf.std_dev(2),
            true_latent: c_st.e,
            ekf_latent: self.ekf.state[1],
            ekf_latent_std: self.ekf.std_dev(1),
            actuation: u_throttle,
            innovation: self.ekf.last_innovation,
            reward,
            tracking_metric: c_st.i / SEIR_POP_N,
        }
    }
}

fn set_seir_state(st: &mut SEIRState, e0: f64, i0: f64) {
    st.s = (SEIR_POP_N - e0 - i0).max(0.0);
    st.e = e0;
    st.i = i0;
    st.r = 0.0;
}

fn seir_r_var(sensor_noise: f64) -> f64 {
    let s = 14.0 * sensor_noise.max(0.0);
    s * s
}

// [cite:grpo]
