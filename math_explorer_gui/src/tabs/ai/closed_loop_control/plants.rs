//! Nonlinear twin plants (`LorenzSystem`, `SEIRModel`, `BatemanModel` / `SuperpositionModel`) and EKF coupling.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ekf::{
    FastEkf, LorenzKalmanModel, SeirKalmanModel, lorenz_rhs_and_jac, seir_rhs_and_jac,
};
pub use super::plants_pk::PkPlantTwin;
use super::state::{ClosedLoopState, PlantKind};
use math_explorer::epidemiology::compartmental::{SEIRModel, SEIRState};
use math_explorer::nalgebra::Vector3;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use math_explorer::pure_math::analysis::ode::{OdeSystem, RungeKutta4, TimeStepper};
use math_explorer::rand::RngCore;
use math_explorer::rng::OxidizeRng;

/// Number of RK4 phase samples along the Viswanath period-1 AB Lorenz unstable periodic orbit.
pub const LORENZ_UPO_STEPS: usize = 156;
const LORENZ_SIGMA: f64 = 10.0;
const LORENZ_RHO0: f64 = 28.0;
const LORENZ_BETA: f64 = 8.0 / 3.0;

const SEIR_POP_N: f64 = 1000.0;
const SEIR_BETA0: f64 = 0.78;
const SEIR_SIGMA: f64 = 0.25;
const SEIR_GAMMA: f64 = 0.12;

/// Generates a deterministic standard normal sample $\mathcal{N}(0, 1)$ via Box-Muller using `OxidizeRng`.
pub fn sample_standard_normal(rng: &mut OxidizeRng) -> f64 {
    let u1 =
        ((rng.next_u64() as f64 + 1.0) / (u64::MAX as f64 + 2.0)).clamp(1.0e-12, 1.0 - 1.0e-12);
    let u2 = (rng.next_u64() as f64) / (u64::MAX as f64 + 1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Telemetry returned by a single zero-allocation plant + EKF step.
#[derive(Debug, Clone, Copy)]
pub struct PlantStepOutput {
    /// Elapsed plant simulation time.
    pub time: f64,
    /// Primary scalar output of the uncontrolled open-loop twin.
    pub uncontrolled_primary: f64,
    /// True primary scalar state of the controlled plant.
    pub controlled_primary: f64,
    /// Noisy scalar sensor measurement fed into the EKF.
    pub noisy_measurement: f64,
    /// EKF posterior primary estimate $\hat{y}$.
    pub ekf_primary: f64,
    /// EKF posterior `1-sigma` uncertainty on the primary state.
    pub ekf_primary_std: f64,
    /// True hidden/latent secondary coordinate for phase-portrait & verification diagnostics.
    pub true_latent: f64,
    /// EKF posterior hidden/latent coordinate estimate.
    pub ekf_latent: f64,
    /// EKF posterior `1-sigma` uncertainty on the latent coordinate.
    pub ekf_latent_std: f64,
    /// Applied physical control actuation (`Δρ`, throttle `u`, or oral dose `D`).
    pub actuation: f64,
    /// Scalar EKF measurement innovation $z_k - h(\hat{x}_{k|k-1})$.
    pub innovation: f64,
    /// Dense scalar step reward for GRPO / Q-learning.
    pub reward: f64,
    /// Tracking error metric (orbit tube distance, infected fraction, or `[MEC, MTC]` violation).
    pub tracking_metric: f64,
}

/// Chaotic `LorenzSystem` controlled vs uncontrolled twin with precomputed Viswanath UPO reference.
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
        let init = initial_lorenz_vec(&upo_points[0]);
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
            r_var: lorenz_r_var(cfg.sensor_noise),
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
        let base = initial_lorenz_vec(&self.upo_points[0]);
        let init = Vector3::new(base.x + jitter[0], base.y + jitter[1], base.z + jitter[2]);
        self.controlled.rho = LORENZ_RHO0;
        self.controlled.state.vec = init;
        self.uncontrolled.rho = LORENZ_RHO0;
        self.uncontrolled.state.vec = init;
        self.ekf
            .reset_in_place([init.x - 0.8, init.y + 1.6, init.z - 1.4], [2.5, 8.0, 8.0]);
        self.time = 0.0;
    }

    /// Projects a 3D state onto the nearest velocity-consistent phase index along the UPO.
    pub fn nearest_upo_index(&self, pos: &[f64; 3]) -> (usize, f64) {
        let vel = [
            LORENZ_SIGMA * (pos[1] - pos[0]),
            pos[0] * (LORENZ_RHO0 - pos[2]) - pos[1],
            pos[0] * pos[1] - LORENZ_BETA * pos[2],
        ];
        let mut best_idx = 0;
        let mut best_score = f64::INFINITY;
        let mut best_dist_sq = f64::INFINITY;
        for i in 0..LORENZ_UPO_STEPS {
            let p = &self.upo_points[i];
            let v = &self.upo_vel[i];
            let dx = pos[0] - p[0];
            let dy = pos[1] - p[1];
            let dz = pos[2] - p[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let dvx = vel[0] - v[0];
            let dvy = vel[1] - v[1];
            let dvz = vel[2] - v[2];
            let score = dist_sq + 0.01 * (dvx * dvx + dvy * dvy + dvz * dvz);
            if score < best_score {
                best_score = score;
                best_idx = i;
                best_dist_sq = dist_sq;
            }
        }
        (best_idx, best_dist_sq.sqrt())
    }

    /// Extracts normalized 4D policy features and `0..64` discrete bin from the EKF estimate.
    pub fn features_and_bin(&self) -> ([f64; 4], usize) {
        let xh = &self.ekf.state;
        let (idx, tube_dist) = self.nearest_upo_index(xh);
        let p = &self.upo_points[idx];
        let (ex, ey, ez) = (p[0] - xh[0], p[1] - xh[1], p[2] - xh[2]);
        let sgn_x = xh[0] / (xh[0] * xh[0] + 2.0).sqrt();
        let phi = [
            (0.32 * sgn_x * ey).tanh(),
            (0.32 * sgn_x * ex).tanh(),
            (0.12 * sgn_x * ez).tanh(),
            0.0,
        ];
        let xi = sgn_x * (0.72 * ey + 0.28 * ex);
        let xi_bin = quantize_bin(xi, -3.5, 3.5, 8);
        let d_bin = quantize_bin(tube_dist, 0.0, 8.0, 8);
        (phi, xi_bin * 8 + d_bin)
    }

    /// Steps controlled and uncontrolled `LorenzSystem` twins plus `FastEkf<3>` with zero allocations.
    pub fn step(
        &mut self,
        raw_action: f64,
        cfg: &ClosedLoopState,
        rng: &mut OxidizeRng,
    ) -> PlantStepOutput {
        let dt = PlantKind::LorenzChaos.dt();
        let u_max = 4.5 * cfg.actuation_budget.max(0.0);
        let delta_rho = (raw_action.clamp(-1.0, 1.0) * u_max).clamp(-u_max, u_max);
        self.controlled.rho = LORENZ_RHO0 + delta_rho;
        self.controlled.step_with(&mut self.solver_ctrl, dt);
        self.uncontrolled.step_with(&mut self.solver_open, dt);
        self.time += dt;

        let rho_eff = self.controlled.rho;
        let q_diag = (LorenzKalmanModel {
            sigma: LORENZ_SIGMA,
            rho_eff,
            beta: LORENZ_BETA,
            q_var: cfg.process_noise,
            r_var: lorenz_r_var(cfg.sensor_noise),
        })
        .q_diag(dt);
        self.ekf.predict_rk4(dt, &q_diag, |s| {
            lorenz_rhs_and_jac(s, LORENZ_SIGMA, rho_eff, LORENZ_BETA)
        });

        let c_vec = self.controlled.state.vec;
        let noise_std = 0.8 * cfg.sensor_noise.max(0.0);
        let z_meas = c_vec.x + noise_std * sample_standard_normal(rng);
        self.ekf
            .update_unit_obs(0, z_meas, lorenz_r_var(cfg.sensor_noise));

        let (_, tube_dist) = self.nearest_upo_index(&[c_vec.x, c_vec.y, c_vec.z]);
        let norm_u = if u_max > 1.0e-6 {
            delta_rho / u_max
        } else {
            0.0
        };
        let reward =
            2.0 * (-0.25 * tube_dist * tube_dist).exp() - 0.08 * tube_dist - 0.03 * norm_u * norm_u;

        PlantStepOutput {
            time: self.time,
            uncontrolled_primary: self.uncontrolled.state.vec.x,
            controlled_primary: c_vec.x,
            noisy_measurement: z_meas,
            ekf_primary: self.ekf.state[0],
            ekf_primary_std: self.ekf.std_dev(0),
            true_latent: c_vec.y,
            ekf_latent: self.ekf.state[1],
            ekf_latent_std: self.ekf.std_dev(1),
            actuation: delta_rho,
            innovation: self.ekf.last_innovation,
            reward,
            tracking_metric: tube_dist,
        }
    }
}

fn precompute_viswanath_upo(
    dt: f64,
) -> ([[f64; 3]; LORENZ_UPO_STEPS], [[f64; 3]; LORENZ_UPO_STEPS]) {
    let mut sys = LorenzBuilder::new()
        .sigma(LORENZ_SIGMA)
        .rho(LORENZ_RHO0)
        .beta(LORENZ_BETA)
        .build(LorenzState::new(
            -13.763_610_682_134_2,
            -19.578_751_942_451_8,
            27.0,
        ));
    let mut solver = RungeKutta4::new(&sys.state.vec);
    let mut pts = [[0.0; 3]; LORENZ_UPO_STEPS];
    let mut vel = [[0.0; 3]; LORENZ_UPO_STEPS];
    for i in 0..LORENZ_UPO_STEPS {
        let v = sys.state.vec;
        let dv = sys.derivative(0.0, &v);
        pts[i] = [v.x, v.y, v.z];
        vel[i] = [dv.x, dv.y, dv.z];
        sys.step_with(&mut solver, dt);
    }
    (pts, vel)
}

fn initial_lorenz_vec(upo0: &[f64; 3]) -> Vector3<f64> {
    Vector3::new(upo0[0] + 0.55, upo0[1] - 0.75, upo0[2] + 0.45)
}

fn lorenz_r_var(sensor_noise: f64) -> f64 {
    let s = 0.8 * sensor_noise.max(0.0);
    s * s
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
        let frac = positive_control_fraction(raw_action);
        let u_throttle = (frac * u_max).clamp(0.0, u_max);
        let beta_eff = SEIR_BETA0 * (1.0 - u_throttle);
        self.controlled.dynamics.beta = beta_eff;
        self.controlled.step(dt);
        self.uncontrolled.step(dt);
        self.time += dt;

        let q_diag = (SeirKalmanModel {
            pop_n: SEIR_POP_N,
            beta_eff,
            sigma: SEIR_SIGMA,
            gamma: SEIR_GAMMA,
            q_var: cfg.process_noise,
            r_var: seir_r_var(cfg.sensor_noise),
        })
        .q_diag(dt);
        self.ekf.predict_rk4(dt, &q_diag, |s| {
            seir_rhs_and_jac(s, SEIR_POP_N, beta_eff, SEIR_SIGMA, SEIR_GAMMA)
        });

        let c_st = *self.controlled.get_state();
        let noise_std = 14.0 * cfg.sensor_noise.max(0.0);
        let z_meas = (c_st.i + noise_std * sample_standard_normal(rng)).max(0.0);
        self.ekf
            .update_unit_obs(2, z_meas, seir_r_var(cfg.sensor_noise));
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

/// Smoothly maps a normalized policy command `raw_action` in `[-1.0, 1.0]` to a unilateral actuation fraction in `[0.0, 1.0]`.
pub fn positive_control_fraction(raw_action: f64) -> f64 {
    let x = raw_action.clamp(-1.0, 1.0);
    1.0 / (1.0 + (-3.2 * x + 1.4).exp())
}

/// Quantizes `val` linearly over `[min, max]` into `0..bins`.
pub fn quantize_bin(val: f64, min: f64, max: f64, bins: usize) -> usize {
    let span = (max - min).max(1.0e-9);
    let norm = ((val - min) / span).clamp(0.0, 0.999_999);
    (norm * bins as f64) as usize
}

// [cite:grpo]
