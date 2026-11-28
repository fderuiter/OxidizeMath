//! Unstable Periodic Orbit (UPO) OGY control, Viswanath reference trajectory, and closed-loop telemetry.
//!
//! Provides zero-allocation OGY Rayleigh-parameter stabilization primitives coupled to `FastEkf`
//! and `LorenzKalmanModel` over `RungeKutta4`.

use crate::applied::algorithms::kalman::{FastEkf, LorenzKalmanModel, lorenz_rhs_and_jac};
use nalgebra::Vector3;
use oxidize_core::rng::{OxidizeRng, sample_standard_normal_f64};
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};

/// Number of RK4 phase samples along the Viswanath period-1 AB Lorenz unstable periodic orbit.
pub const LORENZ_UPO_STEPS: usize = 156;
/// Canonical Lorenz Prandtl number $\sigma = 10$ for the Viswanath UPO.
pub const LORENZ_SIGMA: f64 = 10.0;
/// Canonical nominal Rayleigh parameter $\rho_0 = 28$ for the Viswanath UPO.
pub const LORENZ_RHO0: f64 = 28.0;
/// Canonical geometric aspect ratio $\beta = 8/3$ for the Viswanath UPO.
pub const LORENZ_BETA: f64 = 8.0 / 3.0;

/// Telemetry returned by a single zero-allocation closed-loop plant + EKF step.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Noise and actuation parameters for one OGY Lorenz UPO control step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpoControlConfig {
    /// Fixed RK4 integration step $\Delta t$.
    pub dt: f64,
    /// Dimensionless actuation authority multiplier (`1.0` = $\pm 4.5$ Rayleigh units).
    pub actuation_budget: f64,
    /// Continuous-time process noise variance scale $Q$.
    pub process_noise: f64,
    /// Dimensionless scalar $x$-sensor noise scale.
    pub sensor_noise: f64,
}

impl Default for UpoControlConfig {
    fn default() -> Self {
        Self {
            dt: 1.558_652_210_716_2 / (LORENZ_UPO_STEPS as f64),
            actuation_budget: 1.0,
            process_noise: 0.02,
            sensor_noise: 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LorenzFlowRhs {
    sigma: f64,
    rho: f64,
    beta: f64,
}

impl OdeSystem<Vector3<f64>> for LorenzFlowRhs {
    fn derivative(&self, _t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            self.sigma * (state.y - state.x),
            state.x * (self.rho - state.z) - state.y,
            state.x * state.y - self.beta * state.z,
        )
    }
}

/// Smoothly maps a normalized policy command `raw_action` in `[-1.0, 1.0]` to a unilateral actuation fraction in `[0.0, 1.0]`.
///
/// # Examples
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::engineering::positive_control_fraction;
///
/// let frac = positive_control_fraction(0.8);
/// assert!(frac > 0.5 && frac <= 1.0);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn positive_control_fraction(raw_action: f64) -> f64 {
    let x = raw_action.clamp(-1.0, 1.0);
    let out = 1.0 / (1.0 + (-3.2 * x + 1.4).exp());
    debug_assert!((0.0..=1.0).contains(&out));
    out
}

/// Quantizes `val` linearly over `[min, max]` into `0..bins`.
///
/// # Examples
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::engineering::quantize_bin;
///
/// assert_eq!(quantize_bin(0.0, -1.0, 1.0, 8), 4);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn quantize_bin(val: f64, min: f64, max: f64, bins: usize) -> usize {
    if bins == 0 {
        return 0;
    }
    let span = (max - min).max(1.0e-9);
    let norm = ((val - min) / span).clamp(0.0, 0.999_999);
    let idx = (norm * bins as f64) as usize;
    debug_assert!(idx < bins);
    idx.min(bins - 1)
}

/// Precomputes the `156`-point Viswanath period-1 AB unstable periodic orbit and tangent velocities via [`RungeKutta4`].
///
/// # Examples
/// ```
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::engineering::{precompute_viswanath_upo, LORENZ_UPO_STEPS};
///
/// let (pts, vel) = precompute_viswanath_upo(1.558_652_210_716_2 / LORENZ_UPO_STEPS as f64);
/// assert!(pts[0][2] > 20.0 && vel[0][0].is_finite());
/// ```
#[must_use]
#[verified_engine::verified]
pub fn precompute_viswanath_upo(
    dt: f64,
) -> ([[f64; 3]; LORENZ_UPO_STEPS], [[f64; 3]; LORENZ_UPO_STEPS]) {
    let sys = LorenzFlowRhs {
        sigma: LORENZ_SIGMA,
        rho: LORENZ_RHO0,
        beta: LORENZ_BETA,
    };
    let mut state = Vector3::new(-13.763_610_682_134_2, -19.578_751_942_451_8, 27.0);
    let mut solver = RungeKutta4::new(&state);
    let mut pts = [[0.0; 3]; LORENZ_UPO_STEPS];
    let mut vel = [[0.0; 3]; LORENZ_UPO_STEPS];
    let mut t = 0.0;
    for i in 0..LORENZ_UPO_STEPS {
        let dv = sys.derivative(t, &state);
        pts[i] = [state.x, state.y, state.z];
        vel[i] = [dv.x, dv.y, dv.z];
        solver.step(&sys, t, &mut state, dt);
        t += dt;
    }
    debug_assert!(pts[LORENZ_UPO_STEPS - 1][0].is_finite());
    (pts, vel)
}

/// Projects a 3D state onto the nearest velocity-consistent phase index along the precomputed Viswanath UPO.
#[must_use]
#[verified_engine::verified]
pub fn nearest_upo_index(
    upo_points: &[[f64; 3]; LORENZ_UPO_STEPS],
    upo_vel: &[[f64; 3]; LORENZ_UPO_STEPS],
    pos: &[f64; 3],
) -> (usize, f64) {
    let vel = [
        LORENZ_SIGMA * (pos[1] - pos[0]),
        pos[0] * (LORENZ_RHO0 - pos[2]) - pos[1],
        pos[0] * pos[1] - LORENZ_BETA * pos[2],
    ];
    let mut best_idx = 0;
    let mut best_score = f64::INFINITY;
    let mut best_dist_sq = f64::INFINITY;
    for i in 0..LORENZ_UPO_STEPS {
        let p = &upo_points[i];
        let v = &upo_vel[i];
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
    let dist = best_dist_sq.max(0.0).sqrt();
    debug_assert!(best_idx < LORENZ_UPO_STEPS && dist.is_finite());
    (best_idx, dist)
}

/// Extracts normalized 4D OGY policy features and a `0..64` discretized bin from the posterior EKF state.
#[must_use]
#[verified_engine::verified]
pub fn upo_features_and_bin(
    ekf_state: &[f64; 3],
    upo_points: &[[f64; 3]; LORENZ_UPO_STEPS],
    upo_vel: &[[f64; 3]; LORENZ_UPO_STEPS],
) -> ([f64; 4], usize) {
    let (idx, tube_dist) = nearest_upo_index(upo_points, upo_vel, ekf_state);
    let p = &upo_points[idx];
    let (ex, ey, ez) = (
        p[0] - ekf_state[0],
        p[1] - ekf_state[1],
        p[2] - ekf_state[2],
    );
    let sgn_x = ekf_state[0] / (ekf_state[0] * ekf_state[0] + 2.0).sqrt();
    let phi = [
        (0.32 * sgn_x * ey).tanh(),
        (0.32 * sgn_x * ex).tanh(),
        (0.12 * sgn_x * ez).tanh(),
        0.0,
    ];
    let xi = sgn_x * (0.72 * ey + 0.28 * ex);
    let xi_bin = quantize_bin(xi, -3.5, 3.5, 8);
    let d_bin = quantize_bin(tube_dist, 0.0, 8.0, 8);
    let bin = xi_bin * 8 + d_bin;
    debug_assert!(bin < 64);
    (phi, bin)
}

/// Computes bounded OGY Rayleigh actuation $\Delta\rho \in [-u_{\max}, u_{\max}]$.
#[must_use]
#[verified_engine::verified]
pub fn compute_ogy_rayleigh_thrust(raw_action: f64, actuation_budget: f64) -> (f64, f64) {
    let u_max = 4.5 * actuation_budget.max(0.0);
    let delta_rho = (raw_action.clamp(-1.0, 1.0) * u_max).clamp(-u_max, u_max);
    debug_assert!(delta_rho.abs() <= u_max + 1.0e-12);
    (delta_rho, u_max)
}

/// Updates `FastEkf<3>` and scores OGY tube stabilization after advancing controlled and open-loop Lorenz states.
#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
pub fn observe_and_score_upo_step(
    ekf: &mut FastEkf<3>,
    upo_points: &[[f64; 3]; LORENZ_UPO_STEPS],
    upo_vel: &[[f64; 3]; LORENZ_UPO_STEPS],
    controlled_vec: Vector3<f64>,
    uncontrolled_vec: Vector3<f64>,
    time: f64,
    delta_rho: f64,
    u_max: f64,
    cfg: UpoControlConfig,
    rng: &mut OxidizeRng,
) -> PlantStepOutput {
    let rho_eff = LORENZ_RHO0 + delta_rho;
    let r_var = lorenz_sensor_variance(cfg.sensor_noise);
    let q_diag = (LorenzKalmanModel {
        sigma: LORENZ_SIGMA,
        rho_eff,
        beta: LORENZ_BETA,
        q_var: cfg.process_noise,
        r_var,
    })
    .q_diag(cfg.dt);
    ekf.predict_rk4(cfg.dt, &q_diag, |s| {
        lorenz_rhs_and_jac(s, LORENZ_SIGMA, rho_eff, LORENZ_BETA)
    });

    let noise_std = 0.8 * cfg.sensor_noise.max(0.0);
    let z_meas = controlled_vec.x + noise_std * sample_standard_normal_f64(rng);
    ekf.update_unit_obs(0, z_meas, r_var);

    let (_, tube_dist) = nearest_upo_index(
        upo_points,
        upo_vel,
        &[controlled_vec.x, controlled_vec.y, controlled_vec.z],
    );
    let norm_u = if u_max > 1.0e-6 {
        delta_rho / u_max
    } else {
        0.0
    };
    let reward =
        2.0 * (-0.25 * tube_dist * tube_dist).exp() - 0.08 * tube_dist - 0.03 * norm_u * norm_u;
    debug_assert!(reward.is_finite());

    PlantStepOutput {
        time,
        uncontrolled_primary: uncontrolled_vec.x,
        controlled_primary: controlled_vec.x,
        noisy_measurement: z_meas,
        ekf_primary: ekf.state[0],
        ekf_primary_std: ekf.std_dev(0),
        true_latent: controlled_vec.y,
        ekf_latent: ekf.state[1],
        ekf_latent_std: ekf.std_dev(1),
        actuation: delta_rho,
        innovation: ekf.last_innovation,
        reward,
        tracking_metric: tube_dist,
    }
}

/// Computes the Lorenz scalar $x$-sensor variance $R = (0.8\,\sigma_{\text{sensor}})^2$.
#[must_use]
#[verified_engine::verified]
pub fn lorenz_sensor_variance(sensor_noise: f64) -> f64 {
    let s = 0.8 * sensor_noise.max(0.0);
    let var = s * s;
    debug_assert!(var >= 0.0);
    var
}

// [cite:graph_parameters_rust]

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_upo_reference_and_ekf_step_zero_alloc() {
        let cfg = UpoControlConfig::default();
        let (pts, vel) = precompute_viswanath_upo(cfg.dt);
        let init = Vector3::new(pts[0][0] + 0.55, pts[0][1] - 0.75, pts[0][2] + 0.45);
        let model = LorenzKalmanModel {
            sigma: LORENZ_SIGMA,
            rho_eff: LORENZ_RHO0,
            beta: LORENZ_BETA,
            q_var: cfg.process_noise,
            r_var: lorenz_sensor_variance(cfg.sensor_noise),
        };
        let mut ekf = FastEkf::from_validated_builder(
            model,
            cfg.dt,
            [init.x - 0.8, init.y + 1.6, init.z - 1.4],
            [2.5, 8.0, 8.0],
        );
        let mut ctrl_vec = init;
        let mut open_vec = init;
        let mut solver_c = RungeKutta4::new(&ctrl_vec);
        let mut solver_o = RungeKutta4::new(&open_vec);
        let mut rng = OxidizeRng::new(19);
        let mut t = 0.0;

        {
            let _guard = crate::VERIFIED_ENGINE_TEST_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            VerifiedEngine::run_verified(|| {
                for _ in 0..8 {
                    let (phi, bin) = upo_features_and_bin(&ekf.state, &pts, &vel);
                    assert!(bin < 64);
                    let (delta_rho, u_max) =
                        compute_ogy_rayleigh_thrust(phi[0], cfg.actuation_budget);
                    let sys_c = LorenzFlowRhs {
                        sigma: LORENZ_SIGMA,
                        rho: LORENZ_RHO0 + delta_rho,
                        beta: LORENZ_BETA,
                    };
                    let sys_o = LorenzFlowRhs {
                        sigma: LORENZ_SIGMA,
                        rho: LORENZ_RHO0,
                        beta: LORENZ_BETA,
                    };
                    solver_c.step(&sys_c, t, &mut ctrl_vec, cfg.dt);
                    solver_o.step(&sys_o, t, &mut open_vec, cfg.dt);
                    t += cfg.dt;
                    let out = observe_and_score_upo_step(
                        &mut ekf, &pts, &vel, ctrl_vec, open_vec, t, delta_rho, u_max, cfg,
                        &mut rng,
                    );
                    assert!(out.ekf_primary.is_finite());
                }
            });
        }
    }
}
