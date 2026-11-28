//! Zero-allocation stack Extended Kalman Filter (`FastEkf`) and 4th-order Runge-Kutta
//! variational tangent integrator (`rk4_step_with_jacobian`).

use engine_linalg::{
    add_scaled_vec_array, identity_array, identity_plus_scaled_array, mat_mul_array,
    mat_mul_transpose_right_array,
};
use nalgebra::{DMatrix, DVector};

use super::{KalmanError, KalmanFilter, KalmanSystem};

/// Minimum scalar measurement variance floor preventing singular innovation covariance when `sensor_noise == 0.0`.
pub const MIN_MEASUREMENT_VARIANCE: f64 = 1.0e-8;
pub(crate) const MIN_COV_DIAG: f64 = 1.0e-12;

/// Propagates an `N`-dimensional ODE state and its discrete 4th-order Runge-Kutta variational
/// tangent Jacobian $F_k = \partial f_{\text{RK4}} / \partial x$ on the stack with zero heap allocations.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::rk4_step_with_jacobian;
///
/// let (next_x, jac) = rk4_step_with_jacobian(&[1.0_f64], 0.1, |x| ([-2.0 * x[0]], [[-2.0]]));
/// assert!((next_x[0] - (-0.2_f64).exp()).abs() < 1.0e-5);
/// assert!((jac[0][0] - (-0.2_f64).exp()).abs() < 1.0e-5);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn rk4_step_with_jacobian<const N: usize, F>(
    state: &[f64; N],
    dt: f64,
    rhs_and_jac: F,
) -> ([f64; N], [[f64; N]; N])
where
    F: Fn(&[f64; N]) -> ([f64; N], [[f64; N]; N]),
{
    debug_assert!(dt.is_finite());
    let half = 0.5 * dt;
    let sixth = dt / 6.0;

    let (k1, a1) = rhs_and_jac(state);
    let x2 = add_scaled_vec_array(state, &k1, half);
    let phi2 = identity_plus_scaled_array(&a1, half);
    let (k2, a2) = rhs_and_jac(&x2);
    let big_k2 = mat_mul_array(&a2, &phi2);

    let x3 = add_scaled_vec_array(state, &k2, half);
    let phi3 = identity_plus_scaled_array(&big_k2, half);
    let (k3, a3) = rhs_and_jac(&x3);
    let big_k3 = mat_mul_array(&a3, &phi3);

    let x4 = add_scaled_vec_array(state, &k3, dt);
    let phi4 = identity_plus_scaled_array(&big_k3, dt);
    let (k4, a4) = rhs_and_jac(&x4);
    let big_k4 = mat_mul_array(&a4, &phi4);

    let mut next_x = *state;
    let mut jac = identity_array::<f64, N>();
    for (i, nx_slot) in next_x.iter_mut().enumerate() {
        *nx_slot += sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        for (j, jac_slot) in jac[i].iter_mut().enumerate() {
            *jac_slot +=
                sixth * (a1[i][j] + 2.0 * big_k2[i][j] + 2.0 * big_k3[i][j] + big_k4[i][j]);
        }
    }
    (next_x, jac)
}

/// Zero-allocation stack Extended Kalman Filter mirroring [`KalmanFilter`] for fixed dimension `N`.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::FastEkf;
///
/// let mut ekf = FastEkf::new([0.0, 1.0], [1.0, 1.0]);
/// ekf.predict_rk4(0.1, &[1.0e-4, 1.0e-4], |x| ([x[1], -x[0]], [[0.0, 1.0], [-1.0, 0.0]]));
/// ekf.update_unit_obs(0, 0.11, 1.0e-2);
/// assert!(ekf.state[0].is_finite());
/// assert!(ekf.std_dev(0) > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastEkf<const N: usize> {
    /// Posterior state estimate $\hat{x}_{k|k}$.
    pub state: [f64; N],
    /// Posterior error covariance $P_{k|k}$.
    pub cov: [[f64; N]; N],
    /// Latest scalar measurement residual $y_k = z_k - h(\hat{x}_{k|k-1})$.
    pub last_innovation: f64,
    /// Latest scalar innovation variance $S_k = H P_{k|k-1} H^T + R$.
    pub last_innovation_var: f64,
}

impl<const N: usize> FastEkf<N> {
    /// Constructs a stack-resident [`FastEkf`] directly from initial state and diagonal covariance.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(init_state: [f64; N], init_cov_diag: [f64; N]) -> Self {
        let mut ekf = Self {
            state: [0.0; N],
            cov: [[0.0; N]; N],
            last_innovation: 0.0,
            last_innovation_var: MIN_MEASUREMENT_VARIANCE,
        };
        ekf.reset_in_place(init_state, init_cov_diag);
        ekf
    }

    /// Validates dimensions via [`KalmanFilter::builder`] at startup and constructs a stack-resident mirror.
    ///
    /// # Errors
    /// Returns [`KalmanError`] if [`KalmanFilter::builder`] dimension checks fail.
    #[verified_engine::verified]
    pub fn try_from_validated_builder<M: KalmanSystem<f64>>(
        model: M,
        dt: f64,
        init_state: [f64; N],
        init_cov_diag: [f64; N],
    ) -> Result<Self, KalmanError> {
        let kf = KalmanFilter::builder(model, dt)
            .initial_state(DVector::from_column_slice(&init_state))
            .initial_covariance(DMatrix::from_diagonal(&DVector::from_column_slice(
                &init_cov_diag,
            )))
            .build()?;
        let mut state = [0.0; N];
        let mut cov = [[0.0; N]; N];
        for (i, (s_slot, cov_row)) in state.iter_mut().zip(cov.iter_mut()).enumerate() {
            *s_slot = kf.state[i];
            for (j, c_slot) in cov_row.iter_mut().enumerate() {
                *c_slot = kf.covariance[(i, j)];
            }
        }
        Ok(Self {
            state,
            cov,
            last_innovation: 0.0,
            last_innovation_var: MIN_MEASUREMENT_VARIANCE,
        })
    }

    /// Validates dimensions via [`KalmanFilter::builder`] at startup, falling back to [`Self::new`].
    #[must_use]
    #[verified_engine::verified]
    pub fn from_validated_builder<M: KalmanSystem<f64>>(
        model: M,
        dt: f64,
        init_state: [f64; N],
        init_cov_diag: [f64; N],
    ) -> Self {
        Self::try_from_validated_builder(model, dt, init_state, init_cov_diag)
            .unwrap_or_else(|_| Self::new(init_state, init_cov_diag))
    }

    /// Resets state and diagonal covariance in place without heap allocations.
    #[verified_engine::verified]
    pub fn reset_in_place(&mut self, init_state: [f64; N], init_cov_diag: [f64; N]) {
        self.state = init_state;
        self.cov = [[0.0; N]; N];
        for (i, (row, &d)) in self.cov.iter_mut().zip(init_cov_diag.iter()).enumerate() {
            row[i] = d.max(MIN_COV_DIAG);
        }
        self.last_innovation = 0.0;
        self.last_innovation_var = MIN_MEASUREMENT_VARIANCE;
    }

    /// Executes EKF state and covariance prediction $P_{k|k-1} = F P_{k-1|k-1} F^T + Q$ on the stack.
    #[verified_engine::verified]
    pub fn predict_rk4<F>(&mut self, dt: f64, q_diag: &[f64; N], rhs_and_jac: F)
    where
        F: Fn(&[f64; N]) -> ([f64; N], [[f64; N]; N]),
    {
        debug_assert!(dt.is_finite());
        let (nx, f_mat) = rk4_step_with_jacobian(&self.state, dt, rhs_and_jac);
        self.state = nx;
        let fp = mat_mul_array(&f_mat, &self.cov);
        let mut next_cov = mat_mul_transpose_right_array(&fp, &f_mat);
        for (i, (row, &q)) in next_cov.iter_mut().zip(q_diag.iter()).enumerate() {
            row[i] += q.max(MIN_COV_DIAG);
        }
        symmetrize_cov(&mut next_cov, false);
        self.cov = next_cov;
    }

    /// Updates state and covariance from scalar observation $z_k = x_{\text{obs}} + v_k$ on the stack.
    #[verified_engine::verified]
    pub fn update_unit_obs(&mut self, obs_idx: usize, z_meas: f64, r_var: f64) {
        debug_assert!(obs_idx < N);
        if obs_idx >= N {
            return;
        }
        let r_safe = r_var.max(MIN_MEASUREMENT_VARIANCE);
        let y = z_meas - self.state[obs_idx];
        let s = (self.cov[obs_idx][obs_idx] + r_safe).max(MIN_COV_DIAG);
        let mut k_gain = [0.0; N];
        for (k_slot, cov_row) in k_gain.iter_mut().zip(self.cov.iter()) {
            *k_slot = cov_row[obs_idx] / s;
        }
        for (st_slot, &kg) in self.state.iter_mut().zip(k_gain.iter()) {
            *st_slot += kg * y;
        }
        let obs_row = self.cov[obs_idx];
        for (cov_row, &kg) in self.cov.iter_mut().zip(k_gain.iter()) {
            for (c_slot, &obs_c) in cov_row.iter_mut().zip(obs_row.iter()) {
                *c_slot -= kg * obs_c;
            }
        }
        symmetrize_cov(&mut self.cov, true);
        self.last_innovation = y;
        self.last_innovation_var = s;
    }

    /// Posterior `1-sigma` standard deviation $\sqrt{P_{ii}}$ for state component `idx`.
    #[must_use]
    #[verified_engine::verified]
    pub fn std_dev(&self, idx: usize) -> f64 {
        debug_assert!(idx < N);
        if idx < N {
            self.cov[idx][idx].max(0.0).sqrt()
        } else {
            0.0
        }
    }
}

#[verified_engine::verified]
fn symmetrize_cov<const N: usize>(cov: &mut [[f64; N]; N], clamp_diag: bool) {
    let copy = *cov;
    for (i, row) in cov.iter_mut().enumerate() {
        for (j, slot) in row.iter_mut().enumerate() {
            if i == j {
                if clamp_diag {
                    *slot = copy[i][i].max(MIN_COV_DIAG);
                }
            } else {
                *slot = 0.5 * (copy[i][j] + copy[j][i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::applied::algorithms::kalman::{LorenzKalmanModel, lorenz_rhs_and_jac};
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn fast_ekf_lorenz_runs_with_zero_allocations() {
        let _guard = crate::VERIFIED_ENGINE_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let model = LorenzKalmanModel {
            sigma: 10.0,
            rho_eff: 28.0,
            beta: 8.0 / 3.0,
            q_var: 0.02,
            r_var: 0.25,
        };
        let mut ekf = FastEkf::from_validated_builder(model, 0.01, [1.0, 1.0, 20.0], [2.0; 3]);
        let q_diag = model.q_diag(0.01);
        let (out, _) = VerifiedEngine::run_verified(|| {
            for step in 0..8 {
                ekf.predict_rk4(0.01, &q_diag, |x| {
                    lorenz_rhs_and_jac(x, 10.0, 28.0, 8.0 / 3.0)
                });
                ekf.update_unit_obs(0, 1.0 + 0.05 * f64::from(step), 0.25);
            }
            ekf.std_dev(0)
        });
        assert!(out > 0.0 && out.is_finite());
    }
}
