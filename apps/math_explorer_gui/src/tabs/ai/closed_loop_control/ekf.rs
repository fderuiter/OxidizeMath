//! Nonlinear `KalmanSystem<f64>` models and zero-allocation stack `FastEkf` mirrors.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use engine_linalg::{
    add_scaled_vec_array, identity_array, identity_plus_scaled_array, mat_mul_array,
    mat_mul_transpose_right_array,
};
use math_explorer::engine_diffeq::kalman::{KalmanFilter, KalmanSystem};
use math_explorer::nalgebra::{DMatrix, DVector};

/// Minimum scalar measurement variance floor preventing singular innovation covariance when `sensor_noise == 0.0`.
pub const MIN_MEASUREMENT_VARIANCE: f64 = 1.0e-8;
const MIN_COV_DIAG: f64 = 1.0e-12;

/// Propagates an `N`-dimensional ODE state and its exact discrete 4th-order Runge-Kutta tangent Jacobian $F_k = \partial f_{\text{RK4}} / \partial x$.
pub fn rk4_step_with_jacobian<const N: usize, F>(
    state: &[f64; N],
    dt: f64,
    rhs_and_jac: F,
) -> ([f64; N], [[f64; N]; N])
where
    F: Fn(&[f64; N]) -> ([f64; N], [[f64; N]; N]),
{
    let half = 0.5 * dt;
    let sixth = dt / 6.0;

    let (k1, a1) = rhs_and_jac(state);
    let x2 = add_scaled_vec(state, &k1, half);
    let phi2 = identity_plus_scaled(&a1, half);
    let (k2, a2) = rhs_and_jac(&x2);
    let big_k2 = mat_mul(&a2, &phi2);

    let x3 = add_scaled_vec(state, &k2, half);
    let phi3 = identity_plus_scaled(&big_k2, half);
    let (k3, a3) = rhs_and_jac(&x3);
    let big_k3 = mat_mul(&a3, &phi3);

    let x4 = add_scaled_vec(state, &k3, dt);
    let phi4 = identity_plus_scaled(&big_k3, dt);
    let (k4, a4) = rhs_and_jac(&x4);
    let big_k4 = mat_mul(&a4, &phi4);

    let mut next_x = *state;
    let mut jac = identity_mat::<N>();
    for (i, nx_slot) in next_x.iter_mut().enumerate() {
        *nx_slot += sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        for (j, jac_slot) in jac[i].iter_mut().enumerate() {
            *jac_slot +=
                sixth * (a1[i][j] + 2.0 * big_k2[i][j] + 2.0 * big_k3[i][j] + big_k4[i][j]);
        }
    }
    (next_x, jac)
}

#[inline]
fn identity_mat<const N: usize>() -> [[f64; N]; N] {
    identity_array::<f64, N>()
}

#[inline]
fn add_scaled_vec<const N: usize>(base: &[f64; N], delta: &[f64; N], scale: f64) -> [f64; N] {
    add_scaled_vec_array(base, delta, scale)
}

#[inline]
fn identity_plus_scaled<const N: usize>(mat: &[[f64; N]; N], scale: f64) -> [[f64; N]; N] {
    identity_plus_scaled_array(mat, scale)
}

#[inline]
fn mat_mul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    mat_mul_array(a, b)
}

#[inline]
fn mat_mul_transpose_right<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    mat_mul_transpose_right_array(a, b)
}

/// Continuous-time Lorenz vector field and analytical $3 \times 3$ Jacobian at effective Rayleigh number `rho_eff`.
pub fn lorenz_rhs_and_jac(
    state: &[f64; 3],
    sigma: f64,
    rho_eff: f64,
    beta: f64,
) -> ([f64; 3], [[f64; 3]; 3]) {
    let (x, y, z) = (state[0], state[1], state[2]);
    let rhs = [sigma * (y - x), x * (rho_eff - z) - y, x * y - beta * z];
    let jac = [[-sigma, sigma, 0.0], [rho_eff - z, -1.0, -x], [y, x, -beta]];
    (rhs, jac)
}

/// Continuous-time SEIR vector field and analytical $4 \times 4$ Jacobian at effective transmission `beta_eff`.
pub fn seir_rhs_and_jac(
    state: &[f64; 4],
    pop_n: f64,
    beta_eff: f64,
    sigma: f64,
    gamma: f64,
) -> ([f64; 4], [[f64; 4]; 4]) {
    let (s, e, inf) = (state[0], state[1], state[2]);
    let inv_n = 1.0 / pop_n.max(1.0);
    let inf_flow = beta_eff * s * inf * inv_n;
    let d_s = beta_eff * inf * inv_n;
    let d_i = beta_eff * s * inv_n;
    let rhs = [
        -inf_flow,
        inf_flow - sigma * e,
        sigma * e - gamma * inf,
        gamma * inf,
    ];
    let jac = [
        [-d_s, 0.0, -d_i, 0.0],
        [d_s, -sigma, d_i, 0.0],
        [0.0, sigma, -gamma, 0.0],
        [0.0, 0.0, gamma, 0.0],
    ];
    (rhs, jac)
}

/// Continuous-time nonlinear ADME pharmacokinetic vector field (`[A_g, C_p, k_e]`) and $3 \times 3$ Jacobian.
pub fn pk_rhs_and_jac(state: &[f64; 3], ka: f64, volume: f64) -> ([f64; 3], [[f64; 3]; 3]) {
    let (ag, cp, ke) = (state[0], state[1], state[2]);
    let inv_v = 1.0 / volume.max(1.0e-3);
    let rhs = [-ka * ag, ka * inv_v * ag - ke * cp, 0.0];
    let jac = [[-ka, 0.0, 0.0], [ka * inv_v, -ke, -cp], [0.0, 0.0, 0.0]];
    (rhs, jac)
}

/// Nonlinear 3D `KalmanSystem<f64>` observer for `LorenzSystem` from scalar $x(t)$ measurements.
#[derive(Debug, Clone, Copy)]
pub struct LorenzKalmanModel {
    /// Prandtl number $\sigma$.
    pub sigma: f64,
    /// Effective Rayleigh parameter $\rho_0 + \Delta\rho$.
    pub rho_eff: f64,
    /// Geometric damping parameter $\beta$.
    pub beta: f64,
    /// Process noise spectral intensity $q$.
    pub q_var: f64,
    /// Scalar measurement noise variance $R$.
    pub r_var: f64,
}

impl KalmanSystem<f64> for LorenzKalmanModel {
    fn predict_state(&self, state: &DVector<f64>, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        let s = [state[0], state[1], state[2]];
        let (nx, jac) = rk4_step_with_jacobian(&s, dt, |x| {
            lorenz_rhs_and_jac(x, self.sigma, self.rho_eff, self.beta)
        });
        (array_to_dvec(&nx), array_to_dmat(&jac))
    }

    fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        scalar_obs_dvec_dmat::<3>(state[0], 0)
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        diag_dmat(&self.q_diag(dt))
    }

    fn measurement_noise(&self) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.r_var.max(MIN_MEASUREMENT_VARIANCE))
    }
}

impl LorenzKalmanModel {
    /// Diagonal process noise entries for `dt`.
    pub fn q_diag(&self, dt: f64) -> [f64; 3] {
        let q = (self.q_var * dt.max(1.0e-4)).max(MIN_COV_DIAG);
        [q, q * 1.5, q * 1.5]
    }
}

/// Nonlinear 4D `KalmanSystem<f64>` observer for `SEIRModel` from scalar infectious $I(t)$ counts.
#[derive(Debug, Clone, Copy)]
pub struct SeirKalmanModel {
    /// Total conserved population $N$.
    pub pop_n: f64,
    /// Throttled transmission rate $\beta_{\text{eff}} = \beta_0(1 - u_t)$.
    pub beta_eff: f64,
    /// Incubation progression rate $\sigma$.
    pub sigma: f64,
    /// Recovery rate $\gamma$.
    pub gamma: f64,
    /// Process noise intensity.
    pub q_var: f64,
    /// Measurement noise variance on infectious count $I(t)$.
    pub r_var: f64,
}

impl KalmanSystem<f64> for SeirKalmanModel {
    fn predict_state(&self, state: &DVector<f64>, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        let s = [state[0], state[1], state[2], state[3]];
        let (nx, jac) = rk4_step_with_jacobian(&s, dt, |x| {
            seir_rhs_and_jac(x, self.pop_n, self.beta_eff, self.sigma, self.gamma)
        });
        (array_to_dvec(&nx), array_to_dmat(&jac))
    }

    fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        scalar_obs_dvec_dmat::<4>(state[2], 2)
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        diag_dmat(&self.q_diag(dt))
    }

    fn measurement_noise(&self) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.r_var.max(MIN_MEASUREMENT_VARIANCE))
    }
}

impl SeirKalmanModel {
    /// Diagonal process noise entries for `dt`.
    pub fn q_diag(&self, dt: f64) -> [f64; 4] {
        let scale = (self.q_var * 80.0 * dt.max(1.0e-4)).max(MIN_COV_DIAG);
        [scale * 1.5, scale * 2.0, scale, scale * 0.5]
    }
}

/// Nonlinear 3D `KalmanSystem<f64>` observer for ADME pharmacokinetics (`[A_g, C_p, k_e]`).
#[derive(Debug, Clone, Copy)]
pub struct PkKalmanModel {
    /// First-order gut absorption rate $k_a$ (`1/h`).
    pub ka: f64,
    /// Apparent volume of distribution $V$ (`L`).
    pub volume: f64,
    /// Oral bioavailability fraction $F \in (0, 1]$.
    pub bioavail: f64,
    /// Oral dose bolus administered at the start of the interval (`mg`).
    pub dose_bolus_mg: f64,
    /// Process noise scale.
    pub q_var: f64,
    /// Plasma concentration assay variance $R$.
    pub r_var: f64,
}

impl KalmanSystem<f64> for PkKalmanModel {
    fn predict_state(&self, state: &DVector<f64>, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        let dosed = [
            state[0] + self.bioavail * self.dose_bolus_mg,
            state[1],
            state[2],
        ];
        let (nx, jac) =
            rk4_step_with_jacobian(&dosed, dt, |x| pk_rhs_and_jac(x, self.ka, self.volume));
        (array_to_dvec(&nx), array_to_dmat(&jac))
    }

    fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        scalar_obs_dvec_dmat::<3>(state[1], 1)
    }

    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        diag_dmat(&self.q_diag(dt))
    }

    fn measurement_noise(&self) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.r_var.max(MIN_MEASUREMENT_VARIANCE))
    }
}

impl PkKalmanModel {
    /// Diagonal process noise entries for `[A_g, C_p, k_e]` over `dt`.
    pub fn q_diag(&self, dt: f64) -> [f64; 3] {
        let base = (self.q_var * dt.max(1.0e-4)).max(MIN_COV_DIAG);
        [base * 4.0, base * 0.04, (base * 0.008).max(1.0e-6)]
    }
}

/// Zero-allocation stack Extended Kalman Filter mirroring `KalmanFilter<f64, M>`.
#[derive(Debug, Clone, Copy)]
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
    /// Validates dimensions via `KalmanFilter::builder` at startup and constructs a stack-resident mirror.
    pub fn from_validated_builder<M: KalmanSystem<f64>>(
        model: M,
        dt: f64,
        init_state: [f64; N],
        init_cov_diag: [f64; N],
    ) -> Self {
        let kf = KalmanFilter::builder(model, dt)
            .initial_state(array_to_dvec(&init_state))
            .initial_covariance(diag_dmat(&init_cov_diag))
            .build()
            .expect("KalmanFilter::builder dimensions are statically matched");
        let mut state = [0.0; N];
        let mut cov = [[0.0; N]; N];
        for (i, (s_slot, cov_row)) in state.iter_mut().zip(cov.iter_mut()).enumerate() {
            *s_slot = kf.state[i];
            for (j, c_slot) in cov_row.iter_mut().enumerate() {
                *c_slot = kf.covariance[(i, j)];
            }
        }
        Self {
            state,
            cov,
            last_innovation: 0.0,
            last_innovation_var: MIN_MEASUREMENT_VARIANCE,
        }
    }

    /// Resets state and diagonal covariance in place without heap allocations.
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
    pub fn predict_rk4<F>(&mut self, dt: f64, q_diag: &[f64; N], rhs_and_jac: F)
    where
        F: Fn(&[f64; N]) -> ([f64; N], [[f64; N]; N]),
    {
        let (nx, f_mat) = rk4_step_with_jacobian(&self.state, dt, rhs_and_jac);
        self.state = nx;
        let fp = mat_mul(&f_mat, &self.cov);
        let mut next_cov = mat_mul_transpose_right(&fp, &f_mat);
        for (i, (row, &q)) in next_cov.iter_mut().zip(q_diag.iter()).enumerate() {
            row[i] += q.max(MIN_COV_DIAG);
        }
        symmetrize_cov(&mut next_cov, false);
        self.cov = next_cov;
    }

    /// Updates state and covariance from scalar observation $z_k = x_{\text{obs}} + v_k$ on the stack.
    pub fn update_unit_obs(&mut self, obs_idx: usize, z_meas: f64, r_var: f64) {
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
    pub fn std_dev(&self, idx: usize) -> f64 {
        self.cov[idx][idx].max(0.0).sqrt()
    }
}

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

fn array_to_dvec<const N: usize>(arr: &[f64; N]) -> DVector<f64> {
    DVector::from_column_slice(arr)
}

fn array_to_dmat<const N: usize>(mat: &[[f64; N]; N]) -> DMatrix<f64> {
    DMatrix::from_fn(N, N, |r, c| mat[r][c])
}

fn diag_dmat<const N: usize>(diag: &[f64; N]) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_column_slice(diag))
}

fn scalar_obs_dvec_dmat<const N: usize>(val: f64, obs_idx: usize) -> (DVector<f64>, DMatrix<f64>) {
    let mut h = DMatrix::zeros(1, N);
    h[(0, obs_idx)] = 1.0;
    (DVector::from_element(1, val), h)
}

// [cite:grpo]
