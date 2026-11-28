//! Nonlinear [`KalmanSystem`] observers and analytical ODE Jacobians for Lorenz, SEIR, and ADME PK plants.

use nalgebra::{DMatrix, DVector};

use super::KalmanSystem;
use super::ekf::{MIN_COV_DIAG, MIN_MEASUREMENT_VARIANCE, rk4_step_with_jacobian};

/// Continuous-time Lorenz vector field and analytical $3 \times 3$ Jacobian at effective Rayleigh number `rho_eff`.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::lorenz_rhs_and_jac;
///
/// let (rhs, jac) = lorenz_rhs_and_jac(&[1.0, 2.0, 3.0], 10.0, 28.0, 8.0 / 3.0);
/// assert!((rhs[0] - 10.0).abs() < 1.0e-12);
/// assert!((jac[0][0] + 10.0).abs() < 1.0e-12);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn lorenz_rhs_and_jac(
    state: &[f64; 3],
    sigma: f64,
    rho_eff: f64,
    beta: f64,
) -> ([f64; 3], [[f64; 3]; 3]) {
    debug_assert!(sigma.is_finite() && rho_eff.is_finite() && beta.is_finite());
    let (x, y, z) = (state[0], state[1], state[2]);
    let rhs = [sigma * (y - x), x * (rho_eff - z) - y, x * y - beta * z];
    let jac = [[-sigma, sigma, 0.0], [rho_eff - z, -1.0, -x], [y, x, -beta]];
    (rhs, jac)
}

/// Continuous-time SEIR vector field and analytical $4 \times 4$ Jacobian at effective transmission `beta_eff`.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::seir_rhs_and_jac;
///
/// let (rhs, _jac) = seir_rhs_and_jac(&[990.0, 5.0, 5.0, 0.0], 1000.0, 0.4, 0.2, 0.1);
/// assert!((rhs[0] + rhs[1] + rhs[2] + rhs[3]).abs() < 1.0e-12);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn seir_rhs_and_jac(
    state: &[f64; 4],
    pop_n: f64,
    beta_eff: f64,
    sigma: f64,
    gamma: f64,
) -> ([f64; 4], [[f64; 4]; 4]) {
    debug_assert!(pop_n.is_finite() && beta_eff.is_finite());
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
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::pk_rhs_and_jac;
///
/// let (rhs, jac) = pk_rhs_and_jac(&[100.0, 2.0, 0.25], 1.2, 20.0);
/// assert!((rhs[0] + 120.0).abs() < 1.0e-12);
/// assert!((jac[1][2] + 2.0).abs() < 1.0e-12);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn pk_rhs_and_jac(state: &[f64; 3], ka: f64, volume: f64) -> ([f64; 3], [[f64; 3]; 3]) {
    debug_assert!(ka.is_finite() && volume.is_finite());
    let (ag, cp, ke) = (state[0], state[1], state[2]);
    let inv_v = 1.0 / volume.max(1.0e-3);
    let rhs = [-ka * ag, ka * inv_v * ag - ke * cp, 0.0];
    let jac = [[-ka, 0.0, 0.0], [ka * inv_v, -ke, -cp], [0.0, 0.0, 0.0]];
    (rhs, jac)
}

/// Nonlinear 3D [`KalmanSystem`] observer for the Lorenz attractor from scalar $x(t)$ measurements.
///
/// Propagates state $\mathbf{x} = [x, y, z]^\top$ and discrete tangent sensitivity $\Phi_k = \partial \psi_{\Delta t}/\partial \mathbf{x}$
/// along $\dot{x} = \sigma(y - x)$, $\dot{y} = x(\rho_{\text{eff}} - z) - y$, $\dot{z} = xy - \beta z$ with scalar
/// observation $h(\mathbf{x}) = x$ per Lorenz (1963), *J. Atmos. Sci.* 20(2), 130-141
/// (`DOI: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2`) and Ott, Grebogi & Yorke (1990),
/// *Phys. Rev. Lett.* 64(11), 1196-1199 (`DOI: 10.1103/PhysRevLett.64.1196`).
///
/// # Stability
/// Uses fourth-order Runge-Kutta variational propagation and clamps process/measurement variances
/// above positive floors (`1e-9` / `1e-6`) to keep innovation and Joseph-form covariances SPD.
///
/// # Domain of Validity
/// Valid for finite parameters $(\sigma, \rho_{\text{eff}}, \beta)$ and integration steps
/// $0 < \Delta t \le 0.05$ below the chaotic folding timescale.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::LorenzKalmanModel;
///
/// let model = LorenzKalmanModel {
///     sigma: 10.0,
///     rho_eff: 28.0,
///     beta: 8.0 / 3.0,
///     q_var: 0.02,
///     r_var: 0.5,
/// };
/// assert!(model.q_diag(0.01)[0] > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
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
    #[verified_engine::verified]
    fn predict_state(&self, state: &DVector<f64>, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        let s = [state[0], state[1], state[2]];
        let (nx, jac) = rk4_step_with_jacobian(&s, dt, |x| {
            lorenz_rhs_and_jac(x, self.sigma, self.rho_eff, self.beta)
        });
        (array_to_dvec(&nx), array_to_dmat(&jac))
    }

    #[verified_engine::verified]
    fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        scalar_obs_dvec_dmat::<3>(state[0], 0)
    }

    #[verified_engine::verified]
    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        diag_dmat(&self.q_diag(dt))
    }

    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.r_var.max(MIN_MEASUREMENT_VARIANCE))
    }
}

impl LorenzKalmanModel {
    /// Diagonal process noise entries for `dt`.
    #[must_use]
    #[verified_engine::verified]
    pub fn q_diag(&self, dt: f64) -> [f64; 3] {
        debug_assert!(dt.is_finite());
        let q = (self.q_var * dt.max(1.0e-4)).max(MIN_COV_DIAG);
        [q, q * 1.5, q * 1.5]
    }
}

/// Nonlinear 4D [`KalmanSystem`] observer for epidemic SEIR dynamics from scalar infectious $I(t)$ counts.
///
/// Propagates compartmental state $\mathbf{x} = [S, E, I, R]^\top$ under bilinear mass-action transmission
/// $\dot{S} = -\beta_{\text{eff}} S I / N$, $\dot{E} = \beta_{\text{eff}} S I / N - \sigma E$,
/// $\dot{I} = \sigma E - \gamma I$, $\dot{R} = \gamma I$ with scalar surveillance $h(\mathbf{x}) = I$
/// per Kermack & McKendrick (1927), *Proc. R. Soc. Lond. A* 115(772), 700-721
/// (`DOI: 10.1098/rspa.1927.0118`).
///
/// # Stability
/// Clamps population $N \ge 1$ in the transmission denominator and enforces strictly positive
/// diagonal process and measurement noise floors so covariance updates remain well-conditioned.
///
/// # Domain of Validity
/// Valid for non-negative compartment populations $S, E, I, R \ge 0$ with $N > 0$, finite rates
/// $\beta_{\text{eff}}, \sigma, \gamma \ge 0$, and step sizes $0 < \Delta t \le 1.0$.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::SeirKalmanModel;
///
/// let model = SeirKalmanModel {
///     pop_n: 1000.0,
///     beta_eff: 0.35,
///     sigma: 0.2,
///     gamma: 0.1,
///     q_var: 0.05,
///     r_var: 4.0,
/// };
/// assert!(model.q_diag(0.1)[2] > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
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
    #[verified_engine::verified]
    fn predict_state(&self, state: &DVector<f64>, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        let s = [state[0], state[1], state[2], state[3]];
        let (nx, jac) = rk4_step_with_jacobian(&s, dt, |x| {
            seir_rhs_and_jac(x, self.pop_n, self.beta_eff, self.sigma, self.gamma)
        });
        (array_to_dvec(&nx), array_to_dmat(&jac))
    }

    #[verified_engine::verified]
    fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        scalar_obs_dvec_dmat::<4>(state[2], 2)
    }

    #[verified_engine::verified]
    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        diag_dmat(&self.q_diag(dt))
    }

    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.r_var.max(MIN_MEASUREMENT_VARIANCE))
    }
}

impl SeirKalmanModel {
    /// Diagonal process noise entries for `dt`.
    #[must_use]
    #[verified_engine::verified]
    pub fn q_diag(&self, dt: f64) -> [f64; 4] {
        debug_assert!(dt.is_finite());
        let scale = (self.q_var * 80.0 * dt.max(1.0e-4)).max(MIN_COV_DIAG);
        [scale * 1.5, scale * 2.0, scale, scale * 0.5]
    }
}

/// Nonlinear 3D [`KalmanSystem`] observer for ADME pharmacokinetics (`[A_g, C_p, k_e]`).
///
/// Propagates gut depot $A_g$, plasma concentration $C_p$, and latent elimination rate $k_e$ via
/// $\dot{A}_g = -k_a A_g$, $\dot{C}_p = \frac{k_a}{V} A_g - k_e C_p$, $\dot{k}_e = 0$ after bolus
/// impulse $A_g^+ = A_g + F D$ with assay observation $h(\mathbf{x}) = C_p$ per Bateman (1910),
/// *Proc. Cambridge Philos. Soc.* 15, 423-427 and Jazwinski (1970, `ISBN: 978-0-12-381550-7`,
/// `docs/math_specs/`). See also `DOI: 10.1007/BF01062108` (Sheiner et al., 1979).
///
/// # Stability
/// Guards apparent volume $V \ge 10^{-3}\,\text{L}$ against division by zero and maintains positive
/// random-walk process variance on $k_e$ for joint state-parameter observability during elimination.
///
/// # Domain of Validity
/// Valid for finite $k_a > 0$, $V > 0$, bioavailability $F \in (0, 1]$, non-negative bolus $D \ge 0$,
/// and sampling intervals $0 < \Delta t \le 2.0\,\text{h}$.
///
/// # Examples
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::algorithms::kalman::PkKalmanModel;
///
/// let model = PkKalmanModel {
///     ka: 1.2,
///     volume: 20.0,
///     bioavail: 0.85,
///     dose_bolus_mg: 100.0,
///     q_var: 0.05,
///     r_var: 0.04,
/// };
/// assert!(model.q_diag(0.5)[0] > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
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
    #[verified_engine::verified]
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

    #[verified_engine::verified]
    fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        scalar_obs_dvec_dmat::<3>(state[1], 1)
    }

    #[verified_engine::verified]
    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        diag_dmat(&self.q_diag(dt))
    }

    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<f64> {
        DMatrix::from_element(1, 1, self.r_var.max(MIN_MEASUREMENT_VARIANCE))
    }
}

impl PkKalmanModel {
    /// Diagonal process noise entries for `[A_g, C_p, k_e]` over `dt`.
    #[must_use]
    #[verified_engine::verified]
    pub fn q_diag(&self, dt: f64) -> [f64; 3] {
        debug_assert!(dt.is_finite());
        let base = (self.q_var * dt.max(1.0e-4)).max(MIN_COV_DIAG);
        [base * 4.0, base * 0.04, (base * 0.008).max(1.0e-6)]
    }
}

#[verified_engine::verified]
fn array_to_dvec<const N: usize>(arr: &[f64; N]) -> DVector<f64> {
    DVector::from_column_slice(arr)
}

#[verified_engine::verified]
fn array_to_dmat<const N: usize>(mat: &[[f64; N]; N]) -> DMatrix<f64> {
    DMatrix::from_fn(N, N, |r, c| mat[r][c])
}

#[verified_engine::verified]
fn diag_dmat<const N: usize>(diag: &[f64; N]) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_column_slice(diag))
}

#[verified_engine::verified]
fn scalar_obs_dvec_dmat<const N: usize>(val: f64, obs_idx: usize) -> (DVector<f64>, DMatrix<f64>) {
    let mut h = DMatrix::zeros(1, N);
    h[(0, obs_idx)] = 1.0;
    (DVector::from_element(1, val), h)
}
