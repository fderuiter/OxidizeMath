//! Forward-mode [`Dual<f64>`] Extended Kalman Filter traits and Jacobian adapters.

use engine_symbolic::Dual;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::ode::{DualArrayState, OdeSystem, rk4_step_dual_jacobian};

use super::KalmanSystem;

/// Evaluates a discrete transition map $f: \mathbb{R}^N \to \mathbb{R}^N$ and its exact
/// Jacobian $F_{ij} = \partial f_i / \partial x_j$ via forward-mode [`Dual<f64>`] seeding.
#[must_use]
#[verified_engine::verified]
pub fn dual_transition_jacobian<const N: usize, F>(
    state: &[f64; N],
    transition: F,
) -> ([f64; N], [[f64; N]; N])
where
    F: Fn(&[Dual<f64>; N]) -> [Dual<f64>; N],
{
    if N == 0 {
        return ([0.0; N], [[0.0; N]; N]);
    }
    let mut pred = [0.0_f64; N];
    let mut jac = [[0.0_f64; N]; N];
    let mut col = 0;
    while col < N {
        let seeded = DualArrayState::seed_column(state, col);
        let out = transition(&seeded.0);
        let mut row = 0;
        while row < N {
            if col == 0 {
                pred[row] = out[row].re;
            }
            jac[row][col] = out[row].eps;
            row += 1;
        }
        col += 1;
    }
    (pred, jac)
}

/// Evaluates an observation map $h: \mathbb{R}^N \to \mathbb{R}^M$ and its exact
/// Jacobian $H_{ij} = \partial h_i / \partial x_j$ via forward-mode [`Dual<f64>`] seeding.
#[must_use]
#[verified_engine::verified]
pub fn dual_measurement_jacobian<const N: usize, const M: usize, H>(
    state: &[f64; N],
    measurement: H,
) -> ([f64; M], [[f64; N]; M])
where
    H: Fn(&[Dual<f64>; N]) -> [Dual<f64>; M],
{
    if N == 0 {
        let zeros = [Dual::ZERO; N];
        let out = measurement(&zeros);
        let mut pred = [0.0_f64; M];
        let mut r = 0;
        while r < M {
            pred[r] = out[r].re;
            r += 1;
        }
        return (pred, [[0.0; N]; M]);
    }
    let mut pred = [0.0_f64; M];
    let mut jac = [[0.0_f64; N]; M];
    let mut col = 0;
    while col < N {
        let seeded = DualArrayState::seed_column(state, col);
        let out = measurement(&seeded.0);
        let mut row = 0;
        while row < M {
            if col == 0 {
                pred[row] = out[row].re;
            }
            jac[row][col] = out[row].eps;
            row += 1;
        }
        col += 1;
    }
    (pred, jac)
}

/// Nonlinear Extended Kalman Filter dynamics and observation model defined directly over
/// [`Dual<f64>`] scalars without hand-coded Jacobian matrices.
pub trait DualKalmanModel<const N: usize, const M: usize> {
    /// Evaluates the discrete state transition $x_k = f(x_{k-1}, \Delta t)$ over [`Dual<f64>`].
    #[verified_engine::verified]
    fn transition_dual(&self, state: &[Dual<f64>; N], dt: f64) -> [Dual<f64>; N];

    /// Evaluates the observation function $z_k = h(x_k)$ over [`Dual<f64>`].
    #[verified_engine::verified]
    fn measurement_dual(&self, state: &[Dual<f64>; N]) -> [Dual<f64>; M];

    /// Returns the process noise covariance matrix $Q_k$ of shape $N \times N$.
    #[verified_engine::verified]
    fn process_noise_matrix(&self, dt: f64) -> DMatrix<f64>;

    /// Returns the measurement noise covariance matrix $R_k$ of shape $M \times M$.
    #[verified_engine::verified]
    fn measurement_noise_matrix(&self) -> DMatrix<f64>;
}

/// Adapter lifting any [`DualKalmanModel<N, M>`] into a [`KalmanSystem<f64>`] by automatically
/// differentiating `transition_dual` and `measurement_dual`.
#[derive(Debug, Clone, Copy)]
pub struct DualExtendedKalmanSystem<const N: usize, const M: usize, D: DualKalmanModel<N, M>>(
    /// Inner `Dual`-lifted nonlinear model.
    pub D,
);

impl<const N: usize, const M: usize, D: DualKalmanModel<N, M>> KalmanSystem<f64>
    for DualExtendedKalmanSystem<N, M, D>
{
    #[verified_engine::verified]
    fn predict_state(&self, state: &DVector<f64>, dt: f64) -> (DVector<f64>, DMatrix<f64>) {
        let mut s = [0.0_f64; N];
        let mut i = 0;
        while i < N {
            s[i] = state[i];
            i += 1;
        }
        let (pred, jac) = dual_transition_jacobian(&s, |x| self.0.transition_dual(x, dt));
        (
            DVector::from_column_slice(&pred),
            DMatrix::from_fn(N, N, |r, c| jac[r][c]),
        )
    }

    #[verified_engine::verified]
    fn predict_measurement(&self, state: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        let mut s = [0.0_f64; N];
        let mut i = 0;
        while i < N {
            s[i] = state[i];
            i += 1;
        }
        let (pred, jac) = dual_measurement_jacobian(&s, |x| self.0.measurement_dual(x));
        (
            DVector::from_column_slice(&pred),
            DMatrix::from_fn(M, N, |r, c| jac[r][c]),
        )
    }

    #[verified_engine::verified]
    fn process_noise(&self, dt: f64) -> DMatrix<f64> {
        self.0.process_noise_matrix(dt)
    }

    #[verified_engine::verified]
    fn measurement_noise(&self) -> DMatrix<f64> {
        self.0.measurement_noise_matrix()
    }
}

/// Propagates a `#[verified]` [`OdeSystem<DualArrayState<N>>`] by one RK4 step and returns
/// `(next_state, transition_jacobian)` as `(DVector<f64>, DMatrix<f64>)` for [`KalmanSystem<f64>`].
#[must_use]
#[verified_engine::verified]
pub fn rk4_ode_kalman_predict<const N: usize, S>(
    system: &S,
    t: f64,
    state: &DVector<f64>,
    dt: f64,
) -> (DVector<f64>, DMatrix<f64>)
where
    S: OdeSystem<DualArrayState<N>> + ?Sized,
{
    let mut s = [0.0_f64; N];
    let mut i = 0;
    while i < N {
        s[i] = state[i];
        i += 1;
    }
    let (nx, jac) = rk4_step_dual_jacobian(system, t, &s, dt);
    (
        DVector::from_column_slice(&nx),
        DMatrix::from_fn(N, N, |r, c| jac[r][c]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::applied::algorithms::kalman::{KalmanError, KalmanFilter};

    #[test]
    #[verified_engine::verified]
    fn test_dual_extended_kalman_system_without_hand_jacobian() -> Result<(), KalmanError> {
        struct SqrtDualModel;
        impl DualKalmanModel<1, 1> for SqrtDualModel {
            #[verified_engine::verified]
            fn transition_dual(&self, state: &[Dual<f64>; 1], _dt: f64) -> [Dual<f64>; 1] {
                [state[0].sqrt()]
            }

            #[verified_engine::verified]
            fn measurement_dual(&self, state: &[Dual<f64>; 1]) -> [Dual<f64>; 1] {
                [state[0] * state[0]]
            }

            #[verified_engine::verified]
            fn process_noise_matrix(&self, _dt: f64) -> DMatrix<f64> {
                DMatrix::from_element(1, 1, 0.1)
            }

            #[verified_engine::verified]
            fn measurement_noise_matrix(&self) -> DMatrix<f64> {
                DMatrix::from_element(1, 1, 0.1)
            }
        }

        let mut kf = KalmanFilter::builder(DualExtendedKalmanSystem(SqrtDualModel), 1.0)
            .initial_state(DVector::from_element(1, 100.0))
            .initial_covariance(DMatrix::from_element(1, 1, 1.0))
            .build()?;
        kf.predict();
        assert!((kf.state[0] - 10.0).abs() < 1.0e-12);
        // F = 1 / (2 * sqrt(100)) = 0.05 -> P = 0.05^2 * 1 + 0.1 = 0.1025
        assert!((kf.covariance[(0, 0)] - 0.1025).abs() < 1.0e-12);
        kf.update(&DVector::from_element(1, 100.0))?;
        assert!((kf.state[0] - 10.0).abs() < 1.0e-12);
        Ok(())
    }
}
