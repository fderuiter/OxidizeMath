// [cite:engine_opt]
//! Control model and twice-differentiable stage/terminal cost abstractions for `iLQR` and `MPC`.

use crate::error::OptError;
use engine_linalg::mat_idx;

/// Nonlinear discrete-time state-space dynamics $x_{k+1} = f_k(x_k, u_k)$ with exact Jacobians.
///
/// $$
/// x_{k+1} = f_k(x_k, u_k), \qquad F_{x,k} = \frac{\partial f_k}{\partial x_k}, \qquad F_{u,k} = \frac{\partial f_k}{\partial u_k}
/// $$
///
/// Reference: Tassa, Mansard & Todorov (2014), DOI: 10.1109/ICRA.2014.6907001.
///
/// # Stability
/// Analytical or tangent-RK4 Jacobians preserve local quadratic Gauss-Newton convergence of single-shooting iLQR.
///
/// # Domain of Validity
/// Requires finite state `x` (`state_dim()`) and control `u` (`control_dim()`) inputs.
pub trait ControlModel {
    /// State vector dimension $n_x \ge 1$.
    fn state_dim(&self) -> usize;
    /// Control input dimension $n_u \ge 1$.
    fn control_dim(&self) -> usize;
    /// Propagates $x_{k+1} = f_k(x_k, u_k)$ and populates row-major `fx` ($n_x \times n_x$) and `fu` ($n_x \times n_u$).
    fn step_and_linearize(
        &self,
        step: usize,
        x: &[f64],
        u: &[f64],
        x_next: &mut [f64],
        fx: &mut [f64],
        fu: &mut [f64],
    ) -> Result<(), OptError>;
}

/// Twice-differentiable finite-horizon stage and terminal cost $\sum_{k=0}^{N-1} \ell_k(x_k, u_k) + \ell_N(x_N)$.
pub trait StageCost {
    /// Evaluates scalar running cost $\ell_k(x_k, u_k)$.
    fn stage_cost(&self, step: usize, x: &[f64], u: &[f64]) -> f64;
    /// Populates $\ell_x$ ($n_x$), $\ell_u$ ($n_u$), $\ell_{xx}$ ($n_x^2$), $\ell_{uu}$ ($n_u^2$), and $\ell_{ux}$ ($n_u \times n_x$).
    #[allow(clippy::too_many_arguments)]
    fn stage_derivatives(
        &self,
        step: usize,
        x: &[f64],
        u: &[f64],
        lx: &mut [f64],
        lu: &mut [f64],
        lxx: &mut [f64],
        luu: &mut [f64],
        lux: &mut [f64],
    );
    /// Evaluates scalar terminal cost $\ell_N(x_N)$.
    fn terminal_cost(&self, x: &[f64]) -> f64;
    /// Populates terminal gradient $\ell_{N,x}$ ($n_x$) and Hessian $\ell_{N,xx}$ ($n_x^2$).
    fn terminal_derivatives(&self, x: &[f64], lx: &mut [f64], lxx: &mut [f64]);
}

/// Diagonal quadratic tracking cost with optional smooth quadratic state-box barrier penalties.
#[derive(Debug, Clone, Copy)]
pub struct QuadraticBoxCost<'a> {
    /// Diagonal state tracking weights $q \in \mathbb{R}^{n_x}_{\ge 0}$.
    pub q_diag: &'a [f64],
    /// Diagonal control effort weights $r \in \mathbb{R}^{n_u}_{> 0}$.
    pub r_diag: &'a [f64],
    /// Diagonal terminal state weights $q_f \in \mathbb{R}^{n_x}_{\ge 0}$.
    pub qf_diag: &'a [f64],
    /// Target state vector $x_{\text{ref}} \in \mathbb{R}^{n_x}$.
    pub x_ref: &'a [f64],
    /// Target control vector $u_{\text{ref}} \in \mathbb{R}^{n_u}$.
    pub u_ref: &'a [f64],
    /// Optional lower state bounds $x_{\min} \in \mathbb{R}^{n_x}$ (`f64::NEG_INFINITY` disables per component).
    pub x_min: Option<&'a [f64]>,
    /// Optional upper state bounds $x_{\max} \in \mathbb{R}^{n_x}$ (`f64::INFINITY` disables per component).
    pub x_max: Option<&'a [f64]>,
    /// Quadratic penalty weight $w_{\text{bound}} \ge 0$ applied outside $[x_{\min}, x_{\max}]$.
    pub bound_weight: f64,
}

impl StageCost for QuadraticBoxCost<'_> {
    fn stage_cost(&self, _step: usize, x: &[f64], u: &[f64]) -> f64 {
        let mut cost = state_quad_and_barrier_cost(
            x,
            self.x_ref,
            self.q_diag,
            self.x_min,
            self.x_max,
            self.bound_weight,
        );
        for (j, &u_j) in u.iter().enumerate() {
            let du = u_j - self.u_ref[j];
            cost += 0.5 * self.r_diag[j] * du * du;
        }
        cost
    }

    fn stage_derivatives(
        &self,
        _step: usize,
        x: &[f64],
        u: &[f64],
        lx: &mut [f64],
        lu: &mut [f64],
        lxx: &mut [f64],
        luu: &mut [f64],
        lux: &mut [f64],
    ) {
        state_quad_and_barrier_derivs(
            x,
            self.x_ref,
            self.q_diag,
            self.x_min,
            self.x_max,
            self.bound_weight,
            lx,
            lxx,
        );
        let nu = u.len();
        luu.fill(0.0);
        lux.fill(0.0);
        for j in 0..nu {
            lu[j] = self.r_diag[j] * (u[j] - self.u_ref[j]);
            luu[mat_idx(j, j, nu)] = self.r_diag[j];
        }
    }

    fn terminal_cost(&self, x: &[f64]) -> f64 {
        state_quad_and_barrier_cost(
            x,
            self.x_ref,
            self.qf_diag,
            self.x_min,
            self.x_max,
            self.bound_weight,
        )
    }

    fn terminal_derivatives(&self, x: &[f64], lx: &mut [f64], lxx: &mut [f64]) {
        state_quad_and_barrier_derivs(
            x,
            self.x_ref,
            self.qf_diag,
            self.x_min,
            self.x_max,
            self.bound_weight,
            lx,
            lxx,
        );
    }
}

#[verified_engine::verified]
fn state_quad_and_barrier_cost(
    x: &[f64],
    x_ref: &[f64],
    weights: &[f64],
    x_min: Option<&[f64]>,
    x_max: Option<&[f64]>,
    bound_weight: f64,
) -> f64 {
    let mut total = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - x_ref[i];
        total += 0.5 * weights[i] * dx * dx;
        if let Some(lo) = x_min
            && lo[i].is_finite()
            && x[i] < lo[i]
        {
            let v = lo[i] - x[i];
            total += 0.5 * bound_weight * v * v;
        }
        if let Some(hi) = x_max
            && hi[i].is_finite()
            && x[i] > hi[i]
        {
            let v = x[i] - hi[i];
            total += 0.5 * bound_weight * v * v;
        }
    }
    total
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn state_quad_and_barrier_derivs(
    x: &[f64],
    x_ref: &[f64],
    weights: &[f64],
    x_min: Option<&[f64]>,
    x_max: Option<&[f64]>,
    bound_weight: f64,
    lx: &mut [f64],
    lxx: &mut [f64],
) {
    let nx = x.len();
    lxx.fill(0.0);
    for i in 0..nx {
        let mut g = weights[i] * (x[i] - x_ref[i]);
        let mut h = weights[i];
        if let Some(lo) = x_min
            && lo[i].is_finite()
            && x[i] < lo[i]
        {
            g -= bound_weight * (lo[i] - x[i]);
            h += bound_weight;
        }
        if let Some(hi) = x_max
            && hi[i].is_finite()
            && x[i] > hi[i]
        {
            g += bound_weight * (x[i] - hi[i]);
            h += bound_weight;
        }
        lx[i] = g;
        lxx[mat_idx(i, i, nx)] = h;
    }
}
