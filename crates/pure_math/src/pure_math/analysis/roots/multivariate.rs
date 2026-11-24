//! Damped multivariate Newton-Raphson solver for fixed-size stack vectors and
//! pre-allocated dynamic state vectors.

use super::AnalysisError;
use crate::pure_math::analysis::dense_linear::{canonical_zero, solve_linear_in_place};
use crate::pure_math::analysis::ode::IndexedState;
use nalgebra::{DVector, SVector};

/// Maximum compile-time stack vector dimension supported by [`MultivariateNewton::solve_fixed`].
pub const MAX_STACK_NEWTON_DIM: usize = 16;

const PIVOT_TOL: f64 = 1e-14;
const MAX_LINE_SEARCH_STEPS: usize = 8;

/// Multivariate Newton-Raphson nonlinear system solver $F(\vec{x}) = \vec{0}$.
///
/// Supports both zero-allocation stack execution on fixed-size arrays / `nalgebra::SVector`
/// (up to `MAX_STACK_NEWTON_DIM = 16`) and pre-allocated zero-allocation workspace execution on
/// dynamic slices, `DVector<f64>`, and [`IndexedState`] vectors constructed via
/// [`Self::with_dimension`].
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::roots::MultivariateNewton;
///
/// let solver = MultivariateNewton::default();
/// // Solve x^2 + y^2 - 4 = 0, x - y = 0 near (1.5, 1.0) => (sqrt(2), sqrt(2))
/// let root = solver
///     .solve_fixed(|x| [x[0] * x[0] + x[1] * x[1] - 4.0, x[0] - x[1]], [1.5, 1.0])
///     .unwrap();
/// assert!((root[0] - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
/// assert!((root[1] - std::f64::consts::SQRT_2).abs() < math_explorer::registry::TOLERANCE_FAST);
/// ```
#[derive(Debug, Clone)]
pub struct MultivariateNewton {
    /// Maximum number of Newton iterations.
    pub max_iterations: usize,
    /// Infinity-norm tolerance on residual and step vectors.
    pub tolerance: f64,
    /// Relative/absolute finite-difference step scale for numerical Jacobians.
    pub fd_step: f64,
    work_x: Vec<f64>,
    work_fx: Vec<f64>,
    work_f_plus: Vec<f64>,
    work_f_minus: Vec<f64>,
    work_trial_x: Vec<f64>,
    work_delta: Vec<f64>,
    work_jac: Vec<f64>,
    work_pivots: Vec<usize>,
}

impl Default for MultivariateNewton {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(
            math_commons::registry::MAX_ITERATIONS,
            math_commons::registry::TOLERANCE_FAST,
        )
    }
}

impl MultivariateNewton {
    /// Creates a new [`MultivariateNewton`] solver without heap allocation.
    #[verified_engine::verified]
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            fd_step: 1e-7,
            work_x: Vec::new(),
            work_fx: Vec::new(),
            work_f_plus: Vec::new(),
            work_f_minus: Vec::new(),
            work_trial_x: Vec::new(),
            work_delta: Vec::new(),
            work_jac: Vec::new(),
            work_pivots: Vec::new(),
        }
    }

    /// Creates a [`MultivariateNewton`] solver with scratch buffers pre-allocated for `dim` equations.
    #[verified_engine::verified]
    pub fn with_dimension(dim: usize, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            fd_step: 1e-7,
            work_x: vec![0.0; dim],
            work_fx: vec![0.0; dim],
            work_f_plus: vec![0.0; dim],
            work_f_minus: vec![0.0; dim],
            work_trial_x: vec![0.0; dim],
            work_delta: vec![0.0; dim],
            work_jac: vec![0.0; dim.saturating_mul(dim)],
            work_pivots: vec![0; dim],
        }
    }

    #[verified_engine::verified]
    fn ensure_workspace(&mut self, dim: usize) {
        if self.work_x.len() < dim {
            self.work_x.resize(dim, 0.0);
            self.work_fx.resize(dim, 0.0);
            self.work_f_plus.resize(dim, 0.0);
            self.work_f_minus.resize(dim, 0.0);
            self.work_trial_x.resize(dim, 0.0);
            self.work_delta.resize(dim, 0.0);
            self.work_pivots.resize(dim, 0);
        }
        let n2 = dim.saturating_mul(dim);
        if self.work_jac.len() < n2 {
            self.work_jac.resize(n2, 0.0);
        }
    }

    /// Solves $F(\vec{x}) = \vec{0}$ for a fixed-size stack array `[f64; N]` (`1 <= N <= 16`)
    /// using central finite-difference Jacobians and zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_fixed<const N: usize, F>(
        &self,
        mut f: F,
        initial: [f64; N],
    ) -> Result<[f64; N], AnalysisError>
    where
        F: FnMut(&[f64; N]) -> [f64; N],
    {
        if N == 0 || N > MAX_STACK_NEWTON_DIM {
            return Err(AnalysisError::InvalidParameters(
                "Fixed-size Newton dimension must lie in 1..=16",
            ));
        }
        let mut x = initial;
        let mut fx = [0.0; MAX_STACK_NEWTON_DIM];
        let mut f_plus = [0.0; MAX_STACK_NEWTON_DIM];
        let mut f_minus = [0.0; MAX_STACK_NEWTON_DIM];
        let mut trial_x = [0.0; MAX_STACK_NEWTON_DIM];
        let mut delta = [0.0; MAX_STACK_NEWTON_DIM];
        let mut jac = [0.0; MAX_STACK_NEWTON_DIM * MAX_STACK_NEWTON_DIM];
        let mut pivots = [0usize; MAX_STACK_NEWTON_DIM];

        let mut eval_slice = |pt: &[f64], out: &mut [f64]| {
            let mut arr = [0.0; N];
            arr.copy_from_slice(&pt[..N]);
            out[..N].copy_from_slice(&f(&arr));
        };

        run_newton_kernel(
            &mut eval_slice,
            &mut x,
            (self.max_iterations, self.tolerance, self.fd_step),
            &mut fx[..N],
            &mut f_plus[..N],
            &mut f_minus[..N],
            &mut trial_x[..N],
            &mut delta[..N],
            &mut jac[..N * N],
            &mut pivots[..N],
        )?;
        Ok(x)
    }

    /// Solves $F(\vec{x}) = \vec{0}$ for a fixed-size array `[f64; N]` using an analytical
    /// row-major Jacobian `jac_fn(&x) -> [[f64; N]; N]` with zero heap allocations.
    #[verified_engine::verified]
    pub fn solve_fixed_with_jacobian<const N: usize, F, J>(
        &self,
        mut f: F,
        mut jac_fn: J,
        initial: [f64; N],
    ) -> Result<[f64; N], AnalysisError>
    where
        F: FnMut(&[f64; N]) -> [f64; N],
        J: FnMut(&[f64; N]) -> [[f64; N]; N],
    {
        if N == 0 || N > MAX_STACK_NEWTON_DIM {
            return Err(AnalysisError::InvalidParameters(
                "Fixed-size Newton dimension must lie in 1..=16",
            ));
        }
        validate_newton_params(&initial, self.tolerance)?;
        let mut x = initial;
        let mut jac_flat = [0.0; MAX_STACK_NEWTON_DIM * MAX_STACK_NEWTON_DIM];
        let mut rhs = [0.0; MAX_STACK_NEWTON_DIM];
        let mut scratch = [0.0; MAX_STACK_NEWTON_DIM];
        let mut pivots = [0usize; MAX_STACK_NEWTON_DIM];

        for _ in 0..self.max_iterations {
            let fx = f(&x);
            if max_abs_finite(&fx)? <= self.tolerance {
                normalize_slice_zeros(&mut x);
                return Ok(x);
            }
            let j_rows = jac_fn(&x);
            for r in 0..N {
                jac_flat[r * N..(r + 1) * N].copy_from_slice(&j_rows[r]);
                rhs[r] = -fx[r];
            }
            if !solve_linear_in_place(
                &mut jac_flat[..N * N],
                &mut rhs[..N],
                &mut pivots[..N],
                &mut scratch[..N],
                N,
                PIVOT_TOL,
            ) {
                return Err(AnalysisError::InvalidParameters(
                    "Singular or non-finite Jacobian in MultivariateNewton",
                ));
            }
            for i in 0..N {
                x[i] += rhs[i];
            }
            if max_abs_finite(&rhs[..N])? <= self.tolerance {
                normalize_slice_zeros(&mut x);
                return Ok(x);
            }
        }
        let final_fx = f(&x);
        Err(AnalysisError::ConvergenceError(
            max_abs_finite(&final_fx).unwrap_or(f64::NAN),
        ))
    }

    /// Solves $F(\vec{x}) = \vec{0}$ for a fixed-size `nalgebra::SVector<f64, N>` (`1 <= N <= 16`).
    #[verified_engine::verified]
    pub fn solve_svector<const N: usize, F>(
        &self,
        mut f: F,
        initial: SVector<f64, N>,
    ) -> Result<SVector<f64, N>, AnalysisError>
    where
        F: FnMut(&SVector<f64, N>) -> SVector<f64, N>,
    {
        let mut init_arr = [0.0; N];
        init_arr.copy_from_slice(initial.as_slice());
        let solved = self.solve_fixed(
            |arr| {
                let out = f(&SVector::<f64, N>::from_column_slice(arr));
                let mut res = [0.0; N];
                res.copy_from_slice(out.as_slice());
                res
            },
            init_arr,
        )?;
        Ok(SVector::<f64, N>::from_column_slice(&solved))
    }

    /// Solves $F(\vec{x}) = \vec{0}$ in-place on `x` using pre-allocated workspace buffers.
    #[verified_engine::verified]
    pub fn solve_in_place<F>(&mut self, mut f: F, x: &mut [f64]) -> Result<(), AnalysisError>
    where
        F: FnMut(&[f64], &mut [f64]),
    {
        let n = x.len();
        if n == 0 {
            return Err(AnalysisError::InvalidParameters("Dimension must be >= 1"));
        }
        self.ensure_workspace(n);
        run_newton_kernel(
            &mut f,
            x,
            (self.max_iterations, self.tolerance, self.fd_step),
            &mut self.work_fx[..n],
            &mut self.work_f_plus[..n],
            &mut self.work_f_minus[..n],
            &mut self.work_trial_x[..n],
            &mut self.work_delta[..n],
            &mut self.work_jac[..n * n],
            &mut self.work_pivots[..n],
        )
    }

    /// Solves $F(\vec{x}) = \vec{0}$ for a dynamic vector slice, returning a newly allocated `Vec<f64>`.
    #[verified_engine::verified]
    pub fn solve_dynamic<F>(&mut self, f: F, initial: &[f64]) -> Result<Vec<f64>, AnalysisError>
    where
        F: FnMut(&[f64], &mut [f64]),
    {
        let mut x = initial.to_vec();
        self.solve_in_place(f, &mut x)?;
        Ok(x)
    }

    /// Solves $F(\vec{x}) = \vec{0}$ in-place on a `nalgebra::DVector<f64>`.
    #[verified_engine::verified]
    pub fn solve_dvector_in_place<F>(
        &mut self,
        f: F,
        x: &mut DVector<f64>,
    ) -> Result<(), AnalysisError>
    where
        F: FnMut(&[f64], &mut [f64]),
    {
        self.solve_in_place(f, x.as_mut_slice())
    }

    /// Solves $F(\vec{x}) = \vec{0}$ in-place on any [`IndexedState`] vector.
    #[verified_engine::verified]
    pub fn solve_state<S, F>(&mut self, mut f: F, state: &mut S) -> Result<(), AnalysisError>
    where
        S: IndexedState,
        F: FnMut(&S, &mut S),
    {
        let n = state.len();
        if n == 0 {
            return Err(AnalysisError::InvalidParameters("Dimension must be >= 1"));
        }
        self.ensure_workspace(n);
        let mut x_scratch = state.clone();
        let mut out_scratch = state.clone();
        let mut x_buf = std::mem::take(&mut self.work_x);
        for (i, dst) in x_buf.iter_mut().take(n).enumerate() {
            *dst = state.get(i);
        }
        let mut adapter = |x_slice: &[f64], out_slice: &mut [f64]| {
            for (idx, &val) in x_slice.iter().take(n).enumerate() {
                x_scratch.set(idx, val);
            }
            f(&x_scratch, &mut out_scratch);
            for (idx, dst) in out_slice.iter_mut().take(n).enumerate() {
                *dst = out_scratch.get(idx);
            }
        };
        let res = run_newton_kernel(
            &mut adapter,
            &mut x_buf[..n],
            (self.max_iterations, self.tolerance, self.fd_step),
            &mut self.work_fx[..n],
            &mut self.work_f_plus[..n],
            &mut self.work_f_minus[..n],
            &mut self.work_trial_x[..n],
            &mut self.work_delta[..n],
            &mut self.work_jac[..n * n],
            &mut self.work_pivots[..n],
        );
        for (i, &val) in x_buf.iter().take(n).enumerate() {
            state.set(i, val);
        }
        self.work_x = x_buf;
        res
    }
}

#[verified_engine::verified]
fn validate_newton_params(x: &[f64], tolerance: f64) -> Result<(), AnalysisError> {
    if !tolerance.is_finite() || tolerance < 0.0 || x.iter().any(|v| !v.is_finite()) {
        return Err(AnalysisError::InvalidParameters(
            "Initial guess and tolerance must be finite",
        ));
    }
    Ok(())
}

#[verified_engine::verified]
fn max_abs_finite(vals: &[f64]) -> Result<f64, AnalysisError> {
    let mut max_v = 0.0_f64;
    for &v in vals {
        if !v.is_finite() {
            return Err(AnalysisError::InvalidParameters(
                "Non-finite residual or step encountered in MultivariateNewton",
            ));
        }
        max_v = max_v.max(v.abs());
    }
    Ok(max_v)
}

#[verified_engine::verified]
fn normalize_slice_zeros(vals: &mut [f64]) {
    for v in vals.iter_mut() {
        *v = canonical_zero(*v);
    }
}

#[verified_engine::verified]
#[allow(clippy::too_many_arguments)]
fn run_newton_kernel(
    f: &mut dyn FnMut(&[f64], &mut [f64]),
    x: &mut [f64],
    (max_iterations, tolerance, fd_step): (usize, f64, f64),
    fx: &mut [f64],
    f_plus: &mut [f64],
    f_minus: &mut [f64],
    trial_x: &mut [f64],
    delta: &mut [f64],
    jac: &mut [f64],
    pivots: &mut [usize],
) -> Result<(), AnalysisError> {
    validate_newton_params(x, tolerance)?;
    let n = x.len();
    let step_scale = if fd_step.is_finite() && fd_step > 0.0 {
        fd_step
    } else {
        1e-7
    };

    f(x, fx);
    let mut res_norm = max_abs_finite(fx)?;
    if res_norm <= tolerance {
        normalize_slice_zeros(x);
        return Ok(());
    }

    for _ in 0..max_iterations {
        assemble_central_jacobian(f, x, f_plus, f_minus, trial_x, jac, n, step_scale)?;
        for i in 0..n {
            delta[i] = -fx[i];
        }
        if !solve_linear_in_place(jac, delta, pivots, f_minus, n, PIVOT_TOL) {
            return Err(AnalysisError::InvalidParameters(
                "Singular or ill-conditioned Jacobian in MultivariateNewton",
            ));
        }
        let step_norm = damp_newton_step(f, x, fx, f_plus, trial_x, delta, &mut res_norm)?;
        if res_norm <= tolerance || step_norm <= tolerance {
            normalize_slice_zeros(x);
            return Ok(());
        }
    }

    Err(AnalysisError::ConvergenceError(res_norm))
}

#[verified_engine::verified]
#[allow(clippy::too_many_arguments)]
fn assemble_central_jacobian(
    f: &mut dyn FnMut(&[f64], &mut [f64]),
    x: &[f64],
    f_plus: &mut [f64],
    f_minus: &mut [f64],
    trial_x: &mut [f64],
    jac: &mut [f64],
    n: usize,
    step_scale: f64,
) -> Result<(), AnalysisError> {
    trial_x.copy_from_slice(x);
    for col in 0..n {
        let h = step_scale * x[col].abs().max(1.0);
        trial_x[col] = x[col] + h;
        f(trial_x, f_plus);
        trial_x[col] = x[col] - h;
        f(trial_x, f_minus);
        trial_x[col] = x[col];
        let inv_2h = 0.5 / h;
        for row in 0..n {
            let entry = (f_plus[row] - f_minus[row]) * inv_2h;
            if !entry.is_finite() {
                return Err(AnalysisError::InvalidParameters(
                    "Non-finite finite-difference Jacobian entry",
                ));
            }
            jac[math_commons::math_kernel::types::flatten_2d_index(col, row, n)] = entry;
        }
    }
    Ok(())
}

#[verified_engine::verified]
fn damp_newton_step(
    f: &mut dyn FnMut(&[f64], &mut [f64]),
    x: &mut [f64],
    fx: &mut [f64],
    f_trial: &mut [f64],
    trial_x: &mut [f64],
    delta: &[f64],
    res_norm: &mut f64,
) -> Result<f64, AnalysisError> {
    let mut alpha = 1.0;
    for attempt in 0..MAX_LINE_SEARCH_STEPS {
        for i in 0..x.len() {
            trial_x[i] = x[i] + alpha * delta[i];
        }
        f(trial_x, f_trial);
        if let Ok(new_norm) = max_abs_finite(f_trial)
            && (new_norm < *res_norm || attempt + 1 == MAX_LINE_SEARCH_STEPS)
        {
            x.copy_from_slice(trial_x);
            fx.copy_from_slice(f_trial);
            *res_norm = new_norm;
            let mut max_step = 0.0_f64;
            for &d in delta {
                max_step = max_step.max((alpha * d).abs());
            }
            return Ok(max_step);
        }
        alpha *= 0.5;
    }
    Err(AnalysisError::InvalidParameters(
        "Damped Newton backtracking encountered only non-finite residuals",
    ))
}

// [cite:advanced_linear_algebra]
