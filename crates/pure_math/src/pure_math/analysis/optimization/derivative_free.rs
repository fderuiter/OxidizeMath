//! Derivative-free optimization: `NelderMead` n-D simplex search.

use super::OptimizationError;
use nalgebra::{DMatrix, DVector};

/// Hyperparameters for derivative-free `NelderMead` simplex minimization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMeadConfig {
    /// Maximum simplex iterations.
    pub max_iterations: usize,
    /// Simplex diameter and value spread stopping tolerance.
    pub tolerance: f64,
    /// Relative initial simplex perturbation scale.
    pub initial_step: f64,
    /// Reflection factor ($\alpha > 0$, typically `1.0`).
    pub alpha: f64,
    /// Expansion factor ($\gamma > 1$, typically `2.0`).
    pub gamma: f64,
    /// Contraction factor ($0 < \rho < 1$, typically `0.5`).
    pub rho: f64,
    /// Shrink factor ($0 < \sigma < 1$, typically `0.5`).
    pub sigma: f64,
}

impl Default for NelderMeadConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_FAST,
            initial_step: 0.1,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Solution summary returned by [`NelderMead::minimize`].
#[derive(Debug, Clone, PartialEq)]
pub struct NelderMeadSolution {
    /// Converged best vertex $x^*$.
    pub x: DVector<f64>,
    /// Minimum objective value $f(x^*)$.
    pub objective: f64,
    /// Number of simplex iterations executed.
    pub iterations: usize,
    /// Optional 2D best-vertex trajectory `(x[0], x[1])`.
    pub iterate_path: Vec<[f64; 2]>,
}

/// Preallocated scratch buffers for zero-allocation `NelderMead` execution.
#[derive(Debug, Clone)]
pub struct NelderMeadWorkspace {
    simplex: DMatrix<f64>,
    f_vals: Vec<f64>,
    order: Vec<usize>,
    centroid: DVector<f64>,
    reflected: DVector<f64>,
    trial: DVector<f64>,
    vertex_buf: DVector<f64>,
    iterate_path: Vec<[f64; 2]>,
}

impl NelderMeadWorkspace {
    /// Preallocates an `(n + 1)`-vertex simplex workspace for dimension `dim`.
    #[verified_engine::verified]
    pub fn new(dim: usize, max_iterations: usize) -> Self {
        Self {
            simplex: DMatrix::zeros(dim, dim + 1),
            f_vals: vec![0.0; dim + 1],
            order: (0..=dim).collect(),
            centroid: DVector::zeros(dim),
            reflected: DVector::zeros(dim),
            trial: DVector::zeros(dim),
            vertex_buf: DVector::zeros(dim),
            iterate_path: Vec::with_capacity(max_iterations + 2),
        }
    }

    /// Returns the recorded 2D best-vertex trajectory slice.
    #[verified_engine::verified]
    pub fn iterate_path_slice(&self) -> &[[f64; 2]] {
        &self.iterate_path
    }
}

/// Derivative-free n-dimensional Nelder-Mead amoeba/simplex minimizer.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::DVector;
/// use math_explorer::pure_math::analysis::optimization::NelderMead;
///
/// let solver = NelderMead::default();
/// let x0 = DVector::from_vec(vec![-1.2, 1.0]);
/// let sol = solver
///     .minimize(
///         |v| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2),
///         &x0,
///     )
///     .unwrap();
/// assert!((sol.x[0] - 1.0).abs() < 1e-3);
/// assert!((sol.x[1] - 1.0).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMead {
    /// Simplex step and stopping configuration.
    pub config: NelderMeadConfig,
}

impl Default for NelderMead {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            config: NelderMeadConfig::default(),
        }
    }
}

fn sort_simplex_order(order: &mut [usize], f_vals: &[f64]) {
    for i in 1..order.len() {
        let key = order[i];
        let key_val = f_vals[key];
        let mut j = i;
        while j > 0 && f_vals[order[j - 1]] > key_val {
            order[j] = order[j - 1];
            j -= 1;
        }
        order[j] = key;
    }
}

fn simplex_converged(ws: &NelderMeadWorkspace, tol: f64) -> bool {
    let best = ws.order[0];
    let worst = ws.order[ws.order.len() - 1];
    if (ws.f_vals[worst] - ws.f_vals[best]).abs() > tol {
        return false;
    }
    let n = ws.simplex.nrows();
    for &idx in &ws.order[1..] {
        for r in 0..n {
            if (ws.simplex[(r, idx)] - ws.simplex[(r, best)]).abs() > tol {
                return false;
            }
        }
    }
    true
}

fn compute_centroid(ws: &mut NelderMeadWorkspace) {
    let n = ws.simplex.nrows();
    ws.centroid.fill(0.0);
    for &idx in &ws.order[..n] {
        for r in 0..n {
            ws.centroid[r] += ws.simplex[(r, idx)];
        }
    }
    let inv_n = 1.0 / (n as f64);
    for r in 0..n {
        ws.centroid[r] *= inv_n;
    }
}

fn affine_combo(
    out: &mut DVector<f64>,
    centroid: &DVector<f64>,
    simplex: &DMatrix<f64>,
    col: usize,
    scale: f64,
) {
    for r in 0..out.len() {
        out[r] = centroid[r] + scale * (centroid[r] - simplex[(r, col)]);
    }
}

fn shrink_simplex<F: FnMut(&DVector<f64>) -> f64>(
    ws: &mut NelderMeadWorkspace,
    sigma: f64,
    f: &mut F,
) -> Result<(), OptimizationError> {
    let best = ws.order[0];
    let n = ws.simplex.nrows();
    for pos in 1..=n {
        let idx = ws.order[pos];
        for r in 0..n {
            let v = ws.simplex[(r, best)] + sigma * (ws.simplex[(r, idx)] - ws.simplex[(r, best)]);
            ws.simplex[(r, idx)] = v;
            ws.vertex_buf[r] = v;
        }
        let val = f(&ws.vertex_buf);
        if !val.is_finite() {
            return Err(OptimizationError::NonFiniteValue);
        }
        ws.f_vals[idx] = val;
    }
    Ok(())
}

fn nelder_mead_step<F: FnMut(&DVector<f64>) -> f64>(
    cfg: &NelderMeadConfig,
    ws: &mut NelderMeadWorkspace,
    f: &mut F,
) -> Result<(), OptimizationError> {
    let n = ws.simplex.nrows();
    let best = ws.order[0];
    let second_worst = ws.order[n - 1];
    let worst = ws.order[n];

    compute_centroid(ws);
    affine_combo(
        &mut ws.reflected,
        &ws.centroid,
        &ws.simplex,
        worst,
        cfg.alpha,
    );
    let f_ref = f(&ws.reflected);
    if !f_ref.is_finite() {
        return Err(OptimizationError::NonFiniteValue);
    }

    if f_ref >= ws.f_vals[best] && f_ref < ws.f_vals[second_worst] {
        ws.simplex.set_column(worst, &ws.reflected);
        ws.f_vals[worst] = f_ref;
        return Ok(());
    }
    if f_ref < ws.f_vals[best] {
        for r in 0..n {
            ws.trial[r] = ws.centroid[r] + cfg.gamma * (ws.reflected[r] - ws.centroid[r]);
        }
        let f_exp = f(&ws.trial);
        if !f_exp.is_finite() {
            return Err(OptimizationError::NonFiniteValue);
        }
        if f_exp < f_ref {
            ws.simplex.set_column(worst, &ws.trial);
            ws.f_vals[worst] = f_exp;
        } else {
            ws.simplex.set_column(worst, &ws.reflected);
            ws.f_vals[worst] = f_ref;
        }
        return Ok(());
    }
    if f_ref < ws.f_vals[worst] {
        for r in 0..n {
            ws.trial[r] = ws.centroid[r] + cfg.rho * (ws.reflected[r] - ws.centroid[r]);
        }
        let f_oc = f(&ws.trial);
        if f_oc.is_finite() && f_oc <= f_ref {
            ws.simplex.set_column(worst, &ws.trial);
            ws.f_vals[worst] = f_oc;
            return Ok(());
        }
    } else {
        for r in 0..n {
            ws.trial[r] = ws.centroid[r] - cfg.rho * (ws.centroid[r] - ws.simplex[(r, worst)]);
        }
        let f_ic = f(&ws.trial);
        if f_ic.is_finite() && f_ic < ws.f_vals[worst] {
            ws.simplex.set_column(worst, &ws.trial);
            ws.f_vals[worst] = f_ic;
            return Ok(());
        }
    }
    shrink_simplex(ws, cfg.sigma, f)
}

impl NelderMead {
    /// Creates a new `NelderMead` solver with custom configuration.
    #[verified_engine::verified]
    pub fn new(config: NelderMeadConfig) -> Self {
        Self { config }
    }

    fn init_simplex<F: FnMut(&DVector<f64>) -> f64>(
        &self,
        x: &DVector<f64>,
        ws: &mut NelderMeadWorkspace,
        f: &mut F,
    ) -> Result<(), OptimizationError> {
        let n = x.len();
        ws.simplex.set_column(0, x);
        let f0 = f(x);
        if !f0.is_finite() {
            return Err(OptimizationError::NonFiniteValue);
        }
        ws.f_vals[0] = f0;
        ws.order[0] = 0;
        for j in 0..n {
            ws.vertex_buf.copy_from(x);
            let step = self.config.initial_step * x[j].abs().max(1.0);
            ws.vertex_buf[j] += step;
            ws.simplex.set_column(j + 1, &ws.vertex_buf);
            let fj = f(&ws.vertex_buf);
            if !fj.is_finite() {
                return Err(OptimizationError::NonFiniteValue);
            }
            ws.f_vals[j + 1] = fj;
            ws.order[j + 1] = j + 1;
        }
        sort_simplex_order(&mut ws.order, &ws.f_vals);
        Ok(())
    }

    /// Minimizes `f` in-place starting from `x` using preallocated `workspace` (zero heap allocations).
    #[verified_engine::verified]
    pub fn minimize_in_place<F>(
        &self,
        mut f: F,
        x: &mut DVector<f64>,
        workspace: &mut NelderMeadWorkspace,
    ) -> Result<(f64, usize), OptimizationError>
    where
        F: FnMut(&DVector<f64>) -> f64,
    {
        let n = x.len();
        if n == 0 {
            return Err(OptimizationError::InvalidParameter(
                "NelderMead requires dimension >= 1",
            ));
        }
        if workspace.simplex.nrows() != n || workspace.simplex.ncols() != n + 1 {
            return Err(OptimizationError::DimensionMismatch {
                expected: n,
                actual: workspace.simplex.nrows(),
            });
        }
        workspace.iterate_path.clear();
        self.init_simplex(x, workspace, &mut f)?;
        if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
            let b = workspace.order[0];
            workspace
                .iterate_path
                .push([workspace.simplex[(0, b)], workspace.simplex[(1, b)]]);
        }
        let mut iters = 0;
        for iter in 1..=self.config.max_iterations {
            if simplex_converged(workspace, self.config.tolerance) {
                break;
            }
            iters = iter;
            nelder_mead_step(&self.config, workspace, &mut f)?;
            sort_simplex_order(&mut workspace.order, &workspace.f_vals);
            let b = workspace.order[0];
            if n >= 2 && workspace.iterate_path.len() < workspace.iterate_path.capacity() {
                workspace
                    .iterate_path
                    .push([workspace.simplex[(0, b)], workspace.simplex[(1, b)]]);
            }
        }
        let best = workspace.order[0];
        for r in 0..n {
            x[r] = workspace.simplex[(r, best)];
        }
        Ok((workspace.f_vals[best], iters))
    }

    /// Allocates a simplex workspace and minimizes `f` starting from `x0`.
    #[verified_engine::verified]
    pub fn minimize<F>(
        &self,
        f: F,
        x0: &DVector<f64>,
    ) -> Result<NelderMeadSolution, OptimizationError>
    where
        F: FnMut(&DVector<f64>) -> f64,
    {
        let mut x = x0.clone();
        let mut ws = NelderMeadWorkspace::new(x.len(), self.config.max_iterations);
        let (objective, iterations) = self.minimize_in_place(f, &mut x, &mut ws)?;
        Ok(NelderMeadSolution {
            x,
            objective,
            iterations,
            iterate_path: ws.iterate_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nelder_mead_rosenbrock_2d() {
        let solver = NelderMead::new(NelderMeadConfig {
            max_iterations: 2_000,
            tolerance: 1e-8,
            ..NelderMeadConfig::default()
        });
        let x0 = DVector::from_vec(vec![-1.2, 1.0]);
        let sol = solver
            .minimize(
                |v| (1.0 - v[0]).powi(2) + 100.0 * (v[1] - v[0] * v[0]).powi(2),
                &x0,
            )
            .expect("converges");
        assert!((sol.x[0] - 1.0).abs() < 1e-4);
        assert!((sol.x[1] - 1.0).abs() < 1e-4);
        assert!(sol.objective < 1e-8);
    }
}
