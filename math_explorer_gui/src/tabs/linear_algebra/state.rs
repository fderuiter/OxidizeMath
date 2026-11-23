#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! State model, bounded matrix synthesis, solver error comparison, and `InteractiveTool`
//! implementation for [`ConditioningWorkbenchTool`].

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::nalgebra::{DMatrix, DVector};
use math_explorer::pure_math::algebra::linear_algebra::{
    cholesky_decomposition, condition_number, effective_rank, low_rank_approximation, matrix_norm,
    moore_penrose_pseudoinverse, qr_decomposition, solve_nnls, solve_normal_equation,
    solve_ridge_regression, svd_decomposition, MatrixNorm,
};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};

/// Minimum matrix row count accepted by UI controls and JSON presets.
pub const MIN_ROWS: usize = 3;
/// Maximum matrix row count accepted by UI controls and JSON presets to prevent UI stalls.
pub const MAX_ROWS: usize = 32;
/// Minimum matrix column count accepted by UI controls and JSON presets.
pub const MIN_COLS: usize = 2;
/// Maximum matrix column count accepted by UI controls and JSON presets to prevent UI stalls.
pub const MAX_COLS: usize = 16;

/// Canonical matrix family demonstrating spectral conditioning and solver stability.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixFamily {
    /// Läuchli matrix $[1 \dots 1; \mu I]$ where explicit $A^T A$ squares $\kappa(A)$ and loses rank.
    Lauchli,
    /// Polynomial Vandermonde matrix $V_{ij} = t_i^j$ with exponential singular-value decay.
    Vandermonde,
    /// Symmetric positive-definite Hilbert matrix $H_{ij} = 1 / (i + j + 1)$.
    Hilbert,
    /// Near-collinear sensor design matrix highlighting SVD truncation, Ridge, and NNLS.
    RankDeficient,
}

impl MatrixFamily {
    /// Short human-readable label for UI buttons.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Lauchli => "Läuchli",
            Self::Vandermonde => "Vandermonde",
            Self::Hilbert => "Hilbert",
            Self::RankDeficient => "Rank-Deficient",
        }
    }
}

/// Bounded serializable configuration for [`ConditioningWorkbenchTool`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConditioningConfig {
    /// Base-10 logarithm of diagonal perturbation $\mu \in [10^{-12}, 10^{-1}]$.
    pub perturbation_log10: f64,
    /// Deterministic observation noise amplitude $\sigma_{\text{noise}} \in [0.0, 1.0]$.
    pub noise_std: f64,
    /// Base-10 logarithm of Tikhonov / Ridge penalty $\lambda \in [10^{-14}, 10^{2}]$.
    pub ridge_lambda_log10: f64,
    /// Row count $m \in [3, 32]$.
    pub rows: usize,
    /// Column count $n \in [2, 16]$.
    pub cols: usize,
    /// Target SVD truncation rank $k \in [1, 16]$ for Eckart-Young reconstruction.
    pub truncation_rank: usize,
    /// Selected canonical test matrix family.
    pub family: MatrixFamily,
}

impl Default for ConditioningConfig {
    fn default() -> Self {
        Self {
            perturbation_log10: -8.0,
            noise_std: 0.0,
            ridge_lambda_log10: -6.0,
            rows: 7,
            cols: 6,
            truncation_rank: 3,
            family: MatrixFamily::Lauchli,
        }
    }
}

impl ConditioningConfig {
    /// Validates numeric finiteness and hard dimension bounds.
    pub fn validate(&self) -> Result<(), String> {
        if !self.perturbation_log10.is_finite()
            || !(-12.0..=-1.0).contains(&self.perturbation_log10)
        {
            return Err("perturbation_log10 must be finite within [-12.0, -1.0]".to_string());
        }
        if !self.noise_std.is_finite() || !(0.0..=1.0).contains(&self.noise_std) {
            return Err("noise_std must be finite within [0.0, 1.0]".to_string());
        }
        if !self.ridge_lambda_log10.is_finite() || !(-14.0..=2.0).contains(&self.ridge_lambda_log10)
        {
            return Err("ridge_lambda_log10 must be finite within [-14.0, 2.0]".to_string());
        }
        if !(MIN_ROWS..=MAX_ROWS).contains(&self.rows) {
            return Err(format!("rows must be within {MIN_ROWS}..={MAX_ROWS}"));
        }
        if !(MIN_COLS..=MAX_COLS).contains(&self.cols) {
            return Err(format!("cols must be within {MIN_COLS}..={MAX_COLS}"));
        }
        if !(1..=MAX_COLS).contains(&self.truncation_rank) {
            return Err(format!("truncation_rank must be within 1..={MAX_COLS}"));
        }
        Ok(())
    }
}

/// Accuracy and feasibility metrics for a single least-squares solver.
#[derive(Clone, Copy, Debug)]
pub struct SolverMetrics {
    /// Display name of the solver.
    pub label: &'static str,
    /// Whether the solver produced a finite estimate without breakdown.
    pub succeeded: bool,
    /// Relative parameter error $\|\hat{x} - x^\star\|_2 / \|x^\star\|_2$.
    pub forward_rel_error: f64,
    /// Relative residual norm $\|A\hat{x} - b\|_2 / \|b\|_2$.
    pub residual_rel_error: f64,
    /// Minimum coordinate $\min_j \hat{x}_j$ (verifies non-negativity for NNLS).
    pub min_entry: f64,
}

/// Cached spectral invariants, factorization residuals, and 5-solver comparison snapshot.
#[derive(Clone, Debug)]
pub struct WorkbenchSnapshot {
    /// Descending singular values $\sigma_1 \ge \dots \ge \sigma_r$.
    pub singular_values: Vec<f64>,
    /// Spectral condition number $\kappa_2(A) = \sigma_{\max} / \sigma_{\min}$.
    pub condition_number: f64,
    /// Effective numerical rank above `TOLERANCE_HIGH`.
    pub effective_rank: usize,
    /// Induced norms $(\|A\|_1, \|A\|_2, \|A\|_\infty, \|A\|_F)$.
    pub norms: [f64; 4],
    /// QR reconstruction residual $\|QR - A\|_F$.
    pub qr_recon_error: f64,
    /// Eckart-Young rank-$k$ truncation error $\|A - A_k\|_F$.
    pub low_rank_error: f64,
    /// Whether $A$ admits an exact Cholesky $LL^T$ factorization.
    pub cholesky_spd: bool,
    /// Side-by-side comparison of Normal Equations, QR, SVD, Ridge, and NNLS.
    pub solvers: [SolverMetrics; 5],
}

fn build_matrix(cfg: &ConditioningConfig) -> DMatrix<f64> {
    let (m, n) = (cfg.rows, cfg.cols);
    let mu = 10.0_f64.powf(cfg.perturbation_log10);
    match cfg.family {
        MatrixFamily::Lauchli => DMatrix::from_fn(m, n, |i, j| {
            if i == 0 {
                1.0
            } else if i - 1 == j {
                mu
            } else {
                0.0
            }
        }),
        MatrixFamily::Vandermonde => DMatrix::from_fn(m, n, |i, j| {
            let t = (i as f64) / ((m.saturating_sub(1)).max(1) as f64);
            t.powi(j as i32)
        }),
        MatrixFamily::Hilbert => DMatrix::from_fn(m, n, |i, j| 1.0 / ((i + j + 1) as f64)),
        MatrixFamily::RankDeficient => DMatrix::from_fn(m, n, |i, j| {
            let phase = ((i + 1) as f64) * 0.37;
            if j == n - 1 && n >= 2 {
                phase.sin() + mu * ((i + j + 1) as f64).cos()
            } else if j == 0 {
                phase.sin()
            } else {
                ((i + 1) as f64 * (j as f64) * 0.29).cos()
            }
        }),
    }
}

fn build_ground_truth_and_rhs(
    cfg: &ConditioningConfig,
    a: &DMatrix<f64>,
) -> (DVector<f64>, DVector<f64>) {
    let x_true = DVector::from_fn(cfg.cols, |j, _| match cfg.family {
        MatrixFamily::RankDeficient => {
            if j % 2 == 0 {
                1.5 / ((j + 1) as f64)
            } else {
                0.0
            }
        }
        _ => 1.0 + 0.5 * ((j % 3) as f64),
    });
    let noise = DVector::from_fn(cfg.rows, |i, _| {
        cfg.noise_std * (((i as f64 + 1.0) * 1.732_050_8).sin())
    });
    let b = a * &x_true + noise;
    (x_true, b)
}

fn explicit_normal_equations_reference(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let a_t = a.transpose();
    let ata = &a_t * a;
    let atb = &a_t * b;
    let sol = match ata.clone().cholesky() {
        Some(chol) => chol.solve(&atb),
        None => ata.try_inverse().map(|inv| inv * atb)?,
    };
    sol.iter().all(|v| v.is_finite()).then_some(sol)
}

fn score_solver(
    label: &'static str,
    estimate: Option<DVector<f64>>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    x_true: &DVector<f64>,
) -> SolverMetrics {
    let x_norm = x_true.norm().max(1e-15);
    let b_norm = b.norm().max(1e-15);
    let Some(x_hat) = estimate.filter(|v| v.iter().all(|e| e.is_finite())) else {
        return SolverMetrics {
            label,
            succeeded: false,
            forward_rel_error: 1.0,
            residual_rel_error: 1.0,
            min_entry: 0.0,
        };
    };
    let fwd = ((&x_hat - x_true).norm() / x_norm).clamp(1e-16, 1e6);
    let res = ((a * &x_hat - b).norm() / b_norm).clamp(1e-16, 1e6);
    let min_entry = x_hat.iter().copied().fold(f64::INFINITY, f64::min);
    SolverMetrics {
        label,
        succeeded: true,
        forward_rel_error: fwd,
        residual_rel_error: res,
        min_entry: if min_entry.is_finite() {
            min_entry
        } else {
            0.0
        },
    }
}

fn evaluate_solvers(
    cfg: &ConditioningConfig,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    x_true: &DVector<f64>,
    max_sigma: f64,
) -> [SolverMetrics; 5] {
    let pinv_tol = f64::EPSILON * (cfg.rows.max(cfg.cols) as f64) * max_sigma.max(1.0);
    let svd_est = moore_penrose_pseudoinverse(a, pinv_tol)
        .ok()
        .map(|pinv| pinv * b);
    let lambda = 10.0_f64.powf(cfg.ridge_lambda_log10);
    [
        score_solver(
            "Normal Eq (A^T A)",
            explicit_normal_equations_reference(a, b),
            a,
            b,
            x_true,
        ),
        score_solver(
            "Householder QR",
            solve_normal_equation(a, b).ok(),
            a,
            b,
            x_true,
        ),
        score_solver("Thin SVD", svd_est, a, b, x_true),
        score_solver(
            "Ridge (Aug-QR)",
            solve_ridge_regression(a, b, lambda).ok(),
            a,
            b,
            x_true,
        ),
        score_solver("NNLS (Lawson-Hanson)", solve_nnls(a, b).ok(), a, b, x_true),
    ]
}

fn evaluate_workbench(cfg: &ConditioningConfig) -> WorkbenchSnapshot {
    let a = build_matrix(cfg);
    let (x_true, b) = build_ground_truth_and_rhs(cfg, &a);
    let singular_values = svd_decomposition(&a)
        .map(|(_, s, _)| s.iter().copied().collect::<Vec<_>>())
        .unwrap_or_default();
    let max_sigma = singular_values.first().copied().unwrap_or(1.0);
    let solvers = evaluate_solvers(cfg, &a, &b, &x_true, max_sigma);
    let norms = [
        matrix_norm(&a, MatrixNorm::One).unwrap_or(0.0),
        matrix_norm(&a, MatrixNorm::Spectral).unwrap_or(0.0),
        matrix_norm(&a, MatrixNorm::Infinity).unwrap_or(0.0),
        matrix_norm(&a, MatrixNorm::Frobenius).unwrap_or(0.0),
    ];
    let k = cfg.truncation_rank.min(cfg.rows.min(cfg.cols));
    WorkbenchSnapshot {
        singular_values,
        condition_number: condition_number(&a).unwrap_or(f64::INFINITY),
        effective_rank: effective_rank(&a, None),
        norms,
        qr_recon_error: qr_decomposition(&a)
            .map(|(q, r)| (&q * &r - &a).norm())
            .unwrap_or(0.0),
        low_rank_error: low_rank_approximation(&a, k)
            .map(|ak| (&a - ak).norm())
            .unwrap_or(0.0),
        cholesky_spd: cholesky_decomposition(&a).is_ok(),
        solvers,
    }
}

/// Interactive spectral conditioning, matrix factorization, and least-squares solver workbench.
pub struct ConditioningWorkbenchTool {
    pub(crate) config: ConditioningConfig,
    pub(crate) snapshot: WorkbenchSnapshot,
    pub(crate) revision: u64,
}

impl Default for ConditioningWorkbenchTool {
    fn default() -> Self {
        let config = ConditioningConfig::default();
        let snapshot = evaluate_workbench(&config);
        Self {
            config,
            snapshot,
            revision: 1,
        }
    }
}

impl ConditioningWorkbenchTool {
    pub(crate) fn recompute(&mut self) {
        self.snapshot = evaluate_workbench(&self.config);
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Validates and applies a new [`ConditioningConfig`].
    pub fn apply_config(&mut self, cfg: ConditioningConfig) -> Result<(), String> {
        cfg.validate()?;
        self.config = cfg;
        self.recompute();
        Ok(())
    }
}

impl TheoryDescribable for ConditioningWorkbenchTool {
    fn theory_description(&self) -> String {
        "Explicit normal equations form A^T A and square the spectral condition number kappa_2(A)^2, \
         causing catastrophic cancellation whenever sigma_min(A) approaches sqrt(eps_mach) * sigma_max(A). \
         Householder QR factorization (A = Q R), thin Singular Value Decomposition (A = U Sigma V^T), \
         augmented-QR Tikhonov Ridge regression, and Lawson-Hanson active-set NNLS operate directly \
         on A without squaring condition numbers."
            .to_string()
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some("A = Q R = U \\Sigma V^T".to_string())
    }

    fn phonetic_description(&self) -> String {
        "Matrix conditioning workbench comparing explicit normal equations against Householder QR, \
         thin singular value decomposition, Tikhonov Ridge regularization, and Lawson-Hanson \
         non-negative least squares"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.); \
         Lawson, C. L., & Hanson, R. J. (1974). Solving Least Squares Problems [advanced_linear_algebra]."
            .to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("overview".to_string(), self.theory_description());
        map.insert("citation".to_string(), self.theory_citation());
        map
    }
}

impl InteractiveTool for ConditioningWorkbenchTool {
    fn name(&self) -> &'static str {
        "Conditioning & Least-Squares Workbench"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn domain(&self) -> &'static str {
        "linear_algebra"
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds {
            max_bytes: 8 * 1024,
            max_depth: 8,
            max_array_len: 64,
        }
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.validate().is_ok()
    }

    fn clamp_parameters(&mut self) -> bool {
        let c = &mut self.config;
        c.perturbation_log10 = if c.perturbation_log10.is_finite() {
            c.perturbation_log10.clamp(-12.0, -1.0)
        } else {
            -8.0
        };
        c.noise_std = if c.noise_std.is_finite() {
            c.noise_std.clamp(0.0, 1.0)
        } else {
            0.0
        };
        c.ridge_lambda_log10 = if c.ridge_lambda_log10.is_finite() {
            c.ridge_lambda_log10.clamp(-14.0, 2.0)
        } else {
            -6.0
        };
        c.rows = c.rows.clamp(MIN_ROWS, MAX_ROWS);
        c.cols = c.cols.clamp(MIN_COLS, MAX_COLS);
        c.truncation_rank = c.truncation_rank.clamp(1, MAX_COLS);
        self.recompute();
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.config).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: ConditioningConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid linear algebra state: {e}"))?;
        self.apply_config(parsed)
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_export_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "linalg_controls_panel");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_controls_panel(ui);
        });
        self.show_solver_errors_panel(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_singular_value_panel(ui);
        });
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

// [cite:advanced_linear_algebra]
