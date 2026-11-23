//! Inverse Planning Optimization and Multi-Objective IMRT Pareto Frontier Solvers.

use crate::error::PhysicsError;
use nalgebra::{DMatrix, DVector};
use pure_math::pure_math::analysis::optimization::{
    BoxBounds, LbfgsBConfig, LbfgsBSolver, LbfgsBWorkspace, ParetoWorkspace,
};

/// Calculates the quadratic objective function cost.
///
/// The cost function penalizes deviations from the prescription dose in the tumor
/// and any dose exceeding the limit in organs at risk (OAR).
#[verified_engine::verified]
pub fn calculate_cost(
    current_dose: &DVector<f64>,
    tumor_indices: &[usize],
    organ_indices: &[usize],
    prescription: f64,
    limit: f64,
    alpha: f64,
    beta: f64,
) -> f64 {
    let mut cost = 0.0;
    for &idx in tumor_indices {
        if idx < current_dose.len() {
            let diff = current_dose[idx] - prescription;
            cost += alpha * diff * diff;
        }
    }
    for &idx in organ_indices {
        if idx < current_dose.len() {
            let dose = current_dose[idx];
            if dose > limit {
                let diff = dose - limit;
                cost += beta * diff * diff;
            }
        }
    }
    cost
}

/// Performs a single Gradient Descent step to update beamlet weights, clipped to `w >= 0`.
#[verified_engine::verified]
pub fn update_weights(
    current_weights: &DVector<f64>,
    gradient: &DVector<f64>,
    eta: f64,
) -> DVector<f64> {
    let raw_update = current_weights - (gradient * eta);
    raw_update.map(|w| if w < 0.0 { 0.0 } else { w })
}

/// Clinical target and penalty parameters for box-constrained IMRT inverse planning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImrtPlanConfig {
    /// Target tumor prescription dose $D_{\text{Rx}}$ (Gy).
    pub prescription_gy: f64,
    /// Organ-at-risk maximum dose threshold $D_{\text{lim}}$ (Gy).
    pub oar_limit_gy: f64,
    /// Tumor squared-deviation penalty weight $\alpha$.
    pub alpha: f64,
    /// Organ-at-risk overdose penalty weight $\beta$.
    pub beta_oar: f64,
    /// Upper fluence weight limit $w_{\max} > 0$ (`0 <= w_b <= w_max`).
    pub max_beam_weight: f64,
}

impl Default for ImrtPlanConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            prescription_gy: 60.0,
            oar_limit_gy: 26.0,
            alpha: 1.0,
            beta_oar: 3.0,
            max_beam_weight: 10.0,
        }
    }
}

/// Single non-dominated IMRT treatment plan on the Tumor-vs-OAR Pareto trade-off frontier.
#[derive(Debug, Clone, PartialEq)]
pub struct ImrtParetoPoint {
    /// Optimal beam weight vector $w^* \in [0, w_{\max}]^B$.
    pub weights: DVector<f64>,
    /// Unweighted tumor squared deviation $\sum_{i \in \text{PTV}} (D_i - D_{\text{Rx}})^2$.
    pub tumor_quadratic_cost: f64,
    /// Unweighted OAR overdose squared penalty $\sum_{i \in \text{OAR}} \max(0, D_i - D_{\text{lim}})^2$.
    pub oar_overdose_cost: f64,
    /// Mean tumor (`PTV`) dose in Gy.
    pub tumor_mean_gy: f64,
    /// Peak organ-at-risk (`OAR`) dose in Gy.
    pub oar_max_gy: f64,
    /// Scalarization OAR penalty ratio $\beta_{\text{OAR}}$ that generated this plan.
    pub beta_oar: f64,
}

/// Preallocated buffers for zero-allocation box-constrained IMRT beam-weight optimization.
#[derive(Debug, Clone)]
pub struct ImrtSolverWorkspace {
    dose_buf: DVector<f64>,
    bounds: BoxBounds,
    lbfgs_ws: LbfgsBWorkspace,
}

impl ImrtSolverWorkspace {
    /// Preallocates voxel dose and `L-BFGS-B` buffers for `num_voxels` and `num_beams`.
    #[verified_engine::verified]
    pub fn new(
        num_voxels: usize,
        num_beams: usize,
        max_beam_weight: f64,
        max_iterations: usize,
    ) -> Result<Self, PhysicsError> {
        let bounds = BoxBounds::uniform(num_beams, 0.0, max_beam_weight.max(1e-3))?;
        Ok(Self {
            dose_buf: DVector::zeros(num_voxels),
            bounds,
            lbfgs_ws: LbfgsBWorkspace::new(num_beams, 8, max_iterations),
        })
    }

    /// Updates the upper beam weight bound in-place without reallocating.
    #[verified_engine::verified]
    pub fn set_max_beam_weight(&mut self, max_beam_weight: f64) {
        let cap = max_beam_weight.max(1e-3);
        self.bounds.upper.fill(cap);
    }
}

fn eval_imrt_cost_and_grad(
    influence: &DMatrix<f64>,
    weights: &DVector<f64>,
    tumor_indices: &[usize],
    organ_indices: &[usize],
    cfg: &ImrtPlanConfig,
    dose_buf: &mut DVector<f64>,
    grad_out: &mut DVector<f64>,
) -> (f64, f64, f64) {
    dose_buf.gemv(1.0, influence, weights, 0.0);
    grad_out.fill(0.0);
    let mut tumor_sq = 0.0;
    let mut oar_sq = 0.0;
    let n_beams = weights.len();

    for &idx in tumor_indices {
        if idx < dose_buf.len() {
            let diff = dose_buf[idx] - cfg.prescription_gy;
            tumor_sq += diff * diff;
            let coeff = 2.0 * cfg.alpha * diff;
            for b in 0..n_beams {
                grad_out[b] += coeff * influence[(idx, b)];
            }
        }
    }
    for &idx in organ_indices {
        if idx < dose_buf.len() {
            let excess = dose_buf[idx] - cfg.oar_limit_gy;
            if excess > 0.0 {
                oar_sq += excess * excess;
                let coeff = 2.0 * cfg.beta_oar * excess;
                for b in 0..n_beams {
                    grad_out[b] += coeff * influence[(idx, b)];
                }
            }
        }
    }
    (
        cfg.alpha * tumor_sq + cfg.beta_oar * oar_sq,
        tumor_sq,
        oar_sq,
    )
}

/// Solves box-constrained IMRT beam weights $w^* \in [0, w_{\max}]^B$ in-place with zero heap allocations
/// using [`LbfgsBSolver`].
#[verified_engine::verified]
pub fn solve_imrt_weights_in_place(
    influence_matrix: &DMatrix<f64>,
    tumor_indices: &[usize],
    organ_indices: &[usize],
    config: &ImrtPlanConfig,
    weights: &mut DVector<f64>,
    workspace: &mut ImrtSolverWorkspace,
) -> Result<f64, PhysicsError> {
    workspace.set_max_beam_weight(config.max_beam_weight);
    let solver = LbfgsBSolver::new(LbfgsBConfig {
        memory: 8,
        max_iterations: 80,
        tolerance: 1e-6,
        max_line_search: 20,
    });
    let dose_buf = &mut workspace.dose_buf;
    let bounds = &workspace.bounds;
    let lbfgs_ws = &mut workspace.lbfgs_ws;
    let (cost, _, _) = solver.solve_in_place(
        |w, g| {
            eval_imrt_cost_and_grad(
                influence_matrix,
                w,
                tumor_indices,
                organ_indices,
                config,
                dose_buf,
                g,
            )
            .0
        },
        bounds,
        weights,
        lbfgs_ws,
    )?;
    Ok(cost)
}

/// Allocates an [`ImrtSolverWorkspace`] and solves constrained IMRT beam weights from `initial_weights`.
#[verified_engine::verified]
pub fn solve_imrt_weights(
    influence_matrix: &DMatrix<f64>,
    tumor_indices: &[usize],
    organ_indices: &[usize],
    config: &ImrtPlanConfig,
    initial_weights: &DVector<f64>,
) -> Result<(DVector<f64>, f64), PhysicsError> {
    let mut weights = initial_weights.clone();
    let mut ws = ImrtSolverWorkspace::new(
        influence_matrix.nrows(),
        influence_matrix.ncols(),
        config.max_beam_weight,
        80,
    )?;
    let cost = solve_imrt_weights_in_place(
        influence_matrix,
        tumor_indices,
        organ_indices,
        config,
        &mut weights,
        &mut ws,
    )?;
    Ok((weights, cost))
}

/// Computes the non-dominated Tumor-vs-OAR Pareto trade-off curve by sweeping OAR penalty ratios
/// across a reusable [`ImrtSolverWorkspace`] and extracting the Pareto frontier.
#[verified_engine::verified]
pub fn compute_imrt_pareto_frontier(
    influence_matrix: &DMatrix<f64>,
    tumor_indices: &[usize],
    organ_indices: &[usize],
    base_config: &ImrtPlanConfig,
    num_points: usize,
) -> Result<Vec<ImrtParetoPoint>, PhysicsError> {
    let count = num_points.max(2);
    let n_voxels = influence_matrix.nrows();
    let n_beams = influence_matrix.ncols();
    let mut ws = ImrtSolverWorkspace::new(n_voxels, n_beams, base_config.max_beam_weight, 80)?;
    let mut w = DVector::from_element(n_beams, 1.0_f64.min(base_config.max_beam_weight));
    let mut grad_tmp = DVector::zeros(n_beams);
    let mut candidates = Vec::with_capacity(count);
    let mut obj_mat = DMatrix::zeros(count, 2);

    for k in 0..count {
        let frac = (k as f64) / ((count - 1) as f64);
        // Log-sweep beta_oar from 0.05 (tumor priority) to 60.0 (strict OAR sparing):
        let beta_oar = 0.05 * (1200.0_f64).powf(frac);
        let cfg = ImrtPlanConfig {
            beta_oar,
            ..*base_config
        };
        let _ = solve_imrt_weights_in_place(
            influence_matrix,
            tumor_indices,
            organ_indices,
            &cfg,
            &mut w,
            &mut ws,
        )?;
        let (_, tumor_sq, oar_sq) = eval_imrt_cost_and_grad(
            influence_matrix,
            &w,
            tumor_indices,
            organ_indices,
            &cfg,
            &mut ws.dose_buf,
            &mut grad_tmp,
        );
        let tumor_mean = if tumor_indices.is_empty() {
            0.0
        } else {
            tumor_indices.iter().map(|&i| ws.dose_buf[i]).sum::<f64>()
                / (tumor_indices.len() as f64)
        };
        let oar_max = organ_indices
            .iter()
            .map(|&i| ws.dose_buf[i])
            .fold(0.0_f64, f64::max);

        obj_mat[(k, 0)] = oar_sq;
        obj_mat[(k, 1)] = tumor_sq;
        candidates.push(ImrtParetoPoint {
            weights: w.clone(),
            tumor_quadratic_cost: tumor_sq,
            oar_overdose_cost: oar_sq,
            tumor_mean_gy: tumor_mean,
            oar_max_gy: oar_max,
            beta_oar,
        });
    }

    let mut pareto_ws = ParetoWorkspace::new(count);
    let front_indices = pareto_ws.extract_indices_in_place(&obj_mat, 1e-4);
    Ok(front_indices
        .iter()
        .map(|&i| candidates[i].clone())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_cost_function_logic() {
        let doses = DVector::from_vec(vec![50.0, 40.0, 10.0, 30.0]);
        let tumor_idx = vec![0, 1];
        let organ_idx = vec![2, 3];
        let cost = calculate_cost(&doses, &tumor_idx, &organ_idx, 50.0, 20.0, 1.0, 1.0);
        assert!((cost - 200.0).abs() < math_commons::registry::TOLERANCE_FAST);

        let doses_safe = DVector::from_vec(vec![50.0, 50.0, 10.0, 15.0]);
        let cost_safe = calculate_cost(&doses_safe, &tumor_idx, &organ_idx, 50.0, 20.0, 1.0, 1.0);
        assert_eq!(cost_safe, 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_gradient_update() {
        let w_old = DVector::from_vec(vec![10.0, 5.0]);
        let grad = DVector::from_vec(vec![2.0, -1.0]);
        let w_new = update_weights(&w_old, &grad, 1.0);
        assert!((w_new[0] - 8.0).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((w_new[1] - 6.0).abs() < math_commons::registry::TOLERANCE_FAST);

        let w_new_clip = update_weights(
            &DVector::from_vec(vec![1.0]),
            &DVector::from_vec(vec![2.0]),
            1.0,
        );
        assert_eq!(w_new_clip[0], 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_constrained_imrt_solver_and_pareto_frontier() {
        // Beam 0 hits both tumor (voxel 0) and OAR (voxel 1); Beam 1 hits tumor cleanly.
        // Set limit = 5 Gy so conflict exists when max_beam_weight = 4.0 (< 6.0 needed by Beam 1 alone).
        let influence = DMatrix::from_row_slice(2, 2, &[10.0, 10.0, 8.0, 0.5]);
        let tumor = [0_usize];
        let oar = [1_usize];
        let cfg = ImrtPlanConfig {
            prescription_gy: 60.0,
            oar_limit_gy: 12.0,
            alpha: 1.0,
            beta_oar: 4.0,
            max_beam_weight: 4.5,
        };
        let init = DVector::from_vec(vec![3.5, 1.0]);
        let init_dose = &influence * &init;
        let init_cost = calculate_cost(&init_dose, &tumor, &oar, 60.0, 12.0, 1.0, 4.0);
        let (opt_w, opt_cost) =
            solve_imrt_weights(&influence, &tumor, &oar, &cfg, &init).expect("imrt solve");
        assert!(opt_cost < init_cost * 0.25);
        assert!(opt_w[1] > opt_w[0]);

        let frontier =
            compute_imrt_pareto_frontier(&influence, &tumor, &oar, &cfg, 9).expect("pareto");
        assert!(frontier.len() >= 3);
        assert!(frontier[0].oar_overdose_cost <= frontier.last().unwrap().oar_overdose_cost);
    }
}
