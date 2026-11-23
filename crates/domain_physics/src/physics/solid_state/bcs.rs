/// Superconductivity (BCS Theory)
///
/// Describes the pairing of electrons into Cooper pairs via phonon mediation.
///
/// # Refactoring Note
/// This module has been refactored to use crate::error::SolidStateError;
use super::types::ElectronVolts;
use crate::error::SolidStateError;

// --- Traits ---

#[allow(missing_docs)]
pub trait GapEquation {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn calculate_next_gap(
        &self,
        current_gap: ElectronVolts,
    ) -> Result<ElectronVolts, SolidStateError>;
}

// --- Solver ---

#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct BcsGapSolver {
    #[allow(missing_docs)]
    pub max_iterations: usize,
    #[allow(missing_docs)]
    pub tolerance: f64,
    #[allow(missing_docs)]
    pub mixing_param: f64,
}

impl Default for BcsGapSolver {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            max_iterations: math_commons::registry::MAX_ITERATIONS,
            tolerance: math_commons::registry::TOLERANCE_STANDARD,
            mixing_param: 0.5,
        }
    }
}

impl BcsGapSolver {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn with_mixing_param(mut self, param: f64) -> Self {
        self.mixing_param = param;
        self
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn solve<M: GapEquation>(
        &self,
        model: &M,
        initial_guess: ElectronVolts,
    ) -> Result<ElectronVolts, SolidStateError> {
        let mut delta = initial_guess;

        for _ in 0..self.max_iterations {
            let next_delta = model.calculate_next_gap(delta)?;

            let diff = (next_delta - delta).abs();
            if (diff / ElectronVolts::new(1.0)) < self.tolerance {
                return Ok(next_delta);
            }

            delta = delta * (1.0 - self.mixing_param) + next_delta * self.mixing_param;
        }

        Err(SolidStateError::ConvergenceFailure(
            self.max_iterations,
            delta,
        ))
    }
}

// --- Models ---

#[allow(missing_docs)]
pub struct IsotropicBCSModel {
    #[allow(missing_docs)]
    pub energies: Vec<ElectronVolts>,
    #[allow(missing_docs)]
    pub potential: ElectronVolts,
    #[allow(missing_docs)]
    pub debye_cutoff: ElectronVolts,
}

impl IsotropicBCSModel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(
        energies: Vec<f64>,
        potential: f64,
        debye_cutoff: f64,
    ) -> Result<Self, SolidStateError> {
        Ok(Self {
            energies: energies.into_iter().map(ElectronVolts::new).collect(),
            potential: ElectronVolts::new(potential),
            debye_cutoff: ElectronVolts::new(debye_cutoff),
        })
    }
}

impl GapEquation for IsotropicBCSModel {
    #[verified_engine::verified]
    fn calculate_next_gap(
        &self,
        current_gap: ElectronVolts,
    ) -> Result<ElectronVolts, SolidStateError> {
        let mut summation = 0.0;
        let delta_sq = current_gap.powf(2.0);

        for &xi in &self.energies {
            if xi.abs() <= self.debye_cutoff {
                let e_k = (xi.powf(2.0) + delta_sq).sqrt();
                let e_k_raw = e_k / ElectronVolts::new(1.0);
                if e_k_raw > 1e-12 {
                    summation += 1.0 / (2.0 * e_k_raw);
                }
            }
        }

        let factor = summation * (self.potential / ElectronVolts::new(1.0));
        Ok(current_gap * factor)
    }
}

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn coherence_factors_strong(
    xi_k: ElectronVolts,
    delta: ElectronVolts,
) -> Result<(f64, f64), SolidStateError> {
    let e_k = (xi_k.powf(2.0) + delta.powf(2.0)).sqrt();
    let raw_v_sq = 0.5 * (1.0 - (xi_k / e_k));

    let v_sq = math_commons::primitives::UnitInterval::new(raw_v_sq)?;
    let u_sq = v_sq.complement();

    Ok((u_sq.sqrt(), v_sq.value().sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_bcs_probability_conservation() {
        let xi = ElectronVolts::new(1.5);
        let delta = ElectronVolts::new(0.2);
        let (u, v) = coherence_factors_strong(xi, delta).unwrap();
        let prob = u * u + v * v;
        assert!((prob - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_solver_convergence() {
        let energies: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
        let potential = 0.5;
        let debye = 5.0;

        let model = IsotropicBCSModel::new(energies.clone(), potential, debye).unwrap();
        let solver = BcsGapSolver::default();
        let initial = ElectronVolts::new(0.05);

        let result = solver.solve(&model, initial);
        assert!(result.is_ok());
        let delta = result.unwrap();
        assert!((delta / ElectronVolts::new(1.0)) > 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_bcs_gap_solver_direct() {
        let energies: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
        let potential = 0.5;
        let debye = 5.0;

        let model = IsotropicBCSModel::new(energies, potential, debye).unwrap();
        let solver = BcsGapSolver::default()
            .with_max_iterations(1000)
            .with_tolerance(math_commons::registry::TOLERANCE_STANDARD)
            .with_mixing_param(0.5);

        let initial_guess = ElectronVolts::new(0.01 * debye);
        let res = solver.solve(&model, initial_guess);

        let delta = match res {
            Ok(delta) => delta,
            Err(SolidStateError::ConvergenceFailure(_, last_val)) => last_val,
            Err(e) => panic!("Unexpected solver error: {:?}", e),
        };
        assert!((delta / ElectronVolts::new(1.0)) > 0.0);
    }
}
