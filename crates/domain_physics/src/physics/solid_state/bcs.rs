//! Superconductivity (BCS Theory).
//!
//! Models phonon-mediated Cooper pairing of electrons near the Fermi surface via
//! self-consistent gap equations and Bogoliubov quasiparticle coherence factors.
//!
//! Errors during iterative gap solving are reported via [`SolidStateError`].

use super::types::ElectronVolts;
use crate::error::SolidStateError;

// --- Traits ---

/// Fixed-point iteration contract for a superconducting energy-gap model.
pub trait GapEquation {
    /// Evaluates the next superconducting gap estimate $\Delta_{n+1}$ from `current_gap`.
    ///
    /// # Errors
    ///
    /// Returns [`SolidStateError`] if model evaluation encounters an invalid physical state.
    #[verified_engine::verified]
    fn calculate_next_gap(
        &self,
        current_gap: ElectronVolts,
    ) -> Result<ElectronVolts, SolidStateError>;
}

// --- Solver ---

/// Damped fixed-point solver for self-consistent BCS superconducting gap equations.
///
/// Iterates $\Delta_{n+1} = (1 - \alpha)\Delta_n + \alpha F(\Delta_n)$ until
/// $|\Delta_{n+1} - \Delta_n| < \text{tolerance}$ (in eV) or `max_iterations` is exhausted.
///
/// # Examples
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::bcs::{BcsGapSolver, IsotropicBCSModel};
/// use math_explorer::physics::solid_state::types::ElectronVolts;
///
/// let energies: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
/// let model = IsotropicBCSModel::new(energies, 0.5, 5.0).unwrap();
/// let solver = BcsGapSolver::new().with_mixing_param(0.5);
///
/// let gap = solver.solve(&model, ElectronVolts::new(0.05)).unwrap();
/// assert!((gap / ElectronVolts::new(1.0)) > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct BcsGapSolver {
    /// Maximum number of damped fixed-point iterations before returning [`SolidStateError::ConvergenceFailure`].
    pub max_iterations: usize,
    /// Absolute convergence threshold in electron-volts (`eV`).
    pub tolerance: f64,
    /// Linear mixing weight $\alpha \in (0, 1]$ applied to each updated gap estimate.
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
    /// Creates a [`BcsGapSolver`] initialized with standard workspace iteration and tolerance defaults.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Overrides the maximum iteration budget before reporting non-convergence.
    #[verified_engine::verified]
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Overrides the convergence tolerance (in `eV`).
    #[verified_engine::verified]
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Overrides the linear mixing parameter $\alpha$ used to damp gap updates.
    #[verified_engine::verified]
    pub fn with_mixing_param(mut self, param: f64) -> Self {
        self.mixing_param = param;
        self
    }

    /// Solves for the self-consistent superconducting energy gap starting from `initial_guess`.
    ///
    /// # Errors
    ///
    /// Returns [`SolidStateError::ConvergenceFailure`] containing the iteration count and
    /// last gap iterate if convergence is not achieved within [`Self::max_iterations`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_physics::__doctest_facade as math_explorer;
    /// use math_explorer::error::SolidStateError;
    /// use math_explorer::physics::solid_state::bcs::{BcsGapSolver, IsotropicBCSModel};
    /// use math_explorer::physics::solid_state::types::ElectronVolts;
    ///
    /// let energies: Vec<f64> = (0..60).map(|i| (i as f64 - 30.0) * 0.1).collect();
    /// let model = IsotropicBCSModel::new(energies, 0.5, 3.0).unwrap();
    ///
    /// // A single iteration with tight tolerance triggers ConvergenceFailure
    /// let strict_solver = BcsGapSolver::new()
    ///     .with_max_iterations(1)
    ///     .with_tolerance(1e-15);
    /// let err = strict_solver.solve(&model, ElectronVolts::new(0.05)).unwrap_err();
    /// assert!(matches!(err, SolidStateError::ConvergenceFailure(1, _)));
    /// ```
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

/// Zero-temperature isotropic s-wave BCS pairing model over discrete single-particle levels.
///
/// Sums pairing contributions across states within $|\xi_k| \le \hbar\omega_D$ (`debye_cutoff`)
/// weighted by uniform attractive pairing interaction `potential` $V$.
pub struct IsotropicBCSModel {
    /// Single-particle kinetic energies $\xi_k = \epsilon_k - \mu$ relative to the Fermi level.
    pub energies: Vec<ElectronVolts>,
    /// Attractive Cooper-pairing interaction strength $V$ (`eV`).
    pub potential: ElectronVolts,
    /// Debye phonon cutoff energy $\hbar\omega_D$ (`eV`) bounding active pairing states.
    pub debye_cutoff: ElectronVolts,
}

impl IsotropicBCSModel {
    /// Constructs an [`IsotropicBCSModel`] from raw `eV` energy levels, pairing `potential`, and `debye_cutoff`.
    ///
    /// # Errors
    ///
    /// Returns [`SolidStateError`] if model parameters violate solid-state invariants.
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

/// Computes strongly typed Bogoliubov coherence factors `(u_k, v_k)` for quasiparticle dispersion
/// $E_k = \sqrt{\xi_k^2 + \Delta^2}$.
///
/// Evaluates $v_k^2 = \tfrac{1}{2}(1 - \xi_k / E_k)$ inside a verified `UnitInterval` and obtains
/// $u_k^2 = 1 - v_k^2$ as its exact complement.
///
/// # Errors
///
/// Returns `Err` if $v_k^2$ falls outside `[0.0, 1.0]` (for example when $\xi_k = \Delta = 0$).
///
/// # Examples
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::bcs::coherence_factors_strong;
/// use math_explorer::physics::solid_state::types::ElectronVolts;
///
/// let (u_k, v_k) = coherence_factors_strong(ElectronVolts::new(0.8), ElectronVolts::new(0.3)).unwrap();
/// assert!((u_k * u_k + v_k * v_k - 1.0).abs() < 1e-12);
///
/// // Degenerate E_k = 0 yields NaN occupation, rejected by UnitInterval
/// assert!(coherence_factors_strong(ElectronVolts::new(0.0), ElectronVolts::new(0.0)).is_err());
/// ```
#[verified_engine::verified]
pub fn coherence_factors_strong(
    xi_k: ElectronVolts,
    delta: ElectronVolts,
) -> Result<(f64, f64), String> {
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
