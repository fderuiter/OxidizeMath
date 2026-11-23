//! Superconductivity (BCS Theory).
//!
//! Models phonon-mediated Cooper pairing of electrons near the Fermi surface via
//! self-consistent finite-temperature gap equations, Bogoliubov quasiparticle coherence
//! factors, and Dynes-broadened NIS tunneling spectroscopy.
//!
//! Errors during iterative gap solving are reported via [`SolidStateError`].

use super::types::{ElectronVolts, Kelvin};
use crate::error::SolidStateError;

#[path = "bcs_tunneling.rs"]
mod tunneling;
pub use tunneling::{dynes_dos_ratio, nis_tunneling_conductance};

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

/// Finite-temperature isotropic $s$-wave BCS pairing model over discrete single-particle levels.
///
/// Sums pairing contributions across states within $|\xi_k| \le \hbar\omega_D$ (`debye_cutoff`)
/// weighted by uniform attractive pairing interaction `potential` $V$ and thermal factor
/// $\tanh(E_k / (2 k_B T))$. Defaults to `temperature = 0.0 K`.
#[derive(Debug, Clone)]
pub struct IsotropicBCSModel {
    /// Single-particle kinetic energies $\xi_k = \epsilon_k - \mu$ relative to the Fermi level.
    pub energies: Vec<ElectronVolts>,
    /// Attractive Cooper-pairing interaction strength $V$ (`eV`).
    pub potential: ElectronVolts,
    /// Debye phonon cutoff energy $\hbar\omega_D$ (`eV`) bounding active pairing states.
    pub debye_cutoff: ElectronVolts,
    /// Bath temperature $T$ (`K`); defaults to `0.0 K` when constructed via [`Self::new`].
    pub temperature: Kelvin,
}

#[inline]
fn thermal_pair_weight(e_k_ev: f64, two_kt_ev: f64) -> f64 {
    if two_kt_ev <= 1e-15 {
        if e_k_ev > 1e-12 {
            1.0 / (2.0 * e_k_ev)
        } else {
            1.0e12
        }
    } else {
        let x = e_k_ev / two_kt_ev;
        let sinc_tanh = if x < 1e-6 {
            1.0 - (x * x) / 3.0
        } else {
            x.tanh() / x
        };
        sinc_tanh / (2.0 * two_kt_ev)
    }
}

impl IsotropicBCSModel {
    /// Constructs an [`IsotropicBCSModel`] at `0.0 K` from raw `eV` energy levels, `potential`, and `debye_cutoff`.
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
            temperature: Kelvin::new(0.0),
        })
    }

    /// Sets the bath temperature $T$ (`K`) used in finite-temperature $\tanh(E_k / (2 k_B T))$ weighting.
    #[verified_engine::verified]
    pub fn with_temperature(mut self, temperature: Kelvin) -> Self {
        self.temperature = temperature;
        self
    }

    #[inline]
    fn evaluate_kernel_at(&self, delta_ev: f64, temperature: Kelvin) -> f64 {
        let v_ev = self.potential.as_f64();
        let cutoff = self.debye_cutoff.as_f64();
        if v_ev <= 0.0 || cutoff <= 0.0 {
            return 0.0;
        }
        let two_kt_ev = 2.0 * temperature.thermal_energy_ev().as_f64();
        let mut sum = 0.0;
        for &xi in &self.energies {
            let xi_ev = xi.as_f64().abs();
            if xi_ev <= cutoff {
                let e_k = xi_ev.hypot(delta_ev);
                sum += thermal_pair_weight(e_k, two_kt_ev);
            }
        }
        v_ev * sum
    }

    #[inline]
    fn zero_seed_condensation_step(&self, k0: f64) -> f64 {
        let cutoff = self.debye_cutoff.as_f64().max(1e-6);
        let probe =
            (0.02 * cutoff * (k0 - 1.0).sqrt().min(1.0)).clamp(1e-6 * cutoff, 0.25 * cutoff);
        let k_probe = self.evaluate_kernel_at(probe, self.temperature);
        probe * k_probe
    }

    /// Evaluates the dimensionless BCS pairing kernel $K(\Delta, T) = V \sum_{|\xi_k| \le \hbar\omega_D} \frac{\tanh(E_k / (2 k_B T))}{2 E_k}$.
    ///
    /// # Errors
    ///
    /// Returns [`SolidStateError::InvalidParameter`] if `temperature < 0` or inputs are non-finite.
    #[verified_engine::verified]
    pub fn pairing_kernel(&self, gap: ElectronVolts) -> Result<f64, SolidStateError> {
        let t_k = self.temperature.as_f64();
        let d_ev = gap.as_f64();
        if !t_k.is_finite() || t_k < 0.0 || !d_ev.is_finite() {
            return Err(SolidStateError::InvalidParameter(
                "Temperature must be non-negative and gap must be finite".to_string(),
            ));
        }
        Ok(self.evaluate_kernel_at(d_ev.abs(), self.temperature))
    }

    /// Solves the linearized Thouless criterion $K(0, T_c) = 1$ for the critical temperature $T_c$ (`K`).
    ///
    /// Returns `0.0 K` if pairing is repulsive (`potential <= 0`) or sub-critical even as $T \to 0^+$.
    ///
    /// # Errors
    ///
    /// Returns [`SolidStateError::InvalidParameter`] if model energies or parameters are non-finite.
    #[verified_engine::verified]
    pub fn critical_temperature(&self) -> Result<Kelvin, SolidStateError> {
        let v_ev = self.potential.as_f64();
        let cutoff = self.debye_cutoff.as_f64();
        if !v_ev.is_finite() || !cutoff.is_finite() {
            return Err(SolidStateError::InvalidParameter(
                "BCS potential and Debye cutoff must be finite".to_string(),
            ));
        }
        if v_ev <= 0.0 || cutoff <= 0.0 || self.energies.is_empty() {
            return Ok(Kelvin::new(0.0));
        }
        let mut lo = 0.0_f64;
        if self.evaluate_kernel_at(0.0, Kelvin::new(1e-9)) <= 1.0 {
            return Ok(Kelvin::new(0.0));
        }
        let mut hi = Kelvin::from_thermal_energy_ev(ElectronVolts::new(cutoff))
            .as_f64()
            .max(1.0);
        for _ in 0..64 {
            if self.evaluate_kernel_at(0.0, Kelvin::new(hi)) <= 1.0 {
                break;
            }
            lo = hi;
            hi *= 2.0;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if self.evaluate_kernel_at(0.0, Kelvin::new(mid)) > 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) <= 1e-12 * (1.0 + hi) {
                break;
            }
        }
        Ok(Kelvin::new(0.5 * (lo + hi)))
    }
}

impl GapEquation for IsotropicBCSModel {
    #[verified_engine::verified]
    fn calculate_next_gap(
        &self,
        current_gap: ElectronVolts,
    ) -> Result<ElectronVolts, SolidStateError> {
        let t_k = self.temperature.as_f64();
        let gap_raw = current_gap.as_f64();
        if !t_k.is_finite() || t_k < 0.0 || !gap_raw.is_finite() {
            return Err(SolidStateError::InvalidParameter(
                "Temperature must be finite and non-negative".to_string(),
            ));
        }

        let k0 = self.evaluate_kernel_at(0.0, self.temperature);
        if k0 <= 1.0 {
            return Ok(ElectronVolts::new(0.0));
        }

        let gap_abs = gap_raw.abs();
        if gap_abs <= 1e-15 {
            return Ok(ElectronVolts::new(self.zero_seed_condensation_step(k0)));
        }

        let factor = self.evaluate_kernel_at(gap_abs, self.temperature);
        Ok(ElectronVolts::new(gap_abs * factor))
    }
}

/// Evaluates Bogoliubov quasiparticle excitation energy $E_k = \sqrt{\xi_k^2 + \Delta^2}$.
#[verified_engine::verified]
pub fn quasiparticle_energy(xi_k: ElectronVolts, delta: ElectronVolts) -> ElectronVolts {
    ElectronVolts::new(xi_k.as_f64().hypot(delta.as_f64()))
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
) -> Result<(f64, f64), SolidStateError> {
    let e_k = (xi_k.powf(2.0) + delta.powf(2.0)).sqrt();
    let raw_v_sq = 0.5 * (1.0 - (xi_k / e_k));

    let v_sq = math_commons::primitives::UnitInterval::new(raw_v_sq)?;
    let u_sq = v_sq.complement();

    Ok((u_sq.sqrt(), v_sq.value().sqrt()))
}

/// Evaluates Bogoliubov coherence factors `(u_k, v_k)` with continuous normal-state limits at $\Delta = 0$.
///
/// For $\Delta \to 0^+$, returns `(0.0, 1.0)` below the Fermi level ($\xi_k < 0$), `(1.0, 0.0)` above
/// the Fermi level ($\xi_k > 0$), and `(1/√2, 1/√2)` ($u_k^2 = v_k^2 = 0.5$) right at $\xi_k = 0$.
///
/// # Errors
///
/// Returns [`SolidStateError::InvalidParameter`] if `xi_k` or `delta` is non-finite.
#[verified_engine::verified]
pub fn coherence_factors_normal_limit(
    xi_k: ElectronVolts,
    delta: ElectronVolts,
) -> Result<(f64, f64), SolidStateError> {
    let xi = xi_k.as_f64();
    let d = delta.as_f64().abs();
    if !xi.is_finite() || !d.is_finite() {
        return Err(SolidStateError::InvalidParameter(
            "Coherence factor energies must be finite".to_string(),
        ));
    }
    let e_k = xi.hypot(d);
    if e_k <= 1e-15 {
        let half_sqrt = std::f64::consts::FRAC_1_SQRT_2;
        return Ok((half_sqrt, half_sqrt));
    }
    if d <= 1e-15 {
        return if xi < 0.0 {
            Ok((0.0, 1.0))
        } else {
            Ok((1.0, 0.0))
        };
    }
    let raw_v_sq = (0.5 * (1.0 - xi / e_k)).clamp(0.0, 1.0);
    let u_sq = 1.0 - raw_v_sq;
    Ok((u_sq.sqrt(), raw_v_sq.sqrt()))
}

// [cite:mmwave_radiotherapy_setup]

#[cfg(test)]
#[path = "bcs_tests.rs"]
mod tests;
