//! Physics strategies for Mean Field Games.
//!
//! This module defines the `Hamiltonian` trait, allowing the solver to be
//! extended with different kinetic energy models (e.g., standard quadratic,
//! relativistic, or non-linear control costs).

/// Represents the Hamiltonian $H(p)$ and its derivative.
///
/// In Mean Field Games, the Hamiltonian corresponds to the Legendre transform
/// of the Lagrangian (running cost of control).
///
/// - **HJB Equation**: $\partial_t u = H(\nabla u) - \nu \Delta u - F$
/// - **Fokker-Planck Equation**: $\partial_t m + \nabla \cdot (m v) - \nu \Delta m = 0$
///   where $v = -\partial_p H(p)$.
pub trait Hamiltonian {
    /// Evaluates the Hamiltonian $H(p)$ at momentum $p$.
    #[verified_engine::verified]
    fn evaluate(&self, p: f64) -> f64;

    /// Evaluates the derivative $\partial_p H(p)$.
    ///
    /// This determines the advection velocity (drift) in the Fokker-Planck equation:
    /// $v = - \partial_p H(p)$.
    #[verified_engine::verified]
    fn derivative(&self, p: f64) -> f64;
}

/// Standard Quadratic Hamiltonian: $H(p) = \frac{p^2}{2m}$.
///
/// Corresponds to $L(\alpha) = \frac{m \alpha^2}{2}$.
///
/// Note: The struct field is named `mass` to represent the mass parameter $m$,
/// not to be confused with the population density $m(x)$.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuadraticHamiltonian {
    #[allow(missing_docs)]
    pub mass: f64,
}

impl Default for QuadraticHamiltonian {
    #[verified_engine::verified]
    fn default() -> Self {
        Self { mass: 1.0 }
    }
}

impl QuadraticHamiltonian {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(mass: f64) -> Self {
        Self { mass }
    }
}

impl Hamiltonian for QuadraticHamiltonian {
    #[verified_engine::verified]
    fn evaluate(&self, p: f64) -> f64 {
        (p * p) / (2.0 * self.mass)
    }

    #[verified_engine::verified]
    fn derivative(&self, p: f64) -> f64 {
        p / self.mass
    }
}
