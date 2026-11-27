//! Band Theory
//!
//! Simple models for electronic band structure.

/// Calculates the energy $E(k)$ for a 1D tight-binding model.
///
/// $E(k) = E_0 - 2t \cos(k a)$
#[verified_engine::verified]
pub fn tight_binding_1d(k: f64, e0: f64, t: f64, a: f64) -> f64 {
    e0 - 2.0 * t * (k * a).cos()
}

/// Calculates the energy $E(k)$ for a free electron model (empty lattice).
///
/// $E(k) = \frac{\hbar^2 k^2}{2m}$
#[verified_engine::verified]
pub fn free_electron_1d(k: f64, hbar: f64, m: f64) -> f64 {
    (hbar * k).powi(2) / (2.0 * m)
}
