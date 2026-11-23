// The Death Rate Equation
// mu(t, a) = nu(a) + C(a) * k(t) * Phi(c(t))

/// Computes the non-negative per-capita mortality rate $\mu(t, a) = \nu(a) + C(a) k(t) \Phi(c(t))$.
///
/// Non-finite inputs return `0.0` and negative rate components are clamped to `0.0` so cohort
/// survival factors $\exp(-\mu \Delta t)$ remain bounded in $(0, 1]$.
///
/// # Arguments
///
/// * `nu_a` - Natural death rate $\nu(a)$ for an individual of age $a$.
/// * `c_a` - Cannibalistic attack rate $C(a)$ on individuals of age $a$.
/// * `k_t` - Total number or biomass of cannibals $k(t)$ at time $t$.
/// * `phi_c_t` - Density-dependent functional response correction factor $\Phi(c(t))$.
///
/// # Returns
///
/// Non-negative per-capita death rate $\mu(t, a) \ge 0$.
///
/// # Example
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::cannibalism::death_rate;
///
/// assert!((death_rate(0.1, 0.2, 10.0, 1.0) - 2.1).abs() < 1e-12);
/// assert_eq!(death_rate(-0.5, -1.0, 10.0, 1.0), 0.0);
/// ```
#[verified_engine::verified]
pub fn death_rate(nu_a: f64, c_a: f64, k_t: f64, phi_c_t: f64) -> f64 {
    if !nu_a.is_finite() || !c_a.is_finite() || !k_t.is_finite() || !phi_c_t.is_finite() {
        return 0.0;
    }
    let natural = nu_a.max(0.0);
    let predation = c_a.max(0.0) * k_t.max(0.0) * phi_c_t.max(0.0);
    let total = natural + predation;
    debug_assert!(total >= 0.0);
    total
}
