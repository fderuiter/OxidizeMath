// The McKendrick-von Foerster Equation
// dn/dt + dn/da = -mu(t, a) * n(t, a)
// n(t, 0) = b(t)

/// Evaluates the Lagrangian characteristic rate of change $\frac{Dn}{Dt} = \frac{\partial n}{\partial t} + \frac{\partial n}{\partial a} = -\mu(t, a) n(t, a)$.
///
/// Returns `0.0` when `t` or `a` is non-finite or `a < 0.0`, and clamps `mu` and `n` to non-negative values.
/// For Eulerian grid transport including $-\frac{\partial n}{\partial a}$, see [`mckendrick_upwind_transport`].
///
/// # Arguments
///
/// * `t` - Simulation time $t \ge 0$.
/// * `a` - Cohort age $a \ge 0$.
/// * `mu` - Per-capita mortality rate $\mu(t, a) \ge 0$.
/// * `n` - Cohort age density $n(t, a) \ge 0$.
///
/// # Returns
///
/// Characteristic mortality rate $-\mu(t, a) n(t, a) \le 0$.
#[verified_engine::verified]
pub fn mckendrick_von_foerster(t: f64, a: f64, mu: f64, n: f64) -> f64 {
    if !t.is_finite() || !a.is_finite() || a < 0.0 || !mu.is_finite() || !n.is_finite() {
        return 0.0;
    }
    let decay = -mu.max(0.0) * n.max(0.0);
    debug_assert!(decay <= 0.0);
    decay
}

/// Evaluates the first-order upwind Eulerian PDE right-hand side
/// $\frac{\partial n}{\partial t}(t, a_j) \approx -\frac{n_j - n_{j-1}}{\Delta a} - \mu_j n_j$.
///
/// # Arguments
///
/// * `n_curr` - Density $n(t, a_j)$ in the current age bin.
/// * `n_prev` - Upstream density $n(t, a_{j-1})$ in the younger age bin (or $b(t)$ at $j = 0$).
/// * `da` - Age bin width $\Delta a > 0$.
/// * `mu` - Per-capita mortality rate $\mu(t, a_j) \ge 0$.
///
/// # Example
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::cannibalism::mckendrick_upwind_transport;
///
/// let rhs = mckendrick_upwind_transport(8.0, 10.0, 0.5, 0.25);
/// assert!((rhs - 2.0).abs() < 1e-12);
/// ```
#[verified_engine::verified]
pub fn mckendrick_upwind_transport(n_curr: f64, n_prev: f64, da: f64, mu: f64) -> f64 {
    if !n_curr.is_finite() || !n_prev.is_finite() || !da.is_finite() || da <= 0.0 || !mu.is_finite()
    {
        return 0.0;
    }
    let curr = n_curr.max(0.0);
    let prev = n_prev.max(0.0);
    let advection = -(curr - prev) / da;
    advection - mu.max(0.0) * curr
}

/// Advances a single cohort bin along the characteristic line $\Delta a = \Delta t$ using
/// unconditional positivity-preserving exponential survival $n_{j-1} \exp(-\mu \Delta t)$.
#[verified_engine::verified]
pub fn characteristic_survival_step(n_prev: f64, mu: f64, dt: f64) -> f64 {
    if !n_prev.is_finite() || !mu.is_finite() || !dt.is_finite() || dt < 0.0 {
        return 0.0;
    }
    let survived = n_prev.max(0.0) * (-mu.max(0.0) * dt).exp();
    debug_assert!(survived >= 0.0);
    survived
}

/// Evaluates a smooth, non-negative oscillatory boundary recruitment profile
/// $b(t) = 100 \left[1 + \frac{1}{4}(1 - \cos t)\right]$ satisfying $b(0) = 100$ and $b(t) \in [100, 150]$.
///
/// For state-dependent McKendrick renewal $b(t) = \int_0^{a_{\max}} \beta(a) n(t, a)\,da$,
/// use [`renewal_birth_rate`].
#[verified_engine::verified]
pub fn birth_rate(t: f64) -> f64 {
    if !t.is_finite() {
        return 0.0;
    }
    let recruitment = 100.0 * (1.0 + 0.25 * (1.0 - t.cos()));
    debug_assert!(recruitment >= 0.0);
    recruitment
}

/// Integrates the boundary renewal condition $b(t) = \sum_{j=0}^{M-1} \beta(a_j)\,n(t, a_j)\,\Delta a$
/// over preallocated age slices without heap allocation.
#[verified_engine::verified]
pub fn renewal_birth_rate(fertility_kernel: &[f64], density: &[f64], da: f64) -> f64 {
    if !da.is_finite() || da <= 0.0 {
        return 0.0;
    }
    let len = fertility_kernel
        .len()
        .min(density.len())
        .min(super::MAX_GRID_SIZE);
    let mut sum = 0.0;
    for j in 0..len {
        let beta = fertility_kernel[j];
        let n = density[j];
        if beta.is_finite() && n.is_finite() {
            sum += beta.max(0.0) * n.max(0.0);
        }
    }
    let births = sum * da;
    debug_assert!(births >= 0.0);
    births
}
