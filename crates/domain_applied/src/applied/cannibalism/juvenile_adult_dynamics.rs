// Juvenile and Adult Dynamics Equations

// Juvenile Dynamics Equation:
// dn/dt + dn/da = -[I_t + C(a) * A(t)] * n(t, a)

// Adult Dynamics Equation:
// dA/dt = n(t, alpha) - f(I(t)) * A(t)

/// Evaluates the characteristic juvenile mortality rate
/// $\frac{Dn}{Dt} = -[I_t + C(a) A(t)] n(t, a)$ from van den Bosch et al.
///
/// Clamps rate and population inputs to non-negative values and returns `0.0` for non-finite inputs.
/// For full Eulerian age transport including $-\frac{\partial n}{\partial a}$, see [`juvenile_upwind_transport`].
#[verified_engine::verified]
pub fn juvenile_dynamics(i_t: f64, c_a: f64, a_t: f64, n_t_a: f64) -> f64 {
    if !i_t.is_finite() || !c_a.is_finite() || !a_t.is_finite() || !n_t_a.is_finite() {
        return 0.0;
    }
    let mu = i_t.max(0.0) + c_a.max(0.0) * a_t.max(0.0);
    let rate = -mu * n_t_a.max(0.0);
    debug_assert!(rate <= 0.0);
    rate
}

/// Evaluates the first-order upwind Eulerian juvenile PDE derivative
/// $\frac{\partial n}{\partial t}(t, a_j) \approx -\frac{n_j - n_{j-1}}{\Delta a} - [I_t + C(a_j) A(t)] n_j$.
#[verified_engine::verified]
pub fn juvenile_upwind_transport(
    n_curr: f64,
    n_prev: f64,
    da: f64,
    i_t: f64,
    c_a: f64,
    a_t: f64,
) -> f64 {
    if !da.is_finite() || da <= 0.0 {
        return 0.0;
    }
    let mu = super::death_rate(i_t, c_a, a_t, 1.0);
    super::mckendrick_upwind_transport(n_curr, n_prev, da, mu)
}

/// Evaluates the adult population rate of change $\frac{dA}{dt} = n(t, \alpha) - f(I(t)) A(t)$.
#[verified_engine::verified]
pub fn adult_dynamics(n_t_alpha: f64, f_i_t: f64, a_t: f64) -> f64 {
    if !n_t_alpha.is_finite() || !f_i_t.is_finite() || !a_t.is_finite() {
        return 0.0;
    }
    n_t_alpha.max(0.0) - f_i_t.max(0.0) * a_t.max(0.0)
}

/// Rate parameters for coupled van den Bosch juvenile-adult characteristic stepping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JuvenileAdultParams {
    /// Per-adult birth rate $\beta_0 \ge 0$ feeding $n(t, 0)$.
    pub birth_rate_per_adult: f64,
    /// Juvenile natural mortality rate $I_t \ge 0$.
    pub juvenile_mortality: f64,
    /// Adult per-capita mortality rate $f(I(t)) \ge 0$.
    pub adult_mortality: f64,
    /// Characteristic age/time step $\Delta a = \Delta t > 0$.
    pub dt: f64,
}

/// Advances a coupled juvenile age distribution `juveniles` ($a \in [0, \alpha)$) and adult pool
/// `adults` by one characteristic step $\Delta a = \Delta t$, returning the updated adult count $A(t + \Delta t)$.
///
/// Uses `scratch` as a preallocated workspace to guarantee zero heap allocation.
///
/// # Example
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::cannibalism::{step_juvenile_adult_coupled, JuvenileAdultParams};
///
/// let mut juv = [10.0, 8.0, 6.0, 4.0];
/// let mut scratch = [0.0; 4];
/// let attack = [0.5, 0.5, 0.1, 0.0];
/// let params = JuvenileAdultParams {
///     birth_rate_per_adult: 15.0,
///     juvenile_mortality: 0.2,
///     adult_mortality: 0.3,
///     dt: 0.25,
/// };
/// let next_adults = step_juvenile_adult_coupled(&mut juv, &mut scratch, &attack, 2.0, params);
/// assert!(next_adults > 2.0);
/// ```
#[verified_engine::verified]
pub fn step_juvenile_adult_coupled(
    juveniles: &mut [f64],
    scratch: &mut [f64],
    attack_kernel: &[f64],
    adults: f64,
    params: JuvenileAdultParams,
) -> f64 {
    let m = juveniles
        .len()
        .min(scratch.len())
        .min(attack_kernel.len())
        .min(super::MAX_GRID_SIZE);
    if m == 0 || !params.dt.is_finite() || params.dt <= 0.0 {
        return adults.max(0.0);
    }
    let a_pos = adults.max(0.0);
    let maturing_density = juveniles[m - 1].max(0.0);
    scratch[0] = params.birth_rate_per_adult.max(0.0) * a_pos;
    for j in 1..m {
        let mu = params.juvenile_mortality.max(0.0) + attack_kernel[j - 1].max(0.0) * a_pos;
        scratch[j] = super::characteristic_survival_step(juveniles[j - 1], mu, params.dt);
    }
    juveniles[..m].copy_from_slice(&scratch[..m]);

    let adult_survival = (-params.adult_mortality.max(0.0) * params.dt).exp();
    let next_adults = a_pos * adult_survival + maturing_density * params.dt;
    debug_assert!(next_adults >= 0.0);
    next_adults
}
