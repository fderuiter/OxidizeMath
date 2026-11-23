//! Zero-allocation upwind solver and operators for the McKendrick-von Foerster age-structured PDE.

use super::death_rate::{
    CannibalismKernel, advect_characteristic_step, sanitize_density, trapezoidal_quadrature,
};

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
/// use [`renewal_birth_rate`] or [`McKendrickSolver::compute_birth_boundary`].
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

/// Pre-allocated upwind characteristic solver for $\partial_t n + \partial_a n = -\mu(t, a, N) n(t, a)$
/// with integral renewal boundary $n(t, 0) = b(t) = \int_0^{a_{\max}} \beta(a, N(t)) n(t, a)\,da$.
#[derive(Debug, Clone)]
pub struct McKendrickSolver {
    ages: Vec<f64>,
    density: Vec<f64>,
    next_density: Vec<f64>,
    mortality: Vec<f64>,
    da: f64,
    time: f64,
    last_birth_rate: f64,
    last_cannibal_load: f64,
    kernel: CannibalismKernel,
}

impl McKendrickSolver {
    /// Allocates age grid and working buffers once so `step_upwind` performs zero heap allocations.
    pub fn new(num_ages: usize, max_age: f64, kernel: CannibalismKernel) -> Self {
        let count = num_ages.clamp(2, 4096);
        let safe_max = if max_age.is_finite() && max_age > 0.0 {
            max_age
        } else {
            10.0
        };
        let da = safe_max / (count - 1) as f64;
        let ages: Vec<f64> = (0..count).map(|i| i as f64 * da).collect();
        Self {
            ages,
            density: vec![0.0; count],
            next_density: vec![0.0; count],
            mortality: vec![0.0; count],
            da,
            time: 0.0,
            last_birth_rate: 0.0,
            last_cannibal_load: 0.0,
            kernel,
        }
    }

    /// Initializes the age-density distribution $n(0, a)$ from a closure and resets simulation time.
    pub fn set_initial_density<F: FnMut(f64) -> f64>(&mut self, mut profile: F) {
        for (slot, &age) in self.density.iter_mut().zip(self.ages.iter()) {
            *slot = sanitize_density(profile(age));
        }
        self.next_density.fill(0.0);
        self.time = 0.0;
        self.refresh_diagnostics();
    }

    /// Copies a slice into the pre-allocated age-density buffer without reallocating.
    #[verified_engine::verified]
    pub fn set_density_slice(&mut self, values: &[f64]) {
        let n = self.density.len();
        let copy_len = n.min(values.len());
        let mut idx = 0;
        while idx < copy_len {
            self.density[idx] = sanitize_density(values[idx]);
            idx += 1;
        }
        while idx < n {
            self.density[idx] = 0.0;
            idx += 1;
        }
        self.time = 0.0;
        self.refresh_diagnostics();
    }

    /// Returns the discretized age grid $[0, a_{\max}]$.
    pub fn ages(&self) -> &[f64] {
        &self.ages
    }

    /// Returns the current age-density distribution $n(t, a)$.
    pub fn density(&self) -> &[f64] {
        &self.density
    }

    /// Returns the most recently evaluated age-specific mortality profile $\mu(t, a)$.
    pub fn mortality(&self) -> &[f64] {
        &self.mortality
    }

    /// Returns the uniform age step $\Delta a$.
    #[verified_engine::verified]
    pub fn da(&self) -> f64 {
        self.da
    }

    /// Returns the current simulation time $t$.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns the most recent boundary birth rate $b(t) = n(t, 0)$.
    #[verified_engine::verified]
    pub fn last_birth_rate(&self) -> f64 {
        self.last_birth_rate
    }

    /// Returns the most recent adult cannibal load $k(t)$.
    #[verified_engine::verified]
    pub fn last_cannibal_load(&self) -> f64 {
        self.last_cannibal_load
    }

    /// Returns the active [`CannibalismKernel`].
    pub fn kernel(&self) -> &CannibalismKernel {
        &self.kernel
    }

    /// Updates the [`CannibalismKernel`] parameters in place without reallocating buffers.
    #[verified_engine::verified]
    pub fn set_kernel(&mut self, kernel: CannibalismKernel) {
        self.kernel = kernel;
        self.refresh_diagnostics();
    }

    /// Computes total population $N(t) = \int_0^{a_{\max}} n(t, a)\,da$ via trapezoidal quadrature.
    #[verified_engine::verified]
    pub fn total_population(&self) -> f64 {
        trapezoidal_quadrature(&self.density, self.da)
    }

    /// Computes juvenile cohort size $J(t) = \int_0^\alpha n(t, a)\,da$ via trapezoidal quadrature.
    #[verified_engine::verified]
    pub fn juvenile_population(&self) -> f64 {
        let n = self.ages.len();
        if n < 2 {
            return 0.0;
        }
        let alpha = sanitize_density(self.kernel.maturation_age);
        let mut sum = 0.0;
        let mut idx = 0;
        while idx + 1 < n {
            let w0 = if self.ages[idx] <= alpha {
                sanitize_density(self.density[idx])
            } else {
                0.0
            };
            let w1 = if self.ages[idx + 1] <= alpha {
                sanitize_density(self.density[idx + 1])
            } else {
                0.0
            };
            sum += 0.5 * (w0 + w1) * self.da;
            idx += 1;
        }
        sanitize_density(sum)
    }

    /// Computes adult cannibal load $k(t) = \int_\alpha^{a_{\max}} n(t, a)\,da$ via trapezoidal quadrature.
    #[verified_engine::verified]
    pub fn cannibal_load(&self) -> f64 {
        let n = self.ages.len();
        if n < 2 {
            return 0.0;
        }
        let alpha = sanitize_density(self.kernel.maturation_age);
        let mut sum = 0.0;
        let mut idx = 0;
        while idx + 1 < n {
            let w0 = if self.ages[idx] >= alpha {
                sanitize_density(self.density[idx])
            } else {
                0.0
            };
            let w1 = if self.ages[idx + 1] >= alpha {
                sanitize_density(self.density[idx + 1])
            } else {
                0.0
            };
            sum += 0.5 * (w0 + w1) * self.da;
            idx += 1;
        }
        sanitize_density(sum)
    }

    /// Computes vulnerable prey load $c(t) = \int_0^{a_{\max}} C(a) n(t, a)\,da$ via trapezoidal quadrature.
    #[verified_engine::verified]
    pub fn prey_load(&self) -> f64 {
        let n = self.ages.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut idx = 0;
        while idx + 1 < n {
            let f0 =
                self.kernel.vulnerability_at(self.ages[idx]) * sanitize_density(self.density[idx]);
            let f1 = self.kernel.vulnerability_at(self.ages[idx + 1])
                * sanitize_density(self.density[idx + 1]);
            sum += 0.5 * (f0 + f1) * self.da;
            idx += 1;
        }
        sanitize_density(sum)
    }

    /// Integrates renewal births $b(t) = \int_0^{a_{\max}} \beta(a, N(t)) n(t, a)\,da$ via trapezoidal quadrature.
    #[verified_engine::verified]
    pub fn compute_birth_boundary(&self, total_pop: f64) -> f64 {
        let n = self.ages.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut idx = 0;
        while idx + 1 < n {
            let b0 = self.kernel.fertility_at(self.ages[idx], total_pop)
                * sanitize_density(self.density[idx]);
            let b1 = self.kernel.fertility_at(self.ages[idx + 1], total_pop)
                * sanitize_density(self.density[idx + 1]);
            sum += 0.5 * (b0 + b1) * self.da;
            idx += 1;
        }
        sanitize_density(sum)
    }

    #[verified_engine::verified]
    fn refresh_diagnostics(&mut self) {
        let total_pop = self.total_population();
        let k_t = self.cannibal_load();
        let c_t = self.prey_load();
        let n = self.ages.len();
        let mut idx = 0;
        while idx < n {
            self.mortality[idx] = self.kernel.mortality_at(self.ages[idx], k_t, c_t);
            idx += 1;
        }
        self.last_cannibal_load = k_t;
        self.last_birth_rate = self.compute_birth_boundary(total_pop);
    }

    #[verified_engine::verified]
    fn step_cfl_substep(&mut self, sub_dt: f64, fixed_birth: Option<f64>) {
        self.refresh_diagnostics();
        let b_t = match fixed_birth {
            Some(val) => sanitize_density(val),
            None => self.last_birth_rate,
        };
        advect_characteristic_step(
            &self.density,
            &self.mortality,
            b_t,
            sub_dt,
            self.da,
            &mut self.next_density,
        );
        core::mem::swap(&mut self.density, &mut self.next_density);
        self.last_birth_rate = b_t;
        self.time += sub_dt;
    }

    /// Advances the McKendrick-von Foerster PDE by `dt` with zero heap allocation,
    /// subdividing into CFL-safe substeps ($\Delta t_{\text{sub}} \le \Delta a$).
    #[verified_engine::verified]
    pub fn step_upwind(&mut self, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 || self.da <= 0.0 {
            return;
        }
        let substeps = ((dt / self.da).ceil() as usize).max(1);
        let sub_dt = dt / substeps as f64;
        let mut step_idx = 0;
        while step_idx < substeps {
            self.step_cfl_substep(sub_dt, None);
            step_idx += 1;
        }
    }

    /// Advances the PDE by `dt` with a prescribed boundary condition $n(t, 0) = b_0$
    /// for characteristic transport and mass-balance verification.
    #[verified_engine::verified]
    pub fn step_upwind_with_boundary(&mut self, dt: f64, boundary_birth: f64) {
        if !dt.is_finite() || dt <= 0.0 || self.da <= 0.0 {
            return;
        }
        let substeps = ((dt / self.da).ceil() as usize).max(1);
        let sub_dt = dt / substeps as f64;
        let mut step_idx = 0;
        while step_idx < substeps {
            self.step_cfl_substep(sub_dt, Some(boundary_birth));
            step_idx += 1;
        }
    }
}
