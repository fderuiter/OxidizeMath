//! Diekmann et al. age-structured cannibalism mortality and saturation kernels.

/// Evaluates the scalar Diekmann per-capita death rate equation
/// $\mu(t, a) = \nu(a) + C(a) k(t) \Phi(c(t))$.
///
/// # Arguments
///
/// * `nu_a` - Natural death rate for an individual of age $a$.
/// * `c_a` - Attack rate of cannibals on individuals of age $a$.
/// * `k_t` - Total number of cannibals at time $t$.
/// * `phi_c_t` - Density-dependent functional saturation factor $\Phi(c(t))$.
#[verified_engine::verified]
pub fn death_rate(nu_a: f64, c_a: f64, k_t: f64, phi_c_t: f64) -> f64 {
    nu_a + c_a * k_t * phi_c_t
}

/// Clamps negative or non-finite population densities to `0.0`.
#[verified_engine::verified]
pub fn sanitize_density(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

/// Integrates a uniformly spaced age-grid profile via composite trapezoidal quadrature.
#[verified_engine::verified]
pub fn trapezoidal_quadrature(values: &[f64], da: f64) -> f64 {
    let n = values.len();
    if n < 2 || !da.is_finite() || da <= 0.0 {
        return 0.0;
    }
    let mut sum = 0.5 * (sanitize_density(values[0]) + sanitize_density(values[n - 1]));
    let mut idx = 1;
    while idx + 1 < n {
        sum += sanitize_density(values[idx]);
        idx += 1;
    }
    sanitize_density(sum * da)
}

/// Advances one CFL-safe characteristic upwind step (`dt <= da`) without heap allocation.
#[verified_engine::verified]
pub fn advect_characteristic_step(
    density: &[f64],
    mortality: &[f64],
    boundary_birth: f64,
    dt: f64,
    da: f64,
    next_density: &mut [f64],
) {
    let n = density.len().min(mortality.len()).min(next_density.len());
    if n == 0 {
        return;
    }
    let lambda = if da > 0.0 && dt.is_finite() && da.is_finite() {
        (dt / da).clamp(0.0, 1.0)
    } else {
        0.0
    };
    next_density[0] = sanitize_density(boundary_birth);
    let mut idx = 1;
    while idx < n {
        let foot = (1.0 - lambda) * sanitize_density(density[idx])
            + lambda * sanitize_density(density[idx - 1]);
        let mu = sanitize_density(mortality[idx]);
        let survival = (-mu * dt.max(0.0)).exp();
        next_density[idx] = sanitize_density(foot * survival);
        idx += 1;
    }
}

/// Diekmann age-structured mortality, juvenile vulnerability, and Ricker saturation kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CannibalismKernel {
    /// Baseline natural per-capita mortality $\nu_0 \ge 0$.
    pub natural_mortality: f64,
    /// Age-dependent senescence slope $\nu_1 \ge 0$ in $\nu(a) = \nu_0 + \nu_1 a$.
    pub senescence_rate: f64,
    /// Cannibalistic attack rate $c_0 \ge 0$ on vulnerable juveniles.
    pub attack_rate: f64,
    /// Maximum age $a_{\text{vuln}} \ge 0$ susceptible to cannibalism.
    pub vulnerability_max_age: f64,
    /// Ricker prey-handling saturation coefficient $s \ge 0$ in $\Phi(c(t))$.
    pub saturation: f64,
    /// Maturation age $\alpha > 0$ separating juveniles from cannibalistic adults.
    pub maturation_age: f64,
    /// Baseline per-capita adult fecundity $\beta_0 \ge 0$ for $a \ge \alpha$.
    pub fecundity: f64,
    /// Ricker density-dependence scale $K > 0$ in $\beta(a, N) = \beta_0 e^{-N/K}$.
    pub carrying_scale: f64,
}

impl Default for CannibalismKernel {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            natural_mortality: 0.15,
            senescence_rate: 0.02,
            attack_rate: 0.35,
            vulnerability_max_age: 3.5,
            saturation: 0.02,
            maturation_age: 4.0,
            fecundity: 3.2,
            carrying_scale: 80.0,
        }
    }
}

impl CannibalismKernel {
    /// Constructs a [`CannibalismKernel`] with explicit demographic and cannibalism rates.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn new(
        natural_mortality: f64,
        senescence_rate: f64,
        attack_rate: f64,
        vulnerability_max_age: f64,
        saturation: f64,
        maturation_age: f64,
        fecundity: f64,
        carrying_scale: f64,
    ) -> Self {
        Self {
            natural_mortality,
            senescence_rate,
            attack_rate,
            vulnerability_max_age,
            saturation,
            maturation_age,
            fecundity,
            carrying_scale,
        }
    }

    /// Evaluates natural mortality $\nu(a) = \max(0, \nu_0 + \nu_1 a)$.
    #[verified_engine::verified]
    pub fn natural_mortality_at(&self, age: f64) -> f64 {
        let base = sanitize_density(self.natural_mortality);
        let slope = sanitize_density(self.senescence_rate);
        sanitize_density(base + slope * sanitize_density(age))
    }

    /// Evaluates juvenile cannibalism vulnerability $C(a)$ on $[0, a_{\text{vuln}}]$.
    #[verified_engine::verified]
    pub fn vulnerability_at(&self, age: f64) -> f64 {
        let c0 = sanitize_density(self.attack_rate);
        let max_v = sanitize_density(self.vulnerability_max_age);
        if c0 == 0.0 || max_v == 0.0 || !age.is_finite() || age < 0.0 || age > max_v {
            0.0
        } else {
            c0
        }
    }

    /// Evaluates Ricker saturation $\Phi(c) = \frac{1 - e^{-s c}}{s c}$ with a smooth $c \to 0$ guard.
    #[verified_engine::verified]
    pub fn ricker_saturation(&self, prey_load: f64) -> f64 {
        let s = sanitize_density(self.saturation);
        let c = sanitize_density(prey_load);
        let sc = s * c;
        if sc <= 1e-8 {
            (1.0 - 0.5 * sc).clamp(0.0, 1.0)
        } else {
            ((1.0 - (-sc).exp()) / sc).clamp(0.0, 1.0)
        }
    }

    /// Evaluates density-regulated adult fertility $\beta(a, N) = \beta_0 e^{-N/K}$ for $a \ge \alpha$.
    #[verified_engine::verified]
    pub fn fertility_at(&self, age: f64, total_population: f64) -> f64 {
        let b0 = sanitize_density(self.fecundity);
        let k = sanitize_density(self.carrying_scale);
        let alpha = sanitize_density(self.maturation_age);
        if b0 == 0.0 || k == 0.0 || !age.is_finite() || age < alpha {
            return 0.0;
        }
        let n_pop = sanitize_density(total_population);
        sanitize_density(b0 * (-n_pop / k).exp())
    }

    /// Evaluates total per-capita mortality $\mu(t, a) = \nu(a) + C(a) k(t) \Phi(c(t))$.
    #[verified_engine::verified]
    pub fn mortality_at(&self, age: f64, cannibal_load: f64, prey_load: f64) -> f64 {
        let nu_a = self.natural_mortality_at(age);
        let c_a = self.vulnerability_at(age);
        let phi = self.ricker_saturation(prey_load);
        sanitize_density(death_rate(nu_a, c_a, sanitize_density(cannibal_load), phi))
    }
}
