//! Coupled van den Bosch et al. juvenile-adult delay PDE-ODE dynamics.

use super::death_rate::{advect_characteristic_step, sanitize_density, trapezoidal_quadrature};

/// Evaluates the scalar reaction term $-[I_t + C(a) A(t)] n(t, a)$ for juvenile mortality.
#[verified_engine::verified]
pub fn juvenile_dynamics(i_t: f64, c_a: f64, a_t: f64, n_t_a: f64) -> f64 {
    -(i_t + c_a * a_t) * n_t_a
}

/// Evaluates the scalar adult net recruitment rate $\frac{dA}{dt} = n(t, \alpha) - f(I(t)) A(t)$.
#[verified_engine::verified]
pub fn adult_dynamics(n_t_alpha: f64, f_i_t: f64, a_t: f64) -> f64 {
    n_t_alpha - f_i_t * a_t
}

/// Demographic and cannibalism parameters for [`JuvenileAdultSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JuvenileAdultParams {
    /// Maturation age $\alpha > 0$ at which juveniles recruit into the adult class.
    pub maturation_age: f64,
    /// Natural per-capita juvenile death rate $I_t \ge 0$.
    pub juvenile_mortality: f64,
    /// Per-adult cannibalistic attack rate $c_0 \ge 0$ on vulnerable juveniles.
    pub attack_rate: f64,
    /// Upper age $a_{\text{vuln}} \in [0, \alpha]$ of juvenile cannibalism vulnerability.
    pub vulnerability_max_age: f64,
    /// Baseline adult mortality rate $\mu_A \ge 0$ in absence of cannibalistic energy intake.
    pub adult_mortality: f64,
    /// Lifeboat survival sensitivity $\eta \ge 0$ in $f(I(t)) = \frac{\mu_A}{1 + \eta I(t)}$.
    pub energy_survival_gain: f64,
    /// Baseline adult fecundity $\beta_0 \ge 0$.
    pub adult_fecundity: f64,
    /// Fecundity conversion efficiency $\gamma \ge 0$ per unit cannibalistic intake $I(t)$.
    pub cannibal_fecundity_gain: f64,
    /// Adult Ricker competition scale $K_A > 0$ in $\exp(-A(t)/K_A)$.
    pub adult_carrying_scale: f64,
}

impl Default for JuvenileAdultParams {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            maturation_age: 4.0,
            juvenile_mortality: 0.08,
            attack_rate: 0.22,
            vulnerability_max_age: 3.2,
            adult_mortality: 1.15,
            energy_survival_gain: 0.03,
            adult_fecundity: 20.0,
            cannibal_fecundity_gain: 0.12,
            adult_carrying_scale: 80.0,
        }
    }
}

/// Zero-allocation coupled PDE-ODE solver for juvenile age advection on $[0, \alpha]$
/// and adult cohort dynamics $\frac{dA}{dt} = n(t, \alpha) - f(I(t)) A(t)$.
#[derive(Debug, Clone)]
pub struct JuvenileAdultSolver {
    ages: Vec<f64>,
    density: Vec<f64>,
    next_density: Vec<f64>,
    mortality: Vec<f64>,
    vulnerability: Vec<f64>,
    adults: f64,
    da: f64,
    time: f64,
    last_maturation_flux: f64,
    last_energy_intake: f64,
    params: JuvenileAdultParams,
}

impl JuvenileAdultSolver {
    /// Pre-allocates juvenile age slices on $[0, \alpha]$ for zero-allocation `step_upwind` execution.
    pub fn new(num_ages: usize, initial_adults: f64, params: JuvenileAdultParams) -> Self {
        let count = num_ages.clamp(2, 4096);
        let alpha = if params.maturation_age.is_finite() && params.maturation_age > 0.0 {
            params.maturation_age
        } else {
            4.0
        };
        let da = alpha / (count - 1) as f64;
        let ages: Vec<f64> = (0..count).map(|i| i as f64 * da).collect();
        let mut solver = Self {
            ages,
            density: vec![0.0; count],
            next_density: vec![0.0; count],
            mortality: vec![0.0; count],
            vulnerability: vec![0.0; count],
            adults: sanitize_density(initial_adults),
            da,
            time: 0.0,
            last_maturation_flux: 0.0,
            last_energy_intake: 0.0,
            params,
        };
        solver.refresh_vulnerability();
        solver
    }

    /// Initializes the juvenile age profile $n(0, a)$ on $[0, \alpha]$ and adult stock $A(0)$.
    pub fn set_initial_state<F: FnMut(f64) -> f64>(&mut self, initial_adults: f64, mut profile: F) {
        for (slot, &age) in self.density.iter_mut().zip(self.ages.iter()) {
            *slot = sanitize_density(profile(age));
        }
        self.next_density.fill(0.0);
        self.adults = sanitize_density(initial_adults);
        self.time = 0.0;
        self.last_maturation_flux = sanitize_density(*self.density.last().unwrap_or(&0.0));
        self.last_energy_intake = self.compute_energy_intake();
    }

    /// Returns the juvenile age grid $[0, \alpha]$.
    pub fn ages(&self) -> &[f64] {
        &self.ages
    }

    /// Returns the juvenile age-density profile $n(t, a)$ on $[0, \alpha]$.
    pub fn juvenile_density(&self) -> &[f64] {
        &self.density
    }

    /// Returns total juvenile abundance $J(t) = \int_0^\alpha n(t, a)\,da$.
    #[verified_engine::verified]
    pub fn total_juveniles(&self) -> f64 {
        trapezoidal_quadrature(&self.density, self.da)
    }

    /// Returns current adult abundance $A(t)$.
    #[verified_engine::verified]
    pub fn adults(&self) -> f64 {
        self.adults
    }

    /// Returns current simulation time $t$.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns uniform juvenile age grid spacing $\Delta a$.
    #[verified_engine::verified]
    pub fn da(&self) -> f64 {
        self.da
    }

    /// Returns maturation recruitment flux $n(t, \alpha)$ entering the adult stage.
    #[verified_engine::verified]
    pub fn last_maturation_flux(&self) -> f64 {
        self.last_maturation_flux
    }

    /// Returns per-adult cannibalistic energy intake $I(t) = \int_0^\alpha C(a) n(t, a)\,da$.
    #[verified_engine::verified]
    pub fn last_energy_intake(&self) -> f64 {
        self.last_energy_intake
    }

    /// Returns active [`JuvenileAdultParams`].
    pub fn params(&self) -> &JuvenileAdultParams {
        &self.params
    }

    /// Updates parameters without heap allocation (preserving current grid length).
    #[verified_engine::verified]
    pub fn set_params(&mut self, params: JuvenileAdultParams) {
        let count = self.ages.len();
        let alpha = if params.maturation_age.is_finite() && params.maturation_age > 0.0 {
            params.maturation_age
        } else {
            4.0
        };
        self.da = alpha / (count.saturating_sub(1).max(1)) as f64;
        let mut idx = 0;
        while idx < count {
            self.ages[idx] = idx as f64 * self.da;
            idx += 1;
        }
        self.params = params;
        self.refresh_vulnerability();
    }

    #[verified_engine::verified]
    fn refresh_vulnerability(&mut self) {
        let c0 = sanitize_density(self.params.attack_rate);
        let max_v = sanitize_density(self.params.vulnerability_max_age);
        let n = self.ages.len();
        let mut idx = 0;
        while idx < n {
            let age = self.ages[idx];
            self.vulnerability[idx] = if c0 > 0.0 && max_v > 0.0 && age <= max_v {
                c0
            } else {
                0.0
            };
            idx += 1;
        }
    }

    #[verified_engine::verified]
    fn compute_energy_intake(&self) -> f64 {
        let n = self.ages.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut idx = 0;
        while idx + 1 < n {
            let e0 = self.vulnerability[idx] * sanitize_density(self.density[idx]);
            let e1 = self.vulnerability[idx + 1] * sanitize_density(self.density[idx + 1]);
            sum += 0.5 * (e0 + e1) * self.da;
            idx += 1;
        }
        sanitize_density(sum)
    }

    #[verified_engine::verified]
    fn step_cfl_substep(&mut self, sub_dt: f64) {
        let n = self.ages.len();
        let a_t = sanitize_density(self.adults);
        let i_nat = sanitize_density(self.params.juvenile_mortality);
        let energy = self.compute_energy_intake();
        let mut idx = 0;
        while idx < n {
            self.mortality[idx] = sanitize_density(i_nat + self.vulnerability[idx] * a_t);
            idx += 1;
        }

        let n_alpha = sanitize_density(self.density[n - 1]);
        let mu_a = sanitize_density(self.params.adult_mortality);
        let eta = sanitize_density(self.params.energy_survival_gain);
        let f_i_t = mu_a / (1.0 + eta * energy);
        let d_a = adult_dynamics(n_alpha, f_i_t, a_t);
        let next_adults = sanitize_density(a_t + sub_dt * d_a);

        let b0 = sanitize_density(self.params.adult_fecundity);
        let gamma = sanitize_density(self.params.cannibal_fecundity_gain);
        let k_a = sanitize_density(self.params.adult_carrying_scale);
        let birth = if k_a > 0.0 {
            sanitize_density((b0 + gamma * energy) * a_t * (-a_t / k_a).exp())
        } else {
            0.0
        };

        advect_characteristic_step(
            &self.density,
            &self.mortality,
            birth,
            sub_dt,
            self.da,
            &mut self.next_density,
        );
        core::mem::swap(&mut self.density, &mut self.next_density);
        self.adults = next_adults;
        self.last_maturation_flux = n_alpha;
        self.last_energy_intake = energy;
        self.time += sub_dt;
    }

    /// Advances juvenile advection and adult recruitment by `dt` without heap allocation,
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
            self.step_cfl_substep(sub_dt);
            step_idx += 1;
        }
    }
}
