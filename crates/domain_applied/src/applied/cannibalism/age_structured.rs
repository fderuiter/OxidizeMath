//! Zero-allocation characteristic-shift solver for the McKendrick-von Foerster cannibalism PDE.
// [cite:cannibalism]

use super::{characteristic_survival_step, death_rate, renewal_birth_rate};

/// Minimum allowed age-grid resolution for [`AgeStructuredSolver`].
pub const MIN_GRID_SIZE: usize = 16;

/// Maximum allowed age-grid resolution for [`AgeStructuredSolver`], bounding preallocated buffers.
pub const MAX_GRID_SIZE: usize = 512;

/// Intraspecific predation regime governing age-specific cannibalism mortality and renewal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredationRegime {
    /// Diekmann et al. (1986) vulnerable-juvenile cannibalism $\mu(t, a) = \nu(a) + C(a) A(t) \Phi(A(t))$.
    Diekmann {
        /// Upper age bound $a_{\text{vul}}$ of vulnerable juveniles ($C(a) = c_0$ on $[0, a_{\text{vul}})$).
        vulnerable_age: f64,
        /// Per-adult cannibalistic attack rate $c_0 \ge 0$ on vulnerable cohorts.
        attack_rate: f64,
        /// Holling Type II saturation handling time $h \ge 0$ in $\Phi(A) = (1 + h A)^{-1}$.
        handling_time: f64,
    },
    /// Stage-structured *Tribolium* egg and callow/pupal cannibalism (Cushing & Costantino).
    Tribolium {
        /// Upper age bound $a_{\text{egg}}$ separating eggs from feeding larvae.
        egg_age: f64,
        /// Lower age bound $a_{\text{pupa}}$ separating feeding larvae from vulnerable pupae/callows.
        pupa_age: f64,
        /// Larval cannibalism rate $c_{el} \ge 0$ on eggs during oviposition.
        c_el: f64,
        /// Adult cannibalism rate $c_{ea} \ge 0$ on eggs during oviposition.
        c_ea: f64,
        /// Adult cannibalism rate $c_{pa} \ge 0$ on callows/pupae ($a \in [a_{\text{pupa}}, \alpha)$).
        c_pa: f64,
        /// Adult per-capita natural mortality rate $\mu_A \ge 0$ for $a \ge \alpha$.
        adult_mortality: f64,
    },
}

/// Configuration parameters for [`AgeStructuredSolver`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgeStructuredConfig {
    /// Number of uniform age bins $M \in [\text{MIN\_GRID\_SIZE}, \text{MAX\_GRID\_SIZE}]$.
    pub grid_size: usize,
    /// Maximum age $a_{\max} > 0$ represented on the grid.
    pub max_age: f64,
    /// Maturation age $\alpha \in (0, a_{\max})$ where individuals enter the adult stage.
    pub maturation_age: f64,
    /// Baseline juvenile natural mortality rate $\mu_0 \ge 0$.
    pub natural_mortality: f64,
    /// Per-adult fecundity rate $\beta_0 \ge 0$.
    pub beta0: f64,
    /// Active cannibalistic predation law.
    pub regime: PredationRegime,
}

impl AgeStructuredConfig {
    /// Subcritical Diekmann configuration whose analytical equilibrium is $A^* \approx 3.6715$.
    #[verified_engine::verified]
    pub fn diekmann_subcritical() -> Self {
        Self {
            grid_size: 100,
            max_age: 5.0,
            maturation_age: 2.0,
            natural_mortality: 0.4,
            beta0: 20.0,
            regime: PredationRegime::Diekmann {
                vulnerable_age: 1.5,
                attack_rate: 0.5,
                handling_time: 0.0,
            },
        }
    }

    /// Supercritical Diekmann Hopf-wave configuration ($\beta_0 = 45.0$) generating cohort waves.
    #[verified_engine::verified]
    pub fn diekmann_hopf() -> Self {
        Self {
            grid_size: 100,
            max_age: 5.0,
            maturation_age: 2.0,
            natural_mortality: 0.4,
            beta0: 45.0,
            regime: PredationRegime::Diekmann {
                vulnerable_age: 0.75,
                attack_rate: 1.0,
                handling_time: 0.03,
            },
        }
    }

    /// *Tribolium* egg-and-callow cannibalism configuration parameterized by $c_{pa}$.
    #[verified_engine::verified]
    pub fn tribolium_cycles(c_pa: f64) -> Self {
        let clamped_cpa = if c_pa.is_finite() { c_pa.max(0.0) } else { 0.7 };
        Self {
            grid_size: 80,
            max_age: 3.2,
            maturation_age: 2.0,
            natural_mortality: 0.2,
            beta0: 140.0,
            regime: PredationRegime::Tribolium {
                egg_age: 0.5,
                pupa_age: 1.4,
                c_el: 1.8,
                c_ea: 1.3,
                c_pa: clamped_cpa,
                adult_mortality: 2.4,
            },
        }
    }

    #[verified_engine::verified]
    fn sanitized(self) -> Self {
        let grid_size = self.grid_size.clamp(MIN_GRID_SIZE, MAX_GRID_SIZE);
        let max_age = if self.max_age.is_finite() && self.max_age > 0.2 {
            self.max_age
        } else {
            5.0
        };
        let maturation_age = if self.maturation_age.is_finite() {
            self.maturation_age.clamp(0.05 * max_age, 0.95 * max_age)
        } else {
            0.4 * max_age
        };
        let natural_mortality = if self.natural_mortality.is_finite() {
            self.natural_mortality.max(0.0)
        } else {
            0.4
        };
        let beta0 = if self.beta0.is_finite() {
            self.beta0.max(0.0)
        } else {
            20.0
        };
        Self {
            grid_size,
            max_age,
            maturation_age,
            natural_mortality,
            beta0,
            regime: self.regime,
        }
    }
}

/// Zero-allocation characteristic-shift solver for the McKendrick-von Foerster cannibalism PDE.
///
/// Preallocates ping-pong cohort density and kernel buffers up to [`MAX_GRID_SIZE`] during
/// initialization so [`AgeStructuredSolver::step`] executes with zero dynamic heap allocations
/// under `VerifiedEngine::run_verified`.
///
/// # Example
///
/// ```rust
/// # use domain_applied::__doctest_facade as math_explorer;
/// use math_explorer::applied::cannibalism::{AgeStructuredConfig, AgeStructuredSolver};
///
/// let mut solver = AgeStructuredSolver::new(AgeStructuredConfig::diekmann_hopf());
/// solver.step_n(50);
/// assert!(solver.adult_population() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct AgeStructuredSolver {
    config: AgeStructuredConfig,
    da: f64,
    time: f64,
    last_births: f64,
    ages: Vec<f64>,
    density: Vec<f64>,
    next_density: Vec<f64>,
    natural_mortality: Vec<f64>,
    fertility_kernel: Vec<f64>,
    attack_kernel: Vec<f64>,
}

impl AgeStructuredSolver {
    /// Creates a new `AgeStructuredSolver` with preallocated buffers bounded by [`MAX_GRID_SIZE`].
    pub fn new(config: AgeStructuredConfig) -> Self {
        let clean = config.sanitized();
        let mut solver = Self {
            config: clean,
            da: clean.max_age / (clean.grid_size as f64),
            time: 0.0,
            last_births: 0.0,
            ages: vec![0.0; MAX_GRID_SIZE],
            density: vec![0.0; MAX_GRID_SIZE],
            next_density: vec![0.0; MAX_GRID_SIZE],
            natural_mortality: vec![0.0; MAX_GRID_SIZE],
            fertility_kernel: vec![0.0; MAX_GRID_SIZE],
            attack_kernel: vec![0.0; MAX_GRID_SIZE],
        };
        solver.rebuild_kernels();
        solver.seed_perturbed_equilibrium(0.35);
        solver
    }

    /// Updates model parameters and kernels in place without heap allocation.
    #[verified_engine::verified]
    pub fn reconfigure(&mut self, config: AgeStructuredConfig) {
        let prev_grid = self.config.grid_size;
        self.config = config.sanitized();
        self.da = self.config.max_age / (self.config.grid_size as f64);
        self.rebuild_kernels();
        if self.config.grid_size != prev_grid {
            self.seed_perturbed_equilibrium(0.35);
        }
    }

    #[verified_engine::verified]
    fn rebuild_kernels(&mut self) {
        let m = self.config.grid_size;
        let da = self.da;
        debug_assert!((MIN_GRID_SIZE..=MAX_GRID_SIZE).contains(&m));
        for j in 0..m {
            let a = (j as f64 + 0.5) * da;
            self.ages[j] = a;
            self.fertility_kernel[j] = if a >= self.config.maturation_age {
                self.config.beta0
            } else {
                0.0
            };
            match self.config.regime {
                PredationRegime::Diekmann {
                    vulnerable_age,
                    attack_rate,
                    ..
                } => {
                    self.natural_mortality[j] = self.config.natural_mortality;
                    self.attack_kernel[j] = if a < vulnerable_age.max(0.0) {
                        attack_rate.max(0.0)
                    } else {
                        0.0
                    };
                }
                PredationRegime::Tribolium {
                    pupa_age,
                    c_pa,
                    adult_mortality,
                    ..
                } => {
                    self.natural_mortality[j] = if a < self.config.maturation_age {
                        self.config.natural_mortality
                    } else {
                        adult_mortality.max(0.0)
                    };
                    self.attack_kernel[j] =
                        if a >= pupa_age.max(0.0) && a < self.config.maturation_age {
                            c_pa.max(0.0)
                        } else {
                            0.0
                        };
                }
            }
        }
    }

    /// Seeds a deterministic non-negative initial age profile perturbed around survival decay.
    #[verified_engine::verified]
    pub fn seed_perturbed_equilibrium(&mut self, perturbation: f64) {
        let m = self.config.grid_size;
        let amp = if perturbation.is_finite() {
            perturbation.clamp(-0.9, 0.9)
        } else {
            0.0
        };
        let base = self.diekmann_equilibrium().unwrap_or(2.0).clamp(0.5, 10.0);
        for j in 0..m {
            let a = self.ages[j];
            let wave = 1.0 + amp * (2.5 * a).sin();
            self.density[j] =
                (base * (-self.config.natural_mortality * a).exp() * wave).max(1.0e-4);
            self.next_density[j] = 0.0;
        }
        self.time = 0.0;
        self.last_births = self.compute_boundary_renewal(self.adult_population());
    }

    /// Overwrites the active cohort density profile from `values` (clamped to $\ge 0$).
    #[verified_engine::verified]
    pub fn set_density_slice(&mut self, values: &[f64]) {
        let m = self.config.grid_size;
        let copy_len = m.min(values.len());
        for (dst, &v) in self.density[..copy_len]
            .iter_mut()
            .zip(values[..copy_len].iter())
        {
            *dst = if v.is_finite() { v.max(0.0) } else { 0.0 };
        }
        self.density[copy_len..m].fill(0.0);
        self.time = 0.0;
    }

    #[verified_engine::verified]
    fn compute_boundary_renewal(&self, adults: f64) -> f64 {
        let m = self.config.grid_size;
        match self.config.regime {
            PredationRegime::Diekmann { .. } => {
                renewal_birth_rate(&self.fertility_kernel[..m], &self.density[..m], self.da)
            }
            PredationRegime::Tribolium { c_el, c_ea, .. } => {
                let raw_births =
                    renewal_birth_rate(&self.fertility_kernel[..m], &self.density[..m], self.da);
                let larvae = self.larval_population();
                let egg_loss = c_el.max(0.0) * larvae + c_ea.max(0.0) * adults.max(0.0);
                raw_births * (-egg_loss).exp()
            }
        }
    }

    /// Advances the age distribution by one characteristic step $\Delta t = \Delta a$ with zero heap allocation.
    #[verified_engine::verified]
    pub fn step(&mut self) {
        let m = self.config.grid_size;
        let dt = self.da;
        debug_assert!(dt > 0.0 && m <= MAX_GRID_SIZE);

        let adults = self.adult_population();
        let phi = match self.config.regime {
            PredationRegime::Diekmann { handling_time, .. } => {
                1.0 / (1.0 + handling_time.max(0.0) * adults)
            }
            PredationRegime::Tribolium { .. } => 1.0,
        };

        let b_t = self.compute_boundary_renewal(adults);
        self.next_density[0] = b_t;
        for j in 1..m {
            let mu = death_rate(
                self.natural_mortality[j - 1],
                self.attack_kernel[j - 1],
                adults,
                phi,
            );
            self.next_density[j] = characteristic_survival_step(self.density[j - 1], mu, dt);
        }

        std::mem::swap(&mut self.density, &mut self.next_density);
        self.last_births = b_t;
        self.time += dt;
    }

    /// Advances the solver by `steps` characteristic steps (bounded per call).
    #[verified_engine::verified]
    pub fn step_n(&mut self, steps: usize) {
        let bounded = steps.min(100_000);
        for _ in 0..bounded {
            self.step();
        }
    }

    /// Computes the closed-form Diekmann adult equilibrium
    /// $A^* = \frac{\ln R_0}{\bar{C} - h \ln R_0}$ when $R_0 > 1$ and $\bar{C} > h \ln R_0$.
    #[verified_engine::verified]
    pub fn diekmann_equilibrium(&self) -> Option<f64> {
        let PredationRegime::Diekmann {
            vulnerable_age,
            attack_rate,
            handling_time,
        } = self.config.regime
        else {
            return None;
        };
        let mu = self.config.natural_mortality.max(0.0);
        let alpha = self.config.maturation_age;
        let a_max = self.config.max_age;
        if self.config.beta0 <= 0.0 || a_max <= alpha {
            return None;
        }
        let adult_expected = if mu > 1.0e-12 {
            ((-mu * alpha).exp() - (-mu * a_max).exp()) / mu
        } else {
            a_max - alpha
        };
        let r0 = self.config.beta0 * adult_expected;
        if r0 <= 1.0 {
            return None;
        }
        let ln_r0 = r0.ln();
        let vul_span = vulnerable_age.clamp(0.0, alpha);
        let c_bar = attack_rate.max(0.0) * vul_span;
        let denom = c_bar - handling_time.max(0.0) * ln_r0;
        if denom <= 1.0e-12 {
            return None;
        }
        Some(ln_r0 / denom)
    }

    /// Returns the active solver configuration.
    #[verified_engine::verified]
    pub fn config(&self) -> &AgeStructuredConfig {
        &self.config
    }

    /// Returns the uniform age/time step $\Delta a = \Delta t$.
    #[verified_engine::verified]
    pub fn da(&self) -> f64 {
        self.da
    }

    /// Returns the current simulation time $t$.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns the most recent boundary renewal rate $b(t) = n(t, 0)$.
    #[verified_engine::verified]
    pub fn last_birth_rate(&self) -> f64 {
        self.last_births
    }

    /// Returns the midpoint age coordinates $a_j \in (0, a_{\max})$.
    #[verified_engine::verified]
    pub fn ages(&self) -> &[f64] {
        &self.ages[..self.config.grid_size]
    }

    /// Returns the active cohort age density slice $n(t, a_j)$.
    #[verified_engine::verified]
    pub fn density(&self) -> &[f64] {
        &self.density[..self.config.grid_size]
    }

    #[verified_engine::verified]
    fn integrate_age_window(&self, min_age: f64, max_age: f64) -> f64 {
        let m = self.config.grid_size;
        let mut sum = 0.0;
        for j in 0..m {
            let a = self.ages[j];
            if a >= min_age && a < max_age {
                sum += self.density[j].max(0.0);
            }
        }
        sum * self.da
    }

    /// Returns total population $\int_0^{a_{\max}} n(t, a)\,da$.
    #[verified_engine::verified]
    pub fn total_population(&self) -> f64 {
        self.integrate_age_window(0.0, self.config.max_age + self.da)
    }

    /// Returns total juvenile population $\int_0^{\alpha} n(t, a)\,da$.
    #[verified_engine::verified]
    pub fn juvenile_population(&self) -> f64 {
        self.integrate_age_window(0.0, self.config.maturation_age)
    }

    /// Returns total adult population $A(t) = \int_{\alpha}^{a_{\max}} n(t, a)\,da$.
    #[verified_engine::verified]
    pub fn adult_population(&self) -> f64 {
        self.integrate_age_window(self.config.maturation_age, self.config.max_age + self.da)
    }

    /// Returns feeding larval population $L(t)$ in *Tribolium* mode (or total juveniles in Diekmann mode).
    #[verified_engine::verified]
    pub fn larval_population(&self) -> f64 {
        match self.config.regime {
            PredationRegime::Diekmann { .. } => self.juvenile_population(),
            PredationRegime::Tribolium {
                egg_age, pupa_age, ..
            } => self.integrate_age_window(egg_age.max(0.0), pupa_age.max(egg_age)),
        }
    }

    /// Returns vulnerable callow/pupal population $P(t)$ in *Tribolium* mode.
    #[verified_engine::verified]
    pub fn callow_population(&self) -> f64 {
        match self.config.regime {
            PredationRegime::Diekmann { vulnerable_age, .. } => {
                self.integrate_age_window(0.0, vulnerable_age.max(0.0))
            }
            PredationRegime::Tribolium { pupa_age, .. } => {
                self.integrate_age_window(pupa_age.max(0.0), self.config.maturation_age)
            }
        }
    }
}
