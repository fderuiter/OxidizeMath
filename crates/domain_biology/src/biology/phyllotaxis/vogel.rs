//! Vogel golden-angle primordia packing, Fibonacci parastichy detection, and nutrient kinetics.

use crate::biology::kinetics::{HillKinetics, KineticsError, KineticsModel, MichaelisMenten};
use std::f64::consts::{PI, TAU};

/// Golden divergence angle $\alpha_\phi = 360^\circ (2 - \varphi) \approx 137.507764^\circ$.
pub const GOLDEN_ANGLE_DEG: f64 = 137.507_764_050_037_85;

/// Golden divergence angle in radians ($\approx 2.39996323\text{ rad}$).
pub const GOLDEN_ANGLE_RAD: f64 = 2.399_963_229_728_653_5;

/// Lucas divergence angle $\approx 99.501553^\circ$, generating Lucas spiral pairs $(7, 11, 18, 29, \dots)$.
pub const LUCAS_ANGLE_DEG: f64 = 99.501_552_809_976_85;

/// Selects the enzyme saturation model coupling substrate availability to developmental rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NutrientKineticsKind {
    /// Hyperbolic non-cooperative Michaelis-Menten uptake ($n = 1$).
    #[default]
    MichaelisMenten,
    /// Cooperative sigmoidal Hill uptake ($n > 0$).
    Hill,
}

/// Biochemical nutrient uptake model coupling substrate concentration `\[S\]` to growth rate.
#[derive(Debug, Clone, Copy)]
pub struct NutrientCoupling {
    kind: NutrientKineticsKind,
    substrate: f64,
    mm: MichaelisMenten,
    hill: HillKinetics,
}

impl NutrientCoupling {
    /// Constructs a validated biochemical nutrient coupling configuration.
    #[verified_engine::verified]
    pub fn new(
        kind: NutrientKineticsKind,
        substrate: f64,
        v_max: f64,
        k_m: f64,
        hill_n: f64,
    ) -> Result<Self, KineticsError> {
        if !substrate.is_finite() || substrate < 0.0 {
            return Err(KineticsError::NegativeSubstrateConcentration);
        }
        let mm = MichaelisMenten::new(v_max, k_m)?;
        let hill = HillKinetics::new(v_max, k_m, hill_n)?;
        Ok(Self {
            kind,
            substrate,
            mm,
            hill,
        })
    }

    /// Evaluates reaction uptake velocity `v([S])` without heap allocations.
    #[verified_engine::verified]
    pub fn reaction_velocity(&self) -> f64 {
        let res = match self.kind {
            NutrientKineticsKind::MichaelisMenten => self.mm.reaction_velocity(self.substrate),
            NutrientKineticsKind::Hill => self.hill.reaction_velocity(self.substrate),
        };
        res.unwrap_or(0.0)
    }

    /// Normalized saturation ratio `v([S]) / V_max` in `[0, 1]`.
    #[verified_engine::verified]
    pub fn normalized_rate(&self) -> f64 {
        let v_max = self.mm.v_max;
        if v_max <= 1e-15 {
            return 0.0;
        }
        (self.reaction_velocity() / v_max).clamp(0.0, 1.0)
    }

    /// Active kinetics family.
    #[verified_engine::verified]
    pub fn kind(&self) -> NutrientKineticsKind {
        self.kind
    }

    /// Current substrate concentration `[S]`.
    #[verified_engine::verified]
    pub fn substrate(&self) -> f64 {
        self.substrate
    }

    /// Maximum velocity $V_{\max}$.
    #[verified_engine::verified]
    pub fn v_max(&self) -> f64 {
        self.mm.v_max
    }

    /// Half-saturation constant $K_m$.
    #[verified_engine::verified]
    pub fn k_m(&self) -> f64 {
        self.mm.k_m
    }

    /// Hill cooperativity coefficient $n$.
    #[verified_engine::verified]
    pub fn hill_n(&self) -> f64 {
        self.hill.n
    }
}

impl Default for NutrientCoupling {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(NutrientKineticsKind::MichaelisMenten, 2.5, 1.0, 1.0, 2.5)
            .expect("valid default nutrient coupling")
    }
}

/// Spatial coordinates and plastochron index of a single botanical primordium (floret/leaf).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Primordium {
    /// Sequential initiation index $k \in \{0, \dots, N-1\}$.
    pub index: usize,
    /// Radial distance $r_k = c \sqrt{k + 0.5 + \phi}$ from the meristem apex.
    pub radius: f64,
    /// Divergence azimuth $\theta_k = k \cdot \alpha$ in radians.
    pub angle_rad: f64,
    /// Cartesian x-coordinate $r_k \cos \theta_k$.
    pub x: f64,
    /// Cartesian y-coordinate $r_k \sin \theta_k$.
    pub y: f64,
}

/// Configuration for Vogel phyllotaxis primordia packing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhyllotaxisConfig {
    /// Divergence angle $\alpha$ in degrees (default: [`GOLDEN_ANGLE_DEG`]).
    pub divergence_angle_deg: f64,
    /// Radial packing scale constant $c > 0$.
    pub scale_c: f64,
    /// Base primordia initiation rate (primordia per unit time at full nutrient saturation).
    pub base_initiation_rate: f64,
}

impl Default for PhyllotaxisConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            divergence_angle_deg: GOLDEN_ANGLE_DEG,
            scale_c: 0.45,
            base_initiation_rate: 12.0,
        }
    }
}

/// Zero-allocation Vogel phyllotaxis state machine with preallocated primordia storage.
#[derive(Debug, Clone)]
pub struct PhyllotaxisState {
    config: PhyllotaxisConfig,
    nutrient: NutrientCoupling,
    primordia: Vec<Primordium>,
    active_count: usize,
    max_primordia: usize,
    plastochron_phase: f64,
    elapsed_time: f64,
}

impl PhyllotaxisState {
    /// Allocates a bounded phyllotaxis state capable of holding up to `max_primordia` florets.
    #[verified_engine::verified]
    pub fn new(
        max_primordia: usize,
        initial_count: usize,
        config: PhyllotaxisConfig,
        nutrient: NutrientCoupling,
    ) -> Self {
        let cap = max_primordia.max(1);
        let active = initial_count.min(cap);
        let mut state = Self {
            config,
            nutrient,
            primordia: vec![Primordium::default(); cap],
            active_count: active,
            max_primordia: cap,
            plastochron_phase: 0.0,
            elapsed_time: 0.0,
        };
        state.recompute_positions_in_place();
        state
    }

    /// Advances plastochron initiation and radial expansion by `dt` with **zero heap allocations**.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let rate_scale = self.nutrient.normalized_rate();
        let delta_plastochrons = dt * self.config.base_initiation_rate.max(0.0) * rate_scale;
        self.plastochron_phase += delta_plastochrons;
        self.elapsed_time += dt;

        let whole_new = self.plastochron_phase.floor() as usize;
        if whole_new > 0 {
            self.plastochron_phase -= whole_new as f64;
            self.active_count = (self.active_count + whole_new).min(self.max_primordia);
        }
        self.recompute_positions_in_place();
    }

    /// Immediately sets the active primordia count (clamped to `max_primordia`) and updates geometry.
    #[verified_engine::verified]
    pub fn set_active_count(&mut self, count: usize) {
        self.active_count = count.min(self.max_primordia);
        self.plastochron_phase = 0.0;
        self.recompute_positions_in_place();
    }

    /// Updates geometric packing configuration and recomputes primordia coordinates in-place.
    #[verified_engine::verified]
    pub fn set_config(&mut self, config: PhyllotaxisConfig) {
        self.config = config;
        self.recompute_positions_in_place();
    }

    /// Updates nutrient uptake coupling configuration.
    #[verified_engine::verified]
    pub fn set_nutrient(&mut self, nutrient: NutrientCoupling) {
        self.nutrient = nutrient;
    }

    #[verified_engine::verified]
    fn recompute_positions_in_place(&mut self) {
        let alpha_rad = self.config.divergence_angle_deg.to_radians();
        let scale_c = self.config.scale_c.max(1e-6);
        let phase = self.plastochron_phase.clamp(0.0, 1.0);

        for k in 0..self.active_count {
            let radial_index = (k as f64) + 0.5 + phase;
            let radius = scale_c * radial_index.sqrt();
            let angle_rad = (k as f64) * alpha_rad;
            let (sin_a, cos_a) = angle_rad.sin_cos();
            self.primordia[k] = Primordium {
                index: k,
                radius,
                angle_rad,
                x: radius * cos_a,
                y: radius * sin_a,
            };
        }
    }

    /// Active slice of packed primordia (`0..active_count`).
    #[verified_engine::verified]
    pub fn primordia(&self) -> &[Primordium] {
        &self.primordia[..self.active_count]
    }

    /// Number of currently initiated primordia.
    #[verified_engine::verified]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Preallocated primordia capacity.
    #[verified_engine::verified]
    pub fn max_primordia(&self) -> usize {
        self.max_primordia
    }

    /// Current phyllotaxis configuration.
    #[verified_engine::verified]
    pub fn config(&self) -> PhyllotaxisConfig {
        self.config
    }

    /// Current biochemical nutrient coupling model.
    #[verified_engine::verified]
    pub fn nutrient(&self) -> NutrientCoupling {
        self.nutrient
    }

    /// Elapsed developmental time.
    #[verified_engine::verified]
    pub fn elapsed_time(&self) -> f64 {
        self.elapsed_time
    }

    /// Detects the visible opposing parastichy spiral pair `(m, n)` (`m <= n`) on the outer Vogel annulus.
    #[verified_engine::verified]
    pub fn detect_parastichy_pair(&self) -> (usize, usize) {
        detect_parastichy_spirals(self.active_count, self.config.divergence_angle_deg)
    }
}

/// Geometrically detects opposing nearest-neighbor parastichy spiral counts `(m, n)` for `count` florets.
#[verified_engine::verified]
pub fn detect_parastichy_spirals(count: usize, divergence_angle_deg: f64) -> (usize, usize) {
    if count < 5 {
        return (1, 1.max(count.min(2)));
    }
    let alpha_rad = divergence_angle_deg.to_radians();
    let k_ref = ((count * 3) / 4).max(3).min(count - 1);
    let r_ref = ((k_ref as f64) + 0.5).sqrt();

    let mut best_pos = (f64::INFINITY, 1usize);
    let mut best_neg = (f64::INFINITY, 1usize);
    let mut best_any = (f64::INFINITY, 1usize);

    for dk in 1..=k_ref {
        let r_nb = (((k_ref - dk) as f64) + 0.5).sqrt();
        let delta_theta = ((dk as f64) * alpha_rad + PI).rem_euclid(TAU) - PI;
        let dist_sq = r_ref * r_ref + r_nb * r_nb - 2.0 * r_ref * r_nb * delta_theta.cos();

        if dist_sq < best_any.0 {
            best_any = (dist_sq, dk);
        }
        if delta_theta > 1e-7 && dist_sq < best_pos.0 {
            best_pos = (dist_sq, dk);
        } else if delta_theta < -1e-7 && dist_sq < best_neg.0 {
            best_neg = (dist_sq, dk);
        }
    }

    let p = if best_pos.0.is_finite() {
        best_pos.1
    } else {
        best_any.1
    };
    let q = if best_neg.0.is_finite() {
        best_neg.1
    } else {
        best_any.1
    };
    (p.min(q), p.max(q))
}

/// Checks whether `(m, n)` (`m <= n`) is a pair of consecutive Fibonacci numbers $(F_k, F_{k+1})$.
#[verified_engine::verified]
pub fn is_consecutive_fibonacci_pair(m: usize, n: usize) -> bool {
    let (lo, hi) = (m.min(n), m.max(n));
    let mut a = 1usize;
    let mut b = 1usize;
    while a <= lo {
        if a == lo && b == hi {
            return true;
        }
        let Some(next) = a.checked_add(b) else {
            break;
        };
        a = b;
        b = next;
    }
    false
}
