//! Biochemical nutrient uptake kinetics coupled to continuous plastochron initiation and L-systems.

// [cite:morphogenesis_and_lenia]

use super::parastichy::ParastichyAnalysis;
use super::vogel::{GOLDEN_ANGLE_DEG, MAX_PRIMORDIA, PrimordiumPoint, VogelPhyllotaxisEngine};
use crate::biology::kinetics::{HillKinetics, KineticsError, KineticsModel, MichaelisMenten};
use crate::biology::lsystem::TurtleConfig;

/// Lucas noble divergence angle $\approx 99.501553^\circ$, generating Lucas $(3, 4, 7, 11, 18)$ spirals.
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

/// Biochemical nutrient uptake model coupling substrate concentration `[S]` to growth velocity.
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

    /// Evaluates reaction uptake velocity $v(\[S\])$ without heap allocations.
    #[verified_engine::verified]
    pub fn reaction_velocity(&self) -> f64 {
        let res = match self.kind {
            NutrientKineticsKind::MichaelisMenten => self.mm.reaction_velocity(self.substrate),
            NutrientKineticsKind::Hill => self.hill.reaction_velocity(self.substrate),
        };
        res.unwrap_or(0.0)
    }

    /// Normalized saturation ratio $v(\[S\]) / V_{\max} \in \[0, 1\]$.
    #[verified_engine::verified]
    pub fn normalized_rate(&self) -> f64 {
        let v_max = self.mm.v_max;
        if v_max <= 1e-15 {
            return 0.0;
        }
        (self.reaction_velocity() / v_max).clamp(0.0, 1.0)
    }

    /// Modulates an L-system [`TurtleConfig`] internode length and branching angle by nutrient saturation.
    #[verified_engine::verified]
    pub fn scaled_turtle_config(&self, base: TurtleConfig) -> TurtleConfig {
        let sat = self.normalized_rate();
        TurtleConfig {
            step_length: (base.step_length * (0.4 + 0.8 * sat)).max(1e-4),
            turn_angle_rad: base.turn_angle_rad * (0.75 + 0.5 * sat),
            initial_heading_rad: base.initial_heading_rad,
            depth_scale: base.depth_scale,
        }
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

    /// Maximum catalytic velocity $V_{\max}$.
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

/// Configuration for continuous-time plastochron initiation on a Vogel phyllotaxis meristem.
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

/// Zero-allocation continuous-time plastochron state machine backed by [`VogelPhyllotaxisEngine`].
#[derive(Debug, Clone)]
pub struct PhyllotaxisState {
    config: PhyllotaxisConfig,
    nutrient: NutrientCoupling,
    engine: VogelPhyllotaxisEngine,
    max_primordia: usize,
    plastochron_phase: f64,
    elapsed_time: f64,
}

impl PhyllotaxisState {
    /// Allocates a bounded phyllotaxis state up to `max_primordia` ($\le 4{,}096$) florets.
    #[verified_engine::verified]
    pub fn new(
        max_primordia: usize,
        initial_count: usize,
        config: PhyllotaxisConfig,
        nutrient: NutrientCoupling,
    ) -> Self {
        let cap = max_primordia.clamp(1, MAX_PRIMORDIA);
        let active = initial_count.clamp(1, cap);
        let div_rad = sanitize_divergence_deg(config.divergence_angle_deg).to_radians();
        let scale = sanitize_scale(config.scale_c);
        let engine = VogelPhyllotaxisEngine::new(active, div_rad, scale)
            .expect("clamped phyllotaxis state parameters are valid");
        Self {
            config,
            nutrient,
            engine,
            max_primordia: cap,
            plastochron_phase: 0.0,
            elapsed_time: 0.0,
        }
    }

    /// Advances plastochron initiation by `dt` with **zero heap allocations**.
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
            let next_count = (self.engine.count() + whole_new).min(self.max_primordia);
            let div_rad = sanitize_divergence_deg(self.config.divergence_angle_deg).to_radians();
            let scale = sanitize_scale(self.config.scale_c);
            let _ = self.engine.set_parameters(next_count, div_rad, scale);
        }
    }

    /// Resets active floret count, configuration, and nutrient model in place without allocating.
    #[verified_engine::verified]
    pub fn reconfigure(
        &mut self,
        count: usize,
        config: PhyllotaxisConfig,
        nutrient: NutrientCoupling,
    ) {
        self.config = config;
        self.nutrient = nutrient;
        self.plastochron_phase = 0.0;
        let clamped = count.clamp(1, self.max_primordia);
        let div_rad = sanitize_divergence_deg(config.divergence_angle_deg).to_radians();
        let scale = sanitize_scale(config.scale_c);
        let _ = self.engine.set_parameters(clamped, div_rad, scale);
    }

    /// Active primordia points slice on the Vogel spiral.
    #[verified_engine::verified]
    pub fn points(&self) -> &[PrimordiumPoint] {
        self.engine.points()
    }

    /// Active primordia count.
    #[verified_engine::verified]
    pub fn active_count(&self) -> usize {
        self.engine.count()
    }

    /// Maximum configured primordia capacity.
    #[verified_engine::verified]
    pub fn max_primordia(&self) -> usize {
        self.max_primordia
    }

    /// Current opposing parastichy spiral pair `(primary, secondary)` with `primary <= secondary`.
    #[verified_engine::verified]
    pub fn detect_parastichy_pair(&self) -> (usize, usize) {
        let a = self.engine.analysis();
        if a.primary_count <= a.secondary_count {
            (a.primary_count, a.secondary_count)
        } else {
            (a.secondary_count, a.primary_count)
        }
    }

    /// Full parastichy and annular packing uniformity analysis from [`VogelPhyllotaxisEngine`].
    #[verified_engine::verified]
    pub fn analysis(&self) -> ParastichyAnalysis {
        self.engine.analysis()
    }

    /// Active biochemical nutrient coupling configuration.
    #[verified_engine::verified]
    pub fn nutrient(&self) -> &NutrientCoupling {
        &self.nutrient
    }

    /// Elapsed developmental simulation time.
    #[verified_engine::verified]
    pub fn elapsed_time(&self) -> f64 {
        self.elapsed_time
    }
}

#[verified_engine::verified]
fn sanitize_divergence_deg(deg: f64) -> f64 {
    if deg.is_finite() {
        deg
    } else {
        GOLDEN_ANGLE_DEG
    }
}

#[verified_engine::verified]
fn sanitize_scale(scale: f64) -> f64 {
    if scale.is_finite() && scale > 1e-6 {
        scale
    } else {
        0.45
    }
}
