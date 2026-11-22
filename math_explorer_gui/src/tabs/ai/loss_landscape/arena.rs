#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Multi-particle optimizer arena racing `SGD`, `Adam`, `HamiltonianLeapfrog2D`
//! (`QuadraticHamiltonian`), `LangevinBaoab2D` (`OxidizeRng`), and optional
//! `#[cfg(feature = "physics")]` `LorenzSystem` chaotic forcing.

use super::surfaces::{LandscapeSurface, SculptBump};
use crate::app::SemanticPalette;
use eframe::egui::Color32;
use math_explorer::applied::game_theory::mean_field::physics::{Hamiltonian, QuadraticHamiltonian};
#[cfg(feature = "physics")]
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState, LorenzSystem};
use math_explorer::pure_math::analysis::evolution::EvolutionEngine;
use math_explorer::pure_math::analysis::momentum_dynamics::{
    clamp_damping, clamp_dt, clamp_mass, clamp_temperature, HamiltonianLeapfrog2D, LangevinBaoab2D,
    PhasePoint2D, MAX_DAMPING, MAX_DT, MAX_MASS, MAX_TEMPERATURE, MIN_DAMPING, MIN_DT, MIN_MASS,
    MIN_TEMPERATURE,
};
use math_explorer::pure_math::analysis::optimization::{Adam, AdamState2D, SGD};
use nalgebra::Vector2;
use oxidize_core::rng::OxidizeRng;

pub const MAX_TRAIL_POINTS: usize = 160;
pub const LR_MIN: f64 = 1e-4;
pub const LR_MAX: f64 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RacerKind {
    Sgd,
    Adam,
    HamiltonianLeapfrog,
    LangevinThermal,
    LorenzChaotic,
}

impl RacerKind {
    pub const ALL: [Self; 5] = [
        Self::Sgd,
        Self::Adam,
        Self::HamiltonianLeapfrog,
        Self::LangevinThermal,
        Self::LorenzChaotic,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sgd => "SGD",
            Self::Adam => "Adam",
            Self::HamiltonianLeapfrog => "Hamiltonian Leapfrog",
            Self::LangevinThermal => "Langevin Thermal",
            Self::LorenzChaotic => "Lorenz Chaotic",
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_available_in_build(self) -> bool {
        !matches!(self, Self::LorenzChaotic) || cfg!(feature = "physics")
    }

    #[must_use]
    pub const fn color(self, palette: &SemanticPalette) -> Color32 {
        match self {
            Self::Sgd => palette.warning,
            Self::Adam => (*palette).series_color(0),
            Self::HamiltonianLeapfrog => palette.success,
            Self::LangevinThermal => (*palette).series_color(3),
            Self::LorenzChaotic => (*palette).series_color(4),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ArenaConfig {
    pub mass: f64,
    pub damping: f64,
    pub temperature: f64,
    pub dt: f64,
    pub learning_rate: f64,
    pub restitution: f64,
    pub rng_seed: u64,
    pub spawn: [f64; 2],
    pub slingshot_velocity: [f64; 2],
    pub enabled_racers: [bool; 5],
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            mass: 1.0,
            damping: 0.45,
            temperature: 1.2,
            dt: 0.02,
            learning_rate: 0.025,
            restitution: 0.6,
            rng_seed: 42,
            spawn: [0.4, 2.4],
            slingshot_velocity: [0.8, -0.6],
            enabled_racers: [true; 5],
        }
    }
}

impl ArenaConfig {
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.mass.is_finite()
            && (MIN_MASS..=MAX_MASS).contains(&self.mass)
            && self.damping.is_finite()
            && (MIN_DAMPING..=MAX_DAMPING).contains(&self.damping)
            && self.temperature.is_finite()
            && (MIN_TEMPERATURE..=MAX_TEMPERATURE).contains(&self.temperature)
            && self.dt.is_finite()
            && (MIN_DT..=MAX_DT).contains(&self.dt)
            && self.learning_rate.is_finite()
            && (LR_MIN..=LR_MAX).contains(&self.learning_rate)
            && self.restitution.is_finite()
            && (0.0..=1.0).contains(&self.restitution)
            && self.spawn.iter().all(|v| v.is_finite() && v.abs() <= 100.0)
            && self
                .slingshot_velocity
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 200.0)
    }

    pub fn clamp_in_place(&mut self) {
        self.mass = clamp_mass(self.mass);
        self.damping = clamp_damping(self.damping);
        self.temperature = clamp_temperature(self.temperature);
        self.dt = clamp_dt(self.dt);
        self.learning_rate = if self.learning_rate.is_finite() {
            self.learning_rate.clamp(LR_MIN, LR_MAX)
        } else {
            0.025
        };
        self.restitution = if self.restitution.is_finite() {
            self.restitution.clamp(0.0, 1.0)
        } else {
            0.6
        };
        for c in &mut self.spawn {
            *c = if c.is_finite() {
                c.clamp(-100.0, 100.0)
            } else {
                0.0
            };
        }
        for v in &mut self.slingshot_velocity {
            *v = if v.is_finite() {
                v.clamp(-200.0, 200.0)
            } else {
                0.0
            };
        }
    }
}

pub struct TrailRing {
    buf: Box<[[f64; 2]; MAX_TRAIL_POINTS]>,
    start: usize,
    len: usize,
}

impl Default for TrailRing {
    fn default() -> Self {
        Self {
            buf: Box::new([[0.0; 2]; MAX_TRAIL_POINTS]),
            start: 0,
            len: 0,
        }
    }
}

impl TrailRing {
    #[inline]
    pub fn clear(&mut self) {
        (self.start, self.len) = (0, 0);
    }

    #[inline]
    pub fn push(&mut self, pt: [f64; 2]) {
        if self.len < MAX_TRAIL_POINTS {
            self.buf[(self.start + self.len) % MAX_TRAIL_POINTS] = pt;
            self.len += 1;
        } else {
            self.buf[self.start] = pt;
            self.start = (self.start + 1) % MAX_TRAIL_POINTS;
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn iter(&self) -> impl Iterator<Item = [f64; 2]> + '_ {
        (0..self.len).map(move |i| self.buf[(self.start + i) % MAX_TRAIL_POINTS])
    }
}

pub struct RacerTrack {
    pub kind: RacerKind,
    pub phase: PhasePoint2D,
    pub adam_state: AdamState2D<f64>,
    pub kinetic_energy: f64,
    pub potential_energy: f64,
    pub trail: TrailRing,
}

impl RacerTrack {
    fn new(kind: RacerKind) -> Self {
        Self {
            kind,
            phase: PhasePoint2D::default(),
            adam_state: AdamState2D::new(),
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            trail: TrailRing::default(),
        }
    }
}

pub struct ParticleArena {
    pub racers: [RacerTrack; 5],
    pub step_count: u64,
    rng: OxidizeRng,
    adam_proto: Adam<f64, ()>,
    #[cfg(feature = "physics")]
    lorenz: LorenzSystem,
}

impl Default for ParticleArena {
    fn default() -> Self {
        Self {
            racers: RacerKind::ALL.map(RacerTrack::new),
            step_count: 0,
            rng: OxidizeRng::new(42),
            adam_proto: Adam::new(0.025).expect("valid default learning rate"),
            #[cfg(feature = "physics")]
            lorenz: LorenzBuilder::new().build(LorenzState::new(10.0, 10.0, 10.0)),
        }
    }
}

impl ParticleArena {
    pub fn reset(
        &mut self,
        cfg: &ArenaConfig,
        surface: LandscapeSurface,
        bumps: &[SculptBump],
        bounds: (f64, f64, f64, f64),
    ) {
        let mass = clamp_mass(cfg.mass);
        let quad_h = QuadraticHamiltonian::new(mass);
        let q0 = Vector2::new(
            cfg.spawn[0].clamp(bounds.0, bounds.1),
            cfg.spawn[1].clamp(bounds.2, bounds.3),
        );
        let p0 = Vector2::new(cfg.slingshot_velocity[0], cfg.slingshot_velocity[1]) * mass;
        let pot0 = surface.evaluate(bumps, q0.x, q0.y);
        let kin0 = quad_h.evaluate(p0.x) + quad_h.evaluate(p0.y);

        self.step_count = 0;
        self.rng = OxidizeRng::new(cfg.rng_seed);
        if let Ok(adam) = Adam::new(cfg.learning_rate.clamp(LR_MIN, LR_MAX)) {
            self.adam_proto = adam;
        }
        #[cfg(feature = "physics")]
        {
            self.lorenz = LorenzBuilder::new()
                .dt(clamp_dt(cfg.dt))
                .build(LorenzState::new(10.0, 10.0, 10.0));
        }

        for racer in &mut self.racers {
            let first_order = matches!(racer.kind, RacerKind::Sgd | RacerKind::Adam);
            racer.phase = if first_order {
                PhasePoint2D::at_rest(q0)
            } else {
                PhasePoint2D::new(q0, p0)
            };
            racer.adam_state = AdamState2D::new();
            racer.potential_energy = pot0;
            racer.kinetic_energy = if first_order { 0.0 } else { kin0 };
            racer.trail.clear();
            racer.trail.push([q0.x, q0.y]);
        }
    }

    pub fn launch_slingshot(
        &mut self,
        cfg: &mut ArenaConfig,
        origin: [f64; 2],
        velocity: [f64; 2],
        surface: LandscapeSurface,
        bumps: &[SculptBump],
        bounds: (f64, f64, f64, f64),
    ) {
        cfg.spawn = [
            (origin[0].clamp(bounds.0, bounds.1) * 10_000.0).round() / 10_000.0,
            (origin[1].clamp(bounds.2, bounds.3) * 10_000.0).round() / 10_000.0,
        ];
        cfg.slingshot_velocity = [
            (velocity[0].clamp(-80.0, 80.0) * 10_000.0).round() / 10_000.0,
            (velocity[1].clamp(-80.0, 80.0) * 10_000.0).round() / 10_000.0,
        ];
        self.reset(cfg, surface, bumps, bounds);
    }

    pub fn step_once(
        &mut self,
        cfg: &ArenaConfig,
        surface: LandscapeSurface,
        bumps: &[SculptBump],
        bounds: (f64, f64, f64, f64),
    ) {
        let (mass, damping) = (clamp_mass(cfg.mass), clamp_damping(cfg.damping));
        let (temperature, dt) = (clamp_temperature(cfg.temperature), clamp_dt(cfg.dt));
        let lr = cfg.learning_rate.clamp(LR_MIN, LR_MAX);
        let restitution = cfg.restitution.clamp(0.0, 1.0);
        let (q_min, q_max) = (
            Vector2::new(bounds.0, bounds.2),
            Vector2::new(bounds.1, bounds.3),
        );

        let quad_h = QuadraticHamiltonian::new(mass);
        let mut sgd = SGD::new(lr);
        self.adam_proto.learning_rate = lr;
        let leapfrog = HamiltonianLeapfrog2D::new(mass, damping);
        let baoab = LangevinBaoab2D::new(mass, damping, temperature);

        #[cfg(feature = "physics")]
        let chaotic_drive = if cfg.enabled_racers[4] {
            self.lorenz.step(dt);
            let lv = self.lorenz.state.vec;
            Vector2::new(lv.x * 0.18, lv.y * 0.14)
        } else {
            Vector2::zeros()
        };

        let mut solvers = StepSolvers {
            sgd: &mut sgd,
            adam: &mut self.adam_proto,
            leapfrog: &leapfrog,
            baoab: &baoab,
            quad_h: &quad_h,
            rng: &mut self.rng,
        };
        let scalars = StepScalars {
            surface,
            bumps,
            dt,
            lr,
            mass,
            restitution,
            q_min,
            q_max,
            #[cfg(feature = "physics")]
            chaotic_drive,
        };

        for (idx, racer) in self.racers.iter_mut().enumerate() {
            if !cfg.enabled_racers[idx] || !racer.kind.is_available_in_build() {
                continue;
            }
            step_racer(racer, &mut solvers, &scalars);
            racer.potential_energy = surface.evaluate(bumps, racer.phase.q.x, racer.phase.q.y);
            racer.trail.push([racer.phase.q.x, racer.phase.q.y]);
        }
        self.step_count += 1;
    }
}

struct StepSolvers<'a> {
    sgd: &'a mut SGD<f64>,
    adam: &'a mut Adam<f64, ()>,
    leapfrog: &'a HamiltonianLeapfrog2D,
    baoab: &'a LangevinBaoab2D,
    quad_h: &'a QuadraticHamiltonian,
    rng: &'a mut OxidizeRng,
}

struct StepScalars<'a> {
    surface: LandscapeSurface,
    bumps: &'a [SculptBump],
    dt: f64,
    lr: f64,
    mass: f64,
    restitution: f64,
    q_min: Vector2<f64>,
    q_max: Vector2<f64>,
    #[cfg(feature = "physics")]
    chaotic_drive: Vector2<f64>,
}

fn step_racer(racer: &mut RacerTrack, s: &mut StepSolvers<'_>, sc: &StepScalars<'_>) {
    let mut grad_fn = |q: Vector2<f64>| sc.surface.gradient(sc.bumps, q.x, q.y);
    match racer.kind {
        RacerKind::Sgd => {
            let prev = racer.phase.q;
            let mut g = grad_fn(prev);
            let _ = s.sgd.step(&mut racer.phase.q, &mut g, s.rng, sc.lr);
            racer.phase.p = (racer.phase.q - prev) * (sc.mass / sc.dt);
            racer
                .phase
                .reflect_bounds(sc.q_min, sc.q_max, sc.restitution);
            racer.kinetic_energy = 0.0;
        }
        RacerKind::Adam => {
            let g = grad_fn(racer.phase.q);
            let mut aux = (g, &mut racer.adam_state);
            let _ = s.adam.step(&mut racer.phase.q, &mut aux, s.rng, sc.lr);
            let raw_q = racer.phase.q;
            racer
                .phase
                .reflect_bounds(sc.q_min, sc.q_max, sc.restitution);
            if (racer.phase.q.x - raw_q.x).abs() > 1e-12 {
                racer.adam_state.m.x *= -sc.restitution;
            }
            if (racer.phase.q.y - raw_q.y).abs() > 1e-12 {
                racer.adam_state.m.y *= -sc.restitution;
            }
            racer.kinetic_energy = 0.0;
        }
        RacerKind::HamiltonianLeapfrog | RacerKind::LangevinThermal | RacerKind::LorenzChaotic => {
            step_momentum_racer(racer, s, sc, &mut grad_fn)
        }
    }
}

fn step_momentum_racer(
    racer: &mut RacerTrack,
    s: &mut StepSolvers<'_>,
    sc: &StepScalars<'_>,
    grad_fn: &mut impl FnMut(Vector2<f64>) -> Vector2<f64>,
) {
    let mut kinetic_deriv = |p_axis: f64| s.quad_h.derivative(p_axis);
    match racer.kind {
        RacerKind::HamiltonianLeapfrog => {
            let _ =
                s.leapfrog
                    .step_with_kinetic(&mut racer.phase, grad_fn, &mut kinetic_deriv, sc.dt);
        }
        RacerKind::LangevinThermal => {
            let _ = s.baoab.step_with_kinetic(
                &mut racer.phase,
                grad_fn,
                &mut kinetic_deriv,
                s.rng,
                sc.dt,
            );
        }
        RacerKind::LorenzChaotic => {
            #[cfg(feature = "physics")]
            {
                let mut forced = |q: Vector2<f64>| grad_fn(q) - sc.chaotic_drive;
                let _ = s.leapfrog.step_with_kinetic(
                    &mut racer.phase,
                    &mut forced,
                    &mut kinetic_deriv,
                    sc.dt,
                );
            }
            #[cfg(not(feature = "physics"))]
            return;
        }
        RacerKind::Sgd | RacerKind::Adam => return,
    }
    racer
        .phase
        .reflect_bounds(sc.q_min, sc.q_max, sc.restitution);
    racer.kinetic_energy = s.quad_h.evaluate(racer.phase.p.x) + s.quad_h.evaluate(racer.phase.p.y);
}
