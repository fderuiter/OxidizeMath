//! Exact analytical geodesic billiards, circle-inversion reflections, and Lyapunov divergence.

use super::billiards_kernel::propagate_particle;
use super::{
    GeodesicArc, HyperbolicError, MAX_BILLIARD_HISTORY, MAX_POLYGON_SIDES, Su11Matrix,
    clamp_to_disk, poincare_distance, regular_polygon_inradius, regular_polygon_poincare_radius,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Reflects `z` across the orthogonal Euclidean circle `(circle_center, circle_radius)` via
/// $\mathcal{I}(z) = C + \frac{R^2}{\overline{z - C}}$, preserving the Poincaré disk metric.
#[must_use]
#[verified_engine::verified]
pub fn circle_inversion(z: Complex64, circle_center: Complex64, circle_radius: f64) -> Complex64 {
    let zc = clamp_to_disk(z);
    let diff = zc - circle_center;
    let norm_sqr = diff.norm_sqr();
    if !norm_sqr.is_finite() || norm_sqr <= 1e-30 || circle_radius <= 1e-15 {
        return zc;
    }
    let r2 = circle_radius * circle_radius;
    clamp_to_disk(circle_center + (r2 / diff.conj()))
}

/// Instantaneous hyperbolic billiard state `(pos, dir)` with `|pos| < 1` and `|dir| = 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilliardParticle {
    /// Position inside the Poincaré disk $\mathbb{D}$.
    pub pos: Complex64,
    /// Unit Euclidean/conformal velocity direction at `pos`.
    pub dir: Complex64,
}

/// Geodesic arc wall bounding one side of a regular $\{p, q\}$ hyperbolic billiard table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilliardWall {
    /// Counter-clockwise start vertex in $\mathbb{D}$.
    pub v0: Complex64,
    /// Counter-clockwise end vertex in $\mathbb{D}$.
    pub v1: Complex64,
    /// Orthogonal circle representation of the geodesic wall.
    pub arc: GeodesicArc,
}

impl Default for BilliardWall {
    #[verified_engine::verified]
    fn default() -> Self {
        let z = Complex64::new(0.0, 0.0);
        Self {
            v0: z,
            v1: z,
            arc: GeodesicArc::through_points(z, z),
        }
    }
}

/// Recorded divergence measurement between the reference and $\epsilon$-perturbed billiard trajectories.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LyapunovSample {
    /// Cumulative intrinsic hyperbolic arc length $s \ge 0$.
    pub path_length: f64,
    /// Current intrinsic geodesic separation $d_{\mathbb{D}}(z_1(s), z_2(s))$.
    pub separation: f64,
    /// Normalized logarithmic separation $\ln(d_{\mathbb{D}}(s) / \epsilon_0)$.
    pub log_ratio: f64,
    /// Finite-time Lyapunov exponent $\lambda(s) = \frac{1}{s}\ln(d_{\mathbb{D}}(s) / \epsilon_0)$.
    pub ftle: f64,
}

/// Configuration parameters for a twin-particle $\{p, q\}$ hyperbolic geodesic billiard system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperbolicBilliardConfig {
    /// Polygon side count $p$.
    pub p: usize,
    /// Tessellation vertex valence $q$ satisfying $(p - 2)(q - 2) > 4$.
    pub q: usize,
    /// Initial launch position inside the central $\{p, q\}$ table.
    pub initial_pos: Complex64,
    /// Initial launch angle in radians.
    pub initial_angle: f64,
    /// Initial hyperbolic twin-particle perturbation $\epsilon \in [10^{-10}, 10^{-1}]$.
    pub epsilon: f64,
}

impl Default for HyperbolicBilliardConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            p: 7,
            q: 3,
            initial_pos: Complex64::new(0.08, -0.05),
            initial_angle: 0.43,
            epsilon: 1e-5,
        }
    }
}

/// Pre-allocated twin-particle geodesic billiard engine inside a regular $\{p, q\}$ hyperbolic polygon.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::differential_geometry::hyperbolic::{
///     HyperbolicBilliardConfig, HyperbolicBilliardSystem,
/// };
///
/// let mut sys = HyperbolicBilliardSystem::new(HyperbolicBilliardConfig::default())
///     .expect("valid {7, 3} billiard table");
/// for _ in 0..80 {
///     sys.step(0.05);
/// }
/// assert!(sys.primary_bounces() > 0);
/// assert!(sys.latest_sample().log_ratio > 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct HyperbolicBilliardSystem {
    config: HyperbolicBilliardConfig,
    walls: [BilliardWall; MAX_POLYGON_SIDES],
    wall_count: usize,
    inradius_poincare: f64,
    primary: BilliardParticle,
    twin: BilliardParticle,
    primary_last_wall: Option<usize>,
    twin_last_wall: Option<usize>,
    effective_epsilon: f64,
    path_length: f64,
    primary_bounces: u64,
    twin_bounces: u64,
    history_cap: usize,
    history_head: usize,
    history_len: usize,
    primary_trail: Vec<Complex64>,
    twin_trail: Vec<Complex64>,
    samples: Vec<LyapunovSample>,
}

impl HyperbolicBilliardSystem {
    /// Creates a pre-allocated billiard system with default [`MAX_BILLIARD_HISTORY`] ring capacity.
    pub fn new(config: HyperbolicBilliardConfig) -> Result<Self, HyperbolicError> {
        Self::with_history_capacity(config, MAX_BILLIARD_HISTORY)
    }

    /// Creates a pre-allocated billiard system with explicit ring-buffer history capacity.
    pub fn with_history_capacity(
        config: HyperbolicBilliardConfig,
        history_capacity: usize,
    ) -> Result<Self, HyperbolicError> {
        let cap = history_capacity.clamp(16, MAX_BILLIARD_HISTORY);
        let z = Complex64::new(0.0, 0.0);
        let mut sys = Self {
            config,
            walls: [BilliardWall::default(); MAX_POLYGON_SIDES],
            wall_count: 0,
            inradius_poincare: 0.2,
            primary: BilliardParticle {
                pos: z,
                dir: Complex64::new(1.0, 0.0),
            },
            twin: BilliardParticle {
                pos: z,
                dir: Complex64::new(1.0, 0.0),
            },
            primary_last_wall: None,
            twin_last_wall: None,
            effective_epsilon: 1e-5,
            path_length: 0.0,
            primary_bounces: 0,
            twin_bounces: 0,
            history_cap: cap,
            history_head: 0,
            history_len: 0,
            primary_trail: vec![z; cap],
            twin_trail: vec![z; cap],
            samples: vec![LyapunovSample::default(); cap],
        };
        sys.reconfigure(config)?;
        Ok(sys)
    }

    /// Rebuilds the $\{p, q\}$ walls and resets both trajectories without reallocating ring buffers.
    #[verified_engine::verified]
    pub fn reconfigure(&mut self, config: HyperbolicBilliardConfig) -> Result<(), HyperbolicError> {
        if !config.epsilon.is_finite() || !(1e-12..=0.25).contains(&config.epsilon) {
            return Err(HyperbolicError::InvalidBilliardConfig {
                reason: "epsilon must be finite and within [1e-12, 0.25]",
            });
        }
        if !config.initial_angle.is_finite()
            || !config.initial_pos.re.is_finite()
            || !config.initial_pos.im.is_finite()
        {
            return Err(HyperbolicError::InvalidBilliardConfig {
                reason: "initial position and angle must be finite",
            });
        }
        let r0 = regular_polygon_poincare_radius(config.p, config.q)?;
        let h = regular_polygon_inradius(config.p, config.q)?;
        self.inradius_poincare = (0.5 * h).tanh();
        self.build_polygon_walls(config.p, r0);
        self.config = config;
        self.reset_particles();
        Ok(())
    }

    #[verified_engine::verified]
    fn build_polygon_walls(&mut self, p: usize, r0: f64) {
        self.wall_count = p;
        let mut k = 0;
        while k < p {
            let a0 = 2.0 * PI * (k as f64) / (p as f64);
            let a1 = 2.0 * PI * (((k + 1) % p) as f64) / (p as f64);
            let v0 = clamp_to_disk(Complex64::new(r0 * a0.cos(), r0 * a0.sin()));
            let v1 = clamp_to_disk(Complex64::new(r0 * a1.cos(), r0 * a1.sin()));
            self.walls[k] = BilliardWall {
                v0,
                v1,
                arc: GeodesicArc::through_points(v0, v1),
            };
            k += 1;
        }
    }

    /// Resets the twin particles to the configured launch state without heap allocations.
    #[verified_engine::verified]
    pub fn reset_particles(&mut self) {
        let safe_max = 0.88 * self.inradius_poincare;
        let mut z0 = self.config.initial_pos;
        if z0.norm() > safe_max {
            z0 *= safe_max / z0.norm();
        }
        let theta = self.config.initial_angle;
        let rot = Complex64::new(theta.cos(), theta.sin());
        let eps = self.config.epsilon;
        let denom = 4.0 + eps * eps;
        let w_local = Complex64::new(-eps * eps / denom, (2.0 * eps) / denom) * rot;
        let u_local = Complex64::new((4.0 - eps * eps) / denom, (4.0 * eps) / denom) * rot;

        let boost = Su11Matrix::translation_from_origin(z0);
        let origin = Complex64::new(0.0, 0.0);
        self.primary = BilliardParticle {
            pos: z0,
            dir: boost.apply_direction(origin, rot),
        };
        self.twin = BilliardParticle {
            pos: boost.apply(w_local),
            dir: boost.apply_direction(w_local, u_local),
        };
        self.primary_last_wall = None;
        self.twin_last_wall = None;
        self.path_length = 0.0;
        self.primary_bounces = 0;
        self.twin_bounces = 0;
        self.effective_epsilon = poincare_distance(self.primary.pos, self.twin.pos).max(1e-15);
        self.history_head = 0;
        self.history_len = 0;
        self.record_sample();
    }

    /// Advances both trajectories by intrinsic hyperbolic arc-length `ds` with exact reflections.
    #[verified_engine::verified]
    pub fn step(&mut self, ds: f64) {
        if !ds.is_finite() || ds <= 0.0 {
            return;
        }
        let step_len = ds.min(0.5);
        let (p1, w1, b1) = propagate_particle(
            &self.walls[..self.wall_count],
            self.primary,
            self.primary_last_wall,
            step_len,
        );
        let (p2, w2, b2) = propagate_particle(
            &self.walls[..self.wall_count],
            self.twin,
            self.twin_last_wall,
            step_len,
        );
        self.primary = p1;
        self.twin = p2;
        self.primary_last_wall = w1;
        self.twin_last_wall = w2;
        self.primary_bounces = self.primary_bounces.saturating_add(b1);
        self.twin_bounces = self.twin_bounces.saturating_add(b2);
        self.path_length += step_len;
        self.record_sample();
    }

    #[verified_engine::verified]
    fn record_sample(&mut self) {
        let sep = poincare_distance(self.primary.pos, self.twin.pos).max(1e-16);
        let log_ratio = (sep / self.effective_epsilon).ln();
        let ftle = if self.path_length > 1e-9 {
            log_ratio / self.path_length
        } else {
            1.0
        };
        let slot = if self.history_len < self.history_cap {
            let idx = self.history_len;
            self.history_len += 1;
            idx
        } else {
            let idx = self.history_head;
            self.history_head = (self.history_head + 1) % self.history_cap;
            idx
        };
        self.primary_trail[slot] = self.primary.pos;
        self.twin_trail[slot] = self.twin.pos;
        self.samples[slot] = LyapunovSample {
            path_length: self.path_length,
            separation: sep,
            log_ratio,
            ftle,
        };
    }

    /// Active geodesic walls of the $\{p, q\}$ billiard table.
    #[must_use]
    pub fn walls(&self) -> &[BilliardWall] {
        &self.walls[..self.wall_count]
    }

    /// Current configuration.
    #[must_use]
    #[verified_engine::verified]
    pub fn config(&self) -> HyperbolicBilliardConfig {
        self.config
    }

    /// Poincaré-disk inradius of the central $\{p, q\}$ table.
    #[must_use]
    #[verified_engine::verified]
    pub fn inradius_poincare(&self) -> f64 {
        self.inradius_poincare
    }

    /// Current primary particle state.
    #[must_use]
    #[verified_engine::verified]
    pub fn primary(&self) -> BilliardParticle {
        self.primary
    }

    /// Current perturbed twin particle state.
    #[must_use]
    #[verified_engine::verified]
    pub fn twin(&self) -> BilliardParticle {
        self.twin
    }

    /// Total wall reflections undergone by the primary particle.
    #[must_use]
    #[verified_engine::verified]
    pub fn primary_bounces(&self) -> u64 {
        self.primary_bounces
    }

    /// Number of recorded trajectory/Lyapunov samples currently stored in the ring buffer.
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_count(&self) -> usize {
        self.history_len
    }

    /// Returns the chronological `index`-th sample (`0..sample_count()`) without allocating.
    #[must_use]
    #[verified_engine::verified]
    pub fn sample_at(&self, index: usize) -> Option<(Complex64, Complex64, LyapunovSample)> {
        if index >= self.history_len {
            return None;
        }
        let phys = (self.history_head + index) % self.history_cap;
        Some((
            self.primary_trail[phys],
            self.twin_trail[phys],
            self.samples[phys],
        ))
    }

    /// Returns the most recently recorded [`LyapunovSample`].
    #[must_use]
    #[verified_engine::verified]
    pub fn latest_sample(&self) -> LyapunovSample {
        if self.history_len == 0 {
            return LyapunovSample::default();
        }
        let phys = (self.history_head + self.history_len - 1) % self.history_cap;
        self.samples[phys]
    }
}

// [cite:modular_polynomials_review]
