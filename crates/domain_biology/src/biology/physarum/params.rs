//! Chemotactic sensory parameters, nutrient terminal descriptors, and simulation telemetry.

use verified_engine::Theory;

/// Minimum sensor angle $\theta_s$ in radians.
pub const MIN_SENSOR_ANGLE: f64 = 0.10;
/// Maximum sensor angle $\theta_s$ in radians.
pub const MAX_SENSOR_ANGLE: f64 = 1.50;
/// Minimum agent turn angle $\theta_r$ in radians.
pub const MIN_TURN_ANGLE: f64 = 0.05;
/// Maximum agent turn angle $\theta_r$ in radians.
pub const MAX_TURN_ANGLE: f64 = 1.50;
/// Minimum forward sensor offset distance $d_s$ in grid cells.
pub const MIN_SENSOR_DISTANCE: f64 = 2.0;
/// Maximum forward sensor offset distance $d_s$ in grid cells.
pub const MAX_SENSOR_DISTANCE: f64 = 32.0;
/// Minimum forward propulsion step $v$ in grid cells per step.
pub const MIN_STEP_SIZE: f64 = 0.20;
/// Maximum forward propulsion step $v$ in grid cells per step.
pub const MAX_STEP_SIZE: f64 = 3.50;

#[verified_engine::verified]
fn clamp_finite(value: f64, min: f64, max: f64, fallback: f64) -> f64 {
    debug_assert!(min <= max);
    debug_assert!(fallback >= min && fallback <= max);
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

/// Governing sensory, motor, diffusion, and decay parameters for [`super::PhysarumEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
#[theory(
    description = "Lagrangian-Eulerian Physarum polycephalum chemotactic sensory-motor and adaptive conductance pruning model.",
    citation = "Jones, J. (2010). Characteristics of pattern formation and evolution in approximations of Physarum polycephalum. Artificial Life, 16(2), 127-153; Tero, A. et al. (2010). Rules for Biologically Inspired Adaptive Network Design. Science, 327(5964), 439-442.",
    maturity = "Validated"
)]
pub struct PhysarumParams {
    /// Angular offset $\theta_s$ of left/right chemoattractant sensors (`0.10..=1.50` rad).
    #[theory(min = 0.10, max = 1.50, step = 0.02)]
    pub sensor_angle: f64,
    /// Heading rotation increment $\theta_r$ per step (`0.05..=1.50` rad).
    #[theory(min = 0.05, max = 1.50, step = 0.02)]
    pub turn_angle: f64,
    /// Sensor lookahead distance $d_s$ in grid cells (`2.0..=32.0`).
    #[theory(min = 2.0, max = 32.0, step = 0.5)]
    pub sensor_distance: f64,
    /// Forward translation speed $v$ in grid cells per step (`0.20..=3.50`).
    #[theory(min = 0.20, max = 3.50, step = 0.05)]
    pub step_size: f64,
    /// Chemoattractant mass deposited per agent step (`0.05..=5.00`).
    #[theory(min = 0.05, max = 5.00, step = 0.05)]
    pub deposition: f64,
    /// Isotropic $3\times 3$ diffusion blend factor $\alpha \in [0, 1]$.
    #[theory(min = 0.0, max = 1.0, step = 0.02)]
    pub diffusion_rate: f64,
    /// Linear trail evaporation rate $\gamma \in [0.005, 0.50]$.
    #[theory(min = 0.005, max = 0.50, step = 0.005)]
    pub decay_rate: f64,
    /// Nonlinear sub-threshold filament pruning rate $\gamma_{\text{prune}} \in [0.0, 0.35]$.
    #[theory(min = 0.0, max = 0.35, step = 0.005)]
    pub prune_rate: f64,
    /// Chemoattractant emission amplitude injected at nutrient beacons (`0.0..=5.0`).
    #[theory(min = 0.0, max = 5.0, step = 0.05)]
    pub beacon_strength: f64,
}

impl PhysarumParams {
    /// Sanitizes and clamps all fields of `self` to finite domain-safe intervals.
    #[verified_engine::verified]
    pub fn sanitized(self) -> Self {
        let out = Self {
            sensor_angle: clamp_finite(self.sensor_angle, MIN_SENSOR_ANGLE, MAX_SENSOR_ANGLE, 0.78),
            turn_angle: clamp_finite(self.turn_angle, MIN_TURN_ANGLE, MAX_TURN_ANGLE, 0.65),
            sensor_distance: clamp_finite(
                self.sensor_distance,
                MIN_SENSOR_DISTANCE,
                MAX_SENSOR_DISTANCE,
                9.0,
            ),
            step_size: clamp_finite(self.step_size, MIN_STEP_SIZE, MAX_STEP_SIZE, 1.15),
            deposition: clamp_finite(self.deposition, 0.05, 5.00, 0.65),
            diffusion_rate: clamp_finite(self.diffusion_rate, 0.0, 1.0, 0.55),
            decay_rate: clamp_finite(self.decay_rate, 0.005, 0.50, 0.07),
            prune_rate: clamp_finite(self.prune_rate, 0.0, 0.35, 0.05),
            beacon_strength: clamp_finite(self.beacon_strength, 0.0, 5.0, 1.20),
        };
        debug_assert!(out.is_within_bounds());
        assert!(out.step_size > 0.0);
        out
    }

    /// Returns `true` when every parameter is finite and lies inside its safe simulation interval.
    #[verified_engine::verified]
    pub fn is_within_bounds(&self) -> bool {
        let angles_ok = self.sensor_angle.is_finite()
            && (MIN_SENSOR_ANGLE..=MAX_SENSOR_ANGLE).contains(&self.sensor_angle)
            && self.turn_angle.is_finite()
            && (MIN_TURN_ANGLE..=MAX_TURN_ANGLE).contains(&self.turn_angle);
        let motion_ok = self.sensor_distance.is_finite()
            && (MIN_SENSOR_DISTANCE..=MAX_SENSOR_DISTANCE).contains(&self.sensor_distance)
            && self.step_size.is_finite()
            && (MIN_STEP_SIZE..=MAX_STEP_SIZE).contains(&self.step_size);
        let field_ok = self.deposition.is_finite()
            && (0.05..=5.00).contains(&self.deposition)
            && self.diffusion_rate.is_finite()
            && (0.0..=1.0).contains(&self.diffusion_rate)
            && self.decay_rate.is_finite()
            && (0.005..=0.50).contains(&self.decay_rate)
            && self.prune_rate.is_finite()
            && (0.0..=0.35).contains(&self.prune_rate)
            && self.beacon_strength.is_finite()
            && (0.0..=5.0).contains(&self.beacon_strength);
        debug_assert!(!angles_ok || self.sensor_angle > 0.0);
        debug_assert!(!motion_ok || self.sensor_distance >= MIN_SENSOR_DISTANCE);
        angles_ok && motion_ok && field_ok
    }
}

impl Default for PhysarumParams {
    #[verified_engine::verified]
    fn default() -> Self {
        let out = Self {
            sensor_angle: 0.78,
            turn_angle: 0.65,
            sensor_distance: 9.0,
            step_size: 1.15,
            deposition: 0.65,
            diffusion_rate: 0.55,
            decay_rate: 0.07,
            prune_rate: 0.05,
            beacon_strength: 1.20,
        };
        debug_assert!(out.is_within_bounds());
        assert!(out.deposition > 0.0);
        out
    }
}

/// Nutrient source / urban station beacon anchoring chemotactic slime-mold arteries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysarumTerminal {
    /// Horizontal grid coordinate $x \in [0, W)$.
    pub x: f64,
    /// Vertical grid coordinate $y \in [0, H)$.
    pub y: f64,
    /// Effective feeding and chemoattractant injection radius in grid cells (`1.5..=8.0`).
    pub radius: f64,
    /// Relative chemoattractant emission weight (`0.1..=4.0`).
    pub weight: f64,
}

impl PhysarumTerminal {
    /// Creates a nutrient terminal clamped to valid interior coordinates on a `width x height` grid.
    #[verified_engine::verified]
    pub fn clamped(x: f64, y: f64, radius: f64, weight: f64, width: usize, height: usize) -> Self {
        let max_x = (width.saturating_sub(3)).max(2) as f64;
        let max_y = (height.saturating_sub(3)).max(2) as f64;
        let out = Self {
            x: clamp_finite(x, 2.0, max_x, 0.5 * max_x),
            y: clamp_finite(y, 2.0, max_y, 0.5 * max_y),
            radius: clamp_finite(radius, 1.5, 8.0, 3.0),
            weight: clamp_finite(weight, 0.1, 4.0, 1.0),
        };
        debug_assert!(out.x.is_finite() && out.y.is_finite());
        assert!(out.radius >= 1.5);
        out
    }
}

impl Default for PhysarumTerminal {
    #[verified_engine::verified]
    fn default() -> Self {
        let out = Self {
            x: 32.0,
            y: 32.0,
            radius: 3.0,
            weight: 1.0,
        };
        debug_assert!(out.radius > 0.0);
        assert!(out.weight > 0.0);
        out
    }
}

/// Live Eulerian-Lagrangian scalar telemetry snapshot updated in place with zero allocations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PhysarumTelemetry {
    /// Integrated chemoattractant trail mass $\sum_{x, y} T(x, y)$ across open cells.
    pub total_trail_mass: f64,
    /// Count of open cells whose trail concentration exceeds the active tube threshold.
    pub active_cells: f64,
    /// Fraction of agents that advanced along open corridors during the latest step.
    pub mean_flux: f64,
    /// Maximum trail concentration $\max_{x, y} T(x, y)$ across the grid.
    pub peak_trail: f64,
    /// Cumulative simulated time.
    pub time: f64,
    /// Total discrete Lagrangian-Eulerian steps executed.
    pub step_count: u64,
}
