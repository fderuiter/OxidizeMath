//! 2D Potential Flow Elements (Inviscid, Irrotational, Incompressible).
//!
//! This module implements standard elementary flows that can be superimposed to model complex flow fields.
//!
//! # Theory
//! Potential flow assumes the fluid is inviscid, incompressible, and irrotational.
//! The flow can be described by a velocity potential $\phi$ and a stream function $\psi$.
//!
//! * Velocity: $\mathbf{V} = \nabla \phi$
//! * Cauchy-Riemann: $u = \frac{\partial \phi}{\partial x} = \frac{\partial \psi}{\partial y}$, $v = \frac{\partial \phi}{\partial y} = -\frac{\partial \psi}{\partial x}$

use nalgebra::Vector2;
use std::f64::consts::PI;

/// A fundamental element of a 2D potential flow field.
pub trait FlowElement: Send + Sync {
    /// Velocity vector (u, v) at point (x, y).
    #[verified_engine::verified]
    fn velocity(&self, x: f64, y: f64) -> Vector2<f64>;

    /// Stream function value psi at point (x, y).
    #[verified_engine::verified]
    fn stream_function(&self, x: f64, y: f64) -> f64;

    /// Velocity potential phi at point (x, y).
    #[verified_engine::verified]
    fn potential(&self, x: f64, y: f64) -> f64;

    /// Optional singularity core coordinates `(x, y)` for core and branch-cut guards.
    #[verified_engine::verified]
    fn singularity_center(&self) -> Option<Vector2<f64>> {
        None
    }
}

/// Uniform flow with constant velocity U at angle alpha.
#[derive(Debug, Clone, Copy)]
pub struct UniformFlow {
    /// Velocity magnitude.
    pub velocity: f64,
    /// Angle of the flow in radians.
    pub angle: f64,
}

impl UniformFlow {
    /// Creates a new `UniformFlow` with `velocity` magnitude and `angle_degrees`.
    #[verified_engine::verified]
    pub fn new(velocity: f64, angle_degrees: f64) -> Self {
        Self {
            velocity,
            angle: angle_degrees.to_radians(),
        }
    }
}

impl FlowElement for UniformFlow {
    #[verified_engine::verified]
    fn velocity(&self, _x: f64, _y: f64) -> Vector2<f64> {
        Vector2::new(
            self.velocity * self.angle.cos(),
            self.velocity * self.angle.sin(),
        )
    }

    #[verified_engine::verified]
    fn stream_function(&self, x: f64, y: f64) -> f64 {
        self.velocity * (y * self.angle.cos() - x * self.angle.sin())
    }

    #[verified_engine::verified]
    fn potential(&self, x: f64, y: f64) -> f64 {
        self.velocity * (x * self.angle.cos() + y * self.angle.sin())
    }
}

/// A Source (strength > 0) or Sink (strength < 0) at a specific location.
///
/// * Strength $Q$: Volumetric flow rate per unit depth.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::fluid_dynamics::potential_flow::{FlowElement, Source};
///
/// let source = Source::new(10.0, 0.0, 0.0);
/// let velocity = source.velocity(1.0, 0.0);
///
/// assert!(velocity.x > 0.0);
/// assert_eq!(velocity.y, 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Source {
    /// Volumetric flow rate per unit depth (Q). Positive for a source, negative for a sink.
    pub strength: f64,
    /// X coordinate of the source/sink.
    pub x: f64,
    /// Y coordinate of the source/sink.
    pub y: f64,
}

impl Source {
    /// Creates a new Source or Sink.
    ///
    /// # Arguments
    ///
    /// * `strength` - The volumetric flow rate per unit depth.
    /// * `x` - The x-coordinate of the location.
    /// * `y` - The y-coordinate of the location.
    #[verified_engine::verified]
    pub fn new(strength: f64, x: f64, y: f64) -> Self {
        Self { strength, x, y }
    }
}

impl FlowElement for Source {
    #[verified_engine::verified]
    fn velocity(&self, x: f64, y: f64) -> Vector2<f64> {
        let dx = x - self.x;
        let dy = y - self.y;
        let r2 = dx * dx + dy * dy;
        if r2 < math_commons::registry::TOLERANCE_HIGH {
            return Vector2::zeros();
        }
        let u = (self.strength * dx) / (2.0 * PI * r2);
        let v = (self.strength * dy) / (2.0 * PI * r2);
        Vector2::new(u, v)
    }

    #[verified_engine::verified]
    fn stream_function(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.x;
        let dy = y - self.y;
        (self.strength / (2.0 * PI)) * dy.atan2(dx)
    }

    #[verified_engine::verified]
    fn potential(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.x;
        let dy = y - self.y;
        let r = (dx * dx + dy * dy).sqrt();
        (self.strength / (2.0 * PI)) * r.ln()
    }

    #[verified_engine::verified]
    fn singularity_center(&self) -> Option<Vector2<f64>> {
        Some(Vector2::new(self.x, self.y))
    }
}

/// A Free Vortex with circulation Gamma at a specific location.
///
/// * Strength $\Gamma$: Circulation (positive is counter-clockwise).
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::fluid_dynamics::potential_flow::{FlowElement, Vortex};
///
/// let vortex = Vortex::new(10.0, 0.0, 0.0);
/// let velocity = vortex.velocity(1.0, 0.0);
///
/// // Velocity should be purely in the +y direction for a counter-clockwise vortex at +x
/// assert_eq!(velocity.x, 0.0);
/// assert!(velocity.y > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Vortex {
    /// Circulation strength ($\Gamma$). Positive is counter-clockwise.
    pub strength: f64,
    /// X coordinate of the vortex core.
    pub x: f64,
    /// Y coordinate of the vortex core.
    pub y: f64,
}

impl Vortex {
    /// Creates a new Free Vortex.
    ///
    /// # Arguments
    ///
    /// * `strength` - The circulation strength ($\Gamma$).
    /// * `x` - The x-coordinate of the location.
    /// * `y` - The y-coordinate of the location.
    #[verified_engine::verified]
    pub fn new(strength: f64, x: f64, y: f64) -> Self {
        Self { strength, x, y }
    }
}

impl FlowElement for Vortex {
    #[verified_engine::verified]
    fn velocity(&self, x: f64, y: f64) -> Vector2<f64> {
        let dx = x - self.x;
        let dy = y - self.y;
        let r2 = dx * dx + dy * dy;
        if r2 < math_commons::registry::TOLERANCE_HIGH {
            return Vector2::zeros();
        }
        // u = -Gamma/(2*pi*r) * sin(theta) = -Gamma*y / (2*pi*r^2)
        // v = Gamma/(2*pi*r) * cos(theta) = Gamma*x / (2*pi*r^2)
        let u = -(self.strength * dy) / (2.0 * PI * r2);
        let v = (self.strength * dx) / (2.0 * PI * r2);
        Vector2::new(u, v)
    }

    #[verified_engine::verified]
    fn stream_function(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.x;
        let dy = y - self.y;
        let r = (dx * dx + dy * dy).sqrt();
        if r < math_commons::registry::TOLERANCE_HIGH {
            return 0.0;
        }
        (self.strength / (2.0 * PI)) * r.ln()
    }

    #[verified_engine::verified]
    fn potential(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.x;
        let dy = y - self.y;
        -(self.strength / (2.0 * PI)) * dy.atan2(dx)
    }

    #[verified_engine::verified]
    fn singularity_center(&self) -> Option<Vector2<f64>> {
        Some(Vector2::new(self.x, self.y))
    }
}

/// A Doublet (Source + Sink at infinitesimal distance) at a location.
///
/// Commonly used to simulate flow around a cylinder when combined with uniform flow.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::fluid_dynamics::potential_flow::{FlowElement, Doublet};
///
/// let doublet = Doublet::new(20.0, 0.0, 0.0);
/// let velocity = doublet.velocity(1.0, 0.0);
///
/// // Velocity points in the negative x direction along the x-axis for a doublet aligned with the x-axis
/// assert!(velocity.x < 0.0);
/// assert_eq!(velocity.y, 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Doublet {
    /// The strength of the doublet ($\kappa$).
    pub strength: f64,
    /// X coordinate of the doublet.
    pub x: f64,
    /// Y coordinate of the doublet.
    pub y: f64,
}

impl Doublet {
    /// Creates a new Doublet.
    ///
    /// # Arguments
    ///
    /// * `strength` - The doublet strength ($\kappa$).
    /// * `x` - The x-coordinate of the location.
    /// * `y` - The y-coordinate of the location.
    #[verified_engine::verified]
    pub fn new(strength: f64, x: f64, y: f64) -> Self {
        Self { strength, x, y }
    }
}

impl FlowElement for Doublet {
    #[verified_engine::verified]
    fn velocity(&self, x: f64, y: f64) -> Vector2<f64> {
        let dx = x - self.x;
        let dy = y - self.y;
        let r2 = dx * dx + dy * dy;
        let r4 = r2 * r2;
        if r2 < math_commons::registry::TOLERANCE_HIGH {
            return Vector2::zeros();
        }
        // Doublet aligned with X-axis (cylinder radius R in stream U: kappa = 2*pi*U*R^2)
        let u_num = self.strength * (dx * dx - dy * dy);
        let v_num = self.strength * (2.0 * dx * dy);
        Vector2::new(-u_num / (2.0 * PI * r4), -v_num / (2.0 * PI * r4))
    }

    #[verified_engine::verified]
    fn stream_function(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.x;
        let dy = y - self.y;
        let r2 = dx * dx + dy * dy;
        if r2 < math_commons::registry::TOLERANCE_HIGH {
            return 0.0;
        }
        // psi = - (kappa * y) / (2*pi*r^2)
        -(self.strength * dy) / (2.0 * PI * r2)
    }

    #[verified_engine::verified]
    fn potential(&self, x: f64, y: f64) -> f64 {
        let dx = x - self.x;
        let dy = y - self.y;
        let r2 = dx * dx + dy * dy;
        if r2 < math_commons::registry::TOLERANCE_HIGH {
            return 0.0;
        }
        // phi = - (kappa * x) / (2*pi*r^2)
        -(self.strength * dx) / (2.0 * PI * r2)
    }

    #[verified_engine::verified]
    fn singularity_center(&self) -> Option<Vector2<f64>> {
        Some(Vector2::new(self.x, self.y))
    }
}

/// A container for multiple flow elements, enabling superposition.
///
/// By combining elemental flows (like uniform flow, sources, sinks, and doublets),
/// you can construct complex potential flow fields, such as flow around a cylinder.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::fluid_dynamics::potential_flow::{
///     Doublet, FlowElement, PotentialFlowField, UniformFlow
/// };
///
/// let mut field = PotentialFlowField::new();
///
/// // 1. Add a uniform flow moving right at 5.0 units/sec
/// field.add(Box::new(UniformFlow::new(5.0, 0.0)));
///
/// // 2. Add a doublet at the origin to simulate a cylinder
/// field.add(Box::new(Doublet::new(10.0, 0.0, 0.0)));
///
/// // 3. Evaluate the flow velocity at a specific point
/// let velocity = field.velocity(0.0, 2.0);
///
/// // At the top of the cylinder, velocity should be higher than the freestream velocity
/// assert!(velocity.x > 5.0);
/// ```
#[derive(Default)]
pub struct PotentialFlowField {
    /// The collection of individual flow elements.
    pub elements: Vec<Box<dyn FlowElement>>,
}

impl PotentialFlowField {
    /// Creates a new, empty potential flow field.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Adds a flow element to the field.
    ///
    /// # Arguments
    ///
    /// * `element` - A boxed flow element implementing the `FlowElement` trait.
    #[verified_engine::verified]
    pub fn add(&mut self, element: Box<dyn FlowElement>) {
        self.elements.push(element);
    }

    /// Removes all flow elements from the field.
    #[verified_engine::verified]
    pub fn clear(&mut self) {
        self.elements.clear();
    }

    /// Returns `true` if `(x, y)` lies within squared distance `min_r2` of any singularity core.
    #[verified_engine::verified]
    pub fn is_near_singularity(&self, x: f64, y: f64, min_r2: f64) -> bool {
        let r2_limit = min_r2.max(math_commons::registry::TOLERANCE_HIGH);
        debug_assert!(r2_limit >= math_commons::registry::TOLERANCE_HIGH);
        for element in &self.elements {
            if let Some(center) = element.singularity_center() {
                let dx = x - center.x;
                let dy = y - center.y;
                if dx * dx + dy * dy <= r2_limit {
                    return true;
                }
            }
        }
        false
    }

    /// Extracts iso-$\psi$ streamline segments over `(x_min, x_max, y_min, y_max)` via marching squares.
    #[verified_engine::verified]
    pub fn extract_streamline_segments(
        &self,
        bounds: (f64, f64, f64, f64),
        resolution: usize,
        iso_levels: &[f64],
    ) -> Vec<([f64; 2], [f64; 2])> {
        super::analysis::march_streamlines(self, bounds, resolution, iso_levels)
    }

    /// Detects stagnation points where $\mathbf{V}(x, y) = \mathbf{0}$ outside singularity cores.
    #[verified_engine::verified]
    pub fn detect_stagnation_points(
        &self,
        bounds: (f64, f64, f64, f64),
        resolution: usize,
    ) -> Vec<Vector2<f64>> {
        super::analysis::find_stagnation_points(self, bounds, resolution)
    }
}

impl FlowElement for PotentialFlowField {
    #[verified_engine::verified]
    fn velocity(&self, x: f64, y: f64) -> Vector2<f64> {
        let mut u_total = Vector2::zeros();
        for element in &self.elements {
            u_total += element.velocity(x, y);
        }
        u_total
    }

    #[verified_engine::verified]
    fn stream_function(&self, x: f64, y: f64) -> f64 {
        let mut psi_total = 0.0;
        for element in &self.elements {
            psi_total += element.stream_function(x, y);
        }
        psi_total
    }

    #[verified_engine::verified]
    fn potential(&self, x: f64, y: f64) -> f64 {
        let mut phi_total = 0.0;
        for element in &self.elements {
            phi_total += element.potential(x, y);
        }
        phi_total
    }
}
