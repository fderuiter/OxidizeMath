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
}

/// Uniform flow with constant velocity U at angle alpha.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct UniformFlow {
    /// Velocity magnitude.
    pub velocity: f64,
    /// Angle of the flow in radians.
    pub angle: f64,
}

impl UniformFlow {
    /// Creates a new UniformFlow element.
    ///
    /// # Arguments
    ///
    /// * `velocity` - The velocity magnitude.
    /// * `angle_degrees` - The angle of the flow in degrees.
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
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

        // For doublet aligned with X-axis:
        // u = -kappa * (x^2 - y^2) / (2*pi*r^4)
        // v = -kappa * (2xy) / (2*pi*r^4)
        // Note: Sign convention varies.
        // If phi = -kappa * x / (2*pi*r^2) (doublet pointing +x)
        // Then u = dphi/dx = -kappa/2pi * (1*r^2 - x*2x*x/r)/r^4 ??? No.
        // d(x/r^2)/dx = (1*r^2 - x*2x)/r^4 = (x^2+y^2 - 2x^2)/r^4 = (y^2-x^2)/r^4
        // So u = -kappa/2pi * (y^2-x^2)/r^4 = kappa/2pi * (x^2-y^2)/r^4

        // Wait, standard result for cylinder radius R in stream U:
        // Doublet strength kappa = 2*pi*U*R^2

        let u_num = self.strength * (dx * dx - dy * dy);
        let v_num = self.strength * (2.0 * dx * dy);

        let u = -u_num / (2.0 * PI * r4);
        let v = -v_num / (2.0 * PI * r4);

        Vector2::new(u, v)
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
