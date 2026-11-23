//! # Fluid Dynamics
//!
//! This module implements the fundamental Conservation Laws of Fluid Dynamics:
//! * **Mass**: Continuity Equation.
//! * **Momentum**: Navier-Stokes and Euler Equations.
//!
//! It uses a **Strategy Pattern** to switch between flow regimes (Viscous vs. Inviscid)
//! and analyzing flow characteristics (Laminar vs. Turbulent).
//!
//! ##  Quick Start: Calculating Acceleration
//!
//! Simulate a fluid element (Water) moving through a pipe and calculate the local acceleration
//! due to pressure gradients and viscosity.
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::fluid_dynamics::types::{FluidProperties, FlowState, SpatialGradients};
//! use math_explorer::physics::fluid_dynamics::conservation::{NavierStokes, MomentumEquation};
//! use math_explorer::nalgebra::{Vector3, Matrix3};
//!
//! // 1. Setup Fluid (Water @ 20°C)
//! let water = FluidProperties::water();
//! // Or create a custom fluid (returns Result)
//! let custom = FluidProperties::new(1000.0, 0.001).expect("Invalid fluid properties");
//!
//! // 2. Setup Flow State
//! // Velocity: 10 m/s in X-direction
//! // Pressure: 1 atm (101325 Pa)
//! let velocity = Vector3::new(10.0, 0.0, 0.0);
//! let state = FlowState::new(velocity, 101325.0);
//!
//! // 3. Define Gradients (Steady Laminar Pipe Flow)
//! // - No velocity change in X (fully developed flow): du/dx = 0
//! // - Pressure drops in X: dp/dx = -50 Pa/m
//! // - Viscous drag (Laplacian): d²u/dy² = -100 (parabolic profile)
//! let grad_u = Matrix3::zeros();
//! let grad_p = Vector3::new(-50.0, 0.0, 0.0);
//! let lap_u = Vector3::new(-100.0, 0.0, 0.0);
//!
//! let gradients = SpatialGradients::new(grad_u, grad_p, lap_u);
//!
//! // 4. Calculate Acceleration using Navier-Stokes (Viscous) Strategy
//! let strategy = NavierStokes;
//! let gravity = Vector3::new(0.0, -9.81, 0.0); // Gravity acts in -Y
//!
//! let accel = strategy.acceleration(&water, &state, &gradients, gravity);
//!
//! println!("Local Acceleration: {:.4?}", accel);
//! // Expect positive X acceleration from pressure (driving force)
//! // damped by viscosity (drag)
//! ```
//!
//! ##  Module Architecture
//!
//! The module is designed around the interaction between fluid properties, flow state, and
//! conservation strategies.
//!
//! ```mermaid
//! classDiagram
//!     class FluidProperties {
//!         +density(): f64
//!         +dynamic_viscosity(): f64
//!         +kinematic_viscosity()
//!     }
//!
//!     class FlowState {
//!         +velocity: Vector3
//!         +pressure: f64
//!     }
//!
//!     class SpatialGradients {
//!         +velocity_gradient: Matrix3
//!         +pressure_gradient: Vector3
//!         +laplacian_velocity: Vector3
//!     }
//!
//!     class MomentumEquation {
//!         <<Interface>>
//!         +acceleration(props, state, grads, g)
//!     }
//!
//!     class NavierStokes {
//!         +acceleration()
//!     }
//!
//!     class Euler {
//!         +acceleration()
//!     }
//!
//!     MomentumEquation <|-- NavierStokes : Viscous
//!     MomentumEquation <|-- Euler : Inviscid
//!
//!     NavierStokes ..> FluidProperties : Uses
//!     NavierStokes ..> FlowState : Uses
//!     NavierStokes ..> SpatialGradients : Uses
//! ```
//!
//! ## Mathematical Background
//!
//! The **Navier-Stokes Equation** for incompressible flow is:
//!
//! $$
//! \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} =
//! -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}
//! $$
//!
//! Where:
//! * $\mathbf{u}$: Velocity vector
//! * $\rho$: Density
//! * $p$: Pressure
//! * $\nu$: Kinematic viscosity
//! * $\mathbf{g}$: Body accelerations (gravity)

pub mod analysis;
pub mod conservation;
pub mod lattice_boltzmann;
pub mod potential_flow;
pub mod potential_flow_streamlines;
pub mod regimes;
pub mod solver;
pub mod turbulence;
#[doc(hidden)]
pub mod types;

#[cfg(test)]
mod tests;

// [cite:dwarf_galaxy_empirical_dependencies]

use pure_math::theory_verification;

theory_verification!(
    module = "fluid_dynamics",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
