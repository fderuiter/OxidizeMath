//! Conservation Laws: Mass and Momentum.
//!
//! Implements the core Partial Differential Equations (PDEs) of Fluid Dynamics.

use super::types::{FlowState, FluidProperties, SpatialGradients};
use nalgebra::Vector3;

/// Calculates the Material Derivative ($D/Dt$) of a scalar property.
///
/// $$\frac{D\phi}{Dt} = \frac{\partial \phi}{\partial t} + \mathbf{u} \cdot \nabla \phi$$
///
/// * `local_change`: Partial time derivative $\frac{\partial \phi}{\partial t}$.
/// * `velocity`: Flow velocity $\mathbf{u}$.
/// * `gradient`: Spatial gradient $\nabla \phi$.
#[verified_engine::verified]
pub fn material_derivative_scalar(
    local_change: f64,
    velocity: Vector3<f64>,
    gradient: Vector3<f64>,
) -> f64 {
    local_change + velocity.dot(&gradient)
}

/// Calculates the Material Derivative ($D/Dt$) of a vector property (e.g., velocity).
///
/// $$\frac{D\mathbf{A}}{Dt} = \frac{\partial \mathbf{A}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{A}$$
///
/// * `local_change`: Partial time derivative $\frac{\partial \mathbf{A}}{\partial t}$.
/// * `velocity`: Flow velocity $\mathbf{u}$.
/// * `gradient_tensor`: Jacobian matrix representing $\nabla \mathbf{A}$ (where $J_{ij} = \partial A_i / \partial x_j$).
#[verified_engine::verified]
pub fn material_derivative_vector(
    local_change: Vector3<f64>,
    velocity: Vector3<f64>,
    gradient_tensor: &nalgebra::Matrix3<f64>,
) -> Vector3<f64> {
    // (\mathbf{u} \cdot \nabla) \mathbf{A} corresponds to Jacobian * velocity vector
    // J = [ dA_x/dx  dA_x/dy  dA_x/dz ]
    //     [ dA_y/dx  dA_y/dy  dA_y/dz ]
    //     [ ...                   ]
    // Result is J * u
    local_change + gradient_tensor * velocity
}

/// Checks the Continuity Equation (Conservation of Mass) for Incompressible Flow.
///
/// $$\nabla \cdot \mathbf{u} = 0$$
///
/// Returns the divergence of the velocity field. Ideally, this should be zero.
#[verified_engine::verified]
pub fn continuity_divergence(velocity_divergence: f64) -> f64 {
    velocity_divergence
}

/// Defines a strategy for calculating the momentum conservation (acceleration).
///
/// This trait allows switching between different flow regimes (e.g., Viscous vs. Inviscid)
/// without changing the integration loop.
pub trait MomentumEquation {
    /// Computes the local acceleration $\frac{\partial \mathbf{u}}{\partial t}$.
    #[verified_engine::verified]
    fn acceleration(
        &self,
        properties: &FluidProperties,
        state: &FlowState,
        gradients: &SpatialGradients,
        body_force_accel: Vector3<f64>,
    ) -> Vector3<f64>;
}

/// Navier-Stokes momentum equation (Viscous Flow).
///
/// $$\frac{\partial \mathbf{u}}{\partial t} = -(\mathbf{u} \cdot \nabla)\mathbf{u} - \frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{g}$$
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NavierStokes;

impl MomentumEquation for NavierStokes {
    #[verified_engine::verified]
    fn acceleration(
        &self,
        properties: &FluidProperties,
        state: &FlowState,
        gradients: &SpatialGradients,
        body_force_accel: Vector3<f64>,
    ) -> Vector3<f64> {
        let nu = properties.kinematic_viscosity();
        let rho = properties.density();

        // Convective term: -(u . del) u
        let convection = -(gradients.velocity_gradient * state.velocity);

        // Pressure term: -(1/rho) grad p
        let pressure_term = -gradients.pressure_gradient / rho;

        // Viscous term: nu * del^2 u
        let viscous_term = gradients.laplacian_velocity * nu;

        // Sum
        convection + pressure_term + viscous_term + body_force_accel
    }
}

/// Euler momentum equation (Inviscid Flow).
///
/// $$\frac{\partial \mathbf{u}}{\partial t} = -(\mathbf{u} \cdot \nabla)\mathbf{u} - \frac{1}{\rho}\nabla p + \mathbf{g}$$
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Euler;

impl MomentumEquation for Euler {
    #[verified_engine::verified]
    fn acceleration(
        &self,
        properties: &FluidProperties,
        state: &FlowState,
        gradients: &SpatialGradients,
        body_force_accel: Vector3<f64>,
    ) -> Vector3<f64> {
        let rho = properties.density();

        // Convective term: -(u . del) u
        let convection = -(gradients.velocity_gradient * state.velocity);

        // Pressure term: -(1/rho) grad p
        let pressure_term = -gradients.pressure_gradient / rho;

        convection + pressure_term + body_force_accel
    }
}
