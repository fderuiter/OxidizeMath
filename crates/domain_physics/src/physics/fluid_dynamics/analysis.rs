//! Analytical tools and derived quantities for Fluid Dynamics.

use super::types::{FlowState, FluidProperties};

/// Calculates the Reynolds Number ($Re$).
///
/// $$Re = \frac{\rho u L}{\mu} = \frac{u L}{\nu}$$
///
/// * `properties`: Fluid properties.
/// * `velocity_magnitude`: Characteristic velocity scale $u$.
/// * `characteristic_length`: Characteristic length scale $L$.
#[verified_engine::verified]
pub fn reynolds_number(
    properties: &FluidProperties,
    velocity_magnitude: f64,
    characteristic_length: f64,
) -> f64 {
    (properties.density() * velocity_magnitude * characteristic_length)
        / properties.dynamic_viscosity()
}

/// Calculates the Bernoulli constant along a streamline for steady, incompressible, inviscid flow.
///
/// $$C = p + \frac{1}{2}\rho v^2 + \rho g h$$
///
/// * `state`: Fluid state ($p, \mathbf{u}$).
/// * `properties`: Fluid properties (specifically density $\rho$).
/// * `height`: Elevation $h$ relative to a datum.
/// * `gravity`: Gravitational acceleration $g$ (magnitude, usually 9.81).
#[verified_engine::verified]
pub fn bernoulli_constant(
    state: &FlowState,
    properties: &FluidProperties,
    height: f64,
    gravity: f64,
) -> f64 {
    let v_sq = state.velocity.norm_squared();
    state.pressure + 0.5 * properties.density() * v_sq + properties.density() * gravity * height
}

/// Calculates Shear Stress ($\tau$) in a Boundary Layer (Newtonian Fluid).
///
/// $$\tau = \mu \frac{\partial u}{\partial y}$$
///
/// * `properties`: Fluid properties (specifically dynamic viscosity $\mu$).
/// * `velocity_gradient_normal`: Gradient of velocity perpendicular to the wall ($\frac{\partial u}{\partial y}$).
#[verified_engine::verified]
pub fn shear_stress(properties: &FluidProperties, velocity_gradient_normal: f64) -> f64 {
    properties.dynamic_viscosity() * velocity_gradient_normal
}
