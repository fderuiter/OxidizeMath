//! Types for Fluid Dynamics.

use crate::error::FluidError;
use nalgebra::{Matrix3, Vector3};

/// Physical properties of the fluid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidProperties {
    /// Density ($\rho$) in kg/m^3.
    density: f64,
    /// Dynamic viscosity ($\mu$) in Pa·s.
    dynamic_viscosity: f64,
}

impl FluidProperties {
    /// Creates a new `FluidProperties` with validation.
    ///
    /// # Errors
    /// Returns `FluidError::InvalidDensity` if `density <= 0`.
    /// Returns `FluidError::InvalidViscosity` if `dynamic_viscosity < 0`.
    #[verified_engine::verified]
    pub fn new(density: f64, dynamic_viscosity: f64) -> Result<Self, FluidError> {
        if density <= 0.0 {
            return Err(FluidError::InvalidDensity { value: density });
        }
        if dynamic_viscosity < 0.0 {
            return Err(FluidError::InvalidViscosity {
                value: dynamic_viscosity,
            });
        }
        Ok(Self {
            density,
            dynamic_viscosity,
        })
    }

    /// Returns a new builder for constructing `FluidProperties`.
    #[verified_engine::verified]
    pub fn builder() -> FluidPropertiesBuilder {
        FluidPropertiesBuilder::default()
    }

    /// Returns the density ($\rho$) in kg/m^3.
    #[verified_engine::verified]
    pub fn density(&self) -> f64 {
        self.density
    }

    /// Returns the dynamic viscosity ($\mu$) in Pa·s.
    #[verified_engine::verified]
    pub fn dynamic_viscosity(&self) -> f64 {
        self.dynamic_viscosity
    }

    /// Calculates the kinematic viscosity ($\nu = \mu / \rho$) in m^2/s.
    #[verified_engine::verified]
    pub fn kinematic_viscosity(&self) -> f64 {
        self.dynamic_viscosity / self.density
    }

    /// Standard properties for Water at 20°C.
    ///
    /// # Panics
    /// Panics if internal values are invalid (should never happen).
    #[verified_engine::verified]
    pub fn water() -> Self {
        Self::new(998.2, 1.002e-3).expect("Standard water properties should be valid")
    }

    /// Standard properties for Air at 15°C (Sea Level).
    ///
    /// # Panics
    /// Panics if internal values are invalid (should never happen).
    #[verified_engine::verified]
    pub fn air() -> Self {
        Self::new(1.225, 1.81e-5).expect("Standard air properties should be valid")
    }
}

/// Builder for `FluidProperties`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FluidPropertiesBuilder {
    density: Option<f64>,
    dynamic_viscosity: Option<f64>,
}

impl FluidPropertiesBuilder {
    /// Sets the density ($\rho$) in kg/m^3.
    #[verified_engine::verified]
    pub fn density(mut self, density: f64) -> Self {
        self.density = Some(density);
        self
    }

    /// Sets the dynamic viscosity ($\mu$) in Pa·s.
    #[verified_engine::verified]
    pub fn dynamic_viscosity(mut self, dynamic_viscosity: f64) -> Self {
        self.dynamic_viscosity = Some(dynamic_viscosity);
        self
    }

    /// Builds the `FluidProperties`.
    ///
    /// # Errors
    /// Returns error if density is not set or invalid, or if viscosity is invalid.
    /// If viscosity is not set, it defaults to 0.0 (inviscid).
    #[verified_engine::verified]
    pub fn build(self) -> Result<FluidProperties, FluidError> {
        let density = self.density.unwrap_or(0.0);
        let dynamic_viscosity = self.dynamic_viscosity.unwrap_or(0.0);
        FluidProperties::new(density, dynamic_viscosity)
    }
}

/// Represents the state of a fluid element at a specific point in space and time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowState {
    /// Velocity vector ($\mathbf{u}$) in m/s.
    pub velocity: Vector3<f64>,
    /// Pressure ($p$) in Pa.
    pub pressure: f64,
}

impl FlowState {
    #[verified_engine::verified]
    pub fn new(velocity: Vector3<f64>, pressure: f64) -> Self {
        Self { velocity, pressure }
    }
}

pure_math::impl_vector_operations!(FlowState, velocity, pressure);

/// Encapsulates spatial derivatives required for momentum equations.
///
/// This prevents "Primitive Obsession" by grouping the gradients of velocity and pressure,
/// ensuring they are passed together consistently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpatialGradients {
    /// Jacobian matrix of velocity ($\nabla \mathbf{u}$).
    pub velocity_gradient: Matrix3<f64>,
    /// Gradient of pressure ($\nabla p$).
    pub pressure_gradient: Vector3<f64>,
    /// Laplacian of velocity ($\nabla^2 \mathbf{u}$).
    pub laplacian_velocity: Vector3<f64>,
}

impl SpatialGradients {
    #[verified_engine::verified]
    pub fn new(
        velocity_gradient: Matrix3<f64>,
        pressure_gradient: Vector3<f64>,
        laplacian_velocity: Vector3<f64>,
    ) -> Self {
        Self {
            velocity_gradient,
            pressure_gradient,
            laplacian_velocity,
        }
    }
}
