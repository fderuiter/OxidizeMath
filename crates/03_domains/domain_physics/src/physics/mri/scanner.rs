//! Spatial Encoding and k-Space Trajectories.

use super::proton;
use nalgebra::Vector3;
use std::f64::consts::PI;

/// Calculates the accumulated phase $\phi(\vec{r}, t)$ given a spatial position and accumulated gradient.
///
/// Formula: $\phi(\vec{r}, t) = \gamma \vec{r} \cdot \int_0^t \vec{G}(\tau) d\tau$
///
/// # Arguments
/// * `position` - Spatial position $\vec{r}$ in meters.
/// * `gradient_integral` - Time integral of the gradient vector $\int \vec{G} dt$ in T·s/m.
///
/// # Returns
/// * Phase angle in radians.
#[verified_engine::verified]
pub fn accumulated_phase(position: Vector3<f64>, gradient_integral: Vector3<f64>) -> f64 {
    proton::GYROMAGNETIC_RATIO * position.dot(&gradient_integral)
}

/// Calculates the current k-space coordinate.
///
/// Formula: $\vec{k}(t) = \frac{\gamma}{2\pi} \int_0^t \vec{G}(\tau) d\tau$
///
/// # Arguments
/// * `gradient_integral` - Time integral of the gradient vector $\int \vec{G} dt$ in T·s/m.
///
/// # Returns
/// * k-space coordinate vector in cycles/meter ($m^{-1}$).
#[verified_engine::verified]
pub fn k_space_coordinate(gradient_integral: Vector3<f64>) -> Vector3<f64> {
    (proton::GYROMAGNETIC_RATIO / (2.0 * PI)) * gradient_integral
}
