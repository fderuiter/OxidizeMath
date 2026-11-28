//! Coordinate Transformations for Radar Data.
//!
//! Handles the conversion from raw spherical coordinates (Sensor Frame) to Cartesian coordinates (Patient Frame).
//!
//! # TI IWR6843 Coordinate Convention
//! - **r**: Depth (range).
//! - **x**: Horizontal azimuth.
//! - **z**: Vertical elevation.
//!
//! Note: This differs from standard mathematical spherical conventions.

use nalgebra::{Point3, Rotation3, Translation3};

/// A raw detection point in spherical/sensor space.
#[derive(Debug, Clone, Copy)]
pub struct SphericalPoint {
    /// Range (distance) in meters.
    pub range: f64,
    /// Azimuth FFT bin index.
    pub azimuth_index: isize,
    /// Elevation FFT bin index.
    pub elevation_index: isize,
}

/// Configuration for the Angle-FFT processing.
#[derive(Debug, Clone, Copy)]
pub struct AngleFftConfig {
    /// Size of the Azimuth FFT (e.g., 64).
    pub n_fft_azimuth: usize,
    /// Size of the Elevation FFT (e.g., 32).
    pub n_fft_elevation: usize,
}

impl AngleFftConfig {
    /// Converts a raw spherical point to Cartesian coordinates in the Sensor Frame.
    ///
    /// Let $w_x$ and $w_z$ be the normalized sine values derived from indices:
    /// $$ w_x = \frac{2 \cdot I_{az}}{N_{az}}, \quad w_z = \frac{2 \cdot I_{el}}{N_{el}} $$
    ///
    /// The Cartesian coordinates $(x, y, z)$ are:
    /// $$ x = r \cdot w_x $$
    /// $$ z = r \cdot w_z $$
    /// $$ y = \sqrt{r^2 - x^2 - z^2} $$
    ///
    /// Note: The TI convention defines $y$ as the depth (range direction) for the *derived* Cartesian frame here,
    /// but the prompt specifies $x, y, z$ as:
    /// "x = r * wx"
    /// "z = r * wz"
    /// "y = sqrt(r^2 - x^2 - z^2)"
    ///
    /// Usually in standard TI radar SDK:
    /// - x is horizontal
    /// - y is depth (range)
    /// - z is vertical
    #[verified_engine::verified]
    pub fn spherical_to_cartesian(&self, point: &SphericalPoint) -> Point3<f64> {
        let w_x = (2.0 * point.azimuth_index as f64) / self.n_fft_azimuth as f64;
        let w_z = (2.0 * point.elevation_index as f64) / self.n_fft_elevation as f64;

        let x = point.range * w_x;
        let z = point.range * w_z;

        // Ensure the term under sqrt is non-negative (handle numerical noise or invalid points)
        let y_sq = point.range.powi(2) - x.powi(2) - z.powi(2);
        let y = if y_sq > 0.0 { y_sq.sqrt() } else { 0.0 };

        Point3::new(x, y, z)
    }
}

/// Represents the rigid body transformation from Sensor Frame to Patient Frame.
#[derive(Debug, Clone)]
pub struct SensorToPatientTransform {
    /// Rotation matrix ($R_{3\times3}$).
    pub rotation: Rotation3<f64>,
    /// Translation vector ($T_{3\times1}$).
    pub translation: Translation3<f64>,
}

impl SensorToPatientTransform {
    /// Creates a new transformation from a rotation and translation.
    #[verified_engine::verified]
    pub fn new(rotation: Rotation3<f64>, translation: Translation3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Transforms a point from Sensor Frame ($P_S$) to Patient Frame ($P_P$).
    ///
    /// $$ P_P = R \cdot P_S + T $$
    #[verified_engine::verified]
    pub fn transform_point(&self, point_sensor: &Point3<f64>) -> Point3<f64> {
        self.translation * (self.rotation * point_sensor)
    }
}

/// Calculates the Angle of Arrival (AoA) estimation.
///
/// $$ \theta = \sin^{-1} \left( \frac{\lambda \Delta \phi}{2 \pi l} \right) $$
///
/// # Arguments
///
/// * `wavelength` ($\lambda$) - The signal wavelength.
/// * `phase_difference` ($\Delta \phi$) - The phase difference between two antennas.
/// * `antenna_distance` ($l$) - The distance between the two antennas.
///
/// # Returns
///
/// * `f64` - The angle in radians.
#[verified_engine::verified]
pub fn angle_of_arrival(
    wavelength: f64,
    phase_difference: f64,
    antenna_distance: f64,
) -> Option<f64> {
    let arg = (wavelength * phase_difference) / (2.0 * std::f64::consts::PI * antenna_distance);
    if arg.abs() > 1.0 {
        None // Invalid argument for arcsin (phase ambiguity or noise)
    } else {
        Some(arg.asin())
    }
}

/// Calculates the Angle Resolution ($\theta_{res}$).
///
/// The minimum angular separation resolvable by the radar.
///
/// $$ \theta_{res} = \frac{\lambda}{N_{RX} l \cos(\theta)} $$
///
/// # Arguments
///
/// * `wavelength` ($\lambda$) - Signal wavelength.
/// * `num_rx_antennas` ($N_{RX}$) - Number of receive antennas (or virtual array size).
/// * `antenna_distance` ($l$) - Spacing between array elements.
/// * `angle` ($\theta$) - The look angle (off-boresight). Resolution degrades as $\theta$ increases.
#[verified_engine::verified]
pub fn angle_resolution(
    wavelength: f64,
    num_rx_antennas: usize,
    antenna_distance: f64,
    angle: f64,
) -> f64 {
    let denominator = num_rx_antennas as f64 * antenna_distance * angle.cos();
    if denominator.abs() < math_commons::registry::TOLERANCE_STANDARD {
        f64::INFINITY // Singularity at grazing angles
    } else {
        wavelength / denominator
    }
}

/// Calculates the Required Angular Separation ($\Delta\theta_{\text{req}}$).
///
/// For distinguishing thoracic vs. abdominal respiratory motion.
///
/// $$ \Delta\theta_{\text{req}} \approx \arctan\left(\frac{d_{\text{TAA}}}{R}\right) $$
///
/// # Arguments
///
/// * `vertical_distance` ($d_{\text{TAA}}$) - Distance between thoracic and abdominal regions (e.g., 20 cm).
/// * `range` ($R$) - Distance from the sensor to the patient.
#[verified_engine::verified]
pub fn required_angular_separation(vertical_distance: f64, range: f64) -> f64 {
    (vertical_distance / range).atan()
}
