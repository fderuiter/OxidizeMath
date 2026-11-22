//! MIMO Beamforming ("The Digital Lens").
//!
//! This module implements Multiple-Input Multiple-Output (MIMO) beamforming to spatially filter
//! radar signals. It allows "steering" the radar beam to focus on specific anatomical regions
//! (e.g., sternum vs. abdomen), resolving the "Human Shape Geometric Error".

use num_complex::Complex;
use std::f64::consts::PI;

/// Handles MIMO beamforming operations.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Beamformer {
    /// Wavelength of the signal ($\lambda$).
    wavelength: f64,
    /// Positions of the antennas relative to the center, in meters ($d_k$).
    /// For a uniform linear array, these might be [0, d, 2d, 3d].
    antenna_positions: Vec<f64>,
    /// Complex weights for tapering/windowing ($w_k$).
    weights: Vec<Complex<f64>>,
}

impl Beamformer {
    /// Creates a new Beamformer for a Uniform Linear Array (ULA).
    ///
    /// # Arguments
    ///
    /// * `num_antennas` - Number of RX antennas ($M$).
    /// * `spacing` - Spacing between antennas (usually $\lambda/2$).
    /// * `wavelength` - Signal wavelength ($\lambda$).
    #[verified_engine::verified]
    pub fn new_ula(num_antennas: usize, spacing: f64, wavelength: f64) -> Self {
        let mut antenna_positions = Vec::with_capacity(num_antennas);
        // Center the array around 0 for symmetry, or start at 0.
        // Let's start at 0 as per standard formula d_k usually implies offset from reference.
        for k in 0..num_antennas {
            antenna_positions.push(k as f64 * spacing);
        }

        // Default weights = 1.0 (Rectangular window)
        let weights = vec![Complex::new(1.0, 0.0); num_antennas];

        Self {
            wavelength,
            antenna_positions,
            weights,
        }
    }

    /// Sets custom weights for the antennas (e.g., Hamming window to reduce sidelobes).
    #[verified_engine::verified]
    pub fn set_weights(&mut self, weights: &[f64]) -> Result<(), &'static str> {
        if weights.len() != self.antenna_positions.len() {
            return Err("Number of weights must match number of antennas");
        }
        self.weights = weights.iter().map(|&w| Complex::new(w, 0.0)).collect();
        Ok(())
    }

    /// performs beamforming on a set of antenna signals to focus on a specific angle.
    ///
    /// $$ y(t) = \sum_{k=1}^{M} w_k \cdot x_k(t) \cdot e^{-j \frac{2\pi}{\lambda} d_k \sin(\theta)} $$
    ///
    /// # Arguments
    ///
    /// * `signals` - The complex signals received at each antenna ($x_k(t)$).
    /// * `angle_rad` - The target angle $\theta$ in radians (0 is broadside).
    ///
    /// # Returns
    ///
    /// The combined beamformed signal.
    #[verified_engine::verified]
    pub fn steer(&self, signals: &[Complex<f64>], angle_rad: f64) -> Complex<f64> {
        if signals.len() != self.antenna_positions.len() {
            // In a real system, handle this gracefully. For now, panic or return 0.
            // Using 0.0 to avoid crashing in tight loops.
            return Complex::new(0.0, 0.0);
        }

        let mut sum = Complex::new(0.0, 0.0);
        let k_wave = 2.0 * PI / self.wavelength;
        let sin_theta = angle_rad.sin();

        for (i, &signal) in signals.iter().enumerate() {
            let d_k = self.antenna_positions[i];
            let w_k = self.weights[i];

            // Phase shift: -j * (2*pi/lambda) * d_k * sin(theta)
            let phase_shift = -k_wave * d_k * sin_theta;
            let steering_vector = Complex::new(0.0, phase_shift).exp();

            sum += w_k * signal * steering_vector;
        }

        sum
    }
}
