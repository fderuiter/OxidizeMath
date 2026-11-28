//! Phase Processing for Respiratory Motion Tracking.
//!
//! This module implements the "Motion" (Phase Extraction) and "Unwrapping" (Phase Unwrapping)
//! steps of the radar signal processing pipeline. It converts complex radar returns into
//! sub-millimeter displacement measurements.

use num_complex::Complex;
use std::f64::consts::PI;

/// Handles phase extraction and unwrapping to track displacement over time.
#[derive(Debug, Clone)]
pub struct PhaseUnwrapper {
    /// The phase from the previous time step ($\phi[n-1]$).
    last_phase: f64,
    /// The accumulated displacement since the start of tracking ($D[n-1]$).
    accumulated_displacement: f64,
    /// The wavelength of the radar signal ($\lambda$).
    wavelength: f64,
    /// Whether this is the first sample (to initialize `last_phase`).
    initialized: bool,
}

impl PhaseUnwrapper {
    /// Creates a new PhaseUnwrapper.
    ///
    /// # Arguments
    ///
    /// * `wavelength` - The wavelength of the radar signal in meters (e.g., ~0.0039 m for 77GHz).
    #[verified_engine::verified]
    pub fn new(wavelength: f64) -> Self {
        Self {
            last_phase: 0.0,
            accumulated_displacement: 0.0,
            wavelength,
            initialized: false,
        }
    }

    /// Processes a new complex sample to update displacement.
    ///
    /// Steps:
    /// 1. Extract Phase ($\phi\[n\]$).
    /// 2. Calculate Differential Phase ($\Delta \phi$).
    /// 3. Unwrap Phase (Handle aliasing).
    /// 4. Update Accumulated Displacement ($D\[n\]$).
    ///
    /// # Arguments
    ///
    /// * `sample` - The complex radar return ($I + jQ$).
    ///
    /// # Returns
    ///
    /// The total accumulated displacement in meters.
    #[verified_engine::verified]
    pub fn process(&mut self, sample: Complex<f64>) -> f64 {
        // Step 1: Phase Extraction
        // phi[n] = arctan(Q / I) -> range (-PI, PI]
        let current_phase = sample.arg();

        if !self.initialized {
            self.last_phase = current_phase;
            self.initialized = true;
            return 0.0;
        }

        // Step 2: Differential Phase
        let mut delta_phi = current_phase - self.last_phase;

        // Step 3: The "Unwrapping"
        // If the phase jump is too large, it means we wrapped around the unit circle.
        if delta_phi > PI {
            delta_phi -= 2.0 * PI;
        } else if delta_phi < -PI {
            delta_phi += 2.0 * PI;
        }

        // Step 4: Calculate Displacement Step
        // D_step = (lambda / 4pi) * delta_phi
        let displacement_step = (self.wavelength * delta_phi) / (4.0 * PI);

        // Update State
        self.accumulated_displacement += displacement_step;
        self.last_phase = current_phase;

        self.accumulated_displacement
    }

    /// Resets the unwrapper state (e.g., if tracking is lost).
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.initialized = false;
        self.accumulated_displacement = 0.0;
        self.last_phase = 0.0;
    }
}
