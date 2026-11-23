//! Radar Physics for FMCW Sensors (TI IWR6843).
//!
//! This module implements the fundamental physics equations for Frequency Modulated Continuous Wave (FMCW)
//! radar systems, specifically focusing on range resolution and Doppler velocity estimation.
//!
//! The TI IWR6843 is a mmWave sensor operating in the 60-64 GHz band.
use math_commons::constants::C;

/// Configuration for the FMCW Radar system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmcwConfig {
    /// Sweep bandwidth in Hz (e.g., 4.0e9 for 4 GHz).
    pub bandwidth: f64,
    /// Center frequency in Hz (e.g., 60.0e9 for 60 GHz).
    pub center_frequency: f64,
    /// Chirp duration (separation time between chirps) in seconds.
    pub chirp_time: f64,
}

impl FmcwConfig {
    /// Creates a new configuration for the TI IWR6843 with default values.
    ///
    /// - Bandwidth: 4 GHz
    /// - Frequency: 60 GHz
    /// - Chirp Time: 50 microseconds
    #[verified_engine::verified]
    pub fn iwr6843_default() -> Self {
        Self {
            bandwidth: 4.0e9,
            center_frequency: 60.0e9,
            chirp_time: 50.0e-6,
        }
    }

    /// Calculates the fundamental range resolution ($\Delta R$).
    ///
    /// The ability to resolve two distinct points in depth is determined by the sweep bandwidth $B$.
    ///
    /// $$ \Delta R = \frac{c}{2B} $$
    #[verified_engine::verified]
    pub fn range_resolution(&self) -> f64 {
        C / (2.0 * self.bandwidth)
    }

    /// Calculates the signal wavelength ($\lambda$).
    ///
    /// $$ \lambda = \frac{c}{f_c} $$
    #[verified_engine::verified]
    pub fn wavelength(&self) -> f64 {
        C / self.center_frequency
    }

    /// Estimates radial velocity ($v$) from the phase shift ($\Delta \phi$) between chirps.
    ///
    /// Doppler processing uses the phase change across a sequence of chirps separated by time $T_c$.
    ///
    /// $$ \Delta \phi = \frac{4\pi v T_c}{\lambda} \implies v = \frac{\Delta \phi \lambda}{4\pi T_c} $$
    ///
    /// # Arguments
    ///
    /// * `phase_shift` - The phase shift in radians.
    #[verified_engine::verified]
    pub fn velocity_from_phase(&self, phase_shift: f64) -> f64 {
        let lambda = self.wavelength();
        (phase_shift * lambda) / (4.0 * std::f64::consts::PI * self.chirp_time)
    }

    /// Estimates target range ($R$) from the IF beat frequency ($\hat{f}_{FFT}$).
    ///
    /// Equation (1) from the Bressler et al. framework:
    #[verified_engine::embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
    #[verified_engine::verified]
    pub fn range_from_beat_frequency(&self, beat_frequency: f64) -> f64 {
        (beat_frequency * C * self.chirp_time) / (2.0 * self.bandwidth)
    }

    /// Estimates physical displacement ($d$) from the phase change ($\Delta \phi$).
    ///
    /// Equation (5) from the Bressler et al. framework:
    /// $$ d = \frac{\lambda \Delta \phi}{4\pi} $$
    ///
    /// Note: This is equivalent to velocity * chirp_time, but expressed directly as displacement.
    ///
    /// # Arguments
    ///
    /// * `phase_shift` - The phase shift in radians.
    #[verified_engine::verified]
    pub fn displacement_from_phase(&self, phase_shift: f64) -> f64 {
        let lambda = self.wavelength();
        (phase_shift * lambda) / (4.0 * std::f64::consts::PI)
    }

    /// Calculates the instantaneous frequency of the chirp signal.
    ///
    /// $$ f(t) = St + f_c $$
    ///
    /// # Arguments
    ///
    /// * `t` - Time within the chirp duration.
    #[verified_engine::verified]
    pub fn chirp_frequency(&self, t: f64) -> f64 {
        let slope = self.bandwidth / self.chirp_time;
        slope * t + self.center_frequency
    }

    /// Simulates the mixer output for FMCW signal generation.
    ///
    /// $$ x_{out} = \sin((\omega_1 - \omega_2)t + (\phi_1 - \phi_2)) $$
    ///
    /// # Arguments
    ///
    /// * `omega1` - Angular velocity of signal 1.
    /// * `omega2` - Angular velocity of signal 2.
    /// * `phi1` - Initial phase of signal 1.
    /// * `phi2` - Initial phase of signal 2.
    /// * `t` - Time.
    #[verified_engine::verified]
    pub fn mixer_output(omega1: f64, omega2: f64, phi1: f64, phi2: f64, t: f64) -> f64 {
        ((omega1 - omega2) * t + (phi1 - phi2)).sin()
    }

    /// Calculates the Maximum Unambiguous Velocity ($v_{max}$).
    ///
    /// Determined by the chirp repetition time ($T_c$).
    ///
    /// $$ v_{max} = \frac{\lambda}{4 T_c} $$
    #[verified_engine::verified]
    pub fn max_unambiguous_velocity(&self) -> f64 {
        let lambda = self.wavelength();
        lambda / (4.0 * self.chirp_time)
    }

    /// Calculates the Phase Delay introduced by a dielectric material.
    ///
    /// $$ \Delta \phi = \frac{4\pi d}{\lambda} (\sqrt{\epsilon_r} - 1) $$
    ///
    /// # Arguments
    ///
    /// * `thickness` ($d$) - Material thickness in meters.
    /// * `dielectric_constant` ($\epsilon_r$) - Relative permittivity.
    #[verified_engine::verified]
    pub fn dielectric_phase_delay(&self, thickness: f64, dielectric_constant: f64) -> f64 {
        let lambda = self.wavelength();
        (4.0 * std::f64::consts::PI * thickness / lambda) * (dielectric_constant.sqrt() - 1.0)
    }

    /// Calculates the FMCW Beat Frequency.
    ///
    /// $$ f_b = \frac{S \cdot 2R}{c} $$
    ///
    /// # Arguments
    ///
    /// * `slope` ($S$) - Frequency slope of the chirp (Hz/s).
    /// * `range` ($R$) - Distance to target (m).
    #[verified_engine::verified]
    pub fn beat_frequency(slope: f64, range: f64) -> f64 {
        (slope * 2.0 * range) / C
    }
}
