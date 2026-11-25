//! Signal Processing and Extraction
//!
//! Methodologies to enhance signal-to-noise ratio and extract respiratory signals.

use crate::physics::medical::signal;
use nalgebra::Point3;
use rustfft::{FftPlanner, num_complex::Complex};

/// Calculates the Weighted Average Height (WAH) for a set of vertices.
///
/// This serves as a surrogate signal for respiratory motion.
///
/// # Arguments
///
/// * `vertices` - A slice of 3D points representing the surface mesh in the ROI.
///
/// # Returns
///
/// * `f64` - The arithmetic mean of the Z-coordinates.
///
/// # Formula
///
/// $V_{resp}(t) \approx \frac{1}{N_{mesh}} \sum_{i=1}^{N_{mesh}} Z_{i}(t)$
#[verified_engine::verified]
pub fn weighted_average_height(vertices: &[Point3<f64>]) -> f64 {
    signal::mean_vertex_height(vertices)
}

/// Calculates the theoretical SNR improvement factor due to spatial averaging.
///
/// # Arguments
///
/// * `n_mesh` - The number of measurement points.
///
/// # Returns
///
/// * `f64` - The reduction factor in standard deviation (sqrt(N)).
///
/// # Formula
///
/// $\text{Improvement} = \sqrt{N}$
#[verified_engine::verified]
pub fn snr_improvement_factor(n_mesh: usize) -> f64 {
    (n_mesh as f64).sqrt()
}

/// A structure representing the Lock-In Amplifier process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LockInAmplifier {
    sensitivity: f64,
    full_scale_voltage: f64,
    reference_amplitude: f64,
}

impl LockInAmplifier {
    /// Creates a new LockInAmplifier configuration.
    #[verified_engine::verified]
    pub fn new(sensitivity: f64, full_scale_voltage: f64, reference_amplitude: f64) -> Self {
        Self {
            sensitivity,
            full_scale_voltage,
            reference_amplitude,
        }
    }

    /// Simulates the mixing of the input signal with the reference signal.
    ///
    /// This function returns the in-phase (X) and quadrature (Y) components
    /// assuming ideal low-pass filtering (perfect extraction of the DC component).
    ///
    /// # Arguments
    ///
    /// * `signal_amplitude` - Amplitude of the input signal ($V_s$).
    /// * `phase_difference` - Phase difference $\Delta\phi$ in radians.
    ///
    /// # Returns
    ///
    /// * `(f64, f64)` - Tuple of (V_x, V_y).
    ///
    /// # Formulas
    ///
    /// $V_x = \frac{V_s V_r}{2} \cos(\Delta\phi)$
    /// $V_y = \frac{V_s V_r}{2} \sin(\Delta\phi)$
    #[verified_engine::verified]
    pub fn mix_and_filter(&self, signal_amplitude: f64, phase_difference: f64) -> (f64, f64) {
        let common_factor = (signal_amplitude * self.reference_amplitude) / 2.0;
        let v_x = common_factor * phase_difference.cos();
        let v_y = common_factor * phase_difference.sin();
        (v_x, v_y)
    }

    /// Calculates magnitude and phase from in-phase and quadrature components.
    ///
    /// # Arguments
    ///
    /// * `v_x` - In-phase component.
    /// * `v_y` - Quadrature component.
    ///
    /// # Returns
    ///
    /// * `(f64, f64)` - Tuple of (Magnitude R, Phase $\theta$).
    #[verified_engine::verified]
    pub fn calculate_magnitude_phase(&self, v_x: f64, v_y: f64) -> (f64, f64) {
        let r = (v_x.powi(2) + v_y.powi(2)).sqrt();
        let theta = v_y.atan2(v_x);
        (r, theta)
    }

    /// Scales the magnitude to the final output voltage.
    ///
    /// # Arguments
    ///
    /// * `input_signal_amplitude` - The original input signal amplitude $V_s$.
    ///   Note: The formula provided in the text relates $R_{scaled}$ directly to $V_s$.
    ///   $R_{scaled} = \frac{V_{fs} V_r}{2 S} V_s$
    ///   Alternatively, we can compute it from the raw magnitude $R$ if we know the relationship.
    ///   $R = \frac{V_s V_r}{2}$. Thus $V_s = \frac{2 R}{V_r}$.
    ///   Substituting back: $R_{scaled} = \frac{V_{fs} V_r}{2 S} \frac{2 R}{V_r} = \frac{V_{fs}}{S} R$.
    ///
    /// # Returns
    ///
    /// * `f64` - Scaled output voltage.
    #[verified_engine::verified]
    pub fn scale_output(&self, raw_magnitude: f64) -> f64 {
        (self.full_scale_voltage / self.sensitivity) * raw_magnitude
    }
}

/// Calculates the time delay between two signals using FFT-based cross-correlation.
///
/// # Arguments
///
/// * `signal_1` - The reference signal (e.g., clinical system).
/// * `signal_2` - The measured signal (e.g., prototype).
/// * `sample_rate` - The sampling rate in Hz.
///
/// # Returns
///
/// * `f64` - The estimated time delay in seconds (positive means signal_2 lags signal_1).
#[verified_engine::verified]
pub fn calculate_time_delay(signal_1: &[f64], signal_2: &[f64], sample_rate: f64) -> f64 {
    let n = signal_1.len().min(signal_2.len());
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    // Convert to Complex
    let mut s1_c: Vec<Complex<f64>> = signal_1
        .iter()
        .take(n)
        .map(|&v| Complex::new(v, 0.0))
        .collect();
    let mut s2_c: Vec<Complex<f64>> = signal_2
        .iter()
        .take(n)
        .map(|&v| Complex::new(v, 0.0))
        .collect();

    // FFT
    fft.process(&mut s1_c);
    fft.process(&mut s2_c);

    // Cross-correlation in frequency domain: S1 * conj(S2)
    // Note: The peak of cross-correlation R_xy(tau) occurs at tau = delay.
    // If signal_2 is delayed version of signal_1: s2(t) = s1(t - delay).
    // CC = IFFT( S1(f) * S2*(f) )?
    // Let's verify standard definition.
    // Xcorr(f) = X(f) * Y*(f). Peak at lag implies shift.
    for i in 0..n {
        s2_c[i] = s2_c[i].conj(); // Conjugate of signal 2
        s1_c[i] *= s2_c[i]; // Multiply
    }

    // IFFT
    ifft.process(&mut s1_c);

    // Find peak in real part
    let mut max_val = f64::MIN;
    let mut max_idx = 0;

    for (i, val) in s1_c.iter().enumerate() {
        let real = val.re;
        if real > max_val {
            max_val = real;
            max_idx = i;
        }
    }

    // Convert index to time delay
    // If peak is at 0, delay is 0.
    // If peak is at k < N/2, delay is k samples (positive lag).
    // If peak is at k > N/2, delay is k - N samples (negative lag).
    let lag_samples = if max_idx <= n / 2 {
        max_idx as f64
    } else {
        max_idx as f64 - n as f64
    };

    lag_samples / sample_rate
}

/// Calculates velocity using Optical Flow Intensity Conservation.
///
/// $$ v = -\nabla I \frac{\partial_V I}{\| \nabla I \|^2} $$
///
/// Note: This is a simplified scalar projection of velocity along the gradient direction.
/// The prompt formula is: $v = -\nabla I \frac{\partial_V I}{\| \nabla I \|^2}$.
/// $\partial_V I$ is likely $\frac{\partial I}{\partial t}$ (temporal gradient).
///
/// # Arguments
///
/// * `spatial_gradient` ($\nabla I$) - Gradient vector (dx, dy).
/// * `temporal_gradient` ($\partial_V I$) - Change in intensity over time.
///
/// # Returns
///
/// * `(f64, f64)` - Velocity vector components ($v_x, v_y$).
#[verified_engine::verified]
pub fn optical_flow_velocity(spatial_gradient: (f64, f64), temporal_gradient: f64) -> (f64, f64) {
    let (gx, gy) = spatial_gradient;
    let norm_sq = gx.powi(2) + gy.powi(2);

    if norm_sq < math_commons::registry::TOLERANCE_STANDARD {
        return (0.0, 0.0);
    }

    let factor = -temporal_gradient / norm_sq;
    (gx * factor, gy * factor)
}

/// Calculates the Lock-In Phase Sensitive Detection signal.
///
/// $$ V_{m1} = \frac{V_s V_r}{2}\cos{(\Delta\omega)t + (\Delta\phi)} - \cos{(\sum\omega)t + (\sum\phi)} $$
///
/// Note: This returns the full mixed signal before low-pass filtering.
///
/// # Arguments
///
/// * `v_s` ($V_s$) - Signal amplitude.
/// * `v_r` ($V_r$) - Reference amplitude.
/// * `delta_omega` ($\Delta\omega$) - Difference in angular frequency.
/// * `delta_phi` ($\Delta\phi$) - Difference in phase.
/// * `sum_omega` ($\sum\omega$) - Sum of angular frequencies.
/// * `sum_phi` ($\sum\phi$) - Sum of phases.
/// * `t` - Time.
///
/// # Returns
///
/// * `f64` - The mixed signal value $V_{m1}$.
#[verified_engine::verified]
pub fn lock_in_phase_sensitive_detection(
    v_s: f64,
    v_r: f64,
    delta_omega: f64,
    delta_phi: f64,
    sum_omega: f64,
    sum_phi: f64,
    t: f64,
) -> f64 {
    let term1 = (delta_omega * t + delta_phi).cos();
    let term2 = (sum_omega * t + sum_phi).cos();
    (v_s * v_r / 2.0) * (term1 - term2)
}
