//! Chirp Z-Transform (CZT) Implementation.
//!
//! This module implements the Chirp Z-Transform algorithm as verified in the Bressler et al. (2024) analysis.
//! It allows for high-resolution spectral analysis within a specific frequency band (zoom FFT),
//! overcoming the resolution limits of the standard FFT.

use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Calculates direct summation CZT as a zero-overhead fallback for tiny bin counts or signals.
#[verified_engine::verified]
fn direct_chirp_z_transform(
    signal: &[Complex<f64>],
    start_freq: f64,
    bandwidth: f64,
    sample_rate: f64,
    output_bins: usize,
) -> Vec<Complex<f64>> {
    let mut output = Vec::with_capacity(output_bins);

    // Iterate over each output frequency bin 'k'
    for k in 0..output_bins {
        let mut sum = Complex::new(0.0, 0.0);

        let freq_step_fraction = k as f64 / output_bins as f64;
        let target_freq = start_freq + bandwidth * freq_step_fraction;
        let normalized_freq = target_freq / sample_rate;

        let alpha = -2.0 * PI * normalized_freq;
        let rotation = Complex::new(0.0, alpha).exp();
        let mut current_basis = Complex::new(1.0, 0.0);

        for &x_n in signal {
            sum += x_n * current_basis;
            current_basis *= rotation;
        }

        output.push(sum);
    }

    output
}

/// Calculates the Chirp Z-Transform (CZT) for a specific frequency band.
///
/// This implementation corresponds to the corrected Equation (2) from the Bressler verification:
#[verified_engine::embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:czt")]
///
/// It uses Bluestein's fast convolution algorithm via FFT, reducing runtime complexity
/// to $O((N + K) \log(N + K))$, with a direct recurrence fallback for small bin counts.
///
/// # Arguments
///
/// * `signal` - The time-domain input signal ($x_n$).
/// * `start_freq` - The starting frequency of the zoom window ($f_0$).
/// * `bandwidth` - The bandwidth of the zoom window ($B$).
/// * `sample_rate` - The sampling rate of the input signal ($f_s$).
/// * `output_bins` - The number of frequency bins in the output ($K$).
///
/// # Returns
///
/// A vector of complex numbers representing the frequency spectrum in the specified band.
#[verified_engine::verified]
pub fn chirp_z_transform(
    signal: &[Complex<f64>],
    start_freq: f64,
    bandwidth: f64,
    sample_rate: f64,
    output_bins: usize,
) -> Vec<Complex<f64>> {
    debug_assert!(!signal.is_empty());
    debug_assert!(sample_rate > 0.0);

    if output_bins == 0 {
        return Vec::new();
    }
    if signal.is_empty() || sample_rate <= 0.0 {
        return vec![Complex::new(0.0, 0.0); output_bins];
    }

    let n_samples = signal.len();

    // Direct recurrence loop fallback for tiny output bin counts or tiny signals
    if output_bins <= 4 || n_samples <= 4 {
        return direct_chirp_z_transform(signal, start_freq, bandwidth, sample_rate, output_bins);
    }

    // Bluestein's fast convolution algorithm
    // Frequency for bin k: f_k = f_0 + B * (k / K)
    // Normalized angular frequencies:
    //   omega_0 = 2 * PI * (start_freq / sample_rate)
    //   delta_omega = 2 * PI * (bandwidth / (sample_rate * output_bins))
    let omega_0 = 2.0 * PI * (start_freq / sample_rate);
    let delta_omega = 2.0 * PI * (bandwidth / (sample_rate * output_bins as f64));

    // Linear convolution of lengths N and K requires length at least N + K - 1.
    let conv_len = (n_samples + output_bins - 1).next_power_of_two();

    // 1. Formulate chirp modulation sequences
    // Modulated signal: y_n = x_n * exp(-j * (n * omega_0 + 0.5 * delta_omega * n^2))
    let mut a_buf = vec![Complex::new(0.0, 0.0); conv_len];
    for (n, &x_n) in signal.iter().enumerate() {
        let n_f = n as f64;
        let phase = -(n_f * omega_0 + 0.5 * delta_omega * n_f * n_f);
        let pre_chirp = Complex::new(0.0, phase).exp();
        a_buf[n] = x_n * pre_chirp;
    }

    // Filter kernel: h_m = exp(j * 0.5 * delta_omega * m^2)
    // Embedded in circular buffer of length conv_len:
    //   h[m] for m in 0..K
    //   h[-m] = h[m] for m in 1..N stored at index conv_len - m
    let mut b_buf = vec![Complex::new(0.0, 0.0); conv_len];
    for (m, item) in b_buf.iter_mut().take(output_bins).enumerate() {
        let m_f = m as f64;
        let phase = 0.5 * delta_omega * m_f * m_f;
        *item = Complex::new(0.0, phase).exp();
    }
    for m in 1..n_samples {
        let m_f = m as f64;
        let phase = 0.5 * delta_omega * m_f * m_f;
        b_buf[conv_len - m] = Complex::new(0.0, phase).exp();
    }

    // 2. Compute fast convolution using FFT primitives
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(conv_len);
    fft.process(&mut a_buf);
    fft.process(&mut b_buf);

    for (a, b) in a_buf.iter_mut().zip(b_buf.iter()) {
        *a *= *b;
    }

    let ifft = planner.plan_fft_inverse(conv_len);
    ifft.process(&mut a_buf);

    // 3. Post-chirp modulation and normalization
    let scale = 1.0 / conv_len as f64;
    let mut output = Vec::with_capacity(output_bins);
    for (k, &val) in a_buf.iter().take(output_bins).enumerate() {
        let k_f = k as f64;
        let phase = -0.5 * delta_omega * k_f * k_f;
        let post_chirp = Complex::new(0.0, phase).exp();
        output.push(val * scale * post_chirp);
    }

    output
}

/// Helper struct to configure CZT using Spatial Parameters ("Fonzi" Step 1).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpatialCztConfig {
    /// Start distance in meters ($A$ parameter equivalent).
    pub start_distance: f64,
    /// Resolution step size in meters ($W$ parameter equivalent).
    pub step_distance: f64,
    /// Number of output bins ($K$).
    pub output_bins: usize,
    /// Radar Bandwidth in Hz ($B$).
    pub bandwidth: f64,
    /// Chirp Time in seconds ($T_c$).
    pub chirp_time: f64,
    /// Speed of light ($c$).
    pub c: f64,
}

impl SpatialCztConfig {
    /// Converts spatial parameters to frequency parameters and runs the CZT.
    ///
    /// # Arguments
    ///
    /// * `signal` - The time-domain input signal.
    /// * `sample_rate` - ADC sample rate ($f_s$).
    #[verified_engine::verified]
    pub fn process(&self, signal: &[Complex<f64>], sample_rate: f64) -> Vec<Complex<f64>> {
        // Frequency per meter slope: S_f = (2 * B) / (c * T_c)
        let slope = (2.0 * self.bandwidth) / (self.c * self.chirp_time);

        // f_start = slope * start_distance
        let start_freq = slope * self.start_distance;

        // Total bandwidth of the window = slope * (step_size * bins)
        let zoom_bandwidth = slope * (self.step_distance * self.output_bins as f64);

        chirp_z_transform(
            signal,
            start_freq,
            zoom_bandwidth,
            sample_rate,
            self.output_bins,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_czt_matches_direct_recurrence() {
        let test_cases = [(8, 8), (16, 32), (64, 64), (128, 32), (32, 128)];
        let sample_rate = 1000.0;
        let start_freq = 50.0;
        let bandwidth = 200.0;

        for (n_samples, n_bins) in test_cases {
            // Generate a multi-tone test signal
            let signal: Vec<Complex<f64>> = (0..n_samples)
                .map(|i| {
                    let t = i as f64 / sample_rate;
                    let v1 = (2.0 * PI * 80.0 * t).sin();
                    let v2 = 0.5 * (2.0 * PI * 150.0 * t).cos();
                    Complex::new(v1 + v2, 0.0)
                })
                .collect();

            let fast = chirp_z_transform(&signal, start_freq, bandwidth, sample_rate, n_bins);
            let direct =
                direct_chirp_z_transform(&signal, start_freq, bandwidth, sample_rate, n_bins);

            assert_eq!(fast.len(), n_bins);
            assert_eq!(direct.len(), n_bins);

            for k in 0..n_bins {
                let diff = (fast[k] - direct[k]).norm();
                assert!(
                    diff < 1e-10,
                    "Discrepancy at bin {} for N={}, K={}: fast={:?}, direct={:?}, diff={}",
                    k,
                    n_samples,
                    n_bins,
                    fast[k],
                    direct[k],
                    diff
                );
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_czt_fallback_and_edge_cases() {
        let signal = vec![Complex::new(1.0, 0.0); 16];

        // Zero output bins
        let empty_out = chirp_z_transform(&signal, 0.0, 100.0, 1000.0, 0);
        assert!(empty_out.is_empty());

        // Tiny signal fallback (e.g. N = 1, 2, 3, 4)
        for n in 1..=4 {
            let small_sig = vec![Complex::new(1.0, 0.5); n];
            let res = chirp_z_transform(&small_sig, 10.0, 50.0, 1000.0, 8);
            let direct = direct_chirp_z_transform(&small_sig, 10.0, 50.0, 1000.0, 8);
            assert_eq!(res.len(), 8);
            for i in 0..8 {
                assert!((res[i] - direct[i]).norm() < 1e-12);
            }
        }

        // Tiny bin count fallback (e.g. K = 1, 2, 4)
        for k in 1..=4 {
            let res = chirp_z_transform(&signal, 10.0, 50.0, 1000.0, k);
            let direct = direct_chirp_z_transform(&signal, 10.0, 50.0, 1000.0, k);
            assert_eq!(res.len(), k);
            for i in 0..k {
                assert!((res[i] - direct[i]).norm() < 1e-12);
            }
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_czt_spectral_peak_accuracy() {
        let n_samples = 256;
        let sample_rate = 10_000.0;
        let target_tone = 1250.0; // 1250 Hz

        let signal: Vec<Complex<f64>> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / sample_rate;
                Complex::new((2.0 * PI * target_tone * t).cos(), 0.0)
            })
            .collect();

        // Zoom window: 1000 Hz to 1500 Hz (bandwidth 500 Hz) with 100 bins
        let start_freq = 1000.0;
        let bandwidth = 500.0;
        let n_bins = 100;
        let spectrum = chirp_z_transform(&signal, start_freq, bandwidth, sample_rate, n_bins);

        let mut max_mag = -1.0;
        let mut peak_idx = 0;
        for (i, val) in spectrum.iter().enumerate() {
            let mag = val.norm();
            if mag > max_mag {
                max_mag = mag;
                peak_idx = i;
            }
        }

        let peak_freq = start_freq + bandwidth * (peak_idx as f64 / n_bins as f64);
        assert!(
            (peak_freq - target_tone).abs() <= (bandwidth / n_bins as f64),
            "Peak frequency {} deviated from target tone {}",
            peak_freq,
            target_tone
        );
    }
}
