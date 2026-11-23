//! Optical Sensing and Virtual Detector Plane Models.
//!
//! Combines radiometric Lambertian emitter intensity and ADC-limited laser range resolution with
//! zero-allocation 2D wavefield irradiance sampling along a movable virtual detector line.

use math_commons::math_kernel::types::flatten_2d_index;
use pure_math::pure_math::analysis::pde::wave::WaveField2D;
use serde::{Deserialize, Serialize};

// [cite:optics]

/// Calculates intensity using the Lambertian Intensity Profile.
///
/// $$ I_{tx}(r_i, \phi_i) = \frac{(n + 1) P_{tx}}{2\pi r_i^2} \cos^n(\phi_i) $$
///
/// # Arguments
///
/// * `power_tx` ($P_{tx}$) - Transmitted optical power.
/// * `distance` ($r_i$) - Distance from source to point.
/// * `angle` ($\phi_i$) - Off-normal angle in radians.
/// * `mode_n` ($n$) - Radiation pattern mode number (related to half-power angle).
///
/// # Returns
///
/// * `f64` - Intensity at the point.
#[verified_engine::verified]
pub fn lambertian_intensity(power_tx: f64, distance: f64, angle: f64, mode_n: f64) -> f64 {
    if distance <= 0.0 {
        return 0.0;
    }
    let numerator = (mode_n + 1.0) * power_tx * angle.cos().powf(mode_n);
    let denominator = 2.0 * std::f64::consts::PI * distance.powi(2);

    numerator / denominator
}

/// Calculates LWS Range Resolution (ADC-Limited).
///
/// $$ \Delta r_{\min} = r_0 - \left[(r_0 - \Delta r_{\max})^{-4} 2^{B_{\min}} + r_0^{-4}\right]^{-1/4} $$
///
/// # Arguments
///
/// * `r0` ($r_0$) - Nominal range.
/// * `delta_r_max` ($\Delta r_{\max}$) - Maximum displacement range.
/// * `bits` ($B_{\min}$) - Number of ADC bits.
///
/// # Returns
///
/// * `f64` - Minimum resolvable displacement ($\Delta r_{\min}$).
#[verified_engine::verified]
pub fn lws_range_resolution(r0: f64, delta_r_max: f64, bits: u32) -> f64 {
    let term1 = (r0 - delta_r_max).powi(-4);
    let term2 = 2.0_f64.powi(bits as i32);
    let term3 = r0.powi(-4);

    let inner = term1 * term2 + term3;
    let bracket = inner.powf(-0.25);

    r0 - bracket
}

/// Configuration of a movable vertical optical detector line across a [`WaveField2D`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Physical x-coordinate of the virtual detector plane.
    pub detector_x: f64,
    /// Reference x-coordinate of the optical source or aperture plane.
    pub reference_x: f64,
    /// Fractional active detector height in `(0.1, 1.0]`.
    pub active_span_fraction: f64,
    /// Calibrated optical transmitter power $P_{tx}$ (W).
    pub transmitted_power: f64,
    /// Lambertian emitter order $n \ge 0$.
    pub lambertian_mode: f64,
    /// Maximum sensor range excursion $\Delta r_{\max}$ (m).
    pub delta_r_max: f64,
    /// Quantizer resolution $B_{\min}$ in bits.
    pub adc_bits: u32,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            detector_x: 0.74,
            reference_x: 0.28,
            active_span_fraction: 0.76,
            transmitted_power: 1.0,
            lambertian_mode: 1.0,
            delta_r_max: 0.08,
            adc_bits: 12,
        }
    }
}

/// Calibrated wave irradiance and optical sensor metrics measured along a detector plane.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::optics::sensing::{
///     DetectorConfig, sample_detector_line_in_place,
/// };
/// use math_explorer::pure_math::analysis::pde::wave::{
///     WaveField2D, WaveFieldConfig2D, WaveSource2D, WaveSourceKind,
/// };
///
/// let mut field = WaveField2D::new(WaveFieldConfig2D::default());
/// let src = WaveSource2D {
///     kind: WaveSourceKind::PlaneWave { x_index: 18, y_min: 20, y_max: 108 },
///     frequency: 12.0,
///     amplitude: 60.0,
///     phase: 0.0,
/// };
/// for _ in 0..80 {
///     field.step(field.cfl_dt(), Some(&src));
/// }
/// let mut profile = vec![0.0; 128];
/// let readout = sample_detector_line_in_place(&field, &DetectorConfig::default(), &mut profile);
/// assert!(readout.lambertian_intensity > 0.0);
/// assert!(readout.lws_range_resolution > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct DetectorReadout {
    /// Physical x-coordinate of the sampled detector column.
    pub detector_x: f64,
    /// Axial separation $r_0 = |x_{\text{det}} - x_{\text{ref}}|$.
    pub axial_range: f64,
    /// Peak smoothed wave irradiance $I_{\max}$ on the detector aperture.
    pub peak_irradiance: f64,
    /// Spatial mean smoothed wave irradiance $\bar{I}$ on the detector aperture.
    pub mean_irradiance: f64,
    /// Minimum smoothed wave irradiance $I_{\min}$ inside the central interference zone.
    pub min_irradiance: f64,
    /// Michelson fringe contrast $(I_{\max} - I_{\min}) / (I_{\max} + I_{\min})$.
    pub fringe_visibility: f64,
    /// Off-normal angle $\phi$ (radians) of the brightest detector bin.
    pub peak_angle_rad: f64,
    /// Radiometric intensity from [`lambertian_intensity`] at `(axial_range, peak_angle_rad)`.
    pub lambertian_intensity: f64,
    /// ADC-limited range resolution from [`lws_range_resolution`] at `axial_range`.
    pub lws_range_resolution: f64,
}

/// Samples smoothed wave irradiance along the vertical detector line into `profile_out`
/// with zero dynamic heap allocation and returns calibrated [`DetectorReadout`] metrics.
#[verified_engine::verified]
pub fn sample_detector_line_in_place(
    field: &WaveField2D,
    cfg: &DetectorConfig,
    profile_out: &mut [f64],
) -> DetectorReadout {
    let fcfg = field.config();
    let w = fcfg.width;
    let h = fcfg.height;
    let margin = fcfg.sponge_width.min(h / 4);
    let col =
        ((cfg.detector_x / fcfg.dx).round() as usize).clamp(margin, w.saturating_sub(margin + 1));
    let actual_x = (col as f64 + 0.5) * fcfg.dx;
    let span_frac = cfg.active_span_fraction.clamp(0.15, 1.0);
    let usable_h = h.saturating_sub(2 * margin).max(2);
    let active_rows = ((usable_h as f64 * span_frac).round() as usize).clamp(2, usable_h);
    let y_start = (h - active_rows) / 2;
    let y_end = y_start + active_rows;

    let irr = field.irradiance();
    let out_len = profile_out.len().min(h);
    for slot in profile_out.iter_mut() {
        *slot = 0.0;
    }

    let mut peak = 0.0_f64;
    let mut min_val = f64::INFINITY;
    let mut sum = 0.0_f64;
    let mut peak_row = h / 2;
    for y in y_start..y_end {
        let v = irr[flatten_2d_index(col, y, w)].max(0.0);
        if y < out_len {
            profile_out[y] = v;
        }
        if v > peak {
            peak = v;
            peak_row = y;
        }
        if v < min_val {
            min_val = v;
        }
        sum += v;
    }
    if !min_val.is_finite() {
        min_val = 0.0;
    }
    let mean = sum / (active_rows as f64);
    let denom = peak + min_val;
    let visibility = if denom > 1e-15 {
        ((peak - min_val) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let axial_range = (actual_x - cfg.reference_x).abs().max(fcfg.dx);
    let center_y = 0.5 * (h as f64) * fcfg.dy;
    let peak_y = (peak_row as f64 + 0.5) * fcfg.dy;
    let peak_angle = ((peak_y - center_y).abs() / axial_range)
        .atan()
        .clamp(0.0, 1.45);
    let safe_delta_r = cfg.delta_r_max.clamp(1e-4, 0.75 * axial_range);
    let safe_bits = cfg.adc_bits.clamp(1, 24);
    let lamb = lambertian_intensity(
        cfg.transmitted_power.max(0.0),
        axial_range,
        peak_angle,
        cfg.lambertian_mode.max(0.0),
    );
    let range_res = lws_range_resolution(axial_range, safe_delta_r, safe_bits).max(0.0);

    DetectorReadout {
        detector_x: actual_x,
        axial_range,
        peak_irradiance: peak,
        mean_irradiance: mean,
        min_irradiance: min_val,
        fringe_visibility: visibility,
        peak_angle_rad: peak_angle,
        lambertian_intensity: lamb,
        lws_range_resolution: range_res,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pure_math::pure_math::analysis::pde::wave::WaveFieldConfig2D;

    #[test]
    #[verified_engine::verified]
    fn test_lambertian() {
        let i = lambertian_intensity(2.0 * std::f64::consts::PI, 1.0, 0.0, 1.0);
        assert!((i - 2.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    fn test_detector_line_sampling_and_range_metrics() {
        let mut field = WaveField2D::new(WaveFieldConfig2D {
            width: 64,
            height: 64,
            dx: 1.0 / 64.0,
            dy: 1.0 / 64.0,
            ..WaveFieldConfig2D::default()
        });
        let src = pure_math::pure_math::analysis::pde::wave::WaveSource2D {
            kind: pure_math::pure_math::analysis::pde::wave::WaveSourceKind::PlaneWave {
                x_index: 14,
                y_min: 14,
                y_max: 50,
            },
            frequency: 12.0,
            amplitude: 80.0,
            phase: 0.0,
        };
        let dt = field.cfl_dt();
        for _ in 0..160 {
            field.step(dt, Some(&src));
        }
        let mut profile = vec![0.0; 64];
        let cfg = DetectorConfig {
            detector_x: 0.65,
            reference_x: 0.22,
            ..DetectorConfig::default()
        };
        let (readout, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            sample_detector_line_in_place(&field, &cfg, &mut profile)
        });
        assert!(readout.peak_irradiance > 0.0);
        assert!(readout.mean_irradiance > 0.0);
        assert!(readout.lambertian_intensity > 0.0);
        assert!(readout.lws_range_resolution > 0.0);
    }
}
