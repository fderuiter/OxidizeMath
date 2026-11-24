//! Closed-form Fraunhofer far-field diffraction profiles and critical-point markers.
//!
//! Evaluates single-slit $\text{sinc}^2\beta$, multi-slit grating interference
//! $\text{sinc}^2\beta\,(\frac{\sin N\gamma}{N\sin\gamma})^2$, and circular Airy disk
//! $(2 J_1(x)/x)^2$ intensity profiles into pre-allocated buffers.

// [cite:optics]

use std::f64::consts::PI;

use pure_math::pure_math::special_functions::bessel::bessel_j;
use serde::{Deserialize, Serialize};

/// First positive root $j_{1,1}$ of $J_1(x) = 0$ governing the Airy first dark ring ($\approx 1.21967\,\lambda/D$).
pub const AIRY_FIRST_ZERO_J1: f64 = 3.831_705_970_207_512_5;

/// Second positive root $j_{1,2}$ of $J_1(x) = 0$.
pub const AIRY_SECOND_ZERO_J1: f64 = 7.015_586_669_815_619;

/// Third positive root $j_{1,3}$ of $J_1(x) = 0$.
pub const AIRY_THIRD_ZERO_J1: f64 = 10.173_468_135_062_722;

/// Diffracting aperture geometry in the Fraunhofer far-field regime.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DiffractionAperture {
    /// Rectangular single slit of width $a$ in micrometers.
    SingleSlit {
        /// Slit width $a$ in micrometers ($\mu\text{m}$).
        slit_width_um: f64,
    },
    /// Periodic $N$-slit transmission grating with slit width $a$ and center-to-center pitch $d$.
    MultiSlitGrating {
        /// Individual slit opening width $a$ in micrometers ($\mu\text{m}$).
        slit_width_um: f64,
        /// Center-to-center grating period $d \ge a$ in micrometers ($\mu\text{m}$).
        slit_pitch_um: f64,
        /// Number of illuminated coherent slits $N \ge 1$.
        slit_count: usize,
    },
    /// Circular pinhole or telescope objective of diameter $D$ producing an Airy pattern.
    CircularAiry {
        /// Aperture diameter $D$ in micrometers ($\mu\text{m}$).
        diameter_um: f64,
    },
}

impl DiffractionAperture {
    /// Human-readable label for UI controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SingleSlit { .. } => "Single Slit (sinc²)",
            Self::MultiSlitGrating { .. } => "Multi-Slit Grating",
            Self::CircularAiry { .. } => "Circular Aperture (Airy J₁)",
        }
    }
}

/// Classification of analytical critical points along a Fraunhofer diffraction pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CriticalPointKind {
    /// Constructive grating principal maximum $\sin\theta_m = m\lambda / d$.
    #[default]
    PrincipalMaximum,
    /// Single-slit envelope dark minimum $\sin\theta_m = m\lambda / a$.
    EnvelopeMinimum,
    /// Circular Airy dark ring $\pi D\sin\theta / \lambda = j_{1,m}$.
    AiryDarkRing,
}

/// Analytical critical-point marker on the observation screen.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CriticalPointMarker {
    /// Diffraction angle $\theta$ in radians.
    pub angle_rad: f64,
    /// Transverse observation screen position $y = L\tan\theta$ in millimeters.
    pub screen_y_mm: f64,
    /// Normalized intensity $I(\theta)/I_0 \in [0, 1]$ at the marker.
    pub intensity: f64,
    /// Signed diffraction or zero order $m$.
    pub order: i32,
    /// Physical marker classification.
    pub kind: CriticalPointKind,
}

/// Observation geometry and wavelength parameters for Fraunhofer sampling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FraunhoferConfig {
    /// Aperture type and micrometer dimensions.
    pub aperture: DiffractionAperture,
    /// Monochromatic vacuum wavelength $\lambda$ in nanometers.
    pub wavelength_nm: f64,
    /// Aperture-to-screen propagation distance $L$ in meters.
    pub screen_distance_m: f64,
    /// Angular half-span $\theta_{\max}$ in radians sampled across $[-\theta_{\max}, +\theta_{\max}]$.
    pub half_angle_rad: f64,
}

/// Pre-allocated output buffer for zero-allocation Fraunhofer profile sampling and critical-point markers.
#[derive(Debug, Clone)]
pub struct FraunhoferProfileBuffer {
    angles_rad: Vec<f64>,
    screen_y_mm: Vec<f64>,
    intensities: Vec<f64>,
    active_samples: usize,
    markers: Vec<CriticalPointMarker>,
    active_markers: usize,
}

impl PartialEq for FraunhoferProfileBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.angles_rad() == other.angles_rad()
            && self.screen_y_mm() == other.screen_y_mm()
            && self.intensities() == other.intensities()
            && self.markers() == other.markers()
    }
}

impl FraunhoferProfileBuffer {
    /// Allocates buffers for up to `max_samples` angular samples and `max_markers` critical-point markers.
    #[must_use]
    pub fn new(max_samples: usize, max_markers: usize) -> Self {
        let cap_s = max_samples.max(3);
        let cap_m = max_markers.max(4);
        Self {
            angles_rad: vec![0.0; cap_s],
            screen_y_mm: vec![0.0; cap_s],
            intensities: vec![0.0; cap_s],
            active_samples: 0,
            markers: vec![CriticalPointMarker::default(); cap_m],
            active_markers: 0,
        }
    }

    /// Sampled diffraction angles $\theta$ in radians.
    #[must_use]
    pub fn angles_rad(&self) -> &[f64] {
        &self.angles_rad[..self.active_samples]
    }

    /// Sampled screen positions $y = L\tan\theta$ in millimeters.
    #[must_use]
    pub fn screen_y_mm(&self) -> &[f64] {
        &self.screen_y_mm[..self.active_samples]
    }

    /// Sampled normalized intensities $I(\theta)/I_0 \in [0, 1]$.
    #[must_use]
    pub fn intensities(&self) -> &[f64] {
        &self.intensities[..self.active_samples]
    }

    /// Analytical critical-point markers within $[-\theta_{\max}, +\theta_{\max}]$.
    #[must_use]
    pub fn markers(&self) -> &[CriticalPointMarker] {
        &self.markers[..self.active_markers]
    }
}

fn sinc_sq(beta: f64) -> f64 {
    let b = beta.abs();
    if b < 1.0e-6 {
        let b2 = b * b;
        (1.0 - b2 / 6.0).powi(2)
    } else {
        let s = beta.sin() / beta;
        s * s
    }
}

fn j1_asymptotic(ax: f64) -> f64 {
    let z = 8.0 * ax;
    let inv_z2 = 1.0 / (z * z);
    let p = 1.0 + 15.0 * 9.0 * inv_z2 * 0.5 - (15.0 * 9.0 * 21.0 * 45.0 / 24.0) * inv_z2 * inv_z2;
    let q = (3.0 / z) * (1.0 - (15.0 * 21.0 / 6.0) * inv_z2);
    let chi = ax - 0.75 * PI;
    (2.0 / (PI * ax)).sqrt() * (p * chi.cos() - q * chi.sin())
}

/// Evaluates normalized circular Airy intensity $\bigl(2 J_1(x) / x\bigr)^2 \in [0, 1]$.
///
/// Reuses `pure_math::special_functions::bessel::bessel_j(1.0, |x|)` on $[10^{-6}, 20]$ and switches
/// to Hankel's asymptotic expansion for $|x| > 20$ to prevent alternating power-series cancellation.
#[verified_engine::verified]
#[must_use]
pub fn airy_normalized_intensity(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1.0e-6 {
        let x2 = ax * ax;
        let amp = 1.0 - 0.125 * x2 + (1.0 / 192.0) * x2 * x2;
        return amp * amp;
    }
    let j1 = if ax <= 20.0 {
        bessel_j(1.0, ax)
    } else {
        j1_asymptotic(ax)
    };
    let amp = 2.0 * j1 / ax;
    (amp * amp).clamp(0.0, 1.0)
}

/// Evaluates normalized Fraunhofer intensity $I(\theta)/I_0 \in [0, 1]$ at angle `theta_rad`.
#[verified_engine::verified]
#[must_use]
pub fn fraunhofer_intensity_at_angle(
    aperture: DiffractionAperture,
    wavelength_nm: f64,
    theta_rad: f64,
) -> f64 {
    let lambda_um = (wavelength_nm * 1.0e-3).max(1.0e-4);
    let sin_t = theta_rad.sin();
    match aperture {
        DiffractionAperture::SingleSlit { slit_width_um } => {
            let beta = PI * slit_width_um.max(1.0e-3) * sin_t / lambda_um;
            sinc_sq(beta).clamp(0.0, 1.0)
        }
        DiffractionAperture::MultiSlitGrating {
            slit_width_um,
            slit_pitch_um,
            slit_count,
        } => {
            let a = slit_width_um.max(1.0e-3);
            let d = slit_pitch_um.max(a);
            let n = slit_count.max(1) as f64;
            let beta = PI * a * sin_t / lambda_um;
            let gamma = PI * d * sin_t / lambda_um;
            let sin_g = gamma.sin();
            let grating_factor = if sin_g.abs() < 1.0e-7 {
                1.0
            } else {
                let ratio = (n * gamma).sin() / (n * sin_g);
                ratio * ratio
            };
            (sinc_sq(beta) * grating_factor).clamp(0.0, 1.0)
        }
        DiffractionAperture::CircularAiry { diameter_um } => {
            let x = PI * diameter_um.max(1.0e-3) * sin_t / lambda_um;
            airy_normalized_intensity(x)
        }
    }
}

/// Samples a closed-form Fraunhofer intensity profile and populates critical-point markers with zero allocations.
#[verified_engine::verified]
pub fn sample_fraunhofer_profile(
    config: &FraunhoferConfig,
    sample_count: usize,
    out: &mut FraunhoferProfileBuffer,
) -> usize {
    let n = sample_count.clamp(3, out.angles_rad.len());
    let half_a = config.half_angle_rad.clamp(1.0e-5, 1.35);
    let l_mm = config.screen_distance_m.max(1.0e-3) * 1_000.0;
    for i in 0..n {
        let u = 2.0 * (i as f64) / ((n - 1) as f64) - 1.0;
        let theta = u * half_a;
        out.angles_rad[i] = theta;
        out.screen_y_mm[i] = l_mm * theta.tan();
        out.intensities[i] =
            fraunhofer_intensity_at_angle(config.aperture, config.wavelength_nm, theta);
    }
    out.active_samples = n;
    out.active_markers = populate_markers(config, half_a, l_mm, &mut out.markers);
    n
}

struct MarkerCtx<'a> {
    config: &'a FraunhoferConfig,
    half_angle: f64,
    l_mm: f64,
}

fn push_marker(
    ctx: &MarkerCtx<'_>,
    markers: &mut [CriticalPointMarker],
    count: &mut usize,
    sin_theta: f64,
    order: i32,
    kind: CriticalPointKind,
) {
    if *count >= markers.len() || sin_theta.abs() >= 0.999 {
        return;
    }
    let theta = sin_theta.asin();
    if theta.abs() > ctx.half_angle {
        return;
    }
    markers[*count] = CriticalPointMarker {
        angle_rad: theta,
        screen_y_mm: ctx.l_mm * theta.tan(),
        intensity: fraunhofer_intensity_at_angle(
            ctx.config.aperture,
            ctx.config.wavelength_nm,
            theta,
        ),
        order,
        kind,
    };
    *count += 1;
}

fn populate_markers(
    config: &FraunhoferConfig,
    half_angle: f64,
    l_mm: f64,
    markers: &mut [CriticalPointMarker],
) -> usize {
    let lambda_um = (config.wavelength_nm * 1.0e-3).max(1.0e-4);
    let ctx = MarkerCtx {
        config,
        half_angle,
        l_mm,
    };
    let mut count = 0usize;
    match config.aperture {
        DiffractionAperture::SingleSlit { slit_width_um } => {
            let a = slit_width_um.max(1.0e-3);
            for m in 1..=6_i32 {
                for sign in [-1_i32, 1] {
                    let ord = sign * m;
                    let s_t = (ord as f64) * lambda_um / a;
                    push_marker(
                        &ctx,
                        markers,
                        &mut count,
                        s_t,
                        ord,
                        CriticalPointKind::EnvelopeMinimum,
                    );
                }
            }
        }
        DiffractionAperture::MultiSlitGrating {
            slit_width_um,
            slit_pitch_um,
            ..
        } => populate_grating_markers(
            &ctx,
            slit_width_um,
            slit_pitch_um,
            lambda_um,
            markers,
            &mut count,
        ),
        DiffractionAperture::CircularAiry { diameter_um } => {
            let d = diameter_um.max(1.0e-3);
            let roots = [AIRY_FIRST_ZERO_J1, AIRY_SECOND_ZERO_J1, AIRY_THIRD_ZERO_J1];
            for (idx, &j_root) in roots.iter().enumerate() {
                let sin_r = j_root * lambda_um / (PI * d);
                for sign in [-1_i32, 1] {
                    let ord = sign * ((idx as i32) + 1);
                    let s_t = (sign as f64) * sin_r;
                    push_marker(
                        &ctx,
                        markers,
                        &mut count,
                        s_t,
                        ord,
                        CriticalPointKind::AiryDarkRing,
                    );
                }
            }
        }
    }
    count
}

fn populate_grating_markers(
    ctx: &MarkerCtx<'_>,
    slit_width_um: f64,
    slit_pitch_um: f64,
    lambda_um: f64,
    markers: &mut [CriticalPointMarker],
    count: &mut usize,
) {
    let a = slit_width_um.max(1.0e-3);
    let d = slit_pitch_um.max(a);
    push_marker(
        ctx,
        markers,
        count,
        0.0,
        0,
        CriticalPointKind::PrincipalMaximum,
    );
    for m in 1..=6_i32 {
        for sign in [-1_i32, 1] {
            let ord = sign * m;
            let s_t = (ord as f64) * lambda_um / d;
            push_marker(
                ctx,
                markers,
                count,
                s_t,
                ord,
                CriticalPointKind::PrincipalMaximum,
            );
        }
    }
    for sign in [-1_i32, 1] {
        let s_t = (sign as f64) * lambda_um / a;
        push_marker(
            ctx,
            markers,
            count,
            s_t,
            sign,
            CriticalPointKind::EnvelopeMinimum,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn airy_disk_first_three_zeros_and_central_peak() {
        assert!((airy_normalized_intensity(0.0) - 1.0).abs() < 1.0e-14);
        assert!(airy_normalized_intensity(AIRY_FIRST_ZERO_J1) < 1.0e-12);
        assert!(airy_normalized_intensity(AIRY_SECOND_ZERO_J1) < 1.0e-12);
        assert!(airy_normalized_intensity(AIRY_THIRD_ZERO_J1) < 1.0e-11);
        assert!(airy_normalized_intensity(28.0).is_finite());
    }

    #[test]
    #[verified_engine::verified]
    fn grating_principal_maxima_and_missing_orders() {
        let ap = DiffractionAperture::MultiSlitGrating {
            slit_width_um: 2.0,
            slit_pitch_um: 6.0,
            slit_count: 8,
        };
        let wl = 600.0; // 0.6 um
        let theta_1 = (0.6_f64 / 6.0).asin();
        let theta_3_missing = (3.0 * 0.6_f64 / 6.0).asin(); // m = 3 coincides with first slit minimum d/a = 3
        let i0 = fraunhofer_intensity_at_angle(ap, wl, 0.0);
        let i1 = fraunhofer_intensity_at_angle(ap, wl, theta_1);
        let i3 = fraunhofer_intensity_at_angle(ap, wl, theta_3_missing);
        assert!((i0 - 1.0).abs() < 1.0e-12);
        assert!(i1 > 0.40);
        assert!(i3 < 1.0e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn single_slit_zeros_and_profile_markers_match_analytic_within_1e6() {
        let ap = DiffractionAperture::SingleSlit { slit_width_um: 4.0 };
        let wl = 500.0; // 0.5 um => lambda / a = 0.125
        for order in [1_i32, -1, 2, -2, 3, -3] {
            let theta_zero = ((order as f64) * 0.125).asin();
            let intensity = fraunhofer_intensity_at_angle(ap, wl, theta_zero);
            assert!(intensity < 1.0e-12, "slit zero order {order}: {intensity}");
        }
        let cfg = FraunhoferConfig {
            aperture: ap,
            wavelength_nm: wl,
            screen_distance_m: 1.0,
            half_angle_rad: 0.45,
        };
        let mut buf = FraunhoferProfileBuffer::new(201, 16);
        assert_eq!(sample_fraunhofer_profile(&cfg, 201, &mut buf), 201);
        let m1 = buf
            .markers()
            .iter()
            .find(|m| m.order == 1)
            .expect("first-order minimum marker present");
        assert!((m1.angle_rad - 0.125_f64.asin()).abs() < 1.0e-12);
        assert!(m1.intensity < 1.0e-12);
    }
}
