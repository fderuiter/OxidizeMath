//! Zero-allocation equatorial gravitational lensing deflection-table kernel.
//!
//! Precomputes signed impact-parameter deflection angles $\Delta\varphi(b)$, horizon capture masks,
//! gravitational redshift factors, and Morris-Thorne two-sheet traversal flags using
//! [`GeodesicSystem::integrate_bounded_rk4`] over [`SchwarzschildBlackHole`],
//! [`KerrEquatorialMetric`], and [`MorrisThorneWormhole`]. Runtime queries (`sample_impact` and
//! `map_image_uv`) execute in $O(1)$ arithmetic with zero heap allocations inside
//! [`verified_engine::engine::VerifiedEngine::run_verified`].

use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::SVector;
use pure_math::pure_math::analysis::ode::{ArrayState, RungeKutta4};
use pure_math::pure_math::tensor::{GeodesicState, GeodesicStepStatus, GeodesicSystem};

use crate::error::HighEnergyError;

use super::kerr::KerrEquatorialMetric;
use super::schwarzschild::SchwarzschildBlackHole;
use super::wormhole::MorrisThorneWormhole;

/// Dimensionless Schwarzschild critical photon impact factor $b_{\text{crit}} / R_s = \frac{3\sqrt{3}}{2}$.
pub const EINSTEIN_RING_IMPACT_FACTOR: f64 = 2.598_076_211_353_316;

/// Default pre-allocated sample count for [`DeflectionTable`] across $b \in [-b_{\max}, +b_{\max}]$.
pub const DEFAULT_DEFLECTION_BINS: usize = 129;

/// Metric geometry driving a [`DeflectionTable`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LensingMetricSpec {
    /// Static Schwarzschild black hole with event horizon radius $R_s > 0$.
    Schwarzschild {
        /// Event horizon scale $R_s = 2GM/c^2$.
        schwarzschild_radius: f64,
    },
    /// Rotating equatorial Kerr black hole with horizon scale $R_s > 0$ and spin $|a| < 0.5 R_s$.
    KerrEquatorial {
        /// Mass scale $R_s = 2GM/c^2$.
        schwarzschild_radius: f64,
        /// Spin length $a = J/(Mc)$ with $|a| < 0.5 R_s$.
        spin_a: f64,
    },
    /// Traversable zero-tidal-force Morris-Thorne wormhole with throat radius $b_0 > 0$.
    MorrisThorne {
        /// Minimum areal radius $b_0$ at $l = 0$.
        throat_radius: f64,
    },
}

impl LensingMetricSpec {
    /// Returns `(b_negative_boundary, b_positive_boundary)` where horizons capture photons
    /// (for black holes) or rays cross the throat into the $l < 0$ universe (for wormholes).
    #[verified_engine::verified]
    pub fn critical_impact_bounds(&self) -> Result<(f64, f64), HighEnergyError> {
        match *self {
            Self::Schwarzschild {
                schwarzschild_radius: rs,
            } => {
                if !rs.is_finite() || rs <= 0.0 {
                    return Err(HighEnergyError::InvalidRadius {
                        radius: rs,
                        limit: 0.0,
                    });
                }
                let b_c = EINSTEIN_RING_IMPACT_FACTOR * rs;
                Ok((-b_c, b_c))
            }
            Self::KerrEquatorial {
                schwarzschild_radius,
                spin_a,
            } => {
                let kerr = KerrEquatorialMetric::new(schwarzschild_radius, spin_a)?;
                let b1 = kerr.critical_impact_parameter(true);
                let b2 = kerr.critical_impact_parameter(false);
                Ok((b1.min(b2), b1.max(b2)))
            }
            Self::MorrisThorne { throat_radius } => {
                let wh = MorrisThorneWormhole::new(throat_radius)?;
                Ok((-wh.throat_radius, wh.throat_radius))
            }
        }
    }
}

/// Precomputed null-ray deflection record at signed impact parameter `impact_b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeflectionSample {
    /// Signed impact parameter $b = \ell / e$.
    pub impact_b: f64,
    /// Net signed bending angle $\hat{\alpha}(b)$ in radians (`0.0` in flat space).
    pub bending_angle: f64,
    /// Total integrated azimuthal sweep $|\Delta\varphi(b)|$ in radians.
    pub total_dphi: f64,
    /// `true` if the photon plunges into a black hole horizon guard.
    pub captured: bool,
    /// Asymptotic exit sheet: `+1` for upper sheet ($r \to \infty$ or $l > 0$), `-1` for $l < 0$ portal, `0` if captured.
    pub sheet_sign: i8,
    /// Effective observed-to-source frequency factor along the ray (`0.0` if captured).
    pub redshift_factor: f64,
}

impl Default for DeflectionSample {
    fn default() -> Self {
        Self {
            impact_b: 0.0,
            bending_angle: 0.0,
            total_dphi: PI,
            captured: false,
            sheet_sign: 1,
            redshift_factor: 1.0,
        }
    }
}

/// Result of mapping a normalized 2D camera pixel `(u, v)` through [`DeflectionTable::map_image_uv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensedRayHit {
    /// Deflected source-plane horizontal coordinate.
    pub source_u: f64,
    /// Deflected source-plane vertical coordinate.
    pub source_v: f64,
    /// `true` when the pixel lies inside the black hole shadow.
    pub captured: bool,
    /// `+1` for our asymptotic universe, `-1` when peering through a wormhole throat ($l < 0$), `0` if captured.
    pub sheet_sign: i8,
    /// Relativistic Einstein-ring / photon-ring brightness boost near the critical curve.
    pub ring_boost: f64,
    /// Observed frequency/intensity factor along the lensed ray.
    pub redshift_factor: f64,
}

/// Pre-allocated lookup table mapping signed impact parameter $b \in [-b_{\max}, +b_{\max}]$ to
/// geodesic deflection angles, capture masks, and wormhole sheet transitions.
#[derive(Debug, Clone)]
pub struct DeflectionTable {
    spec: LensingMetricSpec,
    camera_distance: f64,
    max_impact_b: f64,
    crit_bounds: (f64, f64),
    samples: Vec<DeflectionSample>,
}

impl DeflectionTable {
    /// Pre-allocates a [`DeflectionTable`] with `bins` entries and exact critical bounds without
    /// running initial ODE ray sweeps until [`Self::rebuild_in_place`] is invoked.
    ///
    /// # Errors
    /// Returns [`HighEnergyError`] if `spec` is invalid.
    pub fn with_capacity(spec: LensingMetricSpec, bins: usize) -> Result<Self, HighEnergyError> {
        let crit_bounds = spec.critical_impact_bounds()?;
        Ok(Self {
            spec,
            camera_distance: 20.0,
            max_impact_b: 12.0,
            crit_bounds,
            samples: vec![DeflectionSample::default(); bins.clamp(17, 2049)],
        })
    }

    /// Allocates a [`DeflectionTable`] with `bins` entries and populates it for `spec`.
    ///
    /// # Errors
    /// Returns [`HighEnergyError`] if `spec`, `camera_distance`, or `max_impact_b` is invalid.
    pub fn build(
        spec: LensingMetricSpec,
        camera_distance: f64,
        max_impact_b: f64,
        bins: usize,
    ) -> Result<Self, HighEnergyError> {
        let mut table = Self::with_capacity(spec, bins)?;
        table.rebuild_in_place(spec, camera_distance, max_impact_b)?;
        Ok(table)
    }

    /// Recomputes all deflection bins in place without reallocating `self.samples`.
    ///
    /// # Errors
    /// Returns [`HighEnergyError`] if `spec`, `camera_distance`, or `max_impact_b` is invalid.
    #[verified_engine::verified]
    pub fn rebuild_in_place(
        &mut self,
        spec: LensingMetricSpec,
        camera_distance: f64,
        max_impact_b: f64,
    ) -> Result<(), HighEnergyError> {
        let crit = spec.critical_impact_bounds()?;
        if !camera_distance.is_finite()
            || camera_distance <= 2.0
            || !max_impact_b.is_finite()
            || max_impact_b <= 0.5
        {
            return Err(HighEnergyError::InvalidRadius {
                radius: camera_distance.min(max_impact_b),
                limit: 2.0,
            });
        }
        self.spec = spec;
        self.camera_distance = camera_distance;
        self.max_impact_b = max_impact_b;
        self.crit_bounds = crit;

        let symmetric = !matches!(spec, LensingMetricSpec::KerrEquatorial { .. });
        let n = self.samples.len();
        let denom = (n.saturating_sub(1)).max(1) as f64;
        let mut idx = 0;
        while idx < n {
            let b = -max_impact_b + 2.0 * max_impact_b * ((idx as f64) / denom);
            let mirror = n.saturating_sub(1).saturating_sub(idx);
            self.samples[idx] = if symmetric && mirror < idx {
                let m = self.samples[mirror];
                DeflectionSample {
                    impact_b: b,
                    bending_angle: -m.bending_angle,
                    ..m
                }
            } else {
                shoot_deflection_ray(spec, crit, camera_distance, b)
            };
            idx += 1;
        }
        Ok(())
    }

    /// Active metric specification.
    #[verified_engine::verified]
    pub fn spec(&self) -> LensingMetricSpec {
        self.spec
    }

    /// Critical impact bounds `(b_neg, b_pos)` of the active shadow or wormhole throat.
    #[verified_engine::verified]
    pub fn critical_impact_bounds(&self) -> (f64, f64) {
        self.crit_bounds
    }

    /// Maximum tabulated impact parameter $b_{\max}$.
    #[verified_engine::verified]
    pub fn max_impact_b(&self) -> f64 {
        self.max_impact_b
    }

    /// Read-only slice of tabulated deflection samples.
    #[verified_engine::verified]
    pub fn samples(&self) -> &[DeflectionSample] {
        &self.samples
    }

    /// Zero-allocation $O(1)$ interpolated lookup at signed impact parameter `b_signed`.
    #[verified_engine::verified]
    pub fn sample_impact(&self, b_signed: f64) -> DeflectionSample {
        let n = self.samples.len();
        if n == 0 || !b_signed.is_finite() {
            return DeflectionSample::default();
        }
        let exact_captured = !matches!(self.spec, LensingMetricSpec::MorrisThorne { .. })
            && b_signed > self.crit_bounds.0
            && b_signed < self.crit_bounds.1;
        if exact_captured {
            return captured_sample(b_signed);
        }
        let clamped_b = b_signed.clamp(-self.max_impact_b, self.max_impact_b);
        let u = ((clamped_b + self.max_impact_b) / (2.0 * self.max_impact_b)).clamp(0.0, 1.0);
        let scaled = u * ((n - 1) as f64);
        let i0 = (scaled.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        let t = scaled - (i0 as f64);
        let (s0, s1) = (self.samples[i0], self.samples[i1]);
        if s0.captured != s1.captured || s0.sheet_sign != s1.sheet_sign {
            return if !s0.captured && (t < 0.5 || s1.captured) {
                s0
            } else {
                s1
            };
        }
        DeflectionSample {
            impact_b: b_signed,
            bending_angle: s0.bending_angle + t * (s1.bending_angle - s0.bending_angle),
            total_dphi: s0.total_dphi + t * (s1.total_dphi - s0.total_dphi),
            captured: false,
            sheet_sign: s0.sheet_sign,
            redshift_factor: s0.redshift_factor + t * (s1.redshift_factor - s0.redshift_factor),
        }
    }

    /// Maps normalized 2D image-plane coordinates `(u, v)` (`[-1, 1]`) to deflected source-plane
    /// coordinates with zero heap allocations.
    #[verified_engine::verified]
    pub fn map_image_uv(&self, u: f64, v: f64, fov_b_max: f64) -> LensedRayHit {
        let (bx, by) = (u * fov_b_max, v * fov_b_max);
        let (b_lookup, crit_edge, frame_twist) = self.effective_sky_impact(bx, by);
        let sample = self.sample_impact(b_lookup);
        if sample.captured {
            return LensedRayHit {
                source_u: 0.0,
                source_v: 0.0,
                captured: true,
                sheet_sign: 0,
                ring_boost: 0.0,
                redshift_factor: 0.0,
            };
        }
        let b_rad = (bx * bx + by * by).sqrt().max(1e-9);
        let (dir_x, dir_y) = (bx / b_rad, by / b_rad);
        let (sin_tw, cos_tw) = frame_twist.sin_cos();
        let rot_x = dir_x * cos_tw - dir_y * sin_tw;
        let rot_y = dir_x * sin_tw + dir_y * cos_tw;

        let defl_norm = sample.bending_angle.abs() / PI.max(1.0);
        let radial_shift = if sample.sheet_sign < 0 {
            (b_rad / crit_edge.max(1e-6)) * 0.85
        } else {
            (b_rad / fov_b_max.max(1e-6)) - 0.55 * defl_norm
        };
        let dist_crit = ((b_rad - crit_edge).abs() / crit_edge.max(1e-6)).max(0.0);
        LensedRayHit {
            source_u: radial_shift * rot_x,
            source_v: radial_shift * rot_y,
            captured: false,
            sheet_sign: sample.sheet_sign,
            ring_boost: (-18.0 * dist_crit).exp(),
            redshift_factor: sample.redshift_factor,
        }
    }

    #[verified_engine::verified]
    fn effective_sky_impact(&self, bx: f64, by: f64) -> (f64, f64, f64) {
        let b_rad = (bx * bx + by * by).sqrt();
        match self.spec {
            LensingMetricSpec::Schwarzschild { .. } | LensingMetricSpec::MorrisThorne { .. } => {
                (b_rad, self.crit_bounds.1.abs(), 0.0)
            }
            LensingMetricSpec::KerrEquatorial {
                schwarzschild_radius: rs,
                spin_a,
            } => {
                let cos_psi = if b_rad > 1e-12 { bx / b_rad } else { 1.0 };
                let (b_pos, b_neg) = (self.crit_bounds.1.abs(), self.crit_bounds.0.abs());
                let w_pos = 0.5 * (1.0 + cos_psi);
                let d_flatten = 1.0 - 0.08 * (spin_a.abs() / (0.5 * rs)) * w_pos;
                let local_crit = (w_pos * b_pos + (1.0 - w_pos) * b_neg) * d_flatten;
                let signed_b = if bx >= 0.0 {
                    b_rad * (b_pos / local_crit.max(1e-6))
                } else {
                    -b_rad * (b_neg / local_crit.max(1e-6))
                };
                let twist = (spin_a * rs) / (b_rad * b_rad + 1.5);
                (signed_b, local_crit, twist)
            }
        }
    }
}

#[verified_engine::verified]
fn shoot_deflection_ray(
    spec: LensingMetricSpec,
    crit: (f64, f64),
    cam_dist: f64,
    b: f64,
) -> DeflectionSample {
    match spec {
        LensingMetricSpec::Schwarzschild {
            schwarzschild_radius,
        } => {
            if b > crit.0 && b < crit.1 {
                captured_sample(b)
            } else {
                shoot_bh_4d_ray(schwarzschild_radius, 0.0, cam_dist, b)
            }
        }
        LensingMetricSpec::KerrEquatorial {
            schwarzschild_radius,
            spin_a,
        } => {
            if b > crit.0 && b < crit.1 {
                captured_sample(b)
            } else {
                shoot_bh_4d_ray(schwarzschild_radius, spin_a, cam_dist, b)
            }
        }
        LensingMetricSpec::MorrisThorne { throat_radius } => {
            shoot_wormhole_3d_ray(throat_radius, cam_dist, b)
        }
    }
}

#[verified_engine::verified]
fn captured_sample(b: f64) -> DeflectionSample {
    DeflectionSample {
        impact_b: b,
        bending_angle: 0.0,
        total_dphi: 0.0,
        captured: true,
        sheet_sign: 0,
        redshift_factor: 0.0,
    }
}

#[verified_engine::verified]
fn shoot_bh_4d_ray(rs: f64, spin_a: f64, r_cam: f64, b: f64) -> DeflectionSample {
    let r0 = r_cam.max(4.0 * rs);
    let delta = (r0 * r0 - rs * r0 + spin_a * spin_a).max(1e-9);
    let g33 = r0 * r0 + spin_a * spin_a + (spin_a * spin_a * rs) / r0;
    let ut = (g33 - ((spin_a * rs) / r0) * b) / delta;
    let uphi = ((spin_a * rs) / r0 + (1.0 - rs / r0) * b) / delta;
    let radial_term = ((delta / (r0 * r0)) * (ut - b * uphi)).max(0.0);
    if radial_term <= 1e-14 {
        return DeflectionSample::default();
    }
    let mut state = ArrayState::<8>::from_pos_vel(
        &SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, 0.0),
        &SVector::<f64, 4>::new(ut, -radial_term.sqrt(), 0.0, uphi),
    );
    let mut solver = RungeKutta4::new(&state);
    let d_lambda = 0.12 * rs.max(0.5);

    let status = if spin_a.abs() < 1e-12 {
        let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(rs) else {
            return captured_sample(b);
        };
        GeodesicSystem::new(bh)
            .integrate_bounded_rk4(&mut solver, &mut state, d_lambda, 480, |s| {
                s.0[1] >= r0 && s.0[5] > 0.0
            })
            .1
    } else {
        let Ok(kerr) = KerrEquatorialMetric::new(rs, spin_a) else {
            return captured_sample(b);
        };
        GeodesicSystem::new(kerr)
            .integrate_bounded_rk4(&mut solver, &mut state, d_lambda, 480, |s| {
                s.0[1] >= r0 && s.0[5] > 0.0
            })
            .1
    };
    if matches!(status, GeodesicStepStatus::HaltedSingularity) {
        return captured_sample(b);
    }
    let r_end = state.0[1].max(b.abs() * 1.001);
    let tail_phi = (b.abs() / r0.max(b.abs() * 1.001)).asin() + (b.abs() / r_end).asin();
    let total_dphi = state.0[3].abs() + tail_phi;
    DeflectionSample {
        impact_b: b,
        bending_angle: (total_dphi - PI).max(0.0) * b.signum(),
        total_dphi,
        captured: false,
        sheet_sign: 1,
        redshift_factor: (1.0 + (spin_a * rs) / (r0 * b.abs().max(rs))).clamp(0.25, 2.0),
    }
}

#[verified_engine::verified]
fn shoot_wormhole_3d_ray(b0: f64, l_cam: f64, b: f64) -> DeflectionSample {
    let Ok(wh) = MorrisThorneWormhole::new(b0) else {
        return DeflectionSample::default();
    };
    let l0 = l_cam.max(3.0 * b0);
    let r2 = b0 * b0 + l0 * l0;
    let ul_sq = (1.0 - (b * b) / r2).max(0.0);
    let mut state = ArrayState::<6>::from_pos_vel(
        &SVector::<f64, 3>::new(0.0, l0, 0.0),
        &SVector::<f64, 3>::new(1.0, -ul_sq.sqrt(), b / r2),
    );
    let mut solver = RungeKutta4::new(&state);
    let d_lambda = 0.12 * b0.max(0.5);
    let _ = GeodesicSystem::new(wh).integrate_bounded_rk4(
        &mut solver,
        &mut state,
        d_lambda,
        450,
        |s| (s.0[1] >= l0 && s.0[4] > 0.0) || (s.0[1] <= -l0 && s.0[4] < 0.0),
    );
    let sheet_sign = if state.0[1] < 0.0 || b.abs() < b0 {
        -1
    } else {
        1
    };
    let r_end = (b0 * b0 + state.0[1] * state.0[1])
        .sqrt()
        .max(b.abs() * 1.001);
    let tail = (b.abs() / r2.sqrt().max(b.abs() * 1.001)).asin() + (b.abs() / r_end).asin();
    let total_dphi = state.0[2].abs() + tail;
    DeflectionSample {
        impact_b: b,
        bending_angle: (total_dphi - PI) * b.signum(),
        total_dphi,
        captured: false,
        sheet_sign,
        redshift_factor: 1.0,
    }
}
