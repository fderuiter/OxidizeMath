// [cite:pure_math]
mod polygon;

pub use polygon::*;

use super::{
    CircleInversion, DiskPoint, HalfPlanePoint, HyperbolicError, cayley_boundary_to_real,
    cayley_disk_to_half_plane,
};
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

/// Exact geodesic representation on the Poincaré disk $\overline{\mathbb{D}}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeodesicDisk {
    /// Euclidean diameter passing through the origin along `unit_dir` ($|u| = 1$).
    Diameter {
        /// Unit tangent vector along the diameter.
        unit_dir: Complex64,
    },
    /// Euclidean circle $|z - c|^2 = R^2$ orthogonal to $|z| = 1$ ($|c|^2 = 1 + R^2$).
    OrthogonalArc {
        /// Euclidean circle center $c$ outside the unit disk ($|c| > 1$).
        center: Complex64,
        /// Euclidean circle radius $R = \sqrt{|c|^2 - 1}$.
        radius: f64,
    },
}

impl GeodesicDisk {
    /// Computes the unique Poincaré disk geodesic passing through two distinct points $z_1, z_2 \in \overline{\mathbb{D}}$.
    #[verified_engine::verified]
    pub fn through_points(z1: Complex64, z2: Complex64) -> Result<Self, HyperbolicError> {
        if !z1.re.is_finite()
            || !z1.im.is_finite()
            || !z2.re.is_finite()
            || !z2.im.is_finite()
            || z1.norm_sqr() > 1.0 + 1.0e-9
            || z2.norm_sqr() > 1.0 + 1.0e-9
        {
            return Err(HyperbolicError::InvalidPolygon {
                reason: "geodesic endpoints must be finite and lie in the closed unit disk",
            });
        }
        let diff = z2 - z1;
        let chord = diff.re.hypot(diff.im);
        if chord <= 1.0e-12 {
            return Err(HyperbolicError::InvalidPolygon {
                reason: "geodesic endpoints must be distinct",
            });
        }
        let cross = z1.re * z2.im - z1.im * z2.re;
        if cross.abs() <= 1.0e-11 {
            let unit_dir = Complex64::new(diff.re / chord, diff.im / chord);
            assert!((unit_dir.norm_sqr() - 1.0).abs() < 1.0e-9);
            return Ok(Self::Diameter { unit_dir });
        }
        let b1 = 0.5 * (z1.norm_sqr().min(1.0) + 1.0);
        let b2 = 0.5 * (z2.norm_sqr().min(1.0) + 1.0);
        let cx = (b1 * z2.im - b2 * z1.im) / cross;
        let cy = (z1.re * b2 - z2.re * b1) / cross;
        let center = Complex64::new(cx, cy);
        let rad_sq = center.norm_sqr() - 1.0;
        if !rad_sq.is_finite() || rad_sq <= 1.0e-16 {
            let unit_dir = Complex64::new(diff.re / chord, diff.im / chord);
            return Ok(Self::Diameter { unit_dir });
        }
        let radius = rad_sq.sqrt();
        assert!(radius > 0.0 && radius.is_finite());
        Ok(Self::OrthogonalArc { center, radius })
    }

    /// Converts this geodesic into its Coxeter [`CircleInversion`] reflection mirror.
    #[must_use]
    #[verified_engine::verified]
    pub fn inversion(self) -> CircleInversion {
        match self {
            Self::Diameter { unit_dir } => {
                debug_assert!((unit_dir.norm_sqr() - 1.0).abs() < 1.0e-8);
                CircleInversion::Diameter { unit_dir }
            }
            Self::OrthogonalArc { center, radius } => {
                debug_assert!(radius > 0.0);
                CircleInversion::OrthogonalCircle { center, radius }
            }
        }
    }
}

/// Wraps an angle difference into $(-\pi, \pi]$.
#[verified_engine::verified]
fn wrap_angle_signed(angle: f64) -> f64 {
    assert!(angle.is_finite());
    let mut a = (angle + PI).rem_euclid(TAU) - PI;
    if a <= -PI {
        a += TAU;
    }
    debug_assert!((-PI..=PI).contains(&a));
    a
}

/// Fallible companion to [`sample_disk_geodesic_into`].
#[verified_engine::verified]
pub fn try_sample_disk_geodesic_into(
    z1: Complex64,
    z2: Complex64,
    out: &mut [[f64; 2]],
) -> core::result::Result<usize, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        crate::error::ensure_math(
            z1.re.is_finite() && z2.re.is_finite(),
            "sample_disk_geodesic_into: precondition or invariant violated",
        )?;
        if out.is_empty() {
            return Ok(0);
        }
        if out.len() == 1 {
            out[0] = [z1.re, z1.im];
            return Ok(1);
        }
        let count = out.len();
        let denom = (count - 1) as f64;
        match GeodesicDisk::through_points(z1, z2) {
            Ok(GeodesicDisk::OrthogonalArc { center, radius }) => {
                let d1 = z1 - center;
                let d2 = z2 - center;
                let theta1 = d1.im.atan2(d1.re);
                let delta = wrap_angle_signed(d2.im.atan2(d2.re) - theta1);
                for (i, slot) in out.iter_mut().enumerate() {
                    let t = (i as f64) / denom;
                    let ang = theta1 + t * delta;
                    *slot = [
                        center.re + radius * ang.cos(),
                        center.im + radius * ang.sin(),
                    ];
                }
            }
            _ => {
                for (i, slot) in out.iter_mut().enumerate() {
                    let t = (i as f64) / denom;
                    *slot = [z1.re + t * (z2.re - z1.re), z1.im + t * (z2.im - z1.im)];
                }
            }
        }
        debug_assert_eq!(count, out.len());
        count
    })
}

/// Samples the Poincaré disk geodesic segment from `z1` to `z2` into `out` without heap allocation.
///
/// Returns the number of points written (`out.len()`).
#[verified_engine::verified]
pub fn sample_disk_geodesic_into(z1: Complex64, z2: Complex64, out: &mut [[f64; 2]]) -> usize {
    try_sample_disk_geodesic_into(z1, z2, out)
        .expect("sample_disk_geodesic_into preconditions must hold")
}

/// Exact geodesic representation on the upper half-plane $\mathbb{H}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeodesicHalfPlane {
    /// Vertical Euclidean ray/line $u = x_0$.
    VerticalLine {
        /// Constant real coordinate $x_0$.
        x: f64,
    },
    /// Euclidean semicircle centered at $(c_x, 0)$ on the real axis with radius $R > 0$.
    Semicircle {
        /// Real-axis center coordinate $c_x$.
        center_x: f64,
        /// Euclidean radius $R > 0$.
        radius: f64,
    },
}

impl GeodesicHalfPlane {
    /// Computes the unique upper half-plane geodesic passing through two distinct points $w_1, w_2 \in \mathbb{H}$.
    #[verified_engine::verified]
    pub fn through_points(q1: HalfPlanePoint, q2: HalfPlanePoint) -> Result<Self, HyperbolicError> {
        let w1 = q1.as_complex();
        let w2 = q2.as_complex();
        let diff = w2 - w1;
        if diff.re.hypot(diff.im) <= 1.0e-12 {
            return Err(HyperbolicError::InvalidPolygon {
                reason: "half-plane geodesic endpoints must be distinct",
            });
        }
        let du = w2.re - w1.re;
        if du.abs() <= 1.0e-11 {
            return Ok(Self::VerticalLine {
                x: 0.5 * (w1.re + w2.re),
            });
        }
        let center_x = (w2.norm_sqr() - w1.norm_sqr()) / (2.0 * du);
        let radius = (w1.re - center_x).hypot(w1.im);
        assert!(radius > 0.0 && center_x.is_finite());
        Ok(Self::Semicircle { center_x, radius })
    }
}

/// Fallible companion to [`sample_half_plane_geodesic_into`].
#[verified_engine::verified]
pub fn try_sample_half_plane_geodesic_into(
    q1: HalfPlanePoint,
    q2: HalfPlanePoint,
    out: &mut [[f64; 2]],
) -> core::result::Result<usize, math_commons::error::MathError> {
    #[allow(unused_braces)]
    Ok({
        crate::error::ensure_math(
            q1.im() > 0.0 && q2.im() > 0.0,
            "sample_half_plane_geodesic_into: precondition or invariant violated",
        )?;
        if out.is_empty() {
            return Ok(0);
        }
        if out.len() == 1 {
            out[0] = [q1.re(), q1.im()];
            return Ok(1);
        }
        let count = out.len();
        let denom = (count - 1) as f64;
        let w1 = q1.as_complex();
        let w2 = q2.as_complex();
        match GeodesicHalfPlane::through_points(q1, q2) {
            Ok(GeodesicHalfPlane::Semicircle { center_x, radius }) => {
                let theta1 = w1.im.atan2(w1.re - center_x);
                let theta2 = w2.im.atan2(w2.re - center_x);
                for (i, slot) in out.iter_mut().enumerate() {
                    let t = (i as f64) / denom;
                    let ang = theta1 + t * (theta2 - theta1);
                    *slot = [center_x + radius * ang.cos(), (radius * ang.sin()).max(0.0)];
                }
            }
            _ => {
                let log_ratio = (w2.im / w1.im).ln();
                for (i, slot) in out.iter_mut().enumerate() {
                    let t = (i as f64) / denom;
                    *slot = [w1.re + t * (w2.re - w1.re), w1.im * (t * log_ratio).exp()];
                }
            }
        }
        debug_assert_eq!(count, out.len());
        count
    })
}

/// Samples the upper half-plane geodesic segment between `q1` and `q2` into `out` without heap allocation.
#[verified_engine::verified]
pub fn sample_half_plane_geodesic_into(
    q1: HalfPlanePoint,
    q2: HalfPlanePoint,
    out: &mut [[f64; 2]],
) -> usize {
    try_sample_half_plane_geodesic_into(q1, q2, out)
        .expect("sample_half_plane_geodesic_into preconditions must hold")
}

/// Euclidean circle in $\mathbb{D}$ tangent to $\partial\mathbb{D}$ at an ideal basepoint $\xi = e^{i\theta}$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorocycleDisk {
    /// Unit ideal tangency point $\xi \in \partial\mathbb{D}$ ($|\xi| = 1$).
    pub ideal_base: Complex64,
    /// Euclidean center $c = (1 - r)\xi$ inside $\mathbb{D}$.
    pub euclidean_center: Complex64,
    /// Euclidean radius $r \in (0, 1)$.
    pub euclidean_radius: f64,
}

/// Upper half-plane image of a horocycle under the Cayley transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HorocycleHalfPlane {
    /// Horizontal line $\operatorname{Im}(w) = h > 0$ (when the ideal basepoint is at $\infty$, i.e. $\xi = 1$).
    HorizontalLine {
        /// Constant positive height $h > 0$.
        height: f64,
    },
    /// Euclidean circle tangent to the real axis at `base_x` with Euclidean radius `radius > 0`.
    TangentCircle {
        /// Real-axis tangency coordinate $u_0 \in \mathbb{R}$.
        base_x: f64,
        /// Euclidean radius $R > 0$ (center is $u_0 + i R$).
        radius: f64,
    },
}

impl HorocycleDisk {
    /// Constructs a horocycle tangent to $\partial\mathbb{D}$ at unit direction `ideal_base` with Euclidean radius $r \in (0, 1)$.
    #[verified_engine::verified]
    pub fn from_radius(
        ideal_base: Complex64,
        euclidean_radius: f64,
    ) -> Result<Self, HyperbolicError> {
        let base_norm = ideal_base.re.hypot(ideal_base.im);
        if !base_norm.is_finite() || base_norm <= 1.0e-12 {
            return Err(HyperbolicError::InvalidHorocycle {
                reason: "ideal basepoint must be non-zero and finite",
            });
        }
        if !euclidean_radius.is_finite() || !(1.0e-9..1.0).contains(&euclidean_radius) {
            return Err(HyperbolicError::InvalidHorocycle {
                reason: "Euclidean horocycle radius must lie strictly in (0, 1)",
            });
        }
        let xi = Complex64::new(ideal_base.re / base_norm, ideal_base.im / base_norm);
        let euclidean_center = xi * (1.0 - euclidean_radius);
        assert!((xi.norm_sqr() - 1.0).abs() < 1.0e-9);
        Ok(Self {
            ideal_base: xi,
            euclidean_center,
            euclidean_radius,
        })
    }

    /// Constructs the unique horocycle tangent at `ideal_base` passing through interior point `through`.
    #[verified_engine::verified]
    pub fn through_point(
        ideal_base: Complex64,
        through: DiskPoint,
    ) -> Result<Self, HyperbolicError> {
        let base_norm = ideal_base.re.hypot(ideal_base.im);
        if !base_norm.is_finite() || base_norm <= 1.0e-12 {
            return Err(HyperbolicError::InvalidHorocycle {
                reason: "ideal basepoint must be non-zero and finite",
            });
        }
        let xi = Complex64::new(ideal_base.re / base_norm, ideal_base.im / base_norm);
        let z0 = through.as_complex();
        let diff_sq = (z0 - xi).norm_sqr();
        let denom = diff_sq + (1.0 - z0.norm_sqr());
        assert!(denom > 0.0);
        let r = (diff_sq / denom).clamp(1.0e-8, 1.0 - 1.0e-8);
        Self::from_radius(xi, r)
    }

    /// Fallible companion to [`sample_into`](Self::sample_into).
    #[verified_engine::verified]
    pub fn try_sample_into(
        self,
        out: &mut [[f64; 2]],
    ) -> core::result::Result<usize, math_commons::error::MathError> {
        #[allow(unused_braces)]
        Ok({
            crate::error::ensure_math(
                self.euclidean_radius > 0.0 && self.euclidean_radius < 1.0,
                "sample_into: precondition or invariant violated",
            )?;
            if out.is_empty() {
                return Ok(0);
            }
            let n = out.len();
            let denom = n.saturating_sub(1).max(1) as f64;
            for (i, slot) in out.iter_mut().enumerate() {
                let ang = TAU * (i as f64) / denom;
                *slot = [
                    self.euclidean_center.re + self.euclidean_radius * ang.cos(),
                    self.euclidean_center.im + self.euclidean_radius * ang.sin(),
                ];
            }
            debug_assert_eq!(n, out.len());
            n
        })
    }

    /// Samples the closed horocycle circle in $\mathbb{D}$ into `out` without heap allocation.
    #[verified_engine::verified]
    pub fn sample_into(self, out: &mut [[f64; 2]]) -> usize {
        self.try_sample_into(out)
            .expect("sample_into preconditions must hold")
    }

    /// Fallible companion to [`to_half_plane`](Self::to_half_plane).
    #[verified_engine::verified]
    pub fn try_to_half_plane(
        self,
    ) -> core::result::Result<HorocycleHalfPlane, math_commons::error::MathError> {
        #[allow(unused_braces)]
        Ok({
            let antipodal = DiskPoint::from_complex_clamped(
                self.euclidean_center - self.ideal_base * self.euclidean_radius,
            );
            let hp_antipodal = cayley_disk_to_half_plane(antipodal);
            if let Some(base_x) = cayley_boundary_to_real(self.ideal_base) {
                let du = hp_antipodal.re() - base_x;
                let dv = hp_antipodal.im();
                let radius = ((du * du + dv * dv) / (2.0 * dv)).max(1.0e-8);
                crate::error::ensure_math(
                    radius.is_finite() && radius > 0.0,
                    "to_half_plane: precondition or invariant violated",
                )?;
                debug_assert!(radius > 0.0);
                HorocycleHalfPlane::TangentCircle { base_x, radius }
            } else {
                HorocycleHalfPlane::HorizontalLine {
                    height: hp_antipodal.im(),
                }
            }
        })
    }

    /// Maps this Poincaré disk horocycle into the upper half-plane $\mathbb{H}$ via the Cayley map.
    #[must_use]
    #[verified_engine::verified]
    pub fn to_half_plane(self) -> HorocycleHalfPlane {
        self.try_to_half_plane()
            .expect("to_half_plane preconditions must hold")
    }
}
