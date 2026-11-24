//! Parametric 3D knots, multi-component links, and Alexander braid closures.

// [cite:knot_braid_topology]

use super::KnotError;
use crate::pure_math::algebra::BraidWord;
use nalgebra::Point3;
use std::f64::consts::TAU;

/// A collection of smooth closed 3D curves representing a knot (`components.len() == 1`)
/// or an oriented $k$-component spatial link (`components.len() == k`).
///
/// Each component stores uniformly or smoothly ordered points $P_0, \dots, P_{N-1}$ with
/// an implicit closing segment $P_{N-1} \to P_0$.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedLink3D {
    /// Ordered closed curve polylines, one per link component.
    pub components: Vec<Vec<Point3<f64>>>,
}

impl ClosedLink3D {
    /// Validates and constructs a [`ClosedLink3D`] from closed polylines.
    ///
    /// # Errors
    /// Returns [`KnotError`] if `components` is empty or any component has fewer than 3 samples.
    #[verified_engine::verified]
    pub fn new(components: Vec<Vec<Point3<f64>>>) -> Result<Self, KnotError> {
        if components.is_empty() {
            return Err(KnotError::EmptyCurve);
        }
        for comp in &components {
            if comp.len() < 3 {
                return Err(KnotError::InsufficientSamples {
                    required: 3,
                    actual: comp.len(),
                });
            }
        }
        debug_assert!(!components.is_empty());
        Ok(Self { components })
    }

    /// Constructs the 3D Alexander closure $\hat{\beta}$ of an Artin [`BraidWord`].
    ///
    /// # Errors
    /// Returns [`KnotError`] if `samples_per_turn < 8` or radii are non-positive.
    #[verified_engine::verified]
    pub fn from_braid_closure(
        braid: &BraidWord,
        major_radius: f64,
        strand_spacing: f64,
        samples_per_turn: usize,
    ) -> Result<Self, KnotError> {
        if !major_radius.is_finite() || major_radius <= 0.1 || !strand_spacing.is_finite() {
            return Err(KnotError::InvalidParameters {
                reason: "braid closure radii must be finite and positive",
            });
        }
        if samples_per_turn < 8 {
            return Err(KnotError::InsufficientSamples {
                required: 8,
                actual: samples_per_turn,
            });
        }
        Self::new(braid.alexander_closure_3d(major_radius, strand_spacing, samples_per_turn))
    }

    /// Number of closed components in the spatial link.
    #[must_use]
    #[verified_engine::verified]
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Total sample vertices summed across all components.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_samples(&self) -> usize {
        self.components.iter().map(Vec::len).sum()
    }

    /// Maximum Euclidean distance from the origin across all sampled vertices.
    #[must_use]
    #[verified_engine::verified]
    pub fn bounding_radius(&self) -> f64 {
        let mut max_sq = 0.0_f64;
        for comp in &self.components {
            for pt in comp {
                max_sq = max_sq.max(pt.coords.norm_squared());
            }
        }
        let r = max_sq.sqrt();
        debug_assert!(r.is_finite() && r >= 0.0);
        r
    }
}

#[verified_engine::verified]
fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    debug_assert!(a > 0);
    a
}

/// Parametric $(p, q)$ torus knot or torus link embedded on a standard torus in $\mathbb{R}^3$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TorusKnot {
    /// Longitudinal winding number $p \ge 1$.
    pub p: usize,
    /// Meridional winding number $q \neq 0$ (sign controls handedness/chirality).
    pub q: i32,
    /// Major torus radius $R > r$.
    pub major_radius: f64,
    /// Minor torus radius $r > 0$.
    pub minor_radius: f64,
}

impl TorusKnot {
    /// Constructs a validated $(p, q)$ torus knot or torus link.
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidParameters`] if $p = 0$, $q = 0$, or $0 < r < R$ fails.
    #[verified_engine::verified]
    pub fn new(p: usize, q: i32, major_radius: f64, minor_radius: f64) -> Result<Self, KnotError> {
        if p == 0 || q == 0 {
            return Err(KnotError::InvalidParameters {
                reason: "torus knot winding numbers p and q must be non-zero",
            });
        }
        if !major_radius.is_finite()
            || !minor_radius.is_finite()
            || minor_radius <= 0.0
            || minor_radius >= major_radius
        {
            return Err(KnotError::InvalidParameters {
                reason: "torus knot requires 0 < minor_radius < major_radius",
            });
        }
        Ok(Self {
            p,
            q,
            major_radius,
            minor_radius,
        })
    }

    /// Standard right-handed trefoil $T(2, 3)$ with writhe $+3$.
    #[must_use]
    #[verified_engine::verified]
    pub fn right_trefoil() -> Self {
        Self {
            p: 2,
            q: 3,
            major_radius: 1.8,
            minor_radius: 0.7,
        }
    }

    /// Standard left-handed trefoil $T(2, -3)$ with writhe $-3$.
    #[must_use]
    #[verified_engine::verified]
    pub fn left_trefoil() -> Self {
        Self {
            p: 2,
            q: -3,
            major_radius: 1.8,
            minor_radius: 0.7,
        }
    }

    /// Number of link components $\gcd(p, |q|)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn component_count(&self) -> usize {
        gcd_usize(self.p, self.q.unsigned_abs() as usize)
    }

    /// Evaluates point $\theta \in [0, 2\pi)$ on component `component_idx` with zero heap allocation.
    #[must_use]
    #[verified_engine::verified]
    pub fn eval_component_point(&self, component_idx: usize, theta: f64) -> Point3<f64> {
        let comps = self.component_count();
        let p0 = (self.p / comps) as f64;
        let q0 = f64::from(self.q) / (comps as f64);
        let phase = TAU * (component_idx as f64) / (self.p as f64);
        let phi = p0 * theta;
        let psi = q0 * theta + phase;
        let rho = self.major_radius + self.minor_radius * psi.cos();
        let pt = Point3::new(
            rho * phi.cos(),
            rho * phi.sin(),
            -self.minor_radius * psi.sin(),
        );
        debug_assert!(pt.x.is_finite() && pt.y.is_finite() && pt.z.is_finite());
        pt
    }

    /// Samples `component_idx` uniformly into a pre-allocated buffer `out` with zero heap allocations.
    ///
    /// # Errors
    /// Returns [`KnotError`] if `out.len() < 3` or `component_idx >= self.component_count()`.
    #[verified_engine::verified]
    pub fn sample_component_in_place(
        &self,
        component_idx: usize,
        out: &mut [Point3<f64>],
    ) -> Result<(), KnotError> {
        if component_idx >= self.component_count() {
            return Err(KnotError::InvalidParameters {
                reason: "torus link component index out of range",
            });
        }
        if out.len() < 3 {
            return Err(KnotError::InsufficientSamples {
                required: 3,
                actual: out.len(),
            });
        }
        let n = out.len() as f64;
        for (i, slot) in out.iter_mut().enumerate() {
            let theta = TAU * (i as f64) / n;
            *slot = self.eval_component_point(component_idx, theta);
        }
        Ok(())
    }

    /// Samples all $\gcd(p, |q|)$ components into a [`ClosedLink3D`].
    ///
    /// # Errors
    /// Returns [`KnotError::InsufficientSamples`] if `samples_per_component < 8`.
    #[verified_engine::verified]
    pub fn to_link(&self, samples_per_component: usize) -> Result<ClosedLink3D, KnotError> {
        if samples_per_component < 8 {
            return Err(KnotError::InsufficientSamples {
                required: 8,
                actual: samples_per_component,
            });
        }
        let comps = self.component_count();
        let mut all = Vec::with_capacity(comps);
        for c in 0..comps {
            let mut buf = vec![Point3::origin(); samples_per_component];
            self.sample_component_in_place(c, &mut buf)?;
            all.push(buf);
        }
        ClosedLink3D::new(all)
    }
}

/// Parametric 3D Lissajous knot $L(n_x, n_y, n_z; \phi_x, \phi_y, \phi_z)$.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LissajousKnot {
    /// Integer harmonic frequencies `(nx, ny, nz)`.
    pub frequencies: (usize, usize, usize),
    /// Phase offsets `(px, py, pz)` in radians.
    pub phases: (f64, f64, f64),
    /// Spatial amplitude scale.
    pub scale: f64,
}

impl LissajousKnot {
    /// Constructs a validated [`LissajousKnot`].
    ///
    /// # Errors
    /// Returns [`KnotError::InvalidParameters`] if any frequency is zero or `scale <= 0`.
    #[verified_engine::verified]
    pub fn new(
        frequencies: (usize, usize, usize),
        phases: (f64, f64, f64),
        scale: f64,
    ) -> Result<Self, KnotError> {
        if frequencies.0 == 0 || frequencies.1 == 0 || frequencies.2 == 0 {
            return Err(KnotError::InvalidParameters {
                reason: "Lissajous frequencies must be positive integers",
            });
        }
        if !scale.is_finite()
            || scale <= 0.0
            || !phases.0.is_finite()
            || !phases.1.is_finite()
            || !phases.2.is_finite()
        {
            return Err(KnotError::InvalidParameters {
                reason: "Lissajous scale and phases must be finite with positive scale",
            });
        }
        Ok(Self {
            frequencies,
            phases,
            scale,
        })
    }

    /// Classic $5_2$ three-twist Lissajous knot $(3, 4, 7)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn three_twist() -> Self {
        Self {
            frequencies: (3, 4, 7),
            phases: (0.1, 0.7, 0.0),
            scale: 1.8,
        }
    }

    /// Samples the Lissajous curve into `out` with zero heap allocations.
    ///
    /// # Errors
    /// Returns [`KnotError::InsufficientSamples`] if `out.len() < 8`.
    #[verified_engine::verified]
    pub fn sample_in_place(&self, out: &mut [Point3<f64>]) -> Result<(), KnotError> {
        if out.len() < 8 {
            return Err(KnotError::InsufficientSamples {
                required: 8,
                actual: out.len(),
            });
        }
        let n = out.len() as f64;
        let (nx, ny, nz) = (
            self.frequencies.0 as f64,
            self.frequencies.1 as f64,
            self.frequencies.2 as f64,
        );
        for (i, slot) in out.iter_mut().enumerate() {
            let t = TAU * (i as f64) / n;
            *slot = Point3::new(
                self.scale * (nx * t + self.phases.0).cos(),
                self.scale * (ny * t + self.phases.1).cos(),
                self.scale * (nz * t + self.phases.2).cos(),
            );
        }
        Ok(())
    }

    /// Samples the Lissajous knot into a single-component [`ClosedLink3D`].
    ///
    /// # Errors
    /// Returns [`KnotError::InsufficientSamples`] if `samples < 8`.
    #[verified_engine::verified]
    pub fn to_link(&self, samples: usize) -> Result<ClosedLink3D, KnotError> {
        let mut pts = vec![Point3::origin(); samples.max(1)];
        self.sample_in_place(&mut pts)?;
        ClosedLink3D::new(vec![pts])
    }
}

/// Samples the prime amphichiral figure-eight knot ($4_1$) into a [`ClosedLink3D`].
///
/// # Errors
/// Returns [`KnotError`] if `scale <= 0` or `samples < 16`.
#[verified_engine::verified]
pub fn figure_eight_knot(scale: f64, samples: usize) -> Result<ClosedLink3D, KnotError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(KnotError::InvalidParameters {
            reason: "figure-eight knot scale must be finite and positive",
        });
    }
    if samples < 16 {
        return Err(KnotError::InsufficientSamples {
            required: 16,
            actual: samples,
        });
    }
    let s = scale * 0.65;
    let n = samples as f64;
    let mut pts = Vec::with_capacity(samples);
    for i in 0..samples {
        let theta = TAU * (i as f64) / n;
        let rho = 2.0 + (2.0 * theta).cos();
        pts.push(Point3::new(
            s * rho * (3.0 * theta).cos(),
            s * rho * (3.0 * theta).sin(),
            s * 1.35 * (4.0 * theta).sin(),
        ));
    }
    ClosedLink3D::new(vec![pts])
}

/// Samples the 2-component Hopf link with linking number `+1` (`positive == true`) or `-1`.
///
/// # Errors
/// Returns [`KnotError`] if `radius <= 0` or `samples_per_component < 16`.
#[verified_engine::verified]
pub fn hopf_link(
    radius: f64,
    positive: bool,
    samples_per_component: usize,
) -> Result<ClosedLink3D, KnotError> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(KnotError::InvalidParameters {
            reason: "Hopf link radius must be finite and positive",
        });
    }
    if samples_per_component < 16 {
        return Err(KnotError::InsufficientSamples {
            required: 16,
            actual: samples_per_component,
        });
    }
    let sign = if positive { 1.0 } else { -1.0 };
    let tilt = 0.62_f64;
    let (sin_a, cos_a) = tilt.sin_cos();
    let n = samples_per_component as f64;
    let mut c0 = Vec::with_capacity(samples_per_component);
    let mut c1 = Vec::with_capacity(samples_per_component);
    for i in 0..samples_per_component {
        let th = TAU * (i as f64) / n;
        c0.push(Point3::new(
            -0.5 * radius + radius * th.cos(),
            radius * th.sin(),
            0.0,
        ));
        let z_raw = sign * radius * th.sin();
        c1.push(Point3::new(
            0.5 * radius - radius * th.cos(),
            -z_raw * sin_a,
            z_raw * cos_a,
        ));
    }
    let link = ClosedLink3D::new(vec![c0, c1])?;
    assert_eq!(link.component_count(), 2);
    Ok(link)
}

/// Samples the 3-component Brunnian Borromean rings ($6^3_2$) using mutually orthogonal ellipses.
///
/// # Errors
/// Returns [`KnotError`] if `major_axis <= minor_axis`, `minor_axis <= 0`, or `samples_per_ring < 16`.
#[verified_engine::verified]
pub fn borromean_rings(
    major_axis: f64,
    minor_axis: f64,
    samples_per_ring: usize,
) -> Result<ClosedLink3D, KnotError> {
    if !major_axis.is_finite()
        || !minor_axis.is_finite()
        || minor_axis <= 0.0
        || major_axis <= minor_axis
    {
        return Err(KnotError::InvalidParameters {
            reason: "Borromean rings require 0 < minor_axis < major_axis",
        });
    }
    if samples_per_ring < 16 {
        return Err(KnotError::InsufficientSamples {
            required: 16,
            actual: samples_per_ring,
        });
    }
    let inv_s2 = 1.0 / 2.0_f64.sqrt();
    let inv_s6 = 1.0 / 6.0_f64.sqrt();
    let inv_s3 = 1.0 / 3.0_f64.sqrt();
    let project_so3 = |x: f64, y: f64, z: f64| -> Point3<f64> {
        Point3::new(
            (x - y) * inv_s2,
            (x + y - 2.0 * z) * inv_s6,
            (x + y + z) * inv_s3,
        )
    };
    let n = samples_per_ring as f64;
    let mut r0 = Vec::with_capacity(samples_per_ring);
    let mut r1 = Vec::with_capacity(samples_per_ring);
    let mut r2 = Vec::with_capacity(samples_per_ring);
    for i in 0..samples_per_ring {
        let th = TAU * (i as f64) / n;
        let (st, ct) = th.sin_cos();
        r0.push(project_so3(major_axis * ct, minor_axis * st, 0.0));
        r1.push(project_so3(0.0, major_axis * ct, minor_axis * st));
        r2.push(project_so3(minor_axis * st, 0.0, major_axis * ct));
    }
    let link = ClosedLink3D::new(vec![r0, r1, r2])?;
    assert_eq!(link.component_count(), 3);
    Ok(link)
}
