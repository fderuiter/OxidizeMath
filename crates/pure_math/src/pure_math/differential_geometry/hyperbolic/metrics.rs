//! Intrinsic Poincaré/Klein metrics, geodesic arcs, conformal angles, and Gauss-Bonnet defects.

use super::{
    HYPERBOLIC_CURVATURE, HyperbolicError, HyperbolicModel, Su11Matrix, clamp_to_disk,
    is_hyperbolic_schlafli, klein_to_poincare, poincare_to_klein,
};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Returns the conformal metric factor $\lambda(z) = \frac{2}{1 - |z|^2}$ for $ds = \lambda(z)|dz|$.
#[must_use]
#[verified_engine::verified]
pub fn conformal_metric_factor(z: Complex64) -> f64 {
    let zc = clamp_to_disk(z);
    2.0 / (1.0 - zc.norm_sqr()).max(1e-18)
}

/// Returns the exact constant Gaussian curvature $K = -1.0$ of the unit hyperbolic disk.
#[must_use]
#[verified_engine::verified]
pub fn gaussian_curvature() -> f64 {
    HYPERBOLIC_CURVATURE
}

/// Evaluates Liouville's intrinsic curvature formula $K(z) = -\frac{\Delta \ln \lambda(z)}{\lambda(z)^2}$
/// via second-order central differences on $\ln \lambda(z)$ inside $\mathbb{D}$.
#[must_use]
#[verified_engine::verified]
pub fn intrinsic_gaussian_curvature(z: Complex64) -> f64 {
    let zc = clamp_to_disk(z);
    let margin = (1.0 - zc.norm()).max(1e-8);
    let h = (1e-5_f64).min(0.25 * margin);
    let dx = Complex64::new(h, 0.0);
    let dy = Complex64::new(0.0, h);
    let c0 = conformal_metric_factor(zc).ln();
    let lap_x =
        conformal_metric_factor(zc + dx).ln() - 2.0 * c0 + conformal_metric_factor(zc - dx).ln();
    let lap_y =
        conformal_metric_factor(zc + dy).ln() - 2.0 * c0 + conformal_metric_factor(zc - dy).ln();
    let laplacian = (lap_x + lap_y) / (h * h);
    let lam = conformal_metric_factor(zc);
    -laplacian / (lam * lam)
}

/// Computes the intrinsic hyperbolic distance $d_{\mathbb{D}}(z_1, z_2) = 2 \operatorname{artanh}\!\left|\frac{z_1 - z_2}{1 - \bar{z}_1 z_2}\right|$.
#[must_use]
#[verified_engine::verified]
pub fn poincare_distance(z1: Complex64, z2: Complex64) -> f64 {
    let a = clamp_to_disk(z1);
    let b = clamp_to_disk(z2);
    let den = Complex64::new(1.0, 0.0) - a.conj() * b;
    if den.norm_sqr() <= 1e-30 {
        return 0.0;
    }
    let pseudo_dist = ((a - b) / den).norm().min(1.0 - 1e-12);
    2.0 * pseudo_dist.atanh()
}

/// Computes the intrinsic hyperbolic distance between two Beltrami-Klein coordinates $w_1, w_2 \in \mathbb{K}$.
#[must_use]
#[verified_engine::verified]
pub fn klein_distance(w1: Complex64, w2: Complex64) -> f64 {
    poincare_distance(klein_to_poincare(w1), klein_to_poincare(w2))
}

/// Unique Poincaré geodesic segment connecting `start` and `end`, represented either as a
/// Euclidean circle orthogonal to $|z| = 1$ or a diameter chord through the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicArc {
    /// Start point in Poincaré disk coordinates.
    pub start: Complex64,
    /// End point in Poincaré disk coordinates.
    pub end: Complex64,
    /// Orthogonal Euclidean circle center $C$ ($|C|^2 - R^2 = 1$) when `!is_diameter`.
    pub circle_center: Complex64,
    /// Orthogonal Euclidean circle radius $R$ when `!is_diameter`.
    pub circle_radius: f64,
    /// True when `start`, `end`, and `0` are collinear.
    pub is_diameter: bool,
}

impl GeodesicArc {
    /// Solves the unique orthogonal circle or diameter through `start` and `end` in $\mathbb{D}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn through_points(start: Complex64, end: Complex64) -> Self {
        let p1 = clamp_to_disk(start);
        let p2 = clamp_to_disk(end);
        let cross = p1.re * p2.im - p1.im * p2.re;
        if cross.abs() <= 1e-11 {
            return Self {
                start: p1,
                end: p2,
                circle_center: Complex64::new(0.0, 0.0),
                circle_radius: 0.0,
                is_diameter: true,
            };
        }
        let r1 = 0.5 * (p1.norm_sqr() + 1.0);
        let r2 = 0.5 * (p2.norm_sqr() + 1.0);
        let cx = (r1 * p2.im - r2 * p1.im) / cross;
        let cy = (p1.re * r2 - p2.re * r1) / cross;
        let center = Complex64::new(cx, cy);
        let radius = (center.norm_sqr() - 1.0).max(0.0).sqrt();
        Self {
            start: p1,
            end: p2,
            circle_center: center,
            circle_radius: radius,
            is_diameter: radius <= 1e-12,
        }
    }

    /// Samples `out.len()` points along the geodesic in `model` coordinates without heap allocation.
    #[verified_engine::verified]
    pub fn sample_into(&self, model: HyperbolicModel, out: &mut [Complex64]) -> usize {
        let n = out.len();
        if n == 0 {
            return 0;
        }
        if n == 1 {
            out[0] = match model {
                HyperbolicModel::Poincare => self.start,
                HyperbolicModel::Klein => poincare_to_klein(self.start),
            };
            return 1;
        }
        if matches!(model, HyperbolicModel::Klein) {
            let k1 = poincare_to_klein(self.start);
            let k2 = poincare_to_klein(self.end);
            let denom = (n - 1) as f64;
            for (idx, slot) in out.iter_mut().enumerate() {
                let t = (idx as f64) / denom;
                *slot = clamp_to_disk(k1 * (1.0 - t) + k2 * t);
            }
            return n;
        }
        self.sample_poincare_into(out)
    }

    #[verified_engine::verified]
    fn sample_poincare_into(&self, out: &mut [Complex64]) -> usize {
        let n = out.len();
        let denom = (n - 1) as f64;
        if self.is_diameter {
            for (idx, slot) in out.iter_mut().enumerate() {
                let t = (idx as f64) / denom;
                *slot = clamp_to_disk(self.start * (1.0 - t) + self.end * t);
            }
            return n;
        }
        let d1 = self.start - self.circle_center;
        let d2 = self.end - self.circle_center;
        let theta1 = d1.im.atan2(d1.re);
        let theta2 = d2.im.atan2(d2.re);
        let mut delta = theta2 - theta1;
        if delta > PI {
            delta -= 2.0 * PI;
        } else if delta < -PI {
            delta += 2.0 * PI;
        }
        for (idx, slot) in out.iter_mut().enumerate() {
            let t = (idx as f64) / denom;
            let ang = theta1 + t * delta;
            let pt = self.circle_center
                + Complex64::new(
                    self.circle_radius * ang.cos(),
                    self.circle_radius * ang.sin(),
                );
            *slot = clamp_to_disk(pt);
        }
        n
    }
}

/// Computes the exact conformal interior angle at `vertex` between geodesic rays to `p1` and `p2`.
#[must_use]
#[verified_engine::verified]
pub fn geodesic_interior_angle(vertex: Complex64, p1: Complex64, p2: Complex64) -> f64 {
    let to_origin = Su11Matrix::translation_to_origin(vertex);
    let u1 = to_origin.apply(p1);
    let u2 = to_origin.apply(p2);
    let n1 = u1.norm();
    let n2 = u2.norm();
    if n1 <= 1e-14 || n2 <= 1e-14 {
        return 0.0;
    }
    let cos_angle = ((u1.re * u2.re + u1.im * u2.im) / (n1 * n2)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// Intrinsic Gauss-Bonnet invariants of a geodesic triangle $\triangle ABC \subset \mathbb{H}^2$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussBonnetSummary {
    /// Interior conformal angles $[\alpha, \beta, \gamma]$ in radians.
    pub angles: [f64; 3],
    /// Sum of interior angles $\alpha + \beta + \gamma < \pi$.
    pub angle_sum: f64,
    /// Angular defect $\pi - (\alpha + \beta + \gamma) \ge 0$.
    pub angular_defect: f64,
    /// Intrinsic hyperbolic area $\iint_{\triangle} dA = \frac{\pi - (\alpha + \beta + \gamma)}{-K}$.
    pub area: f64,
    /// Hyperbolic side lengths $[d(A, B), d(B, C), d(C, A)]$.
    pub side_lengths: [f64; 3],
}

/// Evaluates the side lengths, interior angles, angular defect, and Gauss-Bonnet area of $\triangle ABC$.
#[must_use]
#[verified_engine::verified]
pub fn gauss_bonnet_triangle(a: Complex64, b: Complex64, c: Complex64) -> GaussBonnetSummary {
    let alpha = geodesic_interior_angle(a, b, c);
    let beta = geodesic_interior_angle(b, c, a);
    let gamma = geodesic_interior_angle(c, a, b);
    let angle_sum = alpha + beta + gamma;
    let defect = (PI - angle_sum).max(0.0);
    let area = defect / (-HYPERBOLIC_CURVATURE);
    let side_lengths = [
        poincare_distance(a, b),
        poincare_distance(b, c),
        poincare_distance(c, a),
    ];
    GaussBonnetSummary {
        angles: [alpha, beta, gamma],
        angle_sum,
        angular_defect: defect,
        area,
        side_lengths,
    }
}

/// Returns the exact Gauss-Bonnet area $(p - 2)\pi - \frac{2\pi p}{q}$ of a regular $\{p, q\}$ tile.
#[verified_engine::verified]
pub fn regular_polygon_gauss_bonnet_area(p: usize, q: usize) -> Result<f64, HyperbolicError> {
    if !is_hyperbolic_schlafli(p, q) {
        return Err(HyperbolicError::NonHyperbolicSchlafli { p, q });
    }
    let pf = p as f64;
    let qf = q as f64;
    let angle_defect = (pf - 2.0) * PI - (2.0 * PI * pf) / qf;
    Ok(angle_defect / (-HYPERBOLIC_CURVATURE))
}

// [cite:modular_polynomials_review]
