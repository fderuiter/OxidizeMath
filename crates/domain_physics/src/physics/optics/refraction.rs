//! Snell refraction, polarized Fresnel coefficients, Cauchy dispersion, and surface intersections.
//!
//! Provides exact 2D vector refraction/reflection, total internal reflection (TIR),
//! s/p-polarized Fresnel amplitude and power coefficients, and empirical Cauchy glass models.

// [cite:optics]

use serde::{Deserialize, Serialize};

/// Standard optical dielectric media with calibrated Cauchy dispersion coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum GlassMaterial {
    /// Schott N-BK7 borosilicate crown glass ($n_d \approx 1.5168$).
    #[default]
    Bk7Crown,
    /// Schott N-SF11 high-dispersion dense flint glass ($n_d \approx 1.7847$).
    Sf11Flint,
    /// UV-grade fused silica ($\text{SiO}_2$, $n_d \approx 1.4585$).
    FusedSilica,
    /// Crystalline diamond ($n_d \approx 2.417$).
    Diamond,
    /// Liquid water at $20^\circ\text{C}$ ($n_d \approx 1.333$).
    Water,
}

impl GlassMaterial {
    /// Human-readable optical catalog label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Bk7Crown => "N-BK7 Crown Glass",
            Self::Sf11Flint => "SF11 Dense Flint",
            Self::FusedSilica => "Fused Silica (SiO2)",
            Self::Diamond => "Optical Diamond",
            Self::Water => "Distilled Water",
        }
    }

    /// Returns empirical three-term Cauchy coefficients $(A, B, C)$ for $\lambda$ in micrometers.
    #[must_use]
    pub const fn cauchy(self) -> CauchyDispersion {
        match self {
            Self::Bk7Crown => CauchyDispersion {
                a: 1.5046,
                b_um2: 0.00420,
                c_um4: 0.0,
            },
            Self::Sf11Flint => CauchyDispersion {
                a: 1.7432,
                b_um2: 0.01342,
                c_um4: 0.00031,
            },
            Self::FusedSilica => CauchyDispersion {
                a: 1.4440,
                b_um2: 0.00354,
                c_um4: 0.0,
            },
            Self::Diamond => CauchyDispersion {
                a: 2.3780,
                b_um2: 0.01340,
                c_um4: 0.00012,
            },
            Self::Water => CauchyDispersion {
                a: 1.3242,
                b_um2: 0.00306,
                c_um4: 0.0,
            },
        }
    }
}

/// Three-term Cauchy wavelength dispersion model $n(\lambda) = A + B/\lambda^2 + C/\lambda^4$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CauchyDispersion {
    /// Dimensionless long-wavelength asymptotic index $A$.
    pub a: f64,
    /// Second-order dispersion coefficient $B$ in $\mu\text{m}^2$.
    pub b_um2: f64,
    /// Fourth-order UV dispersion coefficient $C$ in $\mu\text{m}^4$.
    pub c_um4: f64,
}

/// Evaluates Cauchy refractive index $n(\lambda)$ for vacuum wavelength `wavelength_nm` in nanometers.
#[verified_engine::verified]
#[must_use]
pub fn cauchy_refractive_index(dispersion: CauchyDispersion, wavelength_nm: f64) -> f64 {
    let lambda_um = (wavelength_nm * 1.0e-3).clamp(0.18, 10.0);
    let inv_l2 = 1.0 / (lambda_um * lambda_um);
    (dispersion.a + dispersion.b_um2 * inv_l2 + dispersion.c_um4 * inv_l2 * inv_l2).max(1.0)
}

/// Computes the critical angle $\theta_c = \arcsin(n_2 / n_1)$ in radians when $n_1 > n_2$.
#[verified_engine::verified]
#[must_use]
pub fn critical_angle(n1: f64, n2: f64) -> Option<f64> {
    if !n1.is_finite() || !n2.is_finite() || n1 <= 0.0 || n2 <= 0.0 || n1 <= n2 {
        return None;
    }
    Some((n2 / n1).clamp(0.0, 1.0).asin())
}

/// Computes Brewster's polarizing angle $\theta_B = \arctan(n_2 / n_1)$ in radians where $R_p(\theta_B) = 0$.
#[verified_engine::verified]
#[must_use]
pub fn brewster_angle(n1: f64, n2: f64) -> f64 {
    if !n1.is_finite() || !n2.is_finite() || n1 <= 0.0 || n2 <= 0.0 {
        return 0.0;
    }
    n2.atan2(n1)
}

/// Computes refracted angle $\theta_t$ in radians via Snell's law $n_1\sin\theta_i = n_2\sin\theta_t$, or `None` under TIR.
#[verified_engine::verified]
#[must_use]
pub fn snell_refract_angle(n1: f64, n2: f64, theta_i_rad: f64) -> Option<f64> {
    if !n1.is_finite() || !n2.is_finite() || !theta_i_rad.is_finite() || n1 <= 0.0 || n2 <= 0.0 {
        return None;
    }
    let sin_t = (n1 / n2) * theta_i_rad.abs().sin();
    if sin_t > 1.0 {
        None
    } else {
        Some(sin_t.clamp(-1.0, 1.0).asin())
    }
}

/// Polarized Fresnel amplitude and power reflection/transmission coefficients at a planar dielectric interface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FresnelCoefficients {
    /// Transverse-electric ($s$-polarized) amplitude reflection coefficient $r_s$.
    pub r_s: f64,
    /// Transverse-electric ($s$-polarized) amplitude transmission coefficient $t_s$.
    pub t_s: f64,
    /// Transverse-magnetic ($p$-polarized) amplitude reflection coefficient $r_p$.
    pub r_p: f64,
    /// Transverse-magnetic ($p$-polarized) amplitude transmission coefficient $t_p$.
    pub t_p: f64,
    /// Power reflectance $R_s = |r_s|^2$ for $s$-polarized light.
    pub reflectance_s: f64,
    /// Power transmittance $T_s = 1 - R_s$ for $s$-polarized light.
    pub transmittance_s: f64,
    /// Power reflectance $R_p = |r_p|^2$ for $p$-polarized light.
    pub reflectance_p: f64,
    /// Power transmittance $T_p = 1 - R_p$ for $p$-polarized light.
    pub transmittance_p: f64,
    /// Unpolarized average power reflectance $R = \frac{1}{2}(R_s + R_p)$.
    pub reflectance_unpolarized: f64,
    /// Unpolarized average power transmittance $T = \frac{1}{2}(T_s + T_p)$.
    pub transmittance_unpolarized: f64,
    /// True when incident angle exceeds the critical angle ($n_1 \sin\theta_i > n_2$).
    pub total_internal_reflection: bool,
}

/// Evaluates polarized Fresnel amplitude and power coefficients conserving $R + T = 1$.
#[verified_engine::verified]
#[must_use]
pub fn fresnel_polarized(n1: f64, n2: f64, theta_i_rad: f64) -> FresnelCoefficients {
    let theta = theta_i_rad
        .abs()
        .clamp(0.0, std::f64::consts::FRAC_PI_2 - 1.0e-12);
    let cos_i = theta.cos();
    let sin_t = (n1 / n2.max(1.0e-12)) * theta.sin();
    if sin_t > 1.0 {
        return FresnelCoefficients {
            r_s: 1.0,
            t_s: 0.0,
            r_p: 1.0,
            t_p: 0.0,
            reflectance_s: 1.0,
            transmittance_s: 0.0,
            reflectance_p: 1.0,
            transmittance_p: 0.0,
            reflectance_unpolarized: 1.0,
            transmittance_unpolarized: 0.0,
            total_internal_reflection: true,
        };
    }
    let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
    let denom_s = (n1 * cos_i + n2 * cos_t).max(1.0e-15);
    let denom_p = (n2 * cos_i + n1 * cos_t).max(1.0e-15);
    let r_s = (n1 * cos_i - n2 * cos_t) / denom_s;
    let t_s = (2.0 * n1 * cos_i) / denom_s;
    let r_p = (n2 * cos_i - n1 * cos_t) / denom_p;
    let t_p = (2.0 * n1 * cos_i) / denom_p;
    let reflectance_s = (r_s * r_s).clamp(0.0, 1.0);
    let reflectance_p = (r_p * r_p).clamp(0.0, 1.0);
    let impedance_ratio = if cos_i > 1.0e-15 {
        (n2 * cos_t) / (n1 * cos_i)
    } else {
        0.0
    };
    let transmittance_s = if cos_i > 1.0e-12 {
        (impedance_ratio * t_s * t_s).clamp(0.0, 1.0)
    } else {
        1.0 - reflectance_s
    };
    let transmittance_p = if cos_i > 1.0e-12 {
        (impedance_ratio * t_p * t_p).clamp(0.0, 1.0)
    } else {
        1.0 - reflectance_p
    };
    FresnelCoefficients {
        r_s,
        t_s,
        r_p,
        t_p,
        reflectance_s,
        transmittance_s,
        reflectance_p,
        transmittance_p,
        reflectance_unpolarized: 0.5 * (reflectance_s + reflectance_p),
        transmittance_unpolarized: 0.5 * (transmittance_s + transmittance_p),
        total_internal_reflection: false,
    }
}

/// Outcome of vector Snell interaction at a dielectric boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnellOutcome {
    /// Refracted ray transmitted into medium $n_2$.
    Refracted {
        /// Unit transmitted direction vector $\hat{\mathbf{t}}$.
        direction: [f64; 2],
        /// Cosine of angle of incidence $\cos\theta_i \in [0, 1]$.
        cos_theta_i: f64,
        /// Cosine of angle of refraction $\cos\theta_t \in [0, 1]$.
        cos_theta_t: f64,
        /// Angle of refraction $\theta_t$ in radians.
        theta_t_rad: f64,
    },
    /// Total internal reflection back into medium $n_1$.
    TotalInternalReflection {
        /// Unit specularly reflected direction vector $\hat{\mathbf{r}}$.
        reflected_direction: [f64; 2],
        /// Cosine of angle of incidence $\cos\theta_i \in [0, 1]$.
        cos_theta_i: f64,
    },
}

/// Normalizes a 2D vector to unit Euclidean length, returning `None` if degenerate.
#[verified_engine::verified]
#[must_use]
pub fn normalize_2d(v: [f64; 2]) -> Option<[f64; 2]> {
    let norm_sq = v[0] * v[0] + v[1] * v[1];
    if !norm_sq.is_finite() || norm_sq < 1.0e-24 {
        return None;
    }
    let inv = 1.0 / norm_sq.sqrt();
    Some([v[0] * inv, v[1] * inv])
}

/// Specularly reflects unit vector `incident_unit` across unit surface normal `normal_unit`.
#[verified_engine::verified]
#[must_use]
pub fn reflect_vector_2d(incident_unit: [f64; 2], normal_unit: [f64; 2]) -> [f64; 2] {
    let dot = incident_unit[0] * normal_unit[0] + incident_unit[1] * normal_unit[1];
    let rx = incident_unit[0] - 2.0 * dot * normal_unit[0];
    let ry = incident_unit[1] - 2.0 * dot * normal_unit[1];
    normalize_2d([rx, ry]).unwrap_or([-incident_unit[0], -incident_unit[1]])
}

/// Computes 2D vector Snell refraction or total internal reflection across a boundary normal.
#[verified_engine::verified]
#[must_use]
pub fn snell_vector_2d(
    incident_unit: [f64; 2],
    normal_unit: [f64; 2],
    n1: f64,
    n2: f64,
) -> SnellOutcome {
    let raw_dot = incident_unit[0] * normal_unit[0] + incident_unit[1] * normal_unit[1];
    let n = if raw_dot <= 0.0 {
        normal_unit
    } else {
        [-normal_unit[0], -normal_unit[1]]
    };
    let cos_i = (-(incident_unit[0] * n[0] + incident_unit[1] * n[1])).clamp(0.0, 1.0);
    let eta = n1 / n2.max(1.0e-12);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        return SnellOutcome::TotalInternalReflection {
            reflected_direction: reflect_vector_2d(incident_unit, n),
            cos_theta_i: cos_i,
        };
    }
    let cos_t = k.sqrt();
    let scale = eta * cos_i - cos_t;
    let tx = eta * incident_unit[0] + scale * n[0];
    let ty = eta * incident_unit[1] + scale * n[1];
    let dir = normalize_2d([tx, ty]).unwrap_or(incident_unit);
    SnellOutcome::Refracted {
        direction: dir,
        cos_theta_i: cos_i,
        cos_theta_t: cos_t,
        theta_t_rad: cos_t.clamp(-1.0, 1.0).acos(),
    }
}

/// Intersects ray `(origin, dir_unit)` with a circular arc centered at `center` of `radius`.
///
/// If `x_hemi_sign` is `Some(s)`, retains intersections satisfying `(hit_x - center_x) * s >= -1e-9`
/// and `|hit_y - center_y| <= aperture_radius`. Returns `(t, hit_point, outward_unit_normal)`.
#[verified_engine::verified]
#[must_use]
pub fn intersect_ray_circle_arc(
    origin: [f64; 2],
    dir_unit: [f64; 2],
    center: [f64; 2],
    radius: f64,
    x_hemi_sign: Option<f64>,
    aperture_radius: f64,
) -> Option<(f64, [f64; 2], [f64; 2])> {
    let ocx = origin[0] - center[0];
    let ocy = origin[1] - center[1];
    let b = ocx * dir_unit[0] + ocy * dir_unit[1];
    let c = ocx * ocx + ocy * ocy - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    for t in [-b - sqrt_d, -b + sqrt_d] {
        if t <= 1.0e-7 {
            continue;
        }
        let hx = origin[0] + t * dir_unit[0];
        let hy = origin[1] + t * dir_unit[1];
        if (hy - center[1]).abs() > aperture_radius + 1.0e-9 {
            continue;
        }
        if let Some(sign) = x_hemi_sign
            && (hx - center[0]) * sign < -1.0e-9
        {
            continue;
        }
        let inv_r = 1.0 / radius.max(1.0e-12);
        return Some((
            t,
            [hx, hy],
            [(hx - center[0]) * inv_r, (hy - center[1]) * inv_r],
        ));
    }
    None
}

/// Intersects ray `(origin, dir_unit)` with finite line segment `[p0, p1]`, returning `(t, hit, unit_normal)`.
#[verified_engine::verified]
#[must_use]
pub fn intersect_ray_segment(
    origin: [f64; 2],
    dir_unit: [f64; 2],
    p0: [f64; 2],
    p1: [f64; 2],
) -> Option<(f64, [f64; 2], [f64; 2])> {
    let sx = p1[0] - p0[0];
    let sy = p1[1] - p0[1];
    let denom = dir_unit[0] * sy - dir_unit[1] * sx;
    if denom.abs() < 1.0e-12 {
        return None;
    }
    let qpx = p0[0] - origin[0];
    let qpy = p0[1] - origin[1];
    let t = (qpx * sy - qpy * sx) / denom;
    let u = (qpx * dir_unit[1] - qpy * dir_unit[0]) / denom;
    if t > 1.0e-7 && (0.0..=1.0).contains(&u) {
        let hit = [origin[0] + t * dir_unit[0], origin[1] + t * dir_unit[1]];
        let normal = normalize_2d([-sy, sx])?;
        Some((t, hit, normal))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn fresnel_brewster_zero_and_power_conservation() {
        let (n1, n2) = (1.0, 1.5168);
        let theta_b = brewster_angle(n1, n2);
        assert!((theta_b - (n2 / n1).atan()).abs() < 1.0e-12);
        let fb = fresnel_polarized(n1, n2, theta_b);
        assert!(fb.reflectance_p < 1.0e-24);
        assert!((fb.reflectance_s + fb.transmittance_s - 1.0).abs() < 1.0e-12);
        assert!((fb.reflectance_p + fb.transmittance_p - 1.0).abs() < 1.0e-12);

        for deg in [0.0_f64, 15.0, 37.5, 60.0, 82.0] {
            let f = fresnel_polarized(n1, n2, deg.to_radians());
            assert!((f.reflectance_s + f.transmittance_s - 1.0).abs() < 1.0e-12);
            assert!((f.reflectance_p + f.transmittance_p - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn total_internal_reflection_above_critical_angle() {
        let (n1, n2) = (1.5168, 1.0);
        let theta_c = critical_angle(n1, n2).expect("dense-to-rare has critical angle");
        assert!((theta_c - (n2 / n1).asin()).abs() < 1.0e-12);
        assert!(snell_refract_angle(n1, n2, theta_c - 0.02).is_some());
        assert!(snell_refract_angle(n1, n2, theta_c + 0.02).is_none());
        let tir = fresnel_polarized(n1, n2, theta_c + 0.02);
        assert!(tir.total_internal_reflection);
        assert!((tir.reflectance_unpolarized - 1.0).abs() < 1.0e-12);
        assert!(tir.transmittance_unpolarized.abs() < 1.0e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn cauchy_normal_dispersion_orders_blue_above_red() {
        let sf11 = GlassMaterial::Sf11Flint.cauchy();
        let n_violet = cauchy_refractive_index(sf11, 405.0);
        let n_red = cauchy_refractive_index(sf11, 680.0);
        assert!(n_violet > n_red + 0.04);
    }
}
