//! Equatorial Kerr radial RK4 effective-potential stepper, Bardeen-Press-Teukolsky ISCO,
//! prograde photon-ring critical impact parameter, and 2D lensing regime classifiers.

use std::f64::consts::PI;

use super::schwarzschild::GeodesicKind;

/// Discrete relativistic lensing and orbital regimes mapped across 2D parameter slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LensingRegime {
    /// Trajectory plunges across the event horizon (or transits through a wormhole throat).
    HorizonCaptureOrTransit = 0,
    /// Multi-radian strong-field photon-ring winding or bound relativistic rosette.
    PhotonRingWinding = 1,
    /// Clean radial turning point with weak/moderate asymptotic deflection.
    WeakDeflectionEscape = 2,
    /// Launch radius lies inside the classical centrifugal barrier ($V_{\text{eff}}(r_0) > \frac{1}{2}e^2$).
    CentrifugalBarrier = 3,
}

impl LensingRegime {
    /// Converts a raw `u8` regime index into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::HorizonCaptureOrTransit,
            1 => Self::PhotonRingWinding,
            2 => Self::WeakDeflectionEscape,
            _ => Self::CentrifugalBarrier,
        }
    }

    /// Human-readable dynamical regime label.
    #[inline]
    #[must_use]
    pub const fn label(self, is_wormhole: bool) -> &'static str {
        match (self, is_wormhole) {
            (Self::HorizonCaptureOrTransit, true) => "Throat Transit (Crosses l = 0)",
            (Self::HorizonCaptureOrTransit, false) => "Horizon Capture (Plunge into r+)",
            (Self::PhotonRingWinding, _) => "Photon-Ring Winding / Rosette",
            (Self::WeakDeflectionEscape, true) => "Same-Sheet Reflection",
            (Self::WeakDeflectionEscape, false) => "Weak-Field Deflection Escape",
            (Self::CentrifugalBarrier, _) => "Turning-Point Barrier",
        }
    }
}

/// Zero-allocation summary returned by 2D relativistic lensing regime evaluators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialAtlasSummary {
    /// Classified orbital or lensing regime.
    pub regime: LensingRegime,
    /// Normalized primary observable in `[0.0, 1.0]` (winding fraction or throat clearance).
    pub primary: f64,
    /// Secondary diagnostic radius ($r_{\min}$, $r_+$, or wormhole throat radius).
    pub secondary: f64,
}

/// Conserved equatorial Kerr radial orbit parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrRadialOrbitParams {
    /// Schwarzschild radius $R_s = 2GM / c^2$.
    pub rs: f64,
    /// Kerr spin length $a = J / (Mc)$.
    pub spin_a: f64,
    /// Conserved specific energy $e$.
    pub energy: f64,
    /// Conserved specific angular momentum or impact parameter $\ell$.
    pub ell: f64,
    /// Mass-shell indicator $\kappa \in \{0, 1\}$ (`0.0` null, `1.0` timelike).
    pub kappa: f64,
}

/// Exact prograde equatorial Kerr critical photon impact parameter $b_{\text{crit}}^+(\chi)$.
///
/// # Examples
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::high_energy::kerr_prograde_critical_b;
///
/// let b_schwarz = kerr_prograde_critical_b(2.0, 0.0);
/// assert!((b_schwarz - 3.0 * 3.0_f64.sqrt()).abs() < 1.0e-12);
/// ```
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn kerr_prograde_critical_b(rs: f64, spin_chi: f64) -> f64 {
    let m = 0.5 * rs.max(0.1);
    let chi = spin_chi.clamp(-0.98, 0.98);
    let b = -m * chi + 6.0 * m * ((-chi).acos() / 3.0).cos();
    debug_assert!(b > 0.0 && b.is_finite());
    b
}

/// Bardeen-Press-Teukolsky (1972) equatorial Kerr ISCO radius $r_{\text{ISCO}}(M, \chi, \ell)$.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn kerr_bpt_isco(m_geom: f64, chi: f64, impact_or_momentum: f64) -> f64 {
    if chi.abs() <= 1e-9 {
        return 6.0 * m_geom;
    }
    let c = chi.clamp(-0.99, 0.99);
    let factor = (1.0 - c * c).max(0.0).cbrt();
    let z1 = 1.0 + factor * ((1.0 + c).max(0.0).cbrt() + (1.0 - c).max(0.0).cbrt());
    let z2 = (3.0 * c * c + z1 * z1).sqrt();
    let root = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).max(0.0).sqrt();
    let out = if c * impact_or_momentum >= 0.0 {
        m_geom * (3.0 + z2 - root)
    } else {
        m_geom * (3.0 + z2 + root)
    };
    debug_assert!(out.is_finite());
    out
}

/// Evaluates the reduced Kerr equatorial effective potential $V_{\text{eff}}(r)$.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn kerr_veff(rs: f64, a: f64, r: f64, e: f64, ell: f64, kappa: f64) -> f64 {
    let inv_r = 1.0 / r.max(1.0e-6);
    let inv_r2 = inv_r * inv_r;
    let diff = ell - a * e;
    let v = 0.5 * (1.0 - rs * inv_r + a * a * inv_r2) * kappa
        + 0.5 * (ell * ell - a * a * e * e) * inv_r2
        - 0.5 * rs * diff * diff * inv_r2 * inv_r;
    debug_assert!(v.is_finite());
    v
}

/// Evaluates radial acceleration $-\partial_r V_{\text{eff}}$ and azimuthal rate $d\varphi / d\lambda$.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn kerr_radial_acc_and_dphi(
    rs: f64,
    a: f64,
    e: f64,
    ell: f64,
    kappa: f64,
    r: f64,
) -> (f64, f64) {
    let inv_r = 1.0 / r.max(0.1);
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    let diff = ell - a * e;
    let dv_dr = (0.5 * rs * inv_r2 - a * a * inv_r3) * kappa - (ell * ell - a * a * e * e) * inv_r3
        + 1.5 * rs * diff * diff * inv_r2 * inv_r2;
    let delta = (r * r - rs * r + a * a).max(1e-6);
    let dphi = ((a * rs * inv_r) * e + (1.0 - rs * inv_r) * ell) / delta;
    debug_assert!(dv_dr.is_finite() && dphi.is_finite());
    (-dv_dr, dphi)
}

/// Advances $(r, v_r, \varphi)$ by one classical 4-stage RK4 step in the Kerr radial potential.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn kerr_radial_rk4_step(
    p: KerrRadialOrbitParams,
    rad: f64,
    vr: f64,
    dt: f64,
) -> (f64, f64, f64) {
    let (a1, p1) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, rad);
    let (r2, v2) = ((rad + 0.5 * dt * vr).max(0.15), vr + 0.5 * dt * a1);
    let (a2, p2) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, r2);
    let (r3, v3) = ((rad + 0.5 * dt * v2).max(0.15), vr + 0.5 * dt * a2);
    let (a3, p3) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, r3);
    let (r4, v4) = ((rad + dt * v3).max(0.15), vr + dt * a3);
    let (a4, p4) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, r4);
    let out = (
        (dt / 6.0) * (vr + 2.0 * v2 + 2.0 * v3 + v4),
        (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4),
        (dt / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4),
    );
    debug_assert!(out.0.is_finite() && out.1.is_finite() && out.2.is_finite());
    out
}

/// Classifies one `(r0, b, chi)` equatorial Kerr or Schwarzschild cell via zero-allocation radial RK4.
///
/// # Examples
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::high_energy::{
///     eval_kerr_lensing_cell, GeodesicKind, LensingRegime,
/// };
///
/// let plunge = eval_kerr_lensing_cell(GeodesicKind::Null, -0.85, 2.0, 0.0, 10.0, 1.0);
/// assert_eq!(plunge.regime, LensingRegime::HorizonCaptureOrTransit);
/// ```
#[must_use]
#[verified_engine::verified]
pub fn eval_kerr_lensing_cell(
    kind: GeodesicKind,
    radial_bias: f64,
    rs: f64,
    chi: f64,
    r0_in: f64,
    b_val: f64,
) -> RadialAtlasSummary {
    let a = 0.5 * rs * chi.clamp(-0.95, 0.95);
    let r_plus = 0.5 * (rs + (rs * rs - 4.0 * a * a).max(0.0).sqrt());
    let r0 = r0_in.max(1.06 * r_plus);
    let kappa = kind.kappa();
    let e = match kind {
        GeodesicKind::Null => 1.0,
        GeodesicKind::Timelike => {
            let delta = (r0 * r0 - rs * r0 + a * a).max(1e-6);
            let g11 = (r0 * r0) / delta;
            let g33 = (r0 * r0 + a * a + (a * a * rs) / r0).max(1e-6);
            let ur = radial_bias * 0.45;
            let ut = ((g33 / delta) * (1.0 + (b_val * b_val) / g33 + g11 * ur * ur))
                .max(1e-6)
                .sqrt();
            ((a * rs) / (r0 * g33)) * b_val + (delta / g33) * ut
        }
    };
    let e_half_sq = 0.5 * e * e;
    let v_init = kerr_veff(rs, a, r0, e, b_val, kappa);
    if v_init > e_half_sq + 1e-6 {
        return RadialAtlasSummary {
            regime: LensingRegime::CentrifugalBarrier,
            primary: 0.0,
            secondary: r0,
        };
    }
    let sign = if radial_bias > 0.0 { 1.0 } else { -1.0 };
    let vr0 = sign * (2.0 * (e_half_sq - v_init).max(0.0)).sqrt();
    let orbit = KerrRadialOrbitParams {
        rs,
        spin_a: a,
        energy: e,
        ell: b_val,
        kappa,
    };
    integrate_kerr_radial_summary(kind, orbit, r0, vr0, r_plus)
}

#[verified_engine::verified]
fn integrate_kerr_radial_summary(
    kind: GeodesicKind,
    orbit: KerrRadialOrbitParams,
    r0: f64,
    vr0: f64,
    r_plus: f64,
) -> RadialAtlasSummary {
    let (mut r, mut vr, mut phi, mut r_min) = (r0, vr0, 0.0_f64, r0);
    for _ in 0..72 {
        let (dr, dvr, dphi) = kerr_radial_rk4_step(orbit, r, vr, 0.24);
        r += dr;
        vr += dvr;
        phi += dphi;
        r_min = r_min.min(r);
        if !r.is_finite() || r <= 1.03 * r_plus {
            return RadialAtlasSummary {
                regime: LensingRegime::HorizonCaptureOrTransit,
                primary: (phi.abs() / (2.0 * PI)).clamp(0.0, 1.0),
                secondary: r_plus,
            };
        }
        if r > 26.0 && vr > 0.0 {
            break;
        }
    }
    let regime = if phi.abs() >= 1.35 * PI || (kind == GeodesicKind::Timelike && r < 16.0) {
        LensingRegime::PhotonRingWinding
    } else {
        LensingRegime::WeakDeflectionEscape
    };
    let primary = (phi.abs() / (2.5 * PI)).clamp(0.0, 1.0);
    debug_assert!((0.0..=1.0).contains(&primary));
    RadialAtlasSummary {
        regime,
        primary,
        secondary: r_min,
    }
}

// [cite:general_relativity_geodesics]

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_kerr_radial_cell_zero_alloc() {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (plunge, _) = VerifiedEngine::run_verified(|| {
            eval_kerr_lensing_cell(GeodesicKind::Null, -0.82, 2.0, 0.65, 10.0, 1.2)
        });
        let (escape, _) = VerifiedEngine::run_verified(|| {
            eval_kerr_lensing_cell(GeodesicKind::Null, -0.82, 2.0, 0.65, 10.0, 8.5)
        });
        assert_eq!(plunge.regime, LensingRegime::HorizonCaptureOrTransit);
        assert_eq!(escape.regime, LensingRegime::WeakDeflectionEscape);
    }
}
