#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation radial RK4 and throat evaluation kernels for [`super::RelativityCartographer`].

use std::f64::consts::PI;

use crate::async_sim::atlas::AtlasCell;
use math_commons::ast::CompiledExpr;

use super::super::sandbox::MIN_REGULAR_LAPSE;
use super::super::GeodesicMode;
use super::{LensingRegime, RelativityCartographer};

#[derive(Clone, Copy)]
struct KerrOrbitParams {
    rs: f64,
    spin_a: f64,
    energy: f64,
    ell: f64,
    kappa: f64,
}

impl RelativityCartographer {
    #[inline]
    pub(super) fn eval_black_hole_cell(
        &self,
        rs: f64,
        chi: f64,
        r0_in: f64,
        b_val: f64,
    ) -> AtlasCell {
        let a = 0.5 * rs * chi.clamp(-0.95, 0.95);
        let r_plus = 0.5 * (rs + (rs * rs - 4.0 * a * a).max(0.0).sqrt());
        let r0 = r0_in.max(1.06 * r_plus);
        let kappa = if self.mode == GeodesicMode::NullPhoton {
            0.0
        } else {
            1.0
        };
        let e = if self.mode == GeodesicMode::NullPhoton {
            1.0
        } else {
            let delta = (r0 * r0 - rs * r0 + a * a).max(1e-6);
            let g11 = (r0 * r0) / delta;
            let g33 = (r0 * r0 + a * a + (a * a * rs) / r0).max(1e-6);
            let ur = self.radial_bias * 0.45;
            let ut = ((g33 / delta) * (1.0 + (b_val * b_val) / g33 + g11 * ur * ur))
                .max(1e-6)
                .sqrt();
            ((a * rs) / (r0 * g33)) * b_val + (delta / g33) * ut
        };
        let e_half_sq = 0.5 * e * e;
        let v_init = kerr_veff(rs, a, r0, e, b_val, kappa);
        if v_init > e_half_sq + 1e-6 {
            return AtlasCell {
                regime: LensingRegime::CentrifugalBarrier as u8,
                primary: 0.0,
                secondary: r0,
            };
        }
        let sign = if self.radial_bias > 0.0 { 1.0 } else { -1.0 };
        let vr0 = sign * (2.0 * (e_half_sq - v_init).max(0.0)).sqrt();
        let orbit = KerrOrbitParams {
            rs,
            spin_a: a,
            energy: e,
            ell: b_val,
            kappa,
        };
        self.integrate_black_hole_radial(orbit, r0, vr0, r_plus)
    }

    #[inline]
    fn integrate_black_hole_radial(
        &self,
        orbit: KerrOrbitParams,
        r0: f64,
        vr0: f64,
        r_plus: f64,
    ) -> AtlasCell {
        let (mut r, mut vr, mut phi, mut r_min) = (r0, vr0, 0.0_f64, r0);
        for _ in 0..72 {
            let (dr, dvr, dphi) = kerr_radial_rk4_step(orbit, r, vr, 0.24);
            r += dr;
            vr += dvr;
            phi += dphi;
            r_min = r_min.min(r);
            if !r.is_finite() || r <= 1.03 * r_plus {
                return AtlasCell {
                    regime: LensingRegime::HorizonCaptureOrTransit as u8,
                    primary: (phi.abs() / (2.0 * PI)).clamp(0.0, 1.0),
                    secondary: r_plus,
                };
            }
            if r > 26.0 && vr > 0.0 {
                break;
            }
        }
        let regime = if phi.abs() >= 1.35 * PI
            || (self.mode == GeodesicMode::TimelikeParticle && r < 16.0)
        {
            LensingRegime::PhotonRingWinding
        } else {
            LensingRegime::WeakDeflectionEscape
        };
        AtlasCell {
            regime: regime as u8,
            primary: (phi.abs() / (2.5 * PI)).clamp(0.0, 1.0),
            secondary: r_min,
        }
    }

    #[inline]
    pub(super) fn eval_custom_lapse_cell(
        &self,
        rs: f64,
        compiled: &CompiledExpr,
        params: &[f64],
        r0_in: f64,
        b_val: f64,
    ) -> AtlasCell {
        let r0 = r0_in.max(0.35);
        let f0 = compiled.eval_2d(r0, rs, params);
        if !f0.is_finite() || f0 <= MIN_REGULAR_LAPSE {
            return AtlasCell {
                regime: LensingRegime::HorizonCaptureOrTransit as u8,
                primary: 1.0,
                secondary: r0,
            };
        }
        let kappa = if self.mode == GeodesicMode::NullPhoton {
            0.0
        } else {
            1.0
        };
        let v0 = 0.5 * f0 * (kappa + (b_val * b_val) / (r0 * r0));
        let e_half_sq = if self.mode == GeodesicMode::NullPhoton {
            0.5
        } else {
            0.5 * (self.radial_bias * 0.45).powi(2) + v0
        };
        if v0 > e_half_sq + 1e-6 {
            return AtlasCell {
                regime: LensingRegime::CentrifugalBarrier as u8,
                primary: 0.0,
                secondary: r0,
            };
        }
        let sign = if self.radial_bias > 0.0 { 1.0 } else { -1.0 };
        let vr0 = sign * (2.0 * (e_half_sq - v0).max(0.0)).sqrt();
        integrate_custom_lapse_radial(rs, compiled, params, kappa, b_val, r0, vr0)
    }

    #[inline]
    pub(super) fn eval_wormhole_cell(
        &self,
        b0: f64,
        custom: Option<&(CompiledExpr, [f64; 8], usize)>,
        l0: f64,
        b_val: f64,
    ) -> AtlasCell {
        let eval_r2 = |l: f64| match custom {
            Some((c, p, n)) => {
                let bv = c.eval_2d(l, b0, &p[..*n]);
                if bv.is_finite() && bv > 0.1 {
                    bv * bv
                } else {
                    0.01
                }
            }
            None => b0 * b0 + l * l,
        };
        let r0_sq = eval_r2(l0);
        let throat_sq = eval_r2(0.0);
        let kinetic_0 = if self.mode == GeodesicMode::NullPhoton {
            1.0 - (b_val * b_val) / r0_sq
        } else {
            self.radial_bias * self.radial_bias
        };
        if kinetic_0 < 0.0 {
            return AtlasCell {
                regime: LensingRegime::CentrifugalBarrier as u8,
                primary: 0.0,
                secondary: l0.abs(),
            };
        }
        let e_kin = kinetic_0 + (b_val * b_val) / r0_sq;
        let transits = e_kin > (b_val * b_val) / throat_sq && l0 * self.radial_bias <= 0.0;
        let ratio = (b_val.abs() / throat_sq.sqrt().max(0.1)).clamp(0.0, 2.0);
        let regime = if transits {
            LensingRegime::HorizonCaptureOrTransit
        } else if (ratio - 1.0).abs() < 0.12 {
            LensingRegime::PhotonRingWinding
        } else {
            LensingRegime::WeakDeflectionEscape
        };
        AtlasCell {
            regime: regime as u8,
            primary: (1.0 - 0.5 * ratio).clamp(0.0, 1.0),
            secondary: throat_sq.sqrt(),
        }
    }
}

#[inline]
fn integrate_custom_lapse_radial(
    rs: f64,
    compiled: &CompiledExpr,
    params: &[f64],
    kappa: f64,
    b_val: f64,
    r0: f64,
    vr0: f64,
) -> AtlasCell {
    let (mut r, mut vr, mut phi, mut r_min) = (r0, vr0, 0.0_f64, r0);
    for _ in 0..64 {
        let h = 1e-4 * r.max(1.0);
        let fp = compiled.eval_2d(r + h, rs, params);
        let fm = compiled.eval_2d((r - h).max(0.06), rs, params);
        if !fp.is_finite() || !fm.is_finite() || fm <= MIN_REGULAR_LAPSE {
            return capture_cell(phi, r.max(0.0));
        }
        let vp = 0.5 * fp * (kappa + (b_val * b_val) / ((r + h) * (r + h)));
        let vm = 0.5 * fm * (kappa + (b_val * b_val) / (r - h).max(0.06).powi(2));
        vr -= ((vp - vm) / (2.0 * h)) * 0.24;
        r += vr * 0.24;
        phi += (b_val / (r * r).max(0.04)) * 0.24;
        r_min = r_min.min(r);
        if r <= 0.2 || compiled.eval_2d(r, rs, params) <= MIN_REGULAR_LAPSE {
            return capture_cell(phi, r.max(0.0));
        }
        if r > 26.0 && vr > 0.0 {
            break;
        }
    }
    let regime = if phi.abs() >= 1.35 * PI {
        LensingRegime::PhotonRingWinding
    } else {
        LensingRegime::WeakDeflectionEscape
    };
    AtlasCell {
        regime: regime as u8,
        primary: (phi.abs() / (2.5 * PI)).clamp(0.0, 1.0),
        secondary: r_min,
    }
}

#[inline]
fn capture_cell(phi: f64, secondary: f64) -> AtlasCell {
    AtlasCell {
        regime: LensingRegime::HorizonCaptureOrTransit as u8,
        primary: (phi.abs() / (2.0 * PI)).clamp(0.0, 1.0),
        secondary,
    }
}

#[inline]
fn kerr_veff(rs: f64, a: f64, r: f64, e: f64, ell: f64, kappa: f64) -> f64 {
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;
    let diff = ell - a * e;
    0.5 * (1.0 - rs * inv_r + a * a * inv_r2) * kappa + 0.5 * (ell * ell - a * a * e * e) * inv_r2
        - 0.5 * rs * diff * diff * inv_r2 * inv_r
}

#[inline]
fn kerr_radial_acc_and_dphi(rs: f64, a: f64, e: f64, ell: f64, kappa: f64, r: f64) -> (f64, f64) {
    let inv_r = 1.0 / r.max(0.1);
    let inv_r2 = inv_r * inv_r;
    let inv_r3 = inv_r2 * inv_r;
    let diff = ell - a * e;
    let dv_dr = (0.5 * rs * inv_r2 - a * a * inv_r3) * kappa - (ell * ell - a * a * e * e) * inv_r3
        + 1.5 * rs * diff * diff * inv_r2 * inv_r2;
    let delta = (r * r - rs * r + a * a).max(1e-6);
    let dphi = ((a * rs * inv_r) * e + (1.0 - rs * inv_r) * ell) / delta;
    (-dv_dr, dphi)
}

#[inline]
fn kerr_radial_rk4_step(p: KerrOrbitParams, rad: f64, vr: f64, dt: f64) -> (f64, f64, f64) {
    let (a1, p1) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, rad);
    let (r2, v2) = ((rad + 0.5 * dt * vr).max(0.15), vr + 0.5 * dt * a1);
    let (a2, p2) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, r2);
    let (r3, v3) = ((rad + 0.5 * dt * v2).max(0.15), vr + 0.5 * dt * a2);
    let (a3, p3) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, r3);
    let (r4, v4) = ((rad + dt * v3).max(0.15), vr + dt * a3);
    let (a4, p4) = kerr_radial_acc_and_dphi(p.rs, p.spin_a, p.energy, p.ell, p.kappa, r4);
    (
        (dt / 6.0) * (vr + 2.0 * v2 + 2.0 * v3 + v4),
        (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4),
        (dt / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4),
    )
}
