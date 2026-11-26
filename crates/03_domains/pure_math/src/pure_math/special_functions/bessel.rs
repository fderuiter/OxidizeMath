// [cite:attention_is_all_you_need_rust]
//! High-precision Bessel functions $J_\nu(x)$, $Y_\nu(x)$, and $j_\ell(x)$ across small,
//! intermediate (Steed CF1/CF2 + Temme), and asymptotic (Hankel) domains.

use statrs::function::gamma::gamma;
use std::f64::consts::{EULER_GAMMA, PI};

const STEED_EPS: f64 = 1e-16;
const STEED_FPMIN: f64 = 1e-300;
const STEED_MAX_ITERS: usize = 10_000;
const TEMME_SPLIT_X: f64 = 2.0;

#[inline]
fn clamp_fpmin(val: f64) -> f64 {
    if val.abs() < STEED_FPMIN {
        if val < 0.0 { -STEED_FPMIN } else { STEED_FPMIN }
    } else {
        val
    }
}

/// Evaluates Hankel's asymptotic expansion (DLMF 10.17.3–10.17.4) for $J_\nu(x)$ and $Y_\nu(x)$
/// when $x \ge 25$ and $x > 0.65\nu^2 + 22$.
#[verified_engine::verified]
fn bessel_hankel_asymptotic(nu: f64, x: f64) -> (f64, f64) {
    let mu = 4.0 * nu * nu;
    let eight_x = 8.0 * x;
    let mut p_sum = 1.0;
    let mut q_sum = 0.0;
    let mut term = 1.0;

    for k in 1..=48_usize {
        let odd = (2 * k - 1) as f64;
        let next_term = term * (mu - odd * odd) / ((k as f64) * eight_x);
        if next_term.abs() > term.abs() && k > 4 {
            break;
        }
        term = next_term;
        let sign = if (k / 2) % 2 == 1 { -1.0 } else { 1.0 };
        if k % 2 == 0 {
            p_sum += sign * term;
        } else {
            q_sum += sign * term;
        }
        if term.abs() < 1e-17 {
            break;
        }
    }

    let phase_turns = (0.5 * nu + 0.25).rem_euclid(2.0);
    let chi = x - phase_turns * PI;
    let (sin_chi, cos_chi) = chi.sin_cos();
    let amp = (2.0 / (PI * x)).sqrt();
    let j_val = amp * (p_sum * cos_chi - q_sum * sin_chi);
    let y_val = amp * (p_sum * sin_chi + q_sum * cos_chi);
    assert!(j_val.is_finite() && y_val.is_finite());
    (j_val, y_val)
}

/// Evaluates Lentz's modified continued fraction CF1 for $J'_\nu(x) / J_\nu(x)$ and scales down `nl` steps.
#[verified_engine::verified]
fn steed_cf1_and_downward(nu: f64, x: f64, nl: usize) -> (f64, f64, f64) {
    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;
    let mut h = clamp_fpmin(nu * xi);
    let mut b = xi2 * nu;
    let mut d = 0.0;
    let mut c = h;
    let mut isign = 1.0_f64;

    for _ in 0..STEED_MAX_ITERS {
        b += xi2;
        d = clamp_fpmin(b - d);
        c = clamp_fpmin(b - 1.0 / c);
        d = 1.0 / d;
        let delt = c * d;
        h *= delt;
        if d < 0.0 {
            isign = -isign;
        }
        if (delt - 1.0).abs() < STEED_EPS {
            break;
        }
    }

    let mut ril = isign * STEED_FPMIN;
    let mut ripl = h * ril;
    let ril1 = ril;
    let mut fact = nu * xi;
    for _ in 0..nl {
        let ritemp = fact * ril + ripl;
        fact -= xi;
        ripl = fact * ritemp - ril;
        ril = ritemp;
    }
    if ril == 0.0 {
        ril = STEED_FPMIN;
    }
    assert!(ril.is_finite());
    (ril, ril1, ripl / ril)
}

/// Evaluates Temme's convergent small-argument series ($x < 2$) for $J_\mu, Y_\mu, Y_{\mu+1}$ with $|\mu| \le 0.5$.
#[verified_engine::verified]
fn temme_small_x(mu: f64, x: f64, f_ratio: f64) -> (f64, f64, f64) {
    let x2 = 0.5 * x;
    let xi = 1.0 / x;
    let xi2 = 2.0 * xi;
    let w = xi2 / PI;
    let pimu = PI * mu;
    let fact = if pimu.abs() < STEED_EPS {
        1.0
    } else {
        pimu / pimu.sin()
    };
    let d_log = -x2.ln();
    let e_val = mu * d_log;
    let fact2 = if e_val.abs() < STEED_EPS {
        1.0
    } else {
        e_val.sinh() / e_val
    };
    let gam_pl = 1.0 / gamma(1.0 + mu);
    let gam_mi = 1.0 / gamma(1.0 - mu);
    let gam1 = if mu.abs() > 1e-6 {
        (gam_mi - gam_pl) / (2.0 * mu)
    } else {
        -EULER_GAMMA
    };
    let gam2 = 0.5 * (gam_mi + gam_pl);
    let mut ff = (2.0 / PI) * fact * (gam1 * e_val.cosh() + gam2 * fact2 * d_log);
    let e_exp = e_val.exp();
    let mut p = e_exp / (PI * gam_pl);
    let mut q = 1.0 / (e_exp * PI * gam_mi);
    let pimu2 = 0.5 * pimu;
    let fact3 = if pimu2.abs() < STEED_EPS {
        1.0
    } else {
        pimu2.sin() / pimu2
    };
    let r = PI * pimu2 * fact3 * fact3;
    let mut c = 1.0;
    let d_sq = -x2 * x2;
    let mut sum_val = ff + r * q;
    let mut sum1 = p;
    let mu2 = mu * mu;

    for i in 1..=256_usize {
        let i_f = i as f64;
        ff = (i_f * ff + p + q) / (i_f * i_f - mu2);
        c *= d_sq / i_f;
        p /= i_f - mu;
        q /= i_f + mu;
        let delt = c * (ff + r * q);
        sum_val += delt;
        sum1 += c * p - i_f * delt;
        if delt.abs() < (1.0 + sum_val.abs()) * STEED_EPS {
            break;
        }
    }
    let rymu = -sum_val;
    let ry1 = -sum1 * xi2;
    let rymup = mu * xi * rymu - ry1;
    let rjmu = w / (rymup - f_ratio * rymu);
    assert!(rjmu.is_finite() && rymu.is_finite());
    (rjmu, rymu, ry1)
}

/// Evaluates Steed's complex continued fraction CF2 ($x \ge 2$) for $J_\mu, Y_\mu, Y_{\mu+1}$.
#[verified_engine::verified]
fn steed_cf2_large_x(mu: f64, x: f64, f_ratio: f64, ril_sign: f64) -> (f64, f64, f64) {
    let xi = 1.0 / x;
    let w = (2.0 * xi) / PI;
    let mut a = 0.25 - mu * mu;
    let mut p = -0.5 * xi;
    let mut q = 1.0;
    let br = 2.0 * x;
    let mut bi = 2.0;
    let mut fact = a * xi / (p * p + q * q);
    let mut cr = br + q * fact;
    let mut ci = bi + p * fact;
    let mut den = br * br + bi * bi;
    let mut dr = br / den;
    let mut di = -bi / den;
    let mut dlr = cr * dr - ci * di;
    let mut dli = cr * di + ci * dr;
    let mut temp = p * dlr - q * dli;
    q = p * dli + q * dlr;
    p = temp;

    for i in 2..=STEED_MAX_ITERS {
        a += (2 * (i - 1)) as f64;
        bi += 2.0;
        dr = a * dr + br;
        di = a * di + bi;
        if dr.abs() + di.abs() < STEED_FPMIN {
            dr = STEED_FPMIN;
        }
        fact = a / (cr * cr + ci * ci);
        cr = br + cr * fact;
        ci = bi - ci * fact;
        if cr.abs() + ci.abs() < STEED_FPMIN {
            cr = STEED_FPMIN;
        }
        den = dr * dr + di * di;
        dr /= den;
        di = -di / den;
        dlr = cr * dr - ci * di;
        dli = cr * di + ci * dr;
        temp = p * dlr - q * dli;
        q = p * dli + q * dlr;
        p = temp;
        if (dlr - 1.0).abs() + dli.abs() < STEED_EPS {
            break;
        }
    }

    let gam = (p - f_ratio) / q;
    let mut rjmu = (w / ((p - f_ratio) * gam + q)).sqrt();
    if ril_sign < 0.0 {
        rjmu = -rjmu;
    }
    let rymu = rjmu * gam;
    let rymup = rymu * (p + q / gam);
    let ry1 = mu * xi * rymu - rymup;
    assert!(rjmu.is_finite() && rymu.is_finite());
    (rjmu, rymu, ry1)
}

/// Evaluates $(J_\nu(x), Y_\nu(x))$ simultaneously for $\nu \ge 0$ and $x > 0$.
#[verified_engine::verified]
fn bessel_jy_nonneg(nu: f64, x: f64) -> (f64, f64) {
    if x >= 25.0 && x > 0.65 * nu * nu + 22.0 {
        return bessel_hankel_asymptotic(nu, x);
    }
    let nl = if x < TEMME_SPLIT_X {
        (nu + 0.5).floor().max(0.0) as usize
    } else {
        (nu - x + 1.5).floor().max(0.0) as usize
    };
    let mu = nu - nl as f64;
    let (ril, ril1, f_ratio) = steed_cf1_and_downward(nu, x, nl);
    let (rjmu, mut ry, mut ry_next) = if x < TEMME_SPLIT_X {
        temme_small_x(mu, x, f_ratio)
    } else {
        steed_cf2_large_x(mu, x, f_ratio, ril)
    };

    let rj = ril1 * (rjmu / ril);
    let xi2 = 2.0 / x;
    for i in 1..=nl {
        let ry_step = (mu + i as f64) * xi2 * ry_next - ry;
        ry = ry_next;
        ry_next = ry_step;
    }
    assert!(rj.is_finite());
    (rj, ry)
}

/// Bessel Function of the First Kind $J_\nu(x)$ accurate to `< 1e-13` across small, intermediate, and asymptotic domains.
#[verified_engine::verified]
pub fn bessel_j(nu: f64, x: f64) -> f64 {
    if !nu.is_finite() || !x.is_finite() {
        return f64::NAN;
    }
    let rounded = nu.round();
    let is_int = (nu - rounded).abs() < 1e-12;
    if x == 0.0 {
        if nu == 0.0 {
            return 1.0;
        }
        if nu > 0.0 || is_int {
            return 0.0;
        }
        return f64::NAN;
    }
    if x < 0.0 && !is_int {
        return f64::NAN;
    }
    let abs_x = x.abs();
    let x_parity = if x < 0.0 && ((rounded as i64).unsigned_abs() % 2 == 1) {
        -1.0
    } else {
        1.0
    };
    if nu >= 0.0 {
        let (j_val, _) = bessel_jy_nonneg(nu, abs_x);
        return x_parity * j_val;
    }
    let abs_nu = -nu;
    let (j_pos, y_pos) = bessel_jy_nonneg(abs_nu, abs_x);
    let val = if is_int {
        let k = (rounded as i64).unsigned_abs();
        if k % 2 == 1 { -j_pos } else { j_pos }
    } else {
        let (sin_p, cos_p) = (abs_nu * PI).sin_cos();
        j_pos * cos_p - y_pos * sin_p
    };
    x_parity * val
}

/// Bessel Function of the Second Kind $Y_\nu(x)$ (Neumann function) accurate across small and asymptotic domains.
#[verified_engine::verified]
pub fn bessel_y(nu: f64, x: f64) -> f64 {
    if !nu.is_finite() || x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::NAN;
    }
    if nu >= 0.0 {
        let (_, y_val) = bessel_jy_nonneg(nu, x);
        return y_val;
    }
    let abs_nu = -nu;
    let (j_pos, y_pos) = bessel_jy_nonneg(abs_nu, x);
    let rounded = nu.round();
    if (nu - rounded).abs() < 1e-12 {
        let k = (rounded as i64).unsigned_abs();
        if k % 2 == 1 { -y_pos } else { y_pos }
    } else {
        let (sin_p, cos_p) = (abs_nu * PI).sin_cos();
        j_pos * sin_p + y_pos * cos_p
    }
}

/// Spherical Bessel Function of the First Kind $j_\ell(x) = \sqrt{\frac{\pi}{2x}} J_{\ell+1/2}(x)$.
#[verified_engine::verified]
pub fn spherical_bessel_j(l: u64, x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x.abs() < 1e-8 {
        if l == 0 {
            return 1.0 - (x * x) / 6.0;
        }
        return 0.0;
    }
    let abs_x = x.abs();
    let val = (PI / (2.0 * abs_x)).sqrt() * bessel_j(l as f64 + 0.5, abs_x);
    if x < 0.0 && l % 2 == 1 { -val } else { val }
}

/// Checks orthogonality of Bessel functions $J_\nu(\alpha x)$ and $J_\nu(\beta x)$ on `[0, 1]`.
#[verified_engine::verified]
pub fn check_orthogonality_bessel(nu: f64, alpha: f64, beta: f64) -> f64 {
    let integrand = |x: f64| x * bessel_j(nu, alpha * x) * bessel_j(nu, beta * x);
    let steps = 1000;
    let mut sum = 0.0;
    let h = 1.0 / steps as f64;

    for i in 0..steps {
        let x = i as f64 * h;
        let x_next = (i + 1) as f64 * h;
        sum += 0.5 * (integrand(x) + integrand(x_next)) * h;
    }
    sum
}

crate::golden_fixture_verification!(
    module = "pure_math",
    test = {
        let j0_40 = bessel_j(0.0, 40.0);
        assert!((j0_40 - 0.007_366_890_584_237_29).abs() < 1e-12);
    }
);
crate::interval_enclosure_verification!(
    module = "pure_math",
    test = {
        let iv = math_commons::primitives::Interval::point(1.0)
            .unwrap()
            .exp();
        assert!(iv.contains(std::f64::consts::E));
    }
);
crate::formal_proof_verification!(
    module = "pure_math",
    proof = "IntervalSoundness.lean",
    test = {
        let a = math_commons::primitives::Interval::new(1.0, 2.0).unwrap();
        let b = math_commons::primitives::Interval::new(0.5, 1.5).unwrap();
        assert!((a - b).contains(0.5));
    }
);

#[cfg(all(test, not(target_arch = "wasm32")))]
mod proptest_suite {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    #[test]
    #[verified_engine::verified]
    fn proptest_bessel_wronskian_and_recurrence_across_all_regimes() {
        let mut runner = TestRunner::new_with_rng(
            Config::with_cases(80),
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        );
        let strategy = (0.25f64..5.5f64, 0.2f64..80.0f64);
        let res = runner.run(&strategy, |(nu, x)| {
            let j_nu = bessel_j(nu, x);
            let j_p1 = bessel_j(nu + 1.0, x);
            let j_m1 = bessel_j(nu - 1.0, x);
            let y_nu = bessel_y(nu, x);
            let y_p1 = bessel_y(nu + 1.0, x);

            let wronskian = j_nu * y_p1 - j_p1 * y_nu;
            let expected_w = -2.0 / (PI * x);
            prop_assert!(
                (wronskian - expected_w).abs() < 1e-11 * expected_w.abs().max(1.0),
                "Wronskian mismatch at nu={nu}, x={x}: got {wronskian}, expected {expected_w}"
            );
            let rec = j_m1 + j_p1 - (2.0 * nu / x) * j_nu;
            prop_assert!(
                rec.abs() < 1e-11 * (2.0 * nu / x).max(1.0),
                "Recurrence mismatch at nu={nu}, x={x}: residual={rec}"
            );
            Ok(())
        });
        assert!(res.is_ok());
    }
}
