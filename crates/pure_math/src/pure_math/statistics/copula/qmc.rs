//! Zero-allocation Gauss-Legendre bivariate normal CDF and Genz seeded QMC multivariate normal CDF.

use crate::pure_math::statistics::distributions::special::{
    sample_uniform_open, standard_normal_cdf, standard_normal_quantile,
};
use nalgebra::DMatrix;
use oxidize_core::rng::OxidizeRng;
use std::f64::consts::TAU;

/// 8 positive abscissas and weights of the symmetric 16-point Gauss-Legendre rule on `[-1, 1]`.
const GL16_NODES: [f64; 8] = [
    0.095_012_509_837_637_44,
    0.281_603_550_779_258_9,
    0.458_016_777_657_227_37,
    0.617_876_244_402_643_8,
    0.755_404_408_355_003,
    0.865_631_202_387_831_8,
    0.944_575_023_073_232_6,
    0.989_400_934_991_649_9,
];

const GL16_WEIGHTS: [f64; 8] = [
    0.189_450_610_455_068_5,
    0.182_603_415_044_923_58,
    0.169_156_519_395_002_54,
    0.149_595_988_816_576_74,
    0.124_628_971_255_533_88,
    0.095_158_511_682_492_79,
    0.062_253_523_938_647_894,
    0.027_152_459_411_754_096,
];

const PRIMES_16: [f64; 16] = [
    2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0, 47.0, 53.0,
];

/// Evaluates $\Phi_2(z_1, z_2; \rho)$ via 16-point Gauss-Legendre quadrature on $\theta \in [0, \arcsin \rho]$.
#[verified_engine::verified]
pub(crate) fn bivariate_standard_normal_cdf(z1: f64, z2: f64, rho: f64) -> f64 {
    let p1 = standard_normal_cdf(z1);
    let p2 = standard_normal_cdf(z2);
    if rho.abs() < 1e-15 {
        return (p1 * p2).clamp(0.0, 1.0);
    }
    let asin_rho = rho.clamp(-1.0, 1.0).asin();
    let half_len = 0.5 * asin_rho;
    let mid = 0.5 * asin_rho;
    let z_sq_sum = z1 * z1 + z2 * z2;
    let two_z1_z2 = 2.0 * z1 * z2;

    let mut integral = 0.0;
    for k in 0..8 {
        let offset = half_len * GL16_NODES[k];
        let w = GL16_WEIGHTS[k];
        integral += w * plackett_trig_integrand(z_sq_sum, two_z1_z2, mid - offset);
        integral += w * plackett_trig_integrand(z_sq_sum, two_z1_z2, mid + offset);
    }
    let raw = p1 * p2 + (half_len * integral) / TAU;
    let upper = p1.min(p2);
    let lower = (p1 + p2 - 1.0).max(0.0);
    raw.clamp(lower, upper)
}

fn plackett_trig_integrand(z_sq_sum: f64, two_z1_z2: f64, theta: f64) -> f64 {
    let (sin_t, cos_t) = theta.sin_cos();
    let denom = 2.0 * cos_t * cos_t;
    if denom <= 1e-30 {
        return 0.0;
    }
    let exponent = -(z_sq_sum - two_z1_z2 * sin_t) / denom;
    if exponent < -745.0 {
        0.0
    } else {
        exponent.exp()
    }
}

/// Evaluates the $d$-dimensional Gaussian copula CDF ($2 \le d \le 16$) via Genz (1992) Cholesky
/// separation-of-variables with Cranley-Patterson randomized Richtmyer Quasi-Monte Carlo.
#[verified_engine::verified]
pub(crate) fn genz_qmc_copula_cdf(
    cholesky: &DMatrix<f64>,
    u_vals: &[f64],
    rng: &mut OxidizeRng,
    n_pairs: usize,
) -> f64 {
    let d = u_vals.len();
    let mut b = [0.0_f64; 16];
    let mut shifts = [0.0_f64; 16];
    let mut alphas = [0.0_f64; 16];
    let mut min_u = 1.0_f64;
    let mut sum_u = 0.0_f64;

    for i in 0..d {
        let ui = u_vals[i];
        min_u = min_u.min(ui);
        sum_u += ui;
        b[i] = if ui >= 1.0 {
            f64::INFINITY
        } else {
            standard_normal_quantile(ui)
        };
        shifts[i] = sample_uniform_open(rng);
        alphas[i] = PRIMES_16[i].sqrt().fract();
    }

    let pairs = n_pairs.max(64);
    let mut acc = 0.0;
    let mut w = [0.5_f64; 16];
    let mut w_anti = [0.5_f64; 16];

    for k in 0..pairs {
        let kf = (k + 1) as f64;
        for j in 0..(d - 1) {
            let u_qmc = (kf * alphas[j] + shifts[j])
                .fract()
                .clamp(1e-12, 1.0 - 1e-12);
            w[j] = u_qmc;
            w_anti[j] = 1.0 - u_qmc;
        }
        acc +=
            0.5 * (genz_integrand(cholesky, &b, &w, d) + genz_integrand(cholesky, &b, &w_anti, d));
    }

    let fh_lower = (sum_u - (d as f64) + 1.0).max(0.0);
    let fh_upper = min_u;
    (acc / (pairs as f64)).clamp(fh_lower, fh_upper)
}

fn genz_integrand(cholesky: &DMatrix<f64>, b: &[f64; 16], w: &[f64; 16], d: usize) -> f64 {
    let mut y = [0.0_f64; 16];
    let e0 = if b[0].is_infinite() && b[0].is_sign_positive() {
        1.0
    } else {
        standard_normal_cdf(b[0] / cholesky[(0, 0)])
    };
    let mut prod = e0;
    if prod <= 0.0 {
        return 0.0;
    }

    for i in 1..d {
        let prev_p = (w[i - 1] * prod_step_e(cholesky, b, &y, i - 1)).clamp(1e-15, 1.0 - 1e-15);
        y[i - 1] = standard_normal_quantile(prev_p);
        let ei = prod_step_e(cholesky, b, &y, i);
        prod *= ei;
        if prod <= 0.0 {
            return 0.0;
        }
    }
    prod
}

fn prod_step_e(cholesky: &DMatrix<f64>, b: &[f64; 16], y: &[f64; 16], i: usize) -> f64 {
    if b[i].is_infinite() && b[i].is_sign_positive() {
        return 1.0;
    }
    let mut partial = 0.0;
    for j in 0..i {
        partial += cholesky[(i, j)] * y[j];
    }
    standard_normal_cdf((b[i] - partial) / cholesky[(i, i)])
}

// [cite:clinical_trials_statistics]
