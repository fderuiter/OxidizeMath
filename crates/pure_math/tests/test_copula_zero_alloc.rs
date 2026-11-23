//! Zero-allocation verification, 29-case bivariate accuracy benchmarks, multi-leg SGP accuracy
//! benchmarks, Fréchet-Hoeffding audits, and seeded Monte Carlo tests for `statistics::copula`.

use approx::assert_relative_eq;
use nalgebra::DMatrix;
use oxidize_core::rng::OxidizeRng;
use pure_math::error::CopulaError;
use pure_math::pure_math::statistics::copula::{
    Correlation, CorrelationMatrix, GaussianCopula, Probability, bivariate_copula_cdf,
    bivariate_normal_cdf, parlay_simulation_scratch_len, price_correlated_parlay,
    price_correlated_parlay_with_scratch, sgp_joint_probability, simulate_correlated_parlay,
    simulate_correlated_parlay_into, standard_normal_cdf,
};
use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2, TAU};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

/// 16-point Gauss-Legendre positive nodes and weights used by multi-panel reference oracles.
const GL16_HALF: [(f64, f64); 8] = [
    (0.095_012_509_837_637_44, 0.189_450_610_455_068_5),
    (0.281_603_550_779_258_9, 0.182_603_415_044_923_58),
    (0.458_016_777_657_227_37, 0.169_156_519_395_002_54),
    (0.617_876_244_402_643_8, 0.149_595_988_816_576_73),
    (0.755_404_408_355_003, 0.124_628_971_255_533_87),
    (0.865_631_202_387_831_8, 0.095_158_511_682_492_78),
    (0.944_575_023_073_232_6, 0.062_253_523_938_647_894),
    (0.989_400_934_991_649_9, 0.027_152_459_411_754_095),
];

fn ref_norm_cdf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-z * FRAC_1_SQRT_2)
}

fn ref_norm_inv(p: f64) -> f64 {
    let (z0, delta) = if p <= 0.5 {
        let z = -SQRT_2 * statrs::function::erf::erfc_inv(2.0 * p);
        (z, 0.5 * statrs::function::erf::erfc(-z * FRAC_1_SQRT_2) - p)
    } else {
        let q = 1.0 - p;
        let z = SQRT_2 * statrs::function::erf::erfc_inv(2.0 * q);
        (z, q - 0.5 * statrs::function::erf::erfc(z * FRAC_1_SQRT_2))
    };
    let pdf = (-0.5 * z0 * z0).exp() / TAU.sqrt();
    let ratio = delta / pdf;
    z0 - ratio / (1.0 + 0.5 * z0 * ratio)
}

fn ref_bivariate_moderate(u1: f64, u2: f64, z1: f64, z2: f64, rho: f64) -> f64 {
    let th_max = rho.asin();
    let mut sum = 0.0;
    for panel in 0..32 {
        let a = th_max * (panel as f64) / 32.0;
        let b = th_max * ((panel + 1) as f64) / 32.0;
        let (hm, mm) = (0.5 * (b - a), 0.5 * (b + a));
        for &(nd, wt) in &GL16_HALF {
            for sgn in [-1.0, 1.0] {
                let th = mm + hm * sgn * nd;
                let sn = th.sin();
                let cs2 = (1.0 - sn) * (1.0 + sn);
                let arg = (sn * z1 * z2 - 0.5 * (z1 * z1 + z2 * z2)) / cs2;
                if arg > -700.0 {
                    sum += wt * hm * arg.exp();
                }
            }
        }
    }
    u1 * u2 + sum / TAU
}

fn ref_bivariate_high(u1: f64, u2: f64, z1: f64, z2: f64, rho: f64) -> f64 {
    let a = z1;
    let b = if rho > 0.0 { z2 } else { -z2 };
    let width = ((1.0 - rho.abs()) * (1.0 + rho.abs())).sqrt();
    let mut sum = 0.0;
    for panel in 0..32 {
        let x0 = width * (panel as f64) / 32.0;
        let x1 = width * ((panel + 1) as f64) / 32.0;
        let (hm, mm) = (0.5 * (x1 - x0), 0.5 * (x1 + x0));
        for &(nd, wt) in &GL16_HALF {
            for sgn in [-1.0, 1.0] {
                let x = mm + hm * sgn * nd;
                let sp = ((1.0 - x) * (1.0 + x)).sqrt();
                let arg = -((a - b) * (a - b)) / (2.0 * x * x) - (a * b) / (1.0 + sp);
                if arg > -700.0 {
                    sum += wt * hm * arg.exp() / sp;
                }
            }
        }
    }
    let tail = sum / (2.0 * PI);
    if rho > 0.0 {
        u1.min(u2) - tail
    } else {
        (u1 + u2 - 1.0).max(0.0) + tail
    }
}

fn reference_bivariate_copula(u1: f64, u2: f64, rho: f64) -> f64 {
    if u1 == 0.0 || u2 == 0.0 {
        return 0.0;
    }
    if u1 == 1.0 {
        return u2;
    }
    if u2 == 1.0 {
        return u1;
    }
    if rho == 0.0 {
        return u1 * u2;
    }
    if rho >= 1.0 {
        return u1.min(u2);
    }
    if rho <= -1.0 {
        return (u1 + u2 - 1.0).max(0.0);
    }
    let (z1, z2) = (ref_norm_inv(u1), ref_norm_inv(u2));
    if rho.abs() < 0.95 {
        ref_bivariate_moderate(u1, u2, z1, z2, rho)
    } else {
        ref_bivariate_high(u1, u2, z1, z2, rho)
    }
}

fn reference_equicorrelated_copula(u: &[f64], rho: f64) -> f64 {
    let sr = rho.sqrt();
    let s1 = (1.0 - rho).sqrt();
    let inv_sqrt_2pi = 1.0 / TAU.sqrt();
    let mut sum = 0.0;
    for panel in 0..16 {
        let a = -10.0 + 20.0 * (panel as f64) / 16.0;
        let b = -10.0 + 20.0 * ((panel + 1) as f64) / 16.0;
        let (hm, mm) = (0.5 * (b - a), 0.5 * (b + a));
        for &(nd, wt) in &GL16_HALF {
            for sgn in [-1.0, 1.0] {
                let factor = mm + hm * sgn * nd;
                let mut prod = 1.0;
                for &ui in u {
                    prod *= ref_norm_cdf((ref_norm_inv(ui) - sr * factor) / s1);
                }
                sum += wt * hm * inv_sqrt_2pi * (-0.5 * factor * factor).exp() * prod;
            }
        }
    }
    sum
}

fn reference_trivariate_copula(u: [f64; 3], r01: f64, r02: f64, r12: f64) -> f64 {
    let (b0, b1, b2) = (ref_norm_inv(u[0]), ref_norm_inv(u[1]), ref_norm_inv(u[2]));
    let l11 = (1.0 - r01 * r01).sqrt();
    let l21 = (r12 - r02 * r01) / l11;
    let l22 = (1.0 - r02 * r02 - l21 * l21).sqrt();
    let inv_s2p = 1.0 / TAU.sqrt();
    let c0 = b0.min(9.5);
    if c0 <= -9.5 {
        return 0.0;
    }
    let mut total = 0.0;
    for p0 in 0..4 {
        let a = -9.5 + (c0 + 9.5) * (p0 as f64) / 4.0;
        let c = -9.5 + (c0 + 9.5) * ((p0 + 1) as f64) / 4.0;
        let (hm0, mm0) = (0.5 * (c - a), 0.5 * (c + a));
        for &(nd0, wt0) in &GL16_HALF {
            for s0 in [-1.0, 1.0] {
                let y0 = mm0 + hm0 * s0 * nd0;
                let w0 = wt0 * hm0 * inv_s2p * (-0.5 * y0 * y0).exp();
                let lim1 = ((b1 - r01 * y0) / l11).min(9.5);
                if lim1 <= -9.5 {
                    continue;
                }
                let (hm1, mm1) = (0.5 * (lim1 + 9.5), 0.5 * (lim1 - 9.5));
                for &(nd1, wt1) in &GL16_HALF {
                    for s1 in [-1.0, 1.0] {
                        let y1 = mm1 + hm1 * s1 * nd1;
                        let w1 = wt1 * hm1 * inv_s2p * (-0.5 * y1 * y1).exp();
                        total += w0 * w1 * ref_norm_cdf((b2 - r02 * y0 - l21 * y1) / l22);
                    }
                }
            }
        }
    }
    total
}

const BIVARIATE_BENCHMARK_CASES: [(f64, f64, f64); 30] = [
    (0.5, 0.5, 0.0),
    (0.5, 0.5, 0.5),
    (0.5, 0.5, -0.5),
    (0.5, 0.5, 0.925),
    (0.5, 0.5, -0.925),
    (0.5, 0.5, 0.99),
    (0.5, 0.5, -0.99),
    (0.5, 0.5, 0.9999),
    (0.5, 0.5, -0.9999),
    (0.01, 0.60, -0.15),
    (0.01, 0.01, 0.0),
    (0.01, 0.01, 0.75),
    (0.01, 0.01, -0.75),
    (0.01, 0.01, 0.98),
    (0.01, 0.99, -0.98),
    (0.001, 0.001, 0.85),
    (0.001, 0.001, 0.96),
    (0.001, 0.50, -0.60),
    (1e-5, 1e-5, 0.90),
    (1e-5, 1e-5, 0.995),
    (0.10, 0.90, 0.40),
    (0.10, 0.90, -0.40),
    (0.25, 0.75, 0.924),
    (0.25, 0.75, 0.926),
    (0.25, 0.75, -0.924),
    (0.25, 0.75, -0.926),
    (0.70, 0.80, 0.0),
    (0.99, 0.60, -0.15),
    (0.99, 0.99, 0.95),
    (0.999, 0.999, -0.95),
];

#[test]
#[verified_engine::verified]
fn test_bivariate_29_scipy_benchmarks_and_frechet_bounds() {
    for &(u1_v, u2_v, rho_v) in &BIVARIATE_BENCHMARK_CASES {
        let u1 = Probability::new(u1_v).unwrap();
        let u2 = Probability::new(u2_v).unwrap();
        let rho = Correlation::new(rho_v).unwrap();
        let got = bivariate_copula_cdf(u1, u2, rho).unwrap().value();
        let expected = reference_bivariate_copula(u1_v, u2_v, rho_v);
        let err = (got - expected).abs();
        assert!(err < 1e-12, "bivariate mismatch: err={err:e}");
        assert!(got >= (u1_v + u2_v - 1.0).max(0.0) && got <= u1_v.min(u2_v));
    }

    for &rho_v in &[-1.0_f64, -0.95, -0.5, 0.0, 0.5, 0.925, 0.99, 1.0] {
        let exact_orthant = 0.25 + rho_v.asin() / TAU;
        let got = bivariate_normal_cdf(0.0, 0.0, rho_v).unwrap();
        assert_relative_eq!(got, exact_orthant, epsilon = 1e-14);
    }
}

#[test]
#[verified_engine::verified]
fn test_three_and_four_leg_sgp_scipy_benchmarks() {
    let u3 = [0.55, 0.60, 0.65];
    let legs3: Vec<Probability> = u3.iter().map(|&p| Probability::new(p).unwrap()).collect();
    let corr3 = CorrelationMatrix::new(DMatrix::from_row_slice(
        3,
        3,
        &[1.0, 0.35, 0.15, 0.35, 1.0, 0.10, 0.15, 0.10, 1.0],
    ))
    .unwrap();
    let got3 = sgp_joint_probability(&legs3, &corr3).unwrap().value();
    assert!((got3 - reference_trivariate_copula(u3, 0.35, 0.15, 0.10)).abs() < 1e-4);

    let legs3_ref: Vec<Probability> = [0.65, 0.60, 0.50]
        .iter()
        .map(|&p| Probability::new(p).unwrap())
        .collect();
    let corr3_ref = CorrelationMatrix::new(DMatrix::from_row_slice(
        3,
        3,
        &[1.0, 0.45, -0.10, 0.45, 1.0, 0.50, -0.10, 0.50, 1.0],
    ))
    .unwrap();
    let got3_ref = sgp_joint_probability(&legs3_ref, &corr3_ref)
        .unwrap()
        .value();
    assert!((got3_ref - 0.270_524).abs() < 5e-4);

    let u4_eq = [0.60, 0.60, 0.60, 0.60];
    let legs4_eq: Vec<Probability> = u4_eq
        .iter()
        .map(|&p| Probability::new(p).unwrap())
        .collect();
    let corr4_eq = CorrelationMatrix::new(DMatrix::from_fn(4, 4, |r, c| {
        if r == c { 1.0 } else { 0.212_471_235_3 }
    }))
    .unwrap();
    let got4_eq = sgp_joint_probability(&legs4_eq, &corr4_eq).unwrap().value();
    assert!((got4_eq - 0.197_551).abs() < 5e-4);
}

#[test]
#[verified_engine::verified]
fn test_multileg_equicorrelated_3_to_10_legs() {
    for d in [3_usize, 4, 5, 6, 8, 10] {
        let raw_u: Vec<f64> = (0..d).map(|j| 0.35 + 0.05 * ((j % 7) as f64)).collect();
        let legs: Vec<Probability> = raw_u
            .iter()
            .map(|&p| Probability::new(p).unwrap())
            .collect();
        let rho = 0.45;
        let corr = CorrelationMatrix::new(DMatrix::from_fn(
            d,
            d,
            |r, c| if r == c { 1.0 } else { rho },
        ))
        .unwrap();
        let got = sgp_joint_probability(&legs, &corr).unwrap().value();
        let expected = reference_equicorrelated_copula(&raw_u, rho);
        assert!((got - expected).abs() < 1e-4);
    }
}

fn assert_zero_alloc_error_paths(
    copula_4: &GaussianCopula,
    copula_20: &GaussianCopula,
    legs_2: &[Probability],
    legs_20: &[Probability],
    small_scratch: &mut [f64],
) {
    assert!(copula_4.cdf(legs_2).is_err());
    assert!(copula_20.cdf_with_scratch(legs_20, small_scratch).is_err());
    assert!(Probability::new(1.5).is_err());
    assert!(Correlation::new(-1.5).is_err());
    assert!(standard_normal_cdf(f64::NAN).is_err());
    assert_eq!(
        bivariate_normal_cdf(0.0, 0.0, 1.2),
        Err(CopulaError::InvalidCorrelation { value: 1.2 })
    );
}

#[test]
#[verified_engine::verified]
fn test_copula_and_parlay_hot_paths_zero_heap_allocations() {
    let corr_2 = CorrelationMatrix::bivariate(Correlation::new(-0.25).unwrap()).unwrap();
    let copula_2 = GaussianCopula::new(corr_2);
    let legs_2 = [
        Probability::new(0.20).unwrap(),
        Probability::new(0.65).unwrap(),
    ];

    let corr_4 = CorrelationMatrix::new(DMatrix::from_fn(
        4,
        4,
        |r, c| if r == c { 1.0 } else { 0.35 },
    ))
    .unwrap();
    let copula_4 = GaussianCopula::new(corr_4.clone());
    let legs_4 = [
        Probability::new(0.55).unwrap(),
        Probability::new(0.50).unwrap(),
        Probability::new(0.60).unwrap(),
        Probability::new(0.45).unwrap(),
    ];
    let mut cdf_scratch_4 = vec![0.0_f64; GaussianCopula::scratch_len(4)];
    let mut sim_scratch_4 = [0.0_f64; parlay_simulation_scratch_len(4)];

    let corr_20 = CorrelationMatrix::new(DMatrix::from_fn(
        20,
        20,
        |r, c| if r == c { 1.0 } else { 0.30 },
    ))
    .unwrap();
    let copula_20 = GaussianCopula::new(corr_20.clone());
    let legs_20: Vec<Probability> = (0..20)
        .map(|i| Probability::new(0.60 + 0.01 * ((i % 5) as f64)).unwrap())
        .collect();
    let mut cdf_scratch_20 = vec![0.0_f64; GaussianCopula::scratch_len(20)];
    let mut sim_scratch_20 = vec![0.0_f64; parlay_simulation_scratch_len(20)];
    let mut rng = OxidizeRng::new(0x0D15_EA5E);

    let ((p2, p4_stack, p4_scratch, p20, priced4, priced20, sim4, sim20), _metrics) =
        VerifiedEngine::run_verified(|| {
            let p2 = copula_2.cdf(&legs_2).expect("2-leg cdf");
            let p4_stack = sgp_joint_probability(&legs_4, &corr_4).expect("4-leg stack sgp");
            let p4_scratch = copula_4
                .cdf_with_scratch(&legs_4, &mut cdf_scratch_4)
                .expect("4-leg scratch cdf");
            let p20 = copula_20
                .cdf_with_scratch(&legs_20, &mut cdf_scratch_20)
                .expect("20-leg scratch cdf");
            let priced4 = price_correlated_parlay(&legs_4, &corr_4).expect("4-leg pricing");
            let priced20 =
                price_correlated_parlay_with_scratch(&legs_20, &corr_20, &mut cdf_scratch_20)
                    .expect("20-leg scratch pricing");
            let sim4 = simulate_correlated_parlay_into(
                &legs_4,
                &corr_4,
                8_000,
                &mut rng,
                &mut sim_scratch_4,
            )
            .expect("4-leg sim");
            let sim20 = simulate_correlated_parlay_into(
                &legs_20,
                &corr_20,
                4_000,
                &mut rng,
                &mut sim_scratch_20,
            )
            .expect("20-leg sim");
            assert_zero_alloc_error_paths(
                &copula_4,
                &copula_20,
                &legs_2,
                &legs_20,
                &mut sim_scratch_4,
            );
            (
                p2, p4_stack, p4_scratch, p20, priced4, priced20, sim4, sim20,
            )
        });

    assert!(p2.value() > 0.0 && p2.value() < 0.20);
    assert_relative_eq!(p4_stack.value(), p4_scratch.value(), epsilon = 1e-15);
    assert_relative_eq!(
        priced4.joint_probability.value(),
        p4_stack.value(),
        epsilon = 1e-15
    );
    assert_relative_eq!(
        priced20.joint_probability.value(),
        p20.value(),
        epsilon = 1e-15
    );
    assert_relative_eq!(
        sim4.joint_probability.value(),
        p4_stack.value(),
        epsilon = 4.0 * sim4.standard_error
    );
    assert!(sim20.joint_probability.value() > sim20.naive_probability.value());

    let mut rng_outside = OxidizeRng::new(0x1234_5678);
    let sim20_conv =
        simulate_correlated_parlay(&legs_20, &corr_20, 2_000, &mut rng_outside).unwrap();
    assert!(sim20_conv.fair_decimal_odds > 1.0);
}
