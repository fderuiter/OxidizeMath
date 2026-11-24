use super::*;
use crate::pure_math::statistics::monte_carlo::MonteCarloError;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_halton_validation_and_radical_inverse() {
    assert_eq!(
        HaltonSequence::new(0).unwrap_err(),
        MonteCarloError::InvalidDimension {
            dimension: 0,
            max_dimension: MAX_HALTON_DIM,
        }
    );
    assert_eq!(
        HaltonSequence::new(MAX_HALTON_DIM + 1).unwrap_err(),
        MonteCarloError::InvalidDimension {
            dimension: MAX_HALTON_DIM + 1,
            max_dimension: MAX_HALTON_DIM,
        }
    );
    assert_eq!(
        HaltonSequence::with_bases(&[2, 4]).unwrap_err(),
        MonteCarloError::InvalidBase { base: 4 }
    );

    assert!((radical_inverse(1, 2) - 0.5).abs() < 1e-15);
    assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-15);
    assert!((radical_inverse(3, 2) - 0.75).abs() < 1e-15);
    assert!((radical_inverse(1, 3) - (1.0 / 3.0)).abs() < 1e-15);
}

#[test]
#[verified_engine::verified]
fn test_sobol_incremental_matches_direct_point_at_across_32d() {
    let mut seq = SobolSequence::new(MAX_SOBOL_DIM).expect("valid 32D Sobol");
    let mut stepped = [0.0; MAX_SOBOL_DIM];
    let mut direct = [0.0; MAX_SOBOL_DIM];

    for idx in 0..256_u32 {
        seq.next_point(&mut stepped).expect("stepped point");
        seq.point_at(idx, &mut direct).expect("direct point");
        for d in 0..MAX_SOBOL_DIM {
            assert!(
                stepped[d] > 0.0 && stepped[d] < 1.0,
                "coordinate out of (0, 1): {}",
                stepped[d]
            );
            assert!(
                (stepped[d] - direct[d]).abs() < 1e-15,
                "mismatch at idx={idx}, dim={d}: {} vs {}",
                stepped[d],
                direct[d]
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_qmc_zero_allocation_hot_loop_under_verified_allocator() {
    let mut halton = HaltonSequence::new(8).expect("8D Halton");
    let mut sobol = SobolSequence::new(8).expect("8D Sobol");
    let mut rng = OxidizeRng::new(12345);
    let mut buf = [0.0_f64; 8];

    let (sum, _) = VerifiedEngine::run_verified(|| {
        halton.apply_cranley_patterson_shift(&mut rng);
        sobol.apply_cranley_patterson_shift(&mut rng);
        let mut acc = 0.0;
        for _ in 0..512 {
            halton.next_point(&mut buf).expect("halton step");
            acc += buf[0] + buf[7];
            sobol.next_point(&mut buf).expect("sobol step");
            acc += buf[0] + buf[7];
        }
        acc
    });
    assert!(sum.is_finite() && sum > 0.0);
}

#[test]
#[verified_engine::verified]
fn test_halton_and_sobol_beat_iid_rmse_on_smooth_5d_integral() {
    // Exact integral of prod_{j=1}^5 (1 + (u_j - 0.5)) over [0, 1]^5 is 1.0;
    // test f(u) = sum_{j=1}^5 exp(u_j) with exact integral 5 * (e - 1).
    let exact = 5.0 * (std::f64::consts::E - 1.0);
    let budget = 256_usize;
    let trials = 32_usize;
    let mut halton = HaltonSequence::new(5).expect("5D Halton");
    let mut sobol = SobolSequence::new(5).expect("5D Sobol");
    let mut pt = [0.0_f64; 5];

    let mut iid_sq = 0.0;
    let mut halton_sq = 0.0;
    let mut sobol_sq = 0.0;

    for t in 0..trials {
        let mut rng = OxidizeRng::new(9000 + t as u64);
        let mut iid_sum = 0.0;
        for _ in 0..budget {
            let mut val = 0.0;
            for _ in 0..5 {
                let u = ((rng.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64);
                val += u.exp();
            }
            iid_sum += val;
        }
        iid_sq += (iid_sum / (budget as f64) - exact).powi(2);

        halton.apply_cranley_patterson_shift(&mut rng);
        let mut h_sum = 0.0;
        for _ in 0..budget {
            halton.next_point(&mut pt).unwrap();
            h_sum += pt.iter().map(|u| u.exp()).sum::<f64>();
        }
        halton_sq += (h_sum / (budget as f64) - exact).powi(2);

        sobol.apply_cranley_patterson_shift(&mut rng);
        let mut s_sum = 0.0;
        for _ in 0..budget {
            sobol.next_point(&mut pt).unwrap();
            s_sum += pt.iter().map(|u| u.exp()).sum::<f64>();
        }
        sobol_sq += (s_sum / (budget as f64) - exact).powi(2);
    }

    let iid_rmse = (iid_sq / trials as f64).sqrt();
    let halton_rmse = (halton_sq / trials as f64).sqrt();
    let sobol_rmse = (sobol_sq / trials as f64).sqrt();
    assert!(
        halton_rmse < iid_rmse * 0.4,
        "Halton RMSE {halton_rmse} should beat IID RMSE {iid_rmse}"
    );
    assert!(
        sobol_rmse < iid_rmse * 0.3,
        "Sobol RMSE {sobol_rmse} should beat IID RMSE {iid_rmse}"
    );
}

// [cite:clinical_trials_statistics]
