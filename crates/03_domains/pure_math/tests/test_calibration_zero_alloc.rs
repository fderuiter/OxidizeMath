//! Zero-allocation runtime verification tests for `pure_math::statistics::calibration`
//! hot kernels under `VerifiedAllocator`.

use math_commons::primitives::{PositiveFloat, UnitInterval};
use pure_math::pure_math::statistics::calibration::{
    DEFAULT_LOG_LOSS_EPSILON, PavaBlock, PlattScaler, ReliabilityBin, TemperatureScaler,
    binary_log_loss, brier_score, ensemble_crps_sorted, fit_pava_sorted_into, gaussian_crps,
    multiclass_log_loss, predict_pava_blocks, reliability_curve_uniform_into,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_calibration_hot_kernels_zero_heap_allocations() {
    let p0 = UnitInterval::new(0.1).unwrap();
    let p1 = UnitInterval::new(0.35).unwrap();
    let p2 = UnitInterval::new(0.65).unwrap();
    let p3 = UnitInterval::new(0.9).unwrap();
    let probs = [p0, p1, p2, p3];
    let mc_probs = [
        UnitInterval::new(0.7).unwrap(),
        UnitInterval::new(0.3).unwrap(),
        UnitInterval::new(0.2).unwrap(),
        UnitInterval::new(0.8).unwrap(),
    ];
    let mc_targets = [0_usize, 1_usize];
    let mc_logits = [2.2, -0.5, -1.1, 2.4];
    let outcomes = [false, false, true, true];
    let sorted_ens = [-1.5, -0.2, 0.4, 1.8];
    let logits = [-2.5, -0.8, 0.9, 2.7];
    let sorted_targets = [0.0, 1.0, 0.0, 1.0];
    let sigma = PositiveFloat::new(1.2).unwrap();

    let mut bins_buf = [ReliabilityBin::default(); 2];
    let mut pava_buf = [PavaBlock::default(); 4];
    let mut out_probs = [p0; 4];
    let mut mc_out = [p0; 2];

    let (total, _metrics) = VerifiedEngine::run_verified(|| {
        let bs = brier_score(&probs, &outcomes).expect("brier");
        let nll = binary_log_loss(&probs, &outcomes, DEFAULT_LOG_LOSS_EPSILON).expect("bin nll");
        let mc_nll =
            multiclass_log_loss(&mc_probs, 2, &mc_targets, DEFAULT_LOG_LOSS_EPSILON).expect("mc");
        let g_crps = gaussian_crps(0.0, sigma, 0.3).expect("gaussian crps");
        let e_crps = ensemble_crps_sorted(&sorted_ens, 0.1).expect("ensemble crps");
        let _ = reliability_curve_uniform_into(&probs, &outcomes, &mut bins_buf).expect("bins");
        let platt = PlattScaler::fit(&logits, &outcomes).expect("platt");
        platt
            .predict_into(&logits, &mut out_probs)
            .expect("platt pred");
        let temp_bin = TemperatureScaler::fit_binary(&logits, &outcomes).expect("temp bin");
        let _ = temp_bin.predict_binary(0.5).expect("temp bin pred");
        let temp_mc =
            TemperatureScaler::fit_multiclass(&mc_logits, 2, &mc_targets).expect("temp mc");
        temp_mc
            .predict_multiclass_into(&mc_logits[..2], &mut mc_out)
            .expect("temp mc pred");
        let m = fit_pava_sorted_into(&logits, &sorted_targets, None, &mut pava_buf).expect("pava");
        let p_mid = predict_pava_blocks(&pava_buf[..m], 0.0).expect("pava pred");
        bs + nll + mc_nll + g_crps + e_crps + p_mid.value()
    });

    assert!(total.is_finite() && total > 0.0);
}

// [cite:clinical_trials_statistics]
