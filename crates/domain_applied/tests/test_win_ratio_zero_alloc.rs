//! Integration test verifying zero-allocation hot paths under `VerifiedAllocator`
//! for `WinRatioAnalysis` and `WinRatioScratch`.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use approx::assert_relative_eq;
use domain_applied::applied::win_ratio::WinRatioScratch;
use domain_applied::applied::win_ratio::pair_comparison::{HigherIsBetter, WinRatioAnalysis};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_win_ratio_scratch_zero_allocation_hot_path() {
    let analysis = WinRatioAnalysis::new()
        .add_strategy(Box::new(HigherIsBetter))
        .add_strategy(Box::new(HigherIsBetter));
    let mut scratch = WinRatioScratch::new(24, 16, 16);

    let ((unmatched, matched, breakdown), _metrics) = VerifiedEngine::run_verified(|| {
        {
            let treat = scratch.treatment_two_tier_mut(20);
            treat[0] = 28.0;
            treat[1] = 4.2;
            treat[2] = 32.0;
            treat[3] = 4.4;
            for i in 2..18 {
                treat[2 * i] = 36.0;
                treat[2 * i + 1] = 3.8 + 0.1 * (i as f64);
            }
            for i in 18..20 {
                treat[2 * i] = 36.0;
                treat[2 * i + 1] = 36.0;
            }
        }
        {
            let ctrl = scratch.control_two_tier_mut(20);
            for i in 0..12 {
                let d = 6.58042 + 0.4 * (i as f64);
                ctrl[2 * i] = d;
                ctrl[2 * i + 1] = d;
            }
            for i in 12..20 {
                ctrl[2 * i] = 36.0;
                ctrl[2 * i + 1] = 36.0;
            }
        }

        let u = analysis.evaluate_flat_unmatched(
            scratch.treatment_two_tier(20),
            scratch.control_two_tier(20),
            2,
        );
        let m = analysis.evaluate_flat_matched(
            scratch.treatment_two_tier(20),
            scratch.control_two_tier(20),
            2,
        );
        let b = scratch.evaluate_staged_two_tier(20, 20, false);
        scratch.set_regime_cell(3, 4, b.stats.unwrap().win_ratio, 3.8623);
        (u, m, b)
    });

    assert_eq!(unmatched, (240, 144));
    assert!(matched.0 + matched.1 <= 20);
    assert_eq!(breakdown.ties, 16);
    let stats = breakdown.stats.expect("non-empty pairs");
    assert_relative_eq!(stats.win_ratio, 1.666_666_7, epsilon = 1e-5);
    assert_relative_eq!(stats.p_value, 4.200_39e-7, epsilon = 1e-10);
}
