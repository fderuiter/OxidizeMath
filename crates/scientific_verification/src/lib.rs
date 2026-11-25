#![doc = include_str!("../README.md")]

/// Tier 3 offline golden reference fixture verification.
pub mod golden;
/// Tier 5 IEEE-754 outward-rounded interval enclosure verification.
pub mod intervals;
/// Tier 6 Lean 4 mechanized proof source & SHA-256 digest verification.
pub mod lean_proofs;
/// Shared Method of Manufactured Solutions (MMS) norms and log-log OLS rate estimators.
pub mod mms;
/// Tier 2 ChaCha-seeded deterministic property & conservation sweeps (host-only).
pub mod proptest_sweeps;
/// Unified 6-tier pyramid runner and certificate metrics aggregator.
pub mod pyramid;
/// Tier 1 convergence certification across all 55 workspace numerical solvers.
pub mod solvers;
/// Tier 4 seeded stochastic moment and confidence-bound verification.
pub mod stochastic;

pub use pyramid::{
    PyramidSummary, TierReport, run_full_pyramid, run_tier1, run_tier2, run_tier3, run_tier4,
    run_tier5, run_tier6,
};

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(test)]
mod tests {
    use super::mms;

    #[test]
    fn ols_convergence_order_recovers_quadratic_rate() {
        let rate = mms::ols_convergence_order(&[0.2, 0.1, 0.05], &[0.04, 0.01, 0.0025]);
        assert!((rate - 2.0).abs() < 1e-12);
    }
}
