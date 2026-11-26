use crate::golden::verify_golden_fixtures;
use crate::intervals::compute_certified_interval_enclosures;
use crate::lean_proofs::verify_and_digest_lean_proofs;
use crate::proptest_sweeps::verify_property_sweeps;
use crate::solvers::verify_all_solvers;
use crate::stochastic::verify_stochastic_moments;
use oxidize_core::traceability::CERTIFIED_SOLVER_COUNT;
use std::collections::BTreeMap;

/// Outcome of executing a single tier of the 6-tier scientific verification pyramid.
#[derive(Clone, Debug)]
pub struct TierReport {
    /// Canonical tier name (`Tier1_Convergence` .. `Tier6_Formal`).
    pub tier_name: &'static str,
    /// Count of certified checks/solvers/proofs in this tier.
    pub passed_checks: usize,
}

/// Complete live certification output across all 6 tiers of the scientific verification pyramid.
#[derive(Clone, Debug)]
pub struct PyramidSummary {
    /// Per-tier certified check counts keyed by canonical tier name.
    pub tier_counts: BTreeMap<String, usize>,
    /// Number of certified numerical solvers (`55`).
    pub certified_solvers: usize,
    /// Maximum IEEE-754 outward-rounded interval enclosure width across certified kernels.
    pub max_interval_width: f64,
    /// Representative `[lo, hi]` interval enclosures for certified kernels.
    pub interval_bounds: BTreeMap<String, [f64; 2]>,
    /// Deterministic SHA-256 digests of all committed `sorry`-free Lean 4 proof modules.
    pub lean4_proof_digests: BTreeMap<String, String>,
}

/// Executes Tier 1: Manufactured-Solution & Grid-Refinement Convergence across all 55 solvers.
pub fn run_tier1() -> Result<TierReport, String> {
    let certs = verify_all_solvers()?;
    if certs.len() != CERTIFIED_SOLVER_COUNT {
        return Err(format!(
            "Expected {CERTIFIED_SOLVER_COUNT} solver certificates, got {}",
            certs.len()
        ));
    }
    Ok(TierReport {
        tier_name: "Tier1_Convergence",
        passed_checks: certs.len(),
    })
}

/// Executes Tier 2: ChaCha-seeded deterministic property & conservation sweeps.
pub fn run_tier2() -> Result<TierReport, String> {
    let passed = verify_property_sweeps()?;
    Ok(TierReport {
        tier_name: "Tier2_Property",
        passed_checks: passed,
    })
}

/// Executes Tier 3: Offline high-precision golden fixture verification (`< 1e-12`).
pub fn run_tier3() -> Result<TierReport, String> {
    let passed = verify_golden_fixtures()?;
    Ok(TierReport {
        tier_name: "Tier3_Golden",
        passed_checks: passed,
    })
}

/// Executes Tier 4: Deterministic seeded stochastic moment & confidence-bound checks.
pub fn run_tier4() -> Result<TierReport, String> {
    let passed = verify_stochastic_moments()?;
    Ok(TierReport {
        tier_name: "Tier4_Stochastic",
        passed_checks: passed,
    })
}

/// Executes Tier 5: IEEE-754 outward-rounded interval enclosure certification.
pub fn run_tier5() -> Result<TierReport, String> {
    let cert = compute_certified_interval_enclosures()?;
    Ok(TierReport {
        tier_name: "Tier5_Interval",
        passed_checks: cert.bounds.len(),
    })
}

/// Executes Tier 6: Lean 4 mechanized proof certificate & SHA-256 digest verification.
pub fn run_tier6() -> Result<TierReport, String> {
    let digests = verify_and_digest_lean_proofs()?;
    Ok(TierReport {
        tier_name: "Tier6_Formal",
        passed_checks: digests.len(),
    })
}

/// Executes all 6 tiers of the scientific verification pyramid and returns certificate telemetry.
pub fn run_full_pyramid() -> Result<PyramidSummary, String> {
    let t1 = run_tier1()?;
    let t2 = run_tier2()?;
    let t3 = run_tier3()?;
    let t4 = run_tier4()?;
    let intervals = compute_certified_interval_enclosures()?;
    let lean_digests = verify_and_digest_lean_proofs()?;

    let mut tier_counts = BTreeMap::new();
    tier_counts.insert(t1.tier_name.to_string(), t1.passed_checks);
    tier_counts.insert(t2.tier_name.to_string(), t2.passed_checks);
    tier_counts.insert(t3.tier_name.to_string(), t3.passed_checks);
    tier_counts.insert(t4.tier_name.to_string(), t4.passed_checks);
    tier_counts.insert("Tier5_Interval".to_string(), intervals.bounds.len());
    tier_counts.insert("Tier6_Formal".to_string(), lean_digests.len());

    Ok(PyramidSummary {
        tier_counts,
        certified_solvers: t1.passed_checks,
        max_interval_width: intervals.max_width,
        interval_bounds: intervals.bounds,
        lean4_proof_digests: lean_digests,
    })
}

#[cfg(test)]
mod tests {
    use super::run_full_pyramid;

    #[test]
    fn full_six_tier_pyramid_certifies_all_55_solvers_intervals_and_lean_proofs() {
        let summary = run_full_pyramid().expect("full 6-tier pyramid must pass");
        assert_eq!(summary.certified_solvers, 55);
        assert_eq!(summary.tier_counts.len(), 6);
        assert!(summary.tier_counts.values().all(|&c| c > 0));
        assert!(summary.max_interval_width > 0.0 && summary.max_interval_width < 1e-11);
        assert_eq!(summary.lean4_proof_digests.len(), 6);
    }
}
