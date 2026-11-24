use crate::gates::run_verification;

/// Returns the canonical `unified_verification` subcommand and display banner for a pyramid CLI verb.
pub fn resolve_pyramid_subcommand(cmd: &str) -> Option<(&'static str, &'static str)> {
    match cmd {
        "verify-tier1" | "verify-mms" => {
            Some(("verify-tier1", "Tier 1: MMS & Grid-Refinement Convergence"))
        }
        "verify-tier2" | "verify-proptest" => Some((
            "verify-tier2",
            "Tier 2: Deterministic Property & Invariant Sweeps",
        )),
        "verify-tier3" | "verify-fixtures" => {
            Some(("verify-tier3", "Tier 3: High-Precision Golden Fixtures"))
        }
        "verify-tier4" | "verify-stochastic" => {
            Some(("verify-tier4", "Tier 4: Stochastic Moment Signatures"))
        }
        "verify-tier5" | "verify-intervals" => Some((
            "verify-tier5",
            "Tier 5: Outward-Rounded Interval Enclosures",
        )),
        "verify-tier6" | "verify-proofs" => {
            Some(("verify-tier6", "Tier 6: Lean 4 Formal Proof Certificates"))
        }
        "verify-pyramid" => Some(("verify-pyramid", "6-Tier Scientific Verification Pyramid")),
        _ => None,
    }
}

/// Dispatches a pyramid verification gate through `unified_verification`.
pub fn run_pyramid_gate(canonical_cmd: &str, banner: &str) {
    println!("=== {banner} ===");
    run_verification(canonical_cmd);
}
