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

/// Prints the 6-tier scientific verification pyramid CLI subcommands for `xtask --help`.
pub fn print_pyramid_help() {
    println!("  verify-tier1             - Run Tier 1 MMS & solver convergence verification");
    println!("  verify-tier2             - Run Tier 2 deterministic property sweeps");
    println!("  verify-tier3             - Run Tier 3 offline golden reference fixtures");
    println!("  verify-tier4             - Run Tier 4 stochastic moment verification");
    println!("  verify-tier5             - Run Tier 5 outward-rounded interval enclosures");
    println!("  verify-tier6             - Run Tier 6 Lean 4 formal proof certificates");
    println!("  verify-pyramid           - Run full 6-tier scientific verification pyramid\n");
}
