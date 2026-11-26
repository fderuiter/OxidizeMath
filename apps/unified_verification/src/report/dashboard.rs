use super::GateEvaluation;

pub(crate) fn print_gate_failures(eval: &GateEvaluation) {
    for err in &eval.gate_errors {
        println!("\n[!] {err}");
    }
    if !eval.unauthorized_debt.is_empty() {
        println!("\n--- Unauthorized Integrity Debt (FAIL) ---");
        for (idx, item) in eval.unauthorized_debt.iter().enumerate() {
            println!("[{}] {}", idx + 1, item);
        }
    }
}

pub(crate) fn print_dashboard(eval: &GateEvaluation) {
    let r = &eval.report;
    println!("\n--- Integrity Report ---");
    println!(
        "Native Execution Coverage: {:.2}%",
        r.native_execution_coverage_pct
    );
    println!("WASM Path Coverage: {:.2}%", r.wasm_path_coverage_pct);
    println!("\n--- High-Integrity Dashboard ---");
    println!(
        "Total Assertion Density: {:.2} asserts/fn",
        r.total_assertion_density
    );
    println!(
        "Verified Modules Density: {:.2} asserts/fn",
        r.verified_modules_density
    );
    println!(
        "Unverified Modules Density: {:.2} asserts/fn",
        r.unverified_modules_density
    );
    println!(
        "Total Production Density: {:.4} asserts/fn",
        r.total_production_assertion_density
    );
    println!(
        "Verified Production Density: {:.4} asserts/fn",
        r.verified_production_assertion_density
    );
    println!(
        "Test Assertion Density: {:.2} asserts/fn",
        r.test_assertion_density
    );
    println!(
        "Scientific Pyramid: {} certified solvers, {:.3e} max interval width, {} Lean 4 proofs",
        r.certified_solvers,
        r.max_interval_width,
        r.lean4_proof_digests.len()
    );
    println!(
        "Authorized whitelist exemptions: {} opt-outs, {} non-domain crates",
        eval.authorized_opt_outs, eval.authorized_unverified_crates
    );
    print_gate_failures(eval);
}
