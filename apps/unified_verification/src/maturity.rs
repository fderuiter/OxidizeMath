//! Workspace maturity qualification runner backed by [`oxidize_core::traceability`].

use oxidize_core::traceability::{
    AstCache, MaturityGateReport, MaturityRegistry, MaturityTier,
    evaluate_maturity_with_registry_and_cache, evaluate_workspace_maturity_with_cache,
};
use std::path::Path;

fn count_tier<'a>(tiers: impl Iterator<Item = &'a MaturityTier>, target: MaturityTier) -> usize {
    tiers.filter(|&&t| t == target).count()
}

/// Prints a formatted summary and any qualification violations for `report`, returning `report.passed()`.
pub fn print_maturity_gate_report(report: &MaturityGateReport) -> bool {
    println!("=== Maturity Tier Qualification Gate ===");
    let c_cert = count_tier(report.registry.crates.values(), MaturityTier::Certified);
    let c_val = count_tier(report.registry.crates.values(), MaturityTier::Validated);
    let c_exp = count_tier(report.registry.crates.values(), MaturityTier::Experimental);
    let m_cert = count_tier(report.registry.modules.values(), MaturityTier::Certified);
    let m_val = count_tier(report.registry.modules.values(), MaturityTier::Validated);
    let m_exp = count_tier(report.registry.modules.values(), MaturityTier::Experimental);
    println!(
        "Crates:  {} Certified | {} Validated | {} Experimental",
        c_cert, c_val, c_exp
    );
    println!(
        "Modules: {} Certified | {} Validated | {} Experimental",
        m_cert, m_val, m_exp
    );

    if report.passed() {
        println!("All maturity qualification gates passed.");
        true
    } else {
        println!("\n[!] Maturity Qualification Failures:");
        for violation in &report.violations {
            println!("  - {violation}");
        }
        false
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn append_lean_kernel_violations(workspace_root: &Path, report: &mut MaturityGateReport) {
    if !workspace_root
        .join("proofs")
        .join("lakefile.lean")
        .is_file()
    {
        return;
    }
    if let Err(err) = scientific_verification::lean_proofs::verify_lean_proofs_in_workspace(
        workspace_root,
        "lake",
        6,
    ) {
        report
            .violations
            .push(format!("Tier6_Formal kernel verification failed: {err}"));
    }
}

#[cfg(target_arch = "wasm32")]
fn append_lean_kernel_violations(_workspace_root: &Path, _report: &mut MaturityGateReport) {}

/// Evaluates workspace maturity qualification in `workspace_root` reusing `cache`.
pub fn check_maturity_in_dir_with_cache(workspace_root: &Path, cache: &mut AstCache) -> bool {
    let mut report = evaluate_workspace_maturity_with_cache(workspace_root, cache);
    append_lean_kernel_violations(workspace_root, &mut report);
    print_maturity_gate_report(&report)
}

/// Evaluates workspace maturity qualification in the active workspace root reusing `cache`.
pub fn check_maturity_with_cache(cache: &mut AstCache) -> bool {
    let root = if Path::new("traceability.toml").is_file() {
        std::path::PathBuf::from(".")
    } else {
        crate::manifest_cache::workspace_root_dir()
    };
    check_maturity_in_dir_with_cache(&root, cache)
}

/// Standalone CLI entrypoint for `unified_verification check-maturity`.
pub fn check_maturity(args: &[String]) -> bool {
    let _root_guard = crate::manifest_cache::WorkspaceRootGuard::enter();
    let mut cache = AstCache::new();
    let mut custom_registry: Option<MaturityRegistry> = None;
    let mut idx = 0;
    while idx < args.len() {
        if args[idx] == "--registry" && idx + 1 < args.len() {
            let path = &args[idx + 1];
            let Ok(content) = std::fs::read_to_string(path) else {
                eprintln!("Failed to read custom registry file: {path}");
                return false;
            };
            custom_registry = Some(MaturityRegistry::parse_toml_str(&content));
            idx += 2;
            continue;
        }
        idx += 1;
    }
    if let Some(reg) = custom_registry {
        let mut report = evaluate_maturity_with_registry_and_cache(Path::new("."), reg, &mut cache);
        append_lean_kernel_violations(Path::new("."), &mut report);
        print_maturity_gate_report(&report)
    } else {
        check_maturity_in_dir_with_cache(Path::new("."), &mut cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxidize_core::traceability::evaluate_maturity_with_registry_and_cache;

    #[test]
    fn check_maturity_passes_baseline_and_fails_cera_or_favoritism_certified_claims() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let root = crate::manifest_cache::workspace_root_dir();
        let mut cache = AstCache::new();
        let base_reg = MaturityRegistry::load_or_embedded(Some(&root));

        let ok_report =
            evaluate_maturity_with_registry_and_cache(&root, base_reg.clone(), &mut cache);
        assert!(
            ok_report.passed(),
            "Expected baseline workspace maturity gates to pass, got: {:?}",
            ok_report.violations
        );

        for target in ["cera_framework", "climate", "favoritism", "favorite_child"] {
            let mut mutated = base_reg.clone();
            mutated
                .modules
                .insert(target.to_string(), MaturityTier::Certified);
            let bad = evaluate_maturity_with_registry_and_cache(&root, mutated, &mut cache);
            assert!(
                !bad.passed(),
                "`{target}` claiming Certified without required proofs must fail check-maturity"
            );
        }

        let mut mutated_crate = base_reg;
        mutated_crate
            .crates
            .insert("domain_climate".to_string(), MaturityTier::Certified);
        let bad_crate = evaluate_maturity_with_registry_and_cache(&root, mutated_crate, &mut cache);
        assert!(
            !bad_crate.passed(),
            "`domain_climate` claiming Certified without required proofs must fail check-maturity"
        );
    }
}
