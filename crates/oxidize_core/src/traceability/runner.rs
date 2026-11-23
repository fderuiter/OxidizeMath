//! Native workspace discovery, dashboard printing, and in-process traceability execution helpers.

use super::{AstCache, TraceabilityEngine, TraceabilityReport};
use crate::vfs::VirtualFileSystem;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

struct LocalFsVfs;

impl VirtualFileSystem for LocalFsVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move { std::fs::read_to_string(&path) })
    }

    fn write_to_file(
        &self,
        path: &str,
        content: &[u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>> {
        let path = path.to_string();
        let content = content.to_vec();
        Box::pin(async move { std::fs::write(&path, content) })
    }

    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move {
            let entries = std::fs::read_dir(&path)?;
            let mut files = Vec::new();
            for entry in entries.flatten() {
                if let Some(s) = entry.file_name().to_str() {
                    files.push(s.to_string());
                }
            }
            Ok(files)
        })
    }
}

const IGNORED_TRACEABILITY_MEMBERS: &[&str] = &[
    "crates/unified_verification",
    "crates/diagnostics",
    "crates/federated_registry",
    "crates/oxidize_core",
    "crates/verified_engine",
    "crates/verified_engine_macros",
    "apps/xtask",
    "crates/math_commons",
    "egui_plot",
];

/// Discovers active workspace module roots subject to traceability verification.
pub fn discover_workspace_code_dirs() -> Vec<String> {
    let (is_root, members) = crate::workspace::get_workspace_members();
    let mut code_dirs = Vec::new();
    let root_prefix = if is_root { "" } else { "../../" };

    for member_str in members {
        if IGNORED_TRACEABILITY_MEMBERS.contains(&member_str.as_str()) {
            continue;
        }
        let member_path = PathBuf::from(root_prefix).join(&member_str);
        let src_path = if member_str == "math_explorer_gui" {
            member_path.join("src").join("tabs")
        } else {
            member_path.join("src")
        };

        if src_path.is_dir() {
            let mut target_dir = crate::path_utils::normalize_path(&src_path);
            if !is_root && target_dir.starts_with("../../") {
                target_dir = target_dir.trim_start_matches("../../").to_string();
            }
            code_dirs.push(target_dir);
        }
    }
    code_dirs
}

fn check_semantic_and_density_errors(report: &TraceabilityReport) -> bool {
    let mut failed = false;
    let mut unverified_semantics: Vec<&String> = report
        .semantic_integrity_status
        .iter()
        .filter_map(|(k, v)| if v != "Verified" { Some(k) } else { None })
        .collect();

    if !unverified_semantics.is_empty() {
        println!("\n[!] Unverified Semantic Integrity Statuses Detected:");
        unverified_semantics.sort();
        for module in unverified_semantics {
            println!("  - {}: Unverified", module);
        }
        failed = true;
    }

    let verified_density = if report.verified_funcs > 0 {
        report.verified_asserts as f64 / report.verified_funcs as f64
    } else {
        0.0
    };
    if report.verified_funcs > 0 && verified_density < 0.4 {
        println!(
            "\n[!] Assertion Density Failure: Verified modules have a density of {:.2} asserts/fn, which is below the minimum required 0.4 asserts/fn.",
            verified_density
        );
        failed = true;
    }
    failed
}

fn report_list_section(
    header: &str,
    subtitle: Option<&str>,
    prefix: &str,
    items: &[String],
) -> bool {
    if items.is_empty() {
        return false;
    }
    println!("\n{header}");
    if let Some(sub) = subtitle {
        println!("{sub}");
    }
    for item in items {
        println!("{prefix}{item}");
    }
    true
}

/// Evaluates all traceability invariants on `report`, printing diagnostics and returning `true` if any check failed.
pub fn check_and_print_traceability_errors(report: &TraceabilityReport) -> bool {
    let mut failed = report_list_section(
        "=== Unlinked Modules ===",
        Some(
            "The following modules contain a verification macro but are missing from the registry:",
        ),
        "   [!] UNLINKED module: ",
        &report.unlinked_code,
    );

    if !report.invalid_links.is_empty() {
        println!("\n=== Validation Failures ===");
        println!("Invalid citations were found (see details above).");
        for (file, cite) in &report.invalid_links {
            println!("   [!] INVALID citation: {} in {}", cite, file);
        }
        failed = true;
    }

    if !report.orphaned_papers.is_empty() {
        if !failed {
            println!("\n=== Validation Failures ===");
        }
        println!("Orphaned papers found: {:?}", report.orphaned_papers);
        failed = true;
    }

    if report_list_section(
        "[!] Unverified Modules (Missing theory_verification! or #[verified]):",
        None,
        "  - ",
        &report.unverified_modules,
    ) {
        println!("\nThreshold not met: False Green detected!");
        failed = true;
    }

    failed |= report_list_section(
        "[!] Invalid Verification Tiers Detected:",
        None,
        "  - ",
        &report.invalid_tiers,
    );
    failed |= report_list_section(
        "[!] Vacuous Bypasses Detected in AI Modules (Zero Initializations):",
        None,
        "  - ",
        &report.vacuous_bypasses,
    );
    failed |= check_semantic_and_density_errors(report);
    failed
}

/// Prints assertion density and semantic integrity dashboard metrics for `report`.
pub fn print_traceability_dashboard(report: &TraceabilityReport) {
    println!("\n=== High-Integrity Dashboard ===");
    let total_density = if report.total_funcs > 0 {
        report.total_asserts as f64 / report.total_funcs as f64
    } else {
        0.0
    };
    let verified_density = if report.verified_funcs > 0 {
        report.verified_asserts as f64 / report.verified_funcs as f64
    } else {
        0.0
    };
    let unverified_funcs = report.total_funcs.saturating_sub(report.verified_funcs);
    let unverified_asserts = report.total_asserts.saturating_sub(report.verified_asserts);
    let unverified_density = if unverified_funcs > 0 {
        unverified_asserts as f64 / unverified_funcs as f64
    } else {
        0.0
    };

    println!("Total Assertion Density: {:.2} asserts/fn", total_density);
    println!(
        "Verified Modules Density: {:.2} asserts/fn",
        verified_density
    );
    println!(
        "Unverified Modules Density: {:.2} asserts/fn",
        unverified_density
    );

    println!("\n=== Semantic Integrity ===");
    for (module, status) in &report.semantic_integrity_status {
        println!("{}: {}", module, status);
    }
}

/// Executes full workspace traceability verification in-process using `cache` and returns `Ok(true)` when all checks pass.
pub fn run_workspace_traceability_with_cache(
    auto_fix: bool,
    cache: &mut AstCache,
) -> Result<bool, std::io::Error> {
    let engine = TraceabilityEngine::new(LocalFsVfs);
    let code_dirs = discover_workspace_code_dirs();
    let all_dirs: Vec<&str> = code_dirs.iter().map(String::as_str).collect();
    let report = futures::executor::block_on(
        engine.scan_repository_with_cache(&all_dirs, "papers", auto_fix, cache),
    )?;

    println!("=== Traceability Report ===");
    println!("Summary: Scanned {} source files.", report.scanned_files);
    print_traceability_dashboard(&report);
    let failed = check_and_print_traceability_errors(&report);
    if !failed {
        println!("All checks passed!");
    }
    Ok(!failed)
}

/// Executes full workspace traceability verification in-process with a fresh [`AstCache`].
pub fn run_workspace_traceability(auto_fix: bool) -> Result<bool, std::io::Error> {
    let mut cache = AstCache::new();
    run_workspace_traceability_with_cache(auto_fix, &mut cache)
}
