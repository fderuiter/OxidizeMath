//! Legacy crate.
#[cfg(not(target_arch = "wasm32"))]
use oxidize_core::traceability::TraceabilityEngine;
#[cfg(not(target_arch = "wasm32"))]
use oxidize_core::vfs::DefaultVfs;
#[cfg(not(target_arch = "wasm32"))]
use std::process;

#[cfg(not(target_arch = "wasm32"))]
fn check_and_print_errors(report: &oxidize_core::traceability::TraceabilityReport) -> bool {
    let mut failed = false;

    if !report.unlinked_code.is_empty() {
        println!("\n=== Unlinked Modules ===");
        println!(
            "The following modules contain a verification macro but are missing from the registry:"
        );
        for file in &report.unlinked_code {
            println!("   [!] UNLINKED module: {}", file);
        }
        failed = true;
    }

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

    if !report.unverified_modules.is_empty() {
        println!("\n[!] Unverified Modules (Missing theory_verification! or #[verified]):");
        for module in &report.unverified_modules {
            println!("  - {}", module);
        }
        println!("\nThreshold not met: False Green detected!");
        failed = true;
    }

    if !report.invalid_tiers.is_empty() {
        println!("\n[!] Invalid Verification Tiers Detected:");
        for err in &report.invalid_tiers {
            println!("  - {}", err);
        }
        failed = true;
    }

    if !report.vacuous_bypasses.is_empty() {
        println!("\n[!] Vacuous Bypasses Detected in AI Modules (Zero Initializations):");
        for file in &report.vacuous_bypasses {
            println!("  - {}", file);
        }
        failed = true;
    }

    let unverified_semantics: Vec<&String> = report
        .semantic_integrity_status
        .iter()
        .filter_map(|(k, v)| if v != "Verified" { Some(k) } else { None })
        .collect();

    if !unverified_semantics.is_empty() {
        println!("\n[!] Unverified Semantic Integrity Statuses Detected:");
        let mut sorted_unverified = unverified_semantics;
        sorted_unverified.sort();
        for module in sorted_unverified {
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

#[cfg(not(target_arch = "wasm32"))]
fn print_dashboard(report: &oxidize_core::traceability::TraceabilityReport) {
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

#[cfg(not(target_arch = "wasm32"))]
fn scan_src_dir(src_path: std::path::PathBuf, is_root: bool, code_dirs: &mut Vec<String>) {
    let mut stack = vec![src_path];

    while let Some(dir) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut has_rs_files = false;
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        stack.push(entry.path());
                    } else if file_type.is_file()
                        && entry.path().extension().is_some_and(|ext| ext == "rs")
                    {
                        has_rs_files = true;
                    }
                }
            }
            if has_rs_files {
                let mut target_dir = dir.to_string_lossy().to_string();
                if !is_root && target_dir.starts_with("../../") {
                    target_dir = target_dir.trim_start_matches("../../").to_string();
                }
                // Convert windows backslashes to forward slashes just in case
                target_dir = target_dir.replace("\\", "/");
                code_dirs.push(target_dir);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn discover_code_dirs() -> Vec<String> {
    let (is_root, members) = oxidize_core::workspace::get_workspace_members();
    let mut code_dirs = Vec::new();
    let root_prefix = if is_root { "" } else { "../../" };

    for member_str in members {
        let ignore_list = [
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
        if ignore_list.contains(&member_str.as_str()) {
            continue;
        }
        let member_path = std::path::PathBuf::from(root_prefix).join(&member_str);
        let src_path = if member_str == "math_explorer_gui" {
            member_path.join("src").join("tabs")
        } else {
            member_path.join("src")
        };

        if src_path.exists() && src_path.is_dir() {
            scan_src_dir(src_path, is_root, &mut code_dirs);
        }
    }
    code_dirs
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let auto_fix = args.iter().any(|arg| arg == "--auto-fix");

    let vfs = DefaultVfs;
    let engine = TraceabilityEngine::new(vfs);

    let code_dirs = discover_code_dirs();

    let all_dirs: Vec<&str> = code_dirs.iter().map(|s| s.as_str()).collect();

    match futures::executor::block_on(engine.scan_repository(&all_dirs, "papers", auto_fix)) {
        Ok(report) => {
            println!("=== Traceability Report ===");
            println!("Summary: Scanned {} source files.", report.scanned_files);

            print_dashboard(&report);

            let failed = check_and_print_errors(&report);

            if failed {
                process::exit(1);
            }

            println!("All checks passed!");
        }
        Err(e) => {
            eprintln!("Error scanning repository: {}", e);
            process::exit(1);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
