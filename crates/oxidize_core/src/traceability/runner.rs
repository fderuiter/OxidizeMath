//! Native workspace discovery, dashboard printing, and in-process traceability execution helpers.

use super::analysis::{format_path_line, invalid_link_key};
use super::{AstCache, TraceabilityEngine, TraceabilityReport};
use crate::vfs::VirtualFileSystem;
use std::future::Future;
use std::path::{Path, PathBuf};
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

pub(super) const IGNORED_TRACEABILITY_MEMBERS: &[&str] = &[
    "crates/unified_verification",
    "crates/diagnostics",
    "crates/federated_registry",
    "crates/oxidize_core",
    "crates/verified_engine",
    "crates/verified_engine_macros",
    "apps/xtask",
    "egui_plot",
];

fn scan_src_dir(src_path: PathBuf, strip_root: Option<&Path>, code_dirs: &mut Vec<String>) {
    let mut stack = vec![src_path];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut has_rs_files = false;
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() && entry.path().extension().is_some_and(|ext| ext == "rs")
            {
                has_rs_files = true;
            }
        }
        if has_rs_files {
            let rel_dir = strip_root
                .and_then(|root| dir.strip_prefix(root).ok())
                .unwrap_or(dir.as_path());
            code_dirs.push(crate::path_utils::normalize_path(rel_dir));
        }
    }
}

/// Discovers active workspace source directories subject to traceability verification.
pub fn discover_workspace_code_dirs() -> Vec<String> {
    let (is_root, members) = crate::workspace::get_workspace_members();
    let mut code_dirs = Vec::new();
    let resolved_root = (!is_root)
        .then(crate::workspace::find_workspace_root)
        .flatten();

    for member_str in members {
        if IGNORED_TRACEABILITY_MEMBERS.contains(&member_str.as_str()) {
            continue;
        }
        let member_path = match &resolved_root {
            Some(root) => root.join(&member_str),
            None => PathBuf::from(&member_str),
        };
        let src_path = if member_str == "math_explorer_gui" {
            member_path.join("src").join("tabs")
        } else {
            member_path.join("src")
        };

        if src_path.is_dir() {
            scan_src_dir(src_path, resolved_root.as_deref(), &mut code_dirs);
        }
    }
    code_dirs
}

fn format_unlinked_and_invalid(report: &TraceabilityReport, lines: &mut Vec<String>) -> bool {
    let mut failed = false;
    if !report.unlinked_module_details.is_empty() || !report.unlinked_code.is_empty() {
        lines.push("\n=== Unlinked Modules ===".to_string());
        lines.push(
            "The following modules contain a verification macro or citation missing from `[links]` in `traceability.toml`:"
                .to_string(),
        );
        if !report.unlinked_module_details.is_empty() {
            for (file, key, line) in &report.unlinked_module_details {
                let loc = format_path_line(file, *line);
                lines.push(format!("   [!] UNLINKED module `{key}` at {loc}"));
                lines.push(format!(
                    "       Hint: Register `{key} = \"<paper>.tex\"` (or `\"spec:<id>\"`) under `[links]` in `traceability.toml`."
                ));
            }
        } else {
            for file in &report.unlinked_code {
                lines.push(format!("   [!] UNLINKED module in {file}"));
                lines.push(
                    "       Hint: Register the module key under `[links]` in `traceability.toml`."
                        .to_string(),
                );
            }
        }
        failed = true;
    }

    if !report.invalid_links.is_empty() {
        lines.push("\n=== Validation Failures ===".to_string());
        lines.push("Invalid citations were found:".to_string());
        for (file, cite) in &report.invalid_links {
            let line = report
                .invalid_link_lines
                .get(&invalid_link_key(file, cite))
                .copied()
                .unwrap_or(0);
            let loc = format_path_line(file, line);
            lines.push(format!("   [!] INVALID citation: `{cite}` at {loc}"));
            lines.push(format!(
                "       Hint: Add `{cite} = \"<paper>.tex\"` under `[links]` in `traceability.toml`, or run `cargo run -p oxidize_core --bin traceability_cli -- --auto-fix`."
            ));
        }
        failed = true;
    }

    if !report.orphaned_papers.is_empty() {
        if !failed {
            lines.push("\n=== Validation Failures ===".to_string());
        }
        lines.push(format!(
            "Orphaned papers found: {:?} (Hint: link each `.tex` paper under `[links]` in `traceability.toml`).",
            report.orphaned_papers
        ));
        failed = true;
    }
    failed
}

fn format_module_and_bypass_errors(report: &TraceabilityReport, lines: &mut Vec<String>) -> bool {
    let mut failed = false;
    if !report.unverified_modules.is_empty() {
        lines.push(
            "\n[!] Unverified Modules (Missing theory_verification! or #[verified]):".to_string(),
        );
        for file in &report.unverified_modules {
            if let Some((fn_name, line)) = report.unverified_module_spans.get(file) {
                let loc = format_path_line(file, *line);
                lines.push(format!("  - {loc} (fn `{fn_name}`)"));
            } else {
                lines.push(format!("  - {file}"));
            }
        }
        lines.push(
            "  Hint: Add `#[verified]` / `theory_verification!` and register `[links]` in `traceability.toml`, or opt out non-mathematical code with `#![cfg_attr(any(), verified(opt_out = \"reason\"))]` (or item `#[verified(opt_out = \"reason\")]`)."
                .to_string(),
        );
        lines.push("\nThreshold not met: False Green detected!".to_string());
        failed = true;
    }

    if !report.invalid_tiers.is_empty() {
        lines.push("\n[!] Invalid Verification Tiers Detected:".to_string());
        for item in &report.invalid_tiers {
            lines.push(format!("  - {item}"));
        }
        lines.push(
            "  Hint: `pure_math` modules must use deterministic `theory_verification!` instead of `stochastic_signature_verification!`."
                .to_string(),
        );
        failed = true;
    }

    if !report.vacuous_bypasses.is_empty() {
        lines.push(
            "\n[!] Vacuous Bypasses Detected in AI Modules (Zero Initializations):".to_string(),
        );
        for file in &report.vacuous_bypasses {
            let line = report.vacuous_bypass_spans.get(file).copied().unwrap_or(0);
            lines.push(format!("  - {}", format_path_line(file, line)));
        }
        lines.push(
            "  Hint: Replace zero initializers (`zeros`, `fill(0)`, `0.0` constants) in `stochastic_test` with deterministic non-zero inputs."
                .to_string(),
        );
        failed = true;
    }
    failed
}

fn format_semantic_and_density_errors(
    report: &TraceabilityReport,
    lines: &mut Vec<String>,
) -> bool {
    let mut failed = false;
    let mut unverified_semantics: Vec<&String> = report
        .semantic_integrity_status
        .iter()
        .filter_map(|(k, v)| if v != "Verified" { Some(k) } else { None })
        .collect();

    if !unverified_semantics.is_empty() {
        lines.push("\n[!] Unverified Semantic Integrity Statuses Detected:".to_string());
        unverified_semantics.sort();
        for module in unverified_semantics {
            lines.push(format!("  - {module}: Unverified"));
        }
        lines.push(
            "  Hint: Add `#[verified]` or `theory_verification!` in a module linked to this key in `traceability.toml`."
                .to_string(),
        );
        failed = true;
    }

    let verified_density = if report.verified_funcs > 0 {
        report.verified_asserts as f64 / report.verified_funcs as f64
    } else {
        0.0
    };
    if report.verified_funcs > 0 && verified_density < 0.4 {
        lines.push(format!(
            "\n[!] Assertion Density Failure: Verified modules have a density of {verified_density:.2} asserts/fn, which is below the minimum required 0.4 asserts/fn."
        ));
        for (file, &[asserts, funcs]) in &report.file_verified_counts {
            if funcs > 0 {
                let file_density = asserts as f64 / funcs as f64;
                if file_density < 0.4 {
                    lines.push(format!(
                        "  - {file}: {file_density:.2} asserts/fn ({asserts} asserts / {funcs} verified fns)"
                    ));
                }
            }
        }
        lines.push(
            "  Hint: Add post-condition `assert!` / `assert_eq!` checks to `#[verified]` functions or mark delegating helpers with `#[verified(opt_out = \"reason\")]`."
                .to_string(),
        );
        failed = true;
    }
    failed
}

/// Formats all traceability diagnostics for `report`, returning `(failed, lines)`.
pub fn format_traceability_errors(report: &TraceabilityReport) -> (bool, Vec<String>) {
    let mut lines = Vec::new();
    let mut failed = format_unlinked_and_invalid(report, &mut lines);
    failed |= format_module_and_bypass_errors(report, &mut lines);
    failed |= format_semantic_and_density_errors(report, &mut lines);
    (failed, lines)
}

/// Evaluates all traceability invariants on `report`, printing diagnostics and returning `true` if any check failed.
pub fn check_and_print_traceability_errors(report: &TraceabilityReport) -> bool {
    let (failed, lines) = format_traceability_errors(report);
    for line in lines {
        println!("{line}");
    }
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
    let (is_root, _) = crate::workspace::get_workspace_members();
    let ws_root = (!is_root)
        .then(crate::workspace::find_workspace_root)
        .flatten();
    let raw_dirs = discover_workspace_code_dirs();
    let resolved_dirs: Vec<String> = match &ws_root {
        Some(root) => raw_dirs
            .iter()
            .map(|d| crate::path_utils::normalize_path(root.join(d)))
            .collect(),
        None => raw_dirs,
    };
    let papers_dir = match &ws_root {
        Some(root) => crate::path_utils::normalize_path(root.join("papers")),
        None => "papers".to_string(),
    };
    let all_dirs: Vec<&str> = resolved_dirs.iter().map(String::as_str).collect();
    let report = futures::executor::block_on(engine.scan_repository_with_cache(
        &all_dirs,
        &papers_dir,
        auto_fix,
        cache,
    ))?;

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
