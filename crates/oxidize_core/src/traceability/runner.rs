//! Native workspace discovery, dashboard printing, and in-process traceability execution helpers.

use super::analysis::{format_path_line, invalid_link_key};
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
    "crates/00_foundation/math_diagnostics",
    "crates/unified_verification",
    "crates/diagnostics",
    "crates/federated_registry",
    "crates/oxidize_core",
    "crates/verified_engine",
    "crates/verified_engine_macros",
    "crates/markdown_tests",
    "apps/xtask",
    "egui_plot",
];

/// Discovers active workspace module roots subject to traceability verification.
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
            let rel_dir = resolved_root
                .as_deref()
                .and_then(|root| src_path.strip_prefix(root).ok())
                .unwrap_or(src_path.as_path());
            code_dirs.push(crate::path_utils::normalize_path(rel_dir));
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
            "  Hint: `pure_math` and `Certified` modules must use deterministic `theory_verification!` instead of `stochastic_signature_verification!`."
                .to_string(),
        );
        failed = true;
    }

    if !report.maturity_violations.is_empty() {
        lines.push("\n[!] Maturity Qualification Violations Detected:".to_string());
        for violation in &report.maturity_violations {
            lines.push(format!("  - {violation}"));
        }
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

fn format_density_floor_errors(
    report: &TraceabilityReport,
    floors: super::DensityFloors,
    lines: &mut Vec<String>,
) -> bool {
    let mut failed = false;
    let verified_density = if report.verified_funcs > 0 {
        report.verified_asserts as f64 / report.verified_funcs as f64
    } else {
        0.0
    };
    if report.verified_funcs > 0 && verified_density < floors.min_verified_density {
        lines.push(format!(
            "\n[!] Assertion Density Failure: Verified modules have a density of {verified_density:.3} asserts/fn, which is below the minimum required {:.3} asserts/fn.",
            floors.min_verified_density
        ));
        for (file, &[asserts, funcs]) in &report.file_verified_counts {
            if funcs > 0 {
                let file_density = asserts as f64 / funcs as f64;
                if file_density < floors.min_verified_density {
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

    let total_density = if report.total_funcs > 0 {
        report.total_asserts as f64 / report.total_funcs as f64
    } else {
        0.0
    };
    if report.total_funcs > 0 && total_density < floors.min_total_density {
        lines.push(format!(
            "\n[!] Assertion Density Failure: Total production assertion density is {total_density:.3} asserts/fn, which is below the minimum required {:.3} asserts/fn.",
            floors.min_total_density
        ));
        failed = true;
    }

    let unverified_funcs = report.total_funcs.saturating_sub(report.verified_funcs);
    let unverified_asserts = report.total_asserts.saturating_sub(report.verified_asserts);
    let unverified_density = if unverified_funcs > 0 {
        unverified_asserts as f64 / unverified_funcs as f64
    } else {
        0.0
    };
    if let Some(min_unverified) = floors.min_unverified_density
        && unverified_funcs > 0
        && unverified_density < min_unverified
    {
        lines.push(format!(
            "\n[!] Assertion Density Failure: Unverified modules have a density of {unverified_density:.3} asserts/fn, which is below the minimum required {min_unverified:.3} asserts/fn."
        ));
        failed = true;
    }
    failed
}

fn format_semantic_and_density_errors(
    report: &TraceabilityReport,
    floors: super::DensityFloors,
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

    failed |= format_density_floor_errors(report, floors, lines);
    failed
}

/// Formats all traceability diagnostics for `report` against explicit `floors`, returning `(failed, lines)`.
pub fn format_traceability_errors_with_floors(
    report: &TraceabilityReport,
    floors: super::DensityFloors,
) -> (bool, Vec<String>) {
    let mut lines = Vec::new();
    let mut failed = format_unlinked_and_invalid(report, &mut lines);
    failed |= format_module_and_bypass_errors(report, &mut lines);
    failed |= format_semantic_and_density_errors(report, floors, &mut lines);
    (failed, lines)
}

/// Formats all traceability diagnostics for `report`, returning `(failed, lines)`.
pub fn format_traceability_errors(report: &TraceabilityReport) -> (bool, Vec<String>) {
    let floors = super::load_workspace_density_floors().unwrap_or_default();
    format_traceability_errors_with_floors(report, floors)
}

/// Evaluates all traceability invariants on `report`, printing diagnostics and returning `true` if any check failed.
pub fn check_and_print_traceability_errors(report: &TraceabilityReport) -> bool {
    let (failed, lines) = format_traceability_errors(report);
    for line in lines {
        println!("{line}");
    }
    failed
}

fn check_and_print_traceability_errors_with_floors(
    report: &TraceabilityReport,
    floors: super::DensityFloors,
) -> bool {
    let (failed, lines) = format_traceability_errors_with_floors(report, floors);
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

/// Aggregated non-test production function and assertion counts across active workspace modules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkspaceAstMetrics {
    /// Number of active non-test Rust source files scanned.
    pub scanned_files: usize,
    /// Total production functions and verification macro blocks across scanned files.
    pub total_funcs: usize,
    /// Total assertion invocations inside production functions and verification macro blocks.
    pub total_asserts: usize,
    /// Total `#[verified]` functions and verification macro blocks across scanned files.
    pub verified_funcs: usize,
    /// Total assertion invocations inside `#[verified]` functions and verification macro blocks.
    pub verified_asserts: usize,
}

fn resolve_workspace_scan_paths() -> (Vec<String>, String) {
    let (is_root, _) = crate::workspace::get_workspace_members();
    let ws_root = (!is_root)
        .then(crate::workspace::find_workspace_root)
        .flatten();
    let raw_dirs = discover_workspace_code_dirs();
    let resolved_dirs = match &ws_root {
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
    (resolved_dirs, papers_dir)
}

/// Collects non-test production function and assertion counts across active workspace modules
/// using `cache`, guaranteeing identical AST counts between `traceability` and `verify-suite`.
pub fn collect_workspace_ast_metrics_with_cache(
    cache: &mut AstCache,
) -> Result<WorkspaceAstMetrics, std::io::Error> {
    let engine = TraceabilityEngine::new(LocalFsVfs);
    let (resolved_dirs, _) = resolve_workspace_scan_paths();
    let all_dirs: Vec<&str> = resolved_dirs.iter().map(String::as_str).collect();
    let mut active_files = std::collections::HashSet::new();
    futures::executor::block_on(engine.gather_active_files(&all_dirs, &mut active_files, cache))?;
    let mut sorted_files: Vec<String> = active_files.into_iter().collect();
    sorted_files.sort();

    let mut metrics = WorkspaceAstMetrics::default();
    for file in sorted_files {
        let norm_file = crate::path_utils::normalize_path(&file);
        let key = AstCache::normalize_key(&norm_file);
        if cache.get(&key).is_none() {
            let content = std::fs::read_to_string(&norm_file)?;
            cache.insert_source(key.clone(), content);
        }
        let Some(ast) = cache.get(&key).and_then(|e| e.ast.as_ref()) else {
            continue;
        };
        if crate::ast_visitor::attrs_require_cfg_test(&ast.attrs) {
            continue;
        }
        metrics.scanned_files += 1;
        let mut visitor = crate::ast_visitor::AstVisitor::new();
        syn::visit::Visit::visit_file(&mut visitor, ast);
        metrics.total_funcs += visitor.total_funcs;
        metrics.total_asserts += visitor.total_asserts;
        metrics.verified_funcs += visitor.verified_funcs;
        metrics.verified_asserts += visitor.verified_asserts;
    }
    Ok(metrics)
}

/// Executes full workspace traceability verification in-process using `cache` and returns `Ok(true)` when all checks pass.
pub fn run_workspace_traceability_with_cache(
    auto_fix: bool,
    cache: &mut AstCache,
) -> Result<bool, std::io::Error> {
    let floors = super::load_workspace_density_floors()?;
    let engine = TraceabilityEngine::new(LocalFsVfs);
    let (resolved_dirs, papers_dir) = resolve_workspace_scan_paths();
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
    let failed = check_and_print_traceability_errors_with_floors(&report, floors);
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
