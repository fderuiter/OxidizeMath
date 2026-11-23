//! Scoped citation and module-registration verification for staged `.rs` files in `batch-staged`.
//!
//! Full 804-file workspace traceability (global assertion density, orphan paper detection, and
//! cross-file semantic integrity) runs in `xtask lint-push`, `xtask lint-all`, and whenever
//! `traceability.toml`, `papers/`, or root workspace configs are staged.

use oxidize_core::ast_visitor::AstVisitor;
use oxidize_core::traceability::AstCache;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

const IGNORED_TRACEABILITY_MEMBERS: &[&str] = &[
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

pub(crate) fn requires_full_traceability(all_staged: &[String], escalate: bool) -> bool {
    escalate
        || all_staged.iter().any(|p| {
            let norm = p.replace('\\', "/");
            let rel = norm.trim_start_matches("./");
            rel == "traceability.toml" || rel.starts_with("papers/")
        })
}

fn is_traced_module_rs_path(rel: &str, workspace_members: &[String]) -> bool {
    if !rel.ends_with(".rs") {
        return false;
    }
    for member in workspace_members {
        if IGNORED_TRACEABILITY_MEMBERS.contains(&member.as_str()) {
            continue;
        }
        let required_prefix = if member == "math_explorer_gui" {
            "math_explorer_gui/src/tabs/".to_string()
        } else {
            format!("{member}/src/")
        };
        if rel.starts_with(&required_prefix) {
            return true;
        }
    }
    false
}

fn is_first_party_rs_path(rel: &str) -> bool {
    rel.ends_with(".rs")
        && (rel.starts_with("crates/")
            || rel.starts_with("apps/")
            || rel.starts_with("math_explorer/")
            || rel.starts_with("math_explorer_gui/"))
        && !rel.starts_with("crates/unified_verification/")
        && !rel.starts_with("egui_plot/")
        && !rel.contains("/target/")
}

fn extract_citations_with_lines(content: &str) -> Vec<(String, usize)> {
    let mut cites = Vec::new();
    let mut search_idx = 0;
    while let Some(start) = content[search_idx..].find("[cite:") {
        let tag_start = search_idx + start;
        let real_start = tag_start + 6;
        let Some(end) = content[real_start..].find(']') else {
            break;
        };
        let paper_name = content[real_start..real_start + end].to_string();
        if !paper_name.is_empty()
            && paper_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
        {
            let line = content[..tag_start].bytes().filter(|&b| b == b'\n').count() + 1;
            cites.push((paper_name, line));
        }
        search_idx = real_start + end + 1;
    }
    cites
}

struct RegistrySnapshot {
    registered_modules: HashSet<String>,
    broken_registry_entries: Vec<(String, String)>,
}

fn load_registry_snapshot(workspace_root: &Path) -> Result<Option<RegistrySnapshot>, String> {
    let registry_path = workspace_root.join("traceability.toml");
    if !registry_path.is_file() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&registry_path)
        .map_err(|e| format!("Failed to read {}: {e}", registry_path.display()))?;
    let table: toml::Table = raw
        .parse()
        .map_err(|e| format!("Invalid TOML in {}: {e}", registry_path.display()))?;

    let mut valid_papers = HashSet::new();
    if let Ok(entries) = fs::read_dir(workspace_root.join("papers")) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str()
                && name.ends_with(".tex")
            {
                valid_papers.insert(name.to_string());
            }
        }
    }

    let mut registered_modules = HashSet::new();
    let mut broken_registry_entries = Vec::new();
    if let Some(links) = table.get("links").and_then(toml::Value::as_table) {
        let mut links_map = HashMap::new();
        for (mod_name, val) in links {
            let Some(target) = val.as_str() else {
                broken_registry_entries.push((mod_name.clone(), "INVALID_FORMAT".to_string()));
                continue;
            };
            links_map.insert(mod_name.clone(), target.to_string());
            if valid_papers.contains(target)
                || target.starts_with("spec:")
                || target.starts_with("registry:")
            {
                registered_modules.insert(mod_name.clone());
            } else {
                broken_registry_entries.push((mod_name.clone(), target.to_string()));
            }
        }
    }
    Ok(Some(RegistrySnapshot {
        registered_modules,
        broken_registry_entries,
    }))
}

fn collect_visitor_traceability_errors(
    rel: &str,
    visitor: AstVisitor,
    registered_modules: &HashSet<String>,
    errors: &mut Vec<String>,
) {
    if visitor.has_vacuous_bypass && rel.contains("domain_ai") {
        let line = visitor.vacuous_bypass_line.unwrap_or(1);
        errors.push(format!(
            "[!] Vacuous bypass detected in AI module at {rel}:{line}"
        ));
    }
    for (module, tier) in &visitor.module_tiers {
        if rel.contains("pure_math") && tier != "Deterministic" {
            let line = visitor.module_spans.get(module).copied().unwrap_or(1);
            errors.push(format!(
                "[!] Invalid verification tier `{tier}` for `{module}` at {rel}:{line}"
            ));
        }
    }
    let is_mod_entry = rel.ends_with("mod.rs") || rel.contains("/tabs/") || rel.ends_with("lib.rs");
    if is_mod_entry {
        for module in &visitor.verified_modules {
            if !registered_modules.contains(module) {
                let line = visitor.module_spans.get(module).copied().unwrap_or(1);
                errors.push(format!(
                    "   [!] UNLINKED module `{module}` at {rel}:{line}\n       Hint: Register `{module} = \"<paper>.tex\"` under `[links]` in `traceability.toml`."
                ));
            }
        }
    }
    let unopted_funcs = visitor.total_funcs.saturating_sub(visitor.opted_out_funcs);
    if unopted_funcs > 0
        && visitor.verified_funcs == 0
        && visitor.verified_modules.is_empty()
        && !visitor.opted_out
    {
        let (fn_name, line) = visitor
            .first_unverified_fn
            .unwrap_or_else(|| ("<unknown>".to_string(), 1));
        errors.push(format!(
            "[!] Unverified module {rel}:{line} (fn `{fn_name}`) missing `#[verified]` / `theory_verification!` or opt-out."
        ));
    }
}

fn check_single_staged_rs_file(
    rel: &str,
    is_traced_module_file: bool,
    registered_modules: &HashSet<String>,
    cache: &mut AstCache,
) -> Vec<String> {
    let mut errors = Vec::new();
    let path = Path::new(rel);
    let Some(entry) = cache.get_or_load_fs(path) else {
        return errors;
    };
    let citations = extract_citations_with_lines(&entry.content);
    let mut valid_cites = Vec::new();
    for (cite, line) in citations {
        if registered_modules.contains(&cite) {
            valid_cites.push(cite);
        } else {
            errors.push(format!(
                "   [!] INVALID citation: `{cite}` at {rel}:{line}\n       Hint: Add `{cite} = \"<paper>.tex\"` under `[links]` in `traceability.toml`."
            ));
        }
    }

    if !is_traced_module_file {
        return errors;
    }
    let Some(ast) = cache.get_or_load_fs(path).and_then(|e| e.ast.as_ref()) else {
        return errors;
    };

    let mut visitor = AstVisitor::new();
    visitor.verified_modules.extend(valid_cites);
    syn::visit::Visit::visit_file(&mut visitor, ast);
    collect_visitor_traceability_errors(rel, visitor, registered_modules, &mut errors);
    errors
}

/// Validates citations and module registrations on staged `.rs` files without scanning all 804 files.
pub(crate) fn run_staged_traceability_with_cache(
    existing_staged: &[String],
    workspace_members: &[String],
    cache: &mut AstCache,
) -> bool {
    let staged_rs: Vec<String> = existing_staged
        .iter()
        .map(|p| p.replace('\\', "/"))
        .map(|p| p.trim_start_matches("./").to_string())
        .filter(|rel| is_first_party_rs_path(rel) && Path::new(rel).is_file())
        .collect();
    if staged_rs.is_empty() {
        return true;
    }

    let workspace_root = crate::manifest_cache::workspace_root_dir();
    let snapshot = match load_registry_snapshot(&workspace_root) {
        Ok(Some(snap)) => snap,
        Ok(None) => return true,
        Err(err) => {
            eprintln!("[!] {err}");
            return false;
        }
    };

    let mut errors = Vec::new();
    for (mod_name, target) in &snapshot.broken_registry_entries {
        errors.push(format!(
            "   [!] INVALID registry link in traceability.toml: `{mod_name}` -> `{target}`"
        ));
    }

    for rel in &staged_rs {
        let is_traced = is_traced_module_rs_path(rel, workspace_members);
        errors.extend(check_single_staged_rs_file(
            rel,
            is_traced,
            &snapshot.registered_modules,
            cache,
        ));
    }

    if errors.is_empty() {
        println!(
            "Scoped traceability check passed ({} staged Rust file(s) verified).",
            staged_rs.len()
        );
        true
    } else {
        println!("=== Scoped Traceability Failures ===");
        for err in &errors {
            println!("{err}");
        }
        false
    }
}
