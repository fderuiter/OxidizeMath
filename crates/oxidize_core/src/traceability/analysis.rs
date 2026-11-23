//! Registry parsing, citation resolution, and AST analysis helpers for [`super::TraceabilityEngine`].

use super::TraceabilityReport;
use std::collections::{HashMap, HashSet};

pub(super) fn invalid_link_key(file: &str, cite: &str) -> String {
    format!("{file}::{cite}")
}

pub(super) fn format_path_line(path: &str, line: usize) -> String {
    if line > 0 {
        format!("{path}:{line}")
    } else {
        path.to_string()
    }
}

pub(super) fn extract_citations_with_lines(content: &str) -> Vec<(String, usize)> {
    let mut cites = Vec::new();
    let mut search_idx = 0;
    while let Some(start) = content[search_idx..].find("[cite:") {
        let tag_start = search_idx + start;
        let real_start = tag_start + 6;
        if let Some(end) = content[real_start..].find(']') {
            let paper_name = content[real_start..real_start + end].to_string();
            if paper_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
                && !paper_name.is_empty()
            {
                let line = content[..tag_start].bytes().filter(|&b| b == b'\n').count() + 1;
                cites.push((paper_name, line));
            }
            search_idx = real_start + end + 1;
        } else {
            break;
        }
    }
    cites
}

pub(super) fn is_module_in_registry_toml(registry_content: &str, module_name: &str) -> bool {
    if let Ok(value) = registry_content.parse::<toml::Table>()
        && let Some(links) = value.get("links").and_then(|v| v.as_table())
        && let Some(paper) = links.get(module_name)
    {
        return paper.as_str().is_some();
    }
    false
}

pub(super) fn apply_registry_table(
    value: &toml::Table,
    valid_papers: &HashSet<String>,
    registered_modules: &mut HashSet<String>,
    registry_links: &mut HashMap<String, String>,
    report: &mut TraceabilityReport,
) {
    let Some(links) = value.get("links").and_then(|v| v.as_table()) else {
        return;
    };
    for (module_name, paper_val) in links {
        registered_modules.insert(module_name.clone());
        let Some(paper_name) = paper_val.as_str() else {
            report
                .invalid_links
                .push((module_name.clone(), "INVALID_FORMAT".to_string()));
            continue;
        };
        let paper_str = paper_name.to_string();
        registry_links.insert(module_name.clone(), paper_str.clone());
        if valid_papers.contains(&paper_str)
            || paper_str.starts_with("spec:")
            || paper_str.starts_with("registry:")
        {
            report
                .paper_coverage
                .entry(paper_str)
                .or_default()
                .push(module_name.clone());
        } else {
            report.invalid_links.push((module_name.clone(), paper_str));
        }
    }
}

pub(super) fn populate_paper_semantic_status(
    valid_papers: &HashSet<String>,
    registry_links: &HashMap<String, String>,
    report: &mut TraceabilityReport,
) {
    for paper in valid_papers {
        let model_name = paper.strip_suffix(".tex").unwrap_or(paper);
        if report.semantic_integrity_status.contains_key(model_name) {
            continue;
        }
        let is_verified = registry_links.iter().any(|(module, target_paper)| {
            let target_clean = target_paper.strip_suffix(".tex").unwrap_or(target_paper);
            (target_paper == paper || target_clean == model_name)
                && matches!(
                    report.semantic_integrity_status.get(module),
                    Some(status) if status == "Verified"
                )
        });
        let status = if is_verified {
            "Verified"
        } else {
            "Unverified"
        };
        report
            .semantic_integrity_status
            .insert(model_name.to_string(), status.to_string());
    }
}

pub(super) fn finalize_report_orphans(report: &mut TraceabilityReport) {
    for (name, linked_code) in &mut report.paper_coverage {
        linked_code.sort();
        linked_code.dedup();
        if linked_code.is_empty() {
            report.orphaned_papers.push(name.clone());
        }
    }
    report.orphaned_papers.sort();
    report.orphaned_papers.dedup();
    report.unlinked_code.sort();
    report.unlinked_code.dedup();
    report.unlinked_module_details.sort();
    report.unlinked_module_details.dedup();
    report.invalid_links.sort();
    report.invalid_links.dedup();
}

pub(super) fn collect_valid_citations(
    file: &str,
    citations: Vec<(String, usize)>,
    registered_modules: &HashSet<String>,
    report: &mut TraceabilityReport,
) -> Vec<String> {
    let mut final_citations = Vec::new();
    for (cite, line) in citations {
        if registered_modules.contains(&cite) {
            final_citations.push(cite);
        } else {
            if line > 0 {
                report
                    .invalid_link_lines
                    .entry(invalid_link_key(file, &cite))
                    .or_insert(line);
            }
            report.invalid_links.push((file.to_string(), cite));
        }
    }
    final_citations
}

pub(super) fn fix_and_collect_citations(
    file: &str,
    content: &mut String,
    citations: Vec<(String, usize)>,
    registered_modules: &HashSet<String>,
    registry_links: &HashMap<String, String>,
    report: &mut TraceabilityReport,
    auto_fix: bool,
) -> (bool, Vec<String>) {
    let mut file_modified = false;
    let mut final_citations = Vec::new();

    for (cite, line) in citations {
        if registered_modules.contains(&cite) {
            final_citations.push(cite);
            continue;
        }
        if let Some(key) = auto_fix
            .then(|| find_best_autofix_key(file, &cite, registry_links))
            .flatten()
        {
            let old_tag = format!("[cite:{cite}]");
            let new_tag = format!("[cite:{key}]");
            *content = content.replace(&old_tag, &new_tag);
            file_modified = true;
            final_citations.push(key);
        } else {
            if line > 0 {
                report
                    .invalid_link_lines
                    .entry(invalid_link_key(file, &cite))
                    .or_insert(line);
            }
            report.invalid_links.push((file.to_string(), cite));
        }
    }

    (file_modified, final_citations)
}

fn find_best_autofix_key(
    file: &str,
    cite: &str,
    registry_links: &HashMap<String, String>,
) -> Option<String> {
    let mut sorted_registry: Vec<_> = registry_links.iter().collect();
    sorted_registry.sort_by_key(|(k, _)| *k);

    let mut best_key = None;
    for (key, val) in sorted_registry {
        let base_val = val.replace(".tex", "").replace("spec:", "");
        if base_val == cite || val == cite {
            best_key = Some(key.clone());
            if file.contains(key) {
                break;
            }
        }
    }
    best_key
}

pub(super) fn analyze_parsed_ast(
    file: &str,
    ast: Option<&syn::File>,
    is_module: bool,
    final_citations: &[String],
    registered_modules: &HashSet<String>,
    report: &mut TraceabilityReport,
) {
    let Some(ast) = ast else {
        return;
    };
    let mut visitor = crate::ast_visitor::AstVisitor::new();
    visitor.verified_modules.extend(final_citations.to_vec());
    syn::visit::Visit::visit_file(&mut visitor, ast);

    report.total_funcs += visitor.total_funcs;
    report.total_asserts += visitor.total_asserts;
    report.verified_funcs += visitor.verified_funcs;
    report.verified_asserts += visitor.verified_asserts;
    if visitor.verified_funcs > 0 {
        report.file_verified_counts.insert(
            file.to_string(),
            [visitor.verified_asserts, visitor.verified_funcs],
        );
    }

    record_visitor_semantic_status(&visitor, final_citations, report);
    record_visitor_violations(file, is_module, &visitor, registered_modules, report);
}

fn record_visitor_semantic_status(
    visitor: &crate::ast_visitor::AstVisitor,
    final_citations: &[String],
    report: &mut TraceabilityReport,
) {
    let mut all_citations = final_citations.to_vec();
    all_citations.extend(visitor.verified_modules.clone());
    all_citations.sort();
    all_citations.dedup();

    let has_verified_fn = visitor.semantic_integrity_funcs > 0 || visitor.verified_funcs > 0;
    for cite in all_citations {
        if has_verified_fn {
            report
                .semantic_integrity_status
                .insert(cite, "Verified".to_string());
        } else {
            report
                .semantic_integrity_status
                .entry(cite)
                .or_insert_with(|| "Unverified".to_string());
        }
    }
}

fn record_visitor_violations(
    file: &str,
    is_module: bool,
    visitor: &crate::ast_visitor::AstVisitor,
    registered_modules: &HashSet<String>,
    report: &mut TraceabilityReport,
) {
    if visitor.has_vacuous_bypass && file.contains("domain_ai") {
        report.vacuous_bypasses.push(file.to_string());
        if let Some(line) = visitor.vacuous_bypass_line.filter(|&l| l > 0) {
            report.vacuous_bypass_spans.insert(file.to_string(), line);
        }
    }
    for (module, tier) in &visitor.module_tiers {
        if file.contains("pure_math") && tier != "Deterministic" {
            let line = visitor.module_spans.get(module).copied().unwrap_or(0);
            let loc = format_path_line(file, line);
            report
                .invalid_tiers
                .push(format!("{module} at {loc} used tier {tier}"));
        }
    }
    if is_module {
        for module in &visitor.verified_modules {
            if !registered_modules.contains(module) {
                let line = visitor.module_spans.get(module).copied().unwrap_or(0);
                report.unlinked_code.push(file.to_string());
                report
                    .unlinked_module_details
                    .push((file.to_string(), module.clone(), line));
            }
        }
    }
    let unopted_funcs = visitor.total_funcs.saturating_sub(visitor.opted_out_funcs);
    if unopted_funcs > 0
        && visitor.verified_funcs == 0
        && visitor.verified_modules.is_empty()
        && !visitor.opted_out
    {
        report.unverified_modules.push(file.to_string());
        if let Some((fn_name, line)) = &visitor.first_unverified_fn {
            report
                .unverified_module_spans
                .insert(file.to_string(), (fn_name.clone(), *line));
        }
    }
}

#[derive(Default)]
struct SubmoduleCollector {
    submodules: Vec<String>,
}

impl<'ast> syn::visit::Visit<'ast> for SubmoduleCollector {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.content.is_none() {
            self.submodules.push(node.ident.to_string());
        }
        syn::visit::visit_item_mod(self, node);
    }
}

pub(super) fn collect_external_submodules(ast: Option<&syn::File>) -> Vec<String> {
    let Some(ast) = ast else {
        return Vec::new();
    };
    let mut collector = SubmoduleCollector::default();
    syn::visit::Visit::visit_file(&mut collector, ast);
    collector.submodules
}

pub(super) fn module_child_dir(norm_file_path: &str) -> String {
    let mut dir_parts: Vec<&str> = norm_file_path.split('/').collect();
    let is_mod_rs = norm_file_path.ends_with("mod.rs")
        || norm_file_path.ends_with("lib.rs")
        || norm_file_path.ends_with("main.rs");
    if is_mod_rs {
        dir_parts.pop();
    } else if let Some(stem) = dir_parts.pop().map(|s| s.strip_suffix(".rs").unwrap_or(s)) {
        dir_parts.push(stem);
    }
    dir_parts.join("/")
}
