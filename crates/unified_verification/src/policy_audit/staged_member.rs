use super::dependencies::{
    DependencyVisitor, collect_expected_dependencies, is_whitelisted_unused_dep,
    manifest_package_name,
};
use super::{
    DeadCodeVisitor, MemberAuditContext, doctests, lint_policy, missing_docs,
    report_unused_dependencies,
};
use crate::whitelist::{ROOT_ONLY_WHITELIST_CANDIDATES, VerificationWhitelist};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

fn normalize_rel(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

fn content_may_reference_dep(content: &str, dep: &str) -> bool {
    if content.contains(dep) || content.contains("include_str") {
        return true;
    }
    if dep == "serde" && (content.contains("Serialize") || content.contains("Deserialize")) {
        return true;
    }
    dep == "thiserror" && content.contains("Error")
}

fn scan_untouched_file_for_deps(
    path: &Path,
    ctx: &MemberAuditContext<'_>,
    required_deps: &HashSet<String>,
    dep_visitor: &mut DependencyVisitor,
    visited_md: &mut HashSet<PathBuf>,
    cache: &mut oxidize_core::traceability::AstCache,
) {
    if required_deps.is_subset(&dep_visitor.used_deps) {
        return;
    }
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let mentions_needed = required_deps
        .iter()
        .filter(|dep| !dep_visitor.used_deps.contains(*dep))
        .any(|dep| content_may_reference_dep(&content, dep));
    if !mentions_needed {
        return;
    }
    if let Some(ast) = cache.get_or_load_fs(path).and_then(|e| e.ast.as_ref()) {
        dep_visitor.visit_file(ast);
        doctests::audit_rs_doctests(ast, path, ctx.member_dir, visited_md, dep_visitor);
    }
}

fn audit_staged_file_full(
    path: &Path,
    file_path: &str,
    ctx: &MemberAuditContext<'_>,
    dep_visitor: &mut DependencyVisitor,
    visited_md: &mut HashSet<PathBuf>,
    matched_missing_docs_keys: &mut HashSet<String>,
    cache: &mut oxidize_core::traceability::AstCache,
) -> bool {
    let Some(ast) = cache.get_or_load_fs(path).and_then(|e| e.ast.as_ref()) else {
        return true;
    };
    let mut ok = true;
    if !dep_visitor.all_deps_matched() {
        dep_visitor.visit_file(ast);
    }
    if !dep_visitor.all_deps_matched() {
        doctests::audit_rs_doctests(ast, path, ctx.member_dir, visited_md, dep_visitor);
    }

    let mut dc_visitor = DeadCodeVisitor::new(file_path, &ctx.whitelist.dead_code);
    dc_visitor.visit_file(ast);
    if dc_visitor.has_unauthorized {
        ok = false;
    }

    if ctx.check_missing_docs
        && !missing_docs::audit_file_missing_docs(
            ast,
            file_path,
            ctx.whitelist,
            matched_missing_docs_keys,
        )
    {
        ok = false;
    }

    if ctx.check_clippy_bypasses {
        let mut clippy_visitor =
            lint_policy::ClippyBypassVisitor::new(file_path.to_string(), ctx.max_fn_lines);
        clippy_visitor.visit_file(ast);
        if clippy_visitor.has_unauthorized {
            ok = false;
        }
    }
    ok
}

fn audit_member_sources_scoped(
    ctx: &MemberAuditContext<'_>,
    staged_rs_set: &HashSet<String>,
    required_deps: &HashSet<String>,
    dep_visitor: &mut DependencyVisitor,
    visited_md: &mut HashSet<PathBuf>,
    cache: &mut oxidize_core::traceability::AstCache,
) -> bool {
    let mut success = true;
    let mut matched_missing_docs_keys = HashSet::new();
    let all_rs = crate::utils::collect_rs_files_filtered(&[ctx.member_dir], &["target"]);
    let mut untouched = Vec::new();

    for path in all_rs {
        let file_path = crate::whitelist::lossy_slash_path(&path);
        let norm = normalize_rel(&file_path);
        if staged_rs_set.contains(&norm) {
            if !audit_staged_file_full(
                &path,
                &file_path,
                ctx,
                dep_visitor,
                visited_md,
                &mut matched_missing_docs_keys,
                cache,
            ) {
                success = false;
            }
        } else {
            if ctx.check_missing_docs {
                missing_docs::record_existing_file_cap(
                    &file_path,
                    ctx.whitelist,
                    &mut matched_missing_docs_keys,
                );
            }
            untouched.push(path);
        }
    }

    for path in &untouched {
        if required_deps.is_subset(&dep_visitor.used_deps) {
            break;
        }
        scan_untouched_file_for_deps(path, ctx, required_deps, dep_visitor, visited_md, cache);
    }

    if ctx.check_missing_docs
        && !missing_docs::check_stale_member_missing_docs_caps(
            ctx.member_dir,
            ctx.whitelist,
            &matched_missing_docs_keys,
        )
    {
        success = false;
    }
    success
}

fn audit_single_member_scoped(
    member: &str,
    staged_rs_set: &HashSet<String>,
    whitelist: &VerificationWhitelist,
    max_fn_lines: usize,
    cache: &mut oxidize_core::traceability::AstCache,
    manifest_cache: &mut crate::manifest_cache::ManifestCache,
) -> bool {
    let member_dir = lint_policy::resolve_member_dir(member);
    let cargo_toml_path = member_dir.join("Cargo.toml");
    let parsed = match manifest_cache.get_or_load(&cargo_toml_path) {
        Ok(cached) => cached.parsed.clone(),
        Err(err) => {
            eprintln!("{err}");
            return false;
        }
    };

    let mut success = manifest_cache.check_manifest_lints_once(member, &cargo_toml_path, &parsed);
    let check_clippy_bypasses = !lint_policy::is_vendored_egui_plot_exempt(member, &parsed);
    let check_missing_docs =
        !missing_docs::is_vendored_missing_docs_exempt(member, &parsed, whitelist);
    let expected_deps = collect_expected_dependencies(&parsed);
    let package_name = manifest_package_name(&parsed);
    let required_deps: HashSet<String> = expected_deps
        .iter()
        .filter(|dep| {
            !is_whitelisted_unused_dep(member, package_name, dep, &whitelist.unused_dependencies)
        })
        .cloned()
        .collect();
    let mut dep_visitor = DependencyVisitor::new(expected_deps.clone());
    let mut visited_md = HashSet::new();

    let ctx = MemberAuditContext {
        member_dir: &member_dir,
        whitelist,
        check_clippy_bypasses,
        check_missing_docs,
        max_fn_lines,
    };
    if !audit_member_sources_scoped(
        &ctx,
        staged_rs_set,
        &required_deps,
        &mut dep_visitor,
        &mut visited_md,
        cache,
    ) {
        success = false;
    }

    if !required_deps.is_subset(&dep_visitor.used_deps)
        && doctests::is_markdown_tests_crate(member, package_name)
    {
        doctests::audit_workspace_markdown_files(&member_dir, &mut visited_md, &mut dep_visitor);
    }

    if !report_unused_dependencies(
        member,
        &cargo_toml_path,
        package_name,
        expected_deps,
        &dep_visitor.used_deps,
        &whitelist.unused_dependencies,
    ) {
        success = false;
    }
    success
}

pub(crate) fn run_staged_policy_audit_with_caches(
    members: &[&str],
    existing_staged: &[String],
    cache: &mut oxidize_core::traceability::AstCache,
    manifest_cache: &mut crate::manifest_cache::ManifestCache,
) -> bool {
    let resolved = if lint_policy::cwd_is_member_crate_dir() {
        crate::whitelist::find_whitelist_path()
    } else {
        crate::whitelist::resolve_whitelist_path(ROOT_ONLY_WHITELIST_CANDIDATES)
    };
    let whitelist_path =
        resolved.unwrap_or_else(|| Path::new(crate::whitelist::WHITELIST_FILE_NAME).to_path_buf());
    let whitelist = match crate::whitelist::load_whitelist_strict(None, &[&whitelist_path]) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[!] {}", err);
            return false;
        }
    };
    let max_fn_lines = lint_policy::load_too_many_lines_threshold();
    let staged_rs_set: HashSet<String> = existing_staged
        .iter()
        .filter(|p| p.ends_with(".rs"))
        .map(|p| normalize_rel(p))
        .collect();

    println!(
        "Running Policy Audit for lint inheritance, Clippy/dead-code bypasses, and unused dependencies..."
    );

    let mut success = true;
    for member in members {
        if !audit_single_member_scoped(
            member,
            &staged_rs_set,
            &whitelist,
            max_fn_lines,
            cache,
            manifest_cache,
        ) {
            success = false;
        }
    }
    success
}
