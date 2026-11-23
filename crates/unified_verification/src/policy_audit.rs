pub(crate) mod dependencies;
pub(crate) mod doctests;
pub(crate) mod lint_policy;

pub use crate::whitelist::DeadCodeEntry;
use crate::whitelist::{PathMatchOptions, ROOT_ONLY_WHITELIST_CANDIDATES, VerificationWhitelist};
use dependencies::{
    DependencyVisitor, collect_expected_dependencies, is_whitelisted_unused_dep,
    manifest_package_name,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::visit::Visit;

fn is_dead_code_lint_name(ident: &syn::Ident) -> bool {
    ident == "dead_code" || ident == "unused" || ident == "warnings"
}

fn is_dead_code_lint_path(path: &syn::Path) -> bool {
    if path.segments.iter().any(|seg| !seg.arguments.is_empty()) {
        return false;
    }
    match path.segments.len() {
        1 => is_dead_code_lint_name(&path.segments[0].ident),
        2 => path.segments[0].ident == "rust" && is_dead_code_lint_name(&path.segments[1].ident),
        _ => false,
    }
}

fn attr_suppresses_dead_code(attr: &syn::Attribute) -> bool {
    crate::ast_helpers::find_suppressed_lint_in_attr(attr, |lint_path| {
        is_dead_code_lint_path(lint_path).then_some(())
    })
    .is_some()
}

struct DeadCodeVisitor {
    file_path: String,
    allowed: Vec<DeadCodeEntry>,
    has_unauthorized: bool,
    violations: Vec<String>,
}

impl DeadCodeVisitor {
    fn new(file_path: String, allowed: Vec<DeadCodeEntry>) -> Self {
        Self {
            file_path,
            allowed,
            has_unauthorized: false,
            violations: Vec::new(),
        }
    }

    fn is_authorized(&self, ident: Option<&str>) -> bool {
        let opts = PathMatchOptions {
            trim_dot_slash: false,
            require_non_empty: false,
            match_file_name: false,
        };
        let actual_path = Path::new(&self.file_path);
        for entry in &self.allowed {
            if crate::whitelist::path_matches_lossy(actual_path, &entry.file, opts) {
                match (&entry.identifier, ident) {
                    (None, _) => return true,
                    (Some(allowed_id), Some(id)) if allowed_id == id => return true,
                    _ => {}
                }
            }
        }
        false
    }

    fn record_violation(&mut self, line: usize, ident: Option<&str>) {
        let loc = crate::ast_helpers::format_file_line(&self.file_path, line);
        let (scope, wl_snippet) = match ident {
            Some(id) => (
                format!("identifier `{id}`"),
                format!("file = \"{}\", identifier = \"{id}\"", self.file_path),
            ),
            None => (
                "crate/module scope".to_string(),
                format!("file = \"{}\"", self.file_path),
            ),
        };
        let msg = format!(
            "[!] Unauthorized dead code bypass at {loc} ({scope})\n  Hint: Gate test-only items with `#[cfg(test)]`, remove unused code, or add `[[dead_code]]` ({wl_snippet}) in `verification_whitelist.toml`."
        );
        println!("{msg}");
        self.violations.push(msg);
        self.has_unauthorized = true;
    }

    fn check_attrs(&mut self, attrs: &[syn::Attribute], ident: Option<&str>) {
        for attr in attrs {
            if attr_suppresses_dead_code(attr) && !self.is_authorized(ident) {
                self.record_violation(attr.span().start().line, ident);
            }
        }
    }
}

impl<'ast> Visit<'ast> for DeadCodeVisitor {
    fn visit_file(&mut self, i: &'ast syn::File) {
        self.check_attrs(&i.attrs, None);
        syn::visit::visit_file(self, i);
    }
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_item_fn(self, i);
    }
    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_struct(self, i);
    }
    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_enum(self, i);
    }
    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_union(self, i);
    }
    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_type(self, i);
    }
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_mod(self, i);
    }
    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_const(self, i);
    }
    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_static(self, i);
    }
    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_trait(self, i);
    }
    fn visit_item_trait_alias(&mut self, i: &'ast syn::ItemTraitAlias) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_trait_alias(self, i);
    }
    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let ident = crate::ast_helpers::impl_base_ident(&i.self_ty);
        self.check_attrs(&i.attrs, ident.as_deref());
        syn::visit::visit_item_impl(self, i);
    }
    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        self.check_attrs(&i.attrs, None);
        syn::visit::visit_item_use(self, i);
    }
    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        let ident = i.ident.as_ref().map(ToString::to_string);
        self.check_attrs(&i.attrs, ident.as_deref());
        syn::visit::visit_item_macro(self, i);
    }
    fn visit_item_foreign_mod(&mut self, i: &'ast syn::ItemForeignMod) {
        self.check_attrs(&i.attrs, None);
        syn::visit::visit_item_foreign_mod(self, i);
    }
    fn visit_field(&mut self, i: &'ast syn::Field) {
        if let Some(ident) = &i.ident {
            self.check_attrs(&i.attrs, Some(&ident.to_string()));
        } else {
            self.check_attrs(&i.attrs, None);
        }
        syn::visit::visit_field(self, i);
    }
    fn visit_variant(&mut self, i: &'ast syn::Variant) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_variant(self, i);
    }
    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_impl_item_fn(self, i);
    }
    fn visit_impl_item_const(&mut self, i: &'ast syn::ImplItemConst) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_impl_item_const(self, i);
    }
    fn visit_impl_item_type(&mut self, i: &'ast syn::ImplItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_impl_item_type(self, i);
    }
    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_trait_item_fn(self, i);
    }
    fn visit_trait_item_const(&mut self, i: &'ast syn::TraitItemConst) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_trait_item_const(self, i);
    }
    fn visit_trait_item_type(&mut self, i: &'ast syn::TraitItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_trait_item_type(self, i);
    }
    fn visit_foreign_item_fn(&mut self, i: &'ast syn::ForeignItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_foreign_item_fn(self, i);
    }
    fn visit_foreign_item_static(&mut self, i: &'ast syn::ForeignItemStatic) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_foreign_item_static(self, i);
    }
    fn visit_foreign_item_type(&mut self, i: &'ast syn::ForeignItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_foreign_item_type(self, i);
    }
    fn visit_local(&mut self, i: &'ast syn::Local) {
        let ident = crate::ast_helpers::pat_ident_name(&i.pat);
        self.check_attrs(&i.attrs, ident.as_deref());
        syn::visit::visit_local(self, i);
    }
}

struct MemberAuditContext<'a> {
    member_dir: &'a Path,
    whitelist: &'a VerificationWhitelist,
    check_clippy_bypasses: bool,
    max_fn_lines: usize,
}

fn audit_member_sources(
    ctx: &MemberAuditContext<'_>,
    dep_visitor: &mut DependencyVisitor,
    visited_md: &mut HashSet<PathBuf>,
    cache: &mut oxidize_core::traceability::AstCache,
) -> bool {
    let mut success = true;
    for path in crate::utils::collect_rs_files_filtered(&[ctx.member_dir], &["target"]) {
        let file_path = crate::whitelist::lossy_slash_path(&path);
        if let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) {
            dep_visitor.visit_file(ast);
            doctests::audit_rs_doctests(ast, &path, ctx.member_dir, visited_md, dep_visitor);

            let mut dc_visitor =
                DeadCodeVisitor::new(file_path.clone(), ctx.whitelist.dead_code.clone());
            dc_visitor.visit_file(ast);
            if dc_visitor.has_unauthorized {
                success = false;
            }

            if ctx.check_clippy_bypasses {
                let mut clippy_visitor =
                    lint_policy::ClippyBypassVisitor::new(file_path, ctx.max_fn_lines);
                clippy_visitor.visit_file(ast);
                if clippy_visitor.has_unauthorized {
                    success = false;
                }
            }
        }
    }
    success
}

fn audit_single_member(
    member: &str,
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
    let expected_deps = collect_expected_dependencies(&parsed);
    let package_name = manifest_package_name(&parsed);
    let mut dep_visitor = DependencyVisitor::new(expected_deps.clone());
    let mut visited_md = HashSet::new();

    let ctx = MemberAuditContext {
        member_dir: &member_dir,
        whitelist,
        check_clippy_bypasses,
        max_fn_lines,
    };
    if !audit_member_sources(&ctx, &mut dep_visitor, &mut visited_md, cache) {
        success = false;
    }
    if crate::api_drift::public_debug::is_mathematical_crate_member(member, package_name)
        && !crate::api_drift::public_debug::check_member_public_debug_with_cache(&member_dir, cache)
    {
        success = false;
    }

    if doctests::is_markdown_tests_crate(member, package_name) {
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

fn report_unused_dependencies(
    member: &str,
    cargo_toml_path: &Path,
    package_name: Option<&str>,
    expected_deps: HashSet<String>,
    used_deps: &HashSet<String>,
    whitelist_entries: &[crate::whitelist::UnusedDependencyEntry],
) -> bool {
    let mut success = true;
    let manifest_rel = crate::whitelist::lossy_slash_path(cargo_toml_path);
    let mut sorted_expected: Vec<String> = expected_deps.into_iter().collect();
    sorted_expected.sort();
    for expected in sorted_expected {
        if !used_deps.contains(&expected)
            && !is_whitelisted_unused_dep(member, package_name, &expected, whitelist_entries)
        {
            println!(
                "[!] Unused dependency detected in {manifest_rel} ({member}): `{expected}`\n  Hint: Remove `{expected}` from `{manifest_rel}`, move test-only dependencies under `[dev-dependencies]` (`#[cfg(test)]`), or add `[[unused_dependencies]]` (`crate_name = \"{member}\"`, `dependency = \"{expected}\"`) in `verification_whitelist.toml`."
            );
            success = false;
        }
    }
    success
}

pub fn run_policy_audit(members: &[&str]) -> bool {
    let mut cache = oxidize_core::traceability::AstCache::new();
    run_policy_audit_with_cache(members, &mut cache)
}

pub fn run_policy_audit_with_cache(
    members: &[&str],
    cache: &mut oxidize_core::traceability::AstCache,
) -> bool {
    let mut manifest_cache = crate::manifest_cache::ManifestCache::new();
    run_policy_audit_with_caches(members, cache, &mut manifest_cache)
}

pub(crate) fn run_policy_audit_with_caches(
    members: &[&str],
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
    run_policy_audit_with_whitelist_and_caches(members, &whitelist_path, cache, manifest_cache)
}

#[cfg(test)]
pub(crate) fn run_policy_audit_with_whitelist(members: &[&str], whitelist_path: &Path) -> bool {
    let mut cache = oxidize_core::traceability::AstCache::new();
    let mut manifest_cache = crate::manifest_cache::ManifestCache::new();
    run_policy_audit_with_whitelist_and_caches(
        members,
        whitelist_path,
        &mut cache,
        &mut manifest_cache,
    )
}

pub(crate) fn run_policy_audit_with_whitelist_and_caches(
    members: &[&str],
    whitelist_path: &Path,
    cache: &mut oxidize_core::traceability::AstCache,
    manifest_cache: &mut crate::manifest_cache::ManifestCache,
) -> bool {
    let whitelist = match crate::whitelist::load_whitelist_strict(None, &[whitelist_path]) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[!] {}", err);
            return false;
        }
    };
    let max_fn_lines = lint_policy::load_too_many_lines_threshold();
    let mut success = true;

    println!(
        "Running Policy Audit for lint inheritance, Clippy/dead-code bypasses, and unused dependencies..."
    );

    for member in members {
        if !audit_single_member(member, &whitelist, max_fn_lines, cache, manifest_cache) {
            success = false;
        }
    }

    success
}

#[cfg(test)]
mod dep_doctest_tests;
#[cfg(test)]
mod tests;
