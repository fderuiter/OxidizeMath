use oxidize_core::traceability::AstCache;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

fn attr_is_cfg_test(attr: &syn::Attribute) -> bool {
    attr.path().is_ident("cfg") && {
        let t = quote::quote!(#attr).to_string().replace(' ', "");
        t == "#[cfg(test)]"
            || t == "#![cfg(test)]"
            || t.contains("cfg(all(test,")
            || t.contains("cfg(any(test,")
    }
}

fn attrs_have_test_gate(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr_is_cfg_test(attr)
            || attr
                .path()
                .segments
                .last()
                .is_some_and(|s| s.ident == "test")
    })
}

fn attr_has_key(attr: &syn::Attribute, key: &str) -> bool {
    !attr.path().is_ident("doc")
        && quote::quote!(#attr)
            .to_string()
            .replace(' ', "")
            .contains(&format!("{key}="))
}

fn attr_is_verified(attr: &syn::Attribute) -> bool {
    let segs = &attr.path().segments;
    (segs.len() == 1 && segs[0].ident == "verified")
        || (segs.len() == 2 && segs[0].ident == "verified_engine" && segs[1].ident == "verified")
}

fn inspect_verified_fn_attrs(attrs: &[syn::Attribute]) -> (bool, bool, bool) {
    let has_opt_out = attrs.iter().any(|a| attr_has_key(a, "opt_out"));
    let has_verified = attrs.iter().any(attr_is_verified);
    let has_alloc_ok = attrs
        .iter()
        .any(|a| attr_is_verified(a) && attr_has_key(a, "alloc_ok"));
    (has_verified, has_alloc_ok, has_opt_out)
}

fn attr_is_global_allocator(attr: &syn::Attribute) -> bool {
    if attr.path().is_ident("global_allocator") {
        return true;
    }
    if attr.path().is_ident("cfg_attr") {
        let compact = quote::quote!(#attr).to_string().replace(' ', "");
        return compact.contains(",global_allocator)");
    }
    false
}

struct GlobalAllocFinder {
    found: bool,
}

impl<'ast> Visit<'ast> for GlobalAllocFinder {
    fn visit_item_static(&mut self, node: &'ast syn::ItemStatic) {
        if node.attrs.iter().any(attr_is_global_allocator) {
            self.found = true;
        }
        visit::visit_item_static(self, node);
    }
}

pub(crate) fn ast_has_global_allocator(ast: &syn::File) -> bool {
    let mut finder = GlobalAllocFinder { found: false };
    finder.visit_file(ast);
    finder.found
}

struct LockCallFinder {
    in_test_scope: bool,
    force_test_file: bool,
    calls: Vec<(String, usize)>,
}

impl LockCallFinder {
    fn record_if_active(&mut self, name: &str, line: usize) {
        if self.force_test_file || self.in_test_scope {
            self.calls.push((name.to_string(), line));
        }
    }
}

impl<'ast> Visit<'ast> for LockCallFinder {
    fn visit_file(&mut self, node: &'ast syn::File) {
        let prev = self.in_test_scope;
        if node.attrs.iter().any(attr_is_cfg_test) {
            self.in_test_scope = true;
        }
        visit::visit_file(self, node);
        self.in_test_scope = prev;
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let prev = self.in_test_scope;
        if attrs_have_test_gate(&node.attrs) {
            self.in_test_scope = true;
        }
        visit::visit_item_mod(self, node);
        self.in_test_scope = prev;
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        let prev = self.in_test_scope;
        if attrs_have_test_gate(&node.attrs) {
            self.in_test_scope = true;
        }
        visit::visit_item_fn(self, node);
        self.in_test_scope = prev;
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(ref ep) = *node.func
            && let Some(last) = ep.path.segments.last()
        {
            let ident = last.ident.to_string();
            if ident == "run_verified" || ident == "lock_allocations" {
                self.record_if_active(&ident, last.ident.span().start().line);
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let ident = node.method.to_string();
        if ident == "run_verified" || ident == "lock_allocations" {
            self.record_if_active(&ident, node.method.span().start().line);
        }
        visit::visit_expr_method_call(self, node);
    }
}

fn is_allocating_assoc_call(type_ident: &str, fn_ident: &str) -> bool {
    match type_ident {
        "Vec" => matches!(fn_ident, "new" | "with_capacity" | "from"),
        "String" => matches!(
            fn_ident,
            "new" | "with_capacity" | "from" | "from_utf8" | "from_utf8_lossy"
        ),
        "Box" | "Arc" | "Rc" | "CString" | "BTreeMap" | "BTreeSet" => fn_ident == "new",
        "HashMap" | "HashSet" | "VecDeque" | "BinaryHeap" => {
            matches!(fn_ident, "new" | "with_capacity")
        }
        "DMatrix" | "DVector" => matches!(
            fn_ident,
            "zeros"
                | "from_element"
                | "from_fn"
                | "from_vec"
                | "from_row_slice"
                | "from_column_slice"
                | "from_iterator"
                | "from_diagonal"
                | "from_diagonal_element"
                | "identity"
                | "repeat"
        ),
        _ => false,
    }
}

struct BodyAllocFinder {
    first_hit: Option<(String, usize)>,
}

impl BodyAllocFinder {
    fn flag(&mut self, desc: impl Into<String>, line: usize) {
        if self.first_hit.is_none() {
            self.first_hit = Some((desc.into(), line));
        }
    }
}

impl<'ast> Visit<'ast> for BodyAllocFinder {
    fn visit_item(&mut self, _node: &'ast syn::Item) {
        // Nested items are audited independently in their own scope.
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        if let Some(last) = node.path.segments.last() {
            let name = last.ident.to_string();
            if name == "vec" && !node.tokens.is_empty() {
                self.flag("vec![...]", last.ident.span().start().line);
            } else if name == "format" {
                self.flag("format!(...)", last.ident.span().start().line);
            }
        }
        visit::visit_macro(self, node);
    }

    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let syn::Expr::Path(ref ep) = *node.func {
            let segs = &ep.path.segments;
            if segs.len() >= 2 {
                let ty = segs[segs.len() - 2].ident.to_string();
                let func = segs[segs.len() - 1].ident.to_string();
                if is_allocating_assoc_call(&ty, &func) {
                    self.flag(
                        format!("{ty}::{func}"),
                        segs[segs.len() - 1].ident.span().start().line,
                    );
                }
            }
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
        let method = node.method.to_string();
        if matches!(
            method.as_str(),
            "to_vec" | "to_string" | "to_owned" | "into_boxed_slice" | "clone_owned" | "collect"
        ) {
            self.flag(format!(".{method}()"), node.method.span().start().line);
        }
        visit::visit_expr_method_call(self, node);
    }
}

struct VerifiedFnAllocVisitor<'a> {
    file_path: &'a str,
    opted_out: bool,
    in_test: bool,
    violations: Vec<String>,
}

impl VerifiedFnAllocVisitor<'_> {
    fn check_fn_block(
        &mut self,
        attrs: &[syn::Attribute],
        sig: &syn::Signature,
        block: &syn::Block,
    ) {
        if self.opted_out || self.in_test || attrs_have_test_gate(attrs) {
            return;
        }
        let (has_verified, has_alloc_ok, has_opt_out) = inspect_verified_fn_attrs(attrs);
        if !has_verified || has_alloc_ok || has_opt_out {
            return;
        }
        let mut finder = BodyAllocFinder { first_hit: None };
        finder.visit_block(block);
        if let Some((construct, line)) = finder.first_hit {
            let report_line = if line > 0 {
                line
            } else {
                sig.ident.span().start().line
            };
            let loc = crate::ast_helpers::format_file_line(self.file_path, report_line);
            self.violations.push(format!(
                "[!] Unannotated heap allocation `{construct}` in `#[verified]` fn `{}` at {loc}\n  Hint: Use pre-allocated scratch buffers for zero-allocation hot paths, or annotate constructors/wrappers/formatters with `#[verified(alloc_ok = \"reason\")]`.",
                sig.ident
            ));
        }
    }
}

impl<'ast> Visit<'ast> for VerifiedFnAllocVisitor<'_> {
    fn visit_file(&mut self, node: &'ast syn::File) {
        if node.attrs.iter().any(|a| attr_has_key(a, "opt_out")) {
            self.opted_out = true;
        }
        if node.attrs.iter().any(attr_is_cfg_test) {
            self.in_test = true;
        }
        visit::visit_file(self, node);
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let prev_opt = self.opted_out;
        let prev_test = self.in_test;
        if node.attrs.iter().any(|a| attr_has_key(a, "opt_out")) {
            self.opted_out = true;
        }
        if attrs_have_test_gate(&node.attrs) {
            self.in_test = true;
        }
        visit::visit_item_mod(self, node);
        self.opted_out = prev_opt;
        self.in_test = prev_test;
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        let prev_opt = self.opted_out;
        let prev_test = self.in_test;
        if node.attrs.iter().any(|a| attr_has_key(a, "opt_out")) {
            self.opted_out = true;
        }
        if attrs_have_test_gate(&node.attrs) {
            self.in_test = true;
        }
        visit::visit_item_impl(self, node);
        self.opted_out = prev_opt;
        self.in_test = prev_test;
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.check_fn_block(&node.attrs, &node.sig, &node.block);
        let prev_test = self.in_test;
        if attrs_have_test_gate(&node.attrs) {
            self.in_test = true;
        }
        visit::visit_item_fn(self, node);
        self.in_test = prev_test;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.check_fn_block(&node.attrs, &node.sig, &node.block);
        visit::visit_impl_item_fn(self, node);
    }

    fn visit_trait_item_fn(&mut self, node: &'ast syn::TraitItemFn) {
        if let Some(ref block) = node.default {
            self.check_fn_block(&node.attrs, &node.sig, block);
        }
        visit::visit_trait_item_fn(self, node);
    }
}

fn is_out_of_line_test_file(slash_path: &str) -> bool {
    slash_path.ends_with("/tests.rs")
        || slash_path.ends_with("_tests.rs")
        || slash_path
            .rsplit('/')
            .next()
            .is_some_and(|name| name.starts_with("tests_"))
}

fn resolve_integration_binary_root(member_dir: &Path, test_file: &Path) -> Option<PathBuf> {
    let tests_dir = member_dir.join("tests");
    let rel = test_file.strip_prefix(&tests_dir).ok()?;
    let first = rel.components().next()?.as_os_str().to_str()?;
    if first.ends_with(".rs") {
        Some(tests_dir.join(first))
    } else {
        let candidate = tests_dir.join(format!("{first}.rs"));
        candidate.is_file().then_some(candidate)
    }
}

fn audit_member_test_allocators(
    member_dir: &Path,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let lib_rs = member_dir.join("src/lib.rs");
    let main_rs = member_dir.join("src/main.rs");
    let crate_root_has_alloc = [&lib_rs, &main_rs].into_iter().any(|root| {
        root.is_file()
            && cache
                .get_or_load_fs(root)
                .and_then(|e| e.ast.as_ref())
                .is_some_and(ast_has_global_allocator)
    });

    let src_dir = member_dir.join("src");
    if src_dir.is_dir() {
        for path in crate::utils::collect_rs_files_filtered(&[&src_dir], &["target"]) {
            let slash = crate::whitelist::lossy_slash_path(&path);
            let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) else {
                continue;
            };
            let mut finder = LockCallFinder {
                in_test_scope: false,
                force_test_file: is_out_of_line_test_file(&slash),
                calls: Vec::new(),
            };
            finder.visit_file(ast);
            if !crate_root_has_alloc {
                for (call_name, line) in finder.calls {
                    let loc = crate::ast_helpers::format_file_line(&slash, line);
                    violations.push(format!(
                        "[!] Test call to `{call_name}` at {loc} without `#[global_allocator]` in crate root (`{}`)\n  Hint: Install `#[cfg(test)] #[global_allocator] static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;` in crate root.",
                        member_dir.display()
                    ));
                }
            }
        }
    }

    audit_integration_test_allocators(member_dir, cache, violations);
}

fn audit_integration_test_allocators(
    member_dir: &Path,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let tests_dir = member_dir.join("tests");
    if !tests_dir.is_dir() {
        return;
    }
    for path in crate::utils::collect_rs_files_filtered(&[&tests_dir], &["target"]) {
        let slash = crate::whitelist::lossy_slash_path(&path);
        let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) else {
            continue;
        };
        let mut finder = LockCallFinder {
            in_test_scope: true,
            force_test_file: true,
            calls: Vec::new(),
        };
        finder.visit_file(ast);
        if finder.calls.is_empty() {
            continue;
        }
        let root_target =
            resolve_integration_binary_root(member_dir, &path).unwrap_or_else(|| path.clone());
        let target_has_alloc = cache
            .get_or_load_fs(&root_target)
            .and_then(|e| e.ast.as_ref())
            .is_some_and(ast_has_global_allocator);
        if !target_has_alloc {
            let root_slash = crate::whitelist::lossy_slash_path(&root_target);
            for (call_name, line) in finder.calls {
                let loc = crate::ast_helpers::format_file_line(&slash, line);
                violations.push(format!(
                    "[!] Integration test call to `{call_name}` at {loc} without `#[global_allocator]` in `{root_slash}`\n  Hint: Add `#[global_allocator] static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;` to `{root_slash}`."
                ));
            }
        }
    }
}

fn audit_member_verified_allocs(
    member_dir: &Path,
    cache: &mut AstCache,
    violations: &mut Vec<String>,
) {
    let src_dir = member_dir.join("src");
    if !src_dir.is_dir() {
        return;
    }
    for path in crate::utils::collect_rs_files_filtered(&[&src_dir], &["target"]) {
        let slash = crate::whitelist::lossy_slash_path(&path);
        if is_out_of_line_test_file(&slash) {
            continue;
        }
        let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) else {
            continue;
        };
        let mut visitor = VerifiedFnAllocVisitor {
            file_path: &slash,
            opted_out: false,
            in_test: false,
            violations: Vec::new(),
        };
        visitor.visit_file(ast);
        violations.extend(visitor.violations);
    }
}

pub(crate) fn collect_verified_alloc_violations(
    members: &[&str],
    cache: &mut AstCache,
) -> Vec<String> {
    let mut violations = Vec::new();
    for &member in members {
        let norm = member.replace('\\', "/");
        if norm == "egui_plot"
            || norm.ends_with("/egui_plot")
            || norm.contains("unified_verification")
        {
            continue;
        }
        let member_dir = crate::policy_audit::lint_policy::resolve_member_dir(member);
        audit_member_test_allocators(&member_dir, cache, &mut violations);
        audit_member_verified_allocs(&member_dir, cache, &mut violations);
    }
    violations
}

pub fn run_verified_alloc_audit_with_cache(members: &[&str], cache: &mut AstCache) -> bool {
    println!(
        "Running Verified Allocator & Zero-Allocation Role Audit across {} members...",
        members.len()
    );
    let violations = collect_verified_alloc_violations(members, cache);
    for msg in &violations {
        println!("{msg}");
    }
    violations.is_empty()
}

#[cfg(test)]
#[path = "verified_alloc_audit_tests.rs"]
mod tests;
