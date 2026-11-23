use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::spanned::Spanned;
use syn::visit::Visit;

#[derive(Default)]
struct MaliciousCodeVisitor {
    has_shell_execution: bool,
    has_network_socket: bool,
    has_suspicious_fs: bool,
}

impl<'ast> Visit<'ast> for MaliciousCodeVisitor {
    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*i.func {
            let path_str = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            if path_str.contains("Command::new") {
                self.has_shell_execution = true;
            }
            if path_str.contains("TcpStream::connect") || path_str.contains("UdpSocket::bind") {
                self.has_network_socket = true;
            }
            if path_str.contains("remove_dir_all") {
                self.has_suspicious_fs = true;
            }
        }
        syn::visit::visit_expr_call(self, i);
    }
}

#[derive(Default)]
struct CloneDetectorVisitor {
    index_clones: Vec<(Option<String>, usize)>,
    color_clones: Vec<(String, usize)>,
    current_fn: Option<String>,
    cast_u8_count: usize,
    mul_255_count: usize,
}

impl CloneDetectorVisitor {
    fn visit_fn_scope<F>(&mut self, sig: &syn::Signature, visit_inner: F)
    where
        F: FnOnce(&mut Self),
    {
        let fn_name = sig.ident.to_string();
        let fn_line = sig.span().start().line;
        let prev_fn = self.current_fn.replace(fn_name.clone());
        let prev_casts = std::mem::take(&mut self.cast_u8_count);
        let prev_muls = std::mem::take(&mut self.mul_255_count);

        visit_inner(self);

        if self.cast_u8_count >= 3 && self.mul_255_count >= 3 {
            self.color_clones.push((fn_name, fn_line));
        }

        self.current_fn = prev_fn;
        self.cast_u8_count = prev_casts;
        self.mul_255_count = prev_muls;
    }

    fn is_a_b_plus_c(expr: &syn::Expr) -> bool {
        if let syn::Expr::Binary(i) = expr
            && let syn::BinOp::Add(_) = i.op
        {
            let left_is_mul =
                matches!(*i.left, syn::Expr::Binary(ref b) if matches!(b.op, syn::BinOp::Mul(_)));
            let right_is_mul =
                matches!(*i.right, syn::Expr::Binary(ref b) if matches!(b.op, syn::BinOp::Mul(_)));

            if left_is_mul || right_is_mul {
                let is_ident = |expr: &syn::Expr| matches!(expr, syn::Expr::Path(p) if p.path.get_ident().is_some());

                let (mul_expr, other_expr) = if left_is_mul {
                    (&*i.left, &*i.right)
                } else {
                    (&*i.right, &*i.left)
                };

                if let syn::Expr::Binary(b) = mul_expr
                    && is_ident(&b.left)
                    && is_ident(&b.right)
                    && is_ident(other_expr)
                {
                    return true;
                }
            }
        }
        false
    }

    // Recursive check if any sub-expression contains A * B + C
    fn contains_a_b_plus_c(expr: &syn::Expr) -> bool {
        if Self::is_a_b_plus_c(expr) {
            return true;
        }
        // Very basic recursion for common wrappers like parenthesis
        match expr {
            syn::Expr::Paren(p) => Self::contains_a_b_plus_c(&p.expr),
            syn::Expr::Block(b) => {
                if let Some(syn::Stmt::Expr(e, _)) = b.block.stmts.last() {
                    Self::contains_a_b_plus_c(e)
                } else {
                    false
                }
            }
            syn::Expr::Cast(c) => Self::contains_a_b_plus_c(&c.expr),
            _ => false,
        }
    }
}

impl<'ast> Visit<'ast> for CloneDetectorVisitor {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.visit_fn_scope(&i.sig, |v| syn::visit::visit_item_fn(v, i));
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        self.visit_fn_scope(&i.sig, |v| syn::visit::visit_impl_item_fn(v, i));
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        self.visit_fn_scope(&i.sig, |v| syn::visit::visit_trait_item_fn(v, i));
    }

    fn visit_expr_index(&mut self, i: &'ast syn::ExprIndex) {
        if Self::contains_a_b_plus_c(&i.index) {
            self.index_clones
                .push((self.current_fn.clone(), i.span().start().line));
        }
        syn::visit::visit_expr_index(self, i);
    }

    fn visit_expr_binary(&mut self, i: &'ast syn::ExprBinary) {
        if let syn::BinOp::Mul(_) = i.op {
            let is_255 = |expr: &syn::Expr| {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Float(f),
                    ..
                }) = expr
                {
                    f.base10_parse::<f64>().unwrap_or(0.0) == 255.0
                } else if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(int_lit),
                    ..
                }) = expr
                {
                    int_lit.base10_parse::<u32>().unwrap_or(0) == 255
                } else {
                    false
                }
            };
            if is_255(&i.left) || is_255(&i.right) {
                self.mul_255_count += 1;
            }
        }
        syn::visit::visit_expr_binary(self, i);
    }

    fn visit_expr_cast(&mut self, i: &'ast syn::ExprCast) {
        if let syn::Type::Path(p) = &*i.ty
            && p.path.is_ident("u8")
        {
            self.cast_u8_count += 1;
        }
        syn::visit::visit_expr_cast(self, i);
    }
}

pub(crate) fn format_clone_findings(
    path_str: &str,
    index_clones: &[(Option<String>, usize)],
    color_clones: &[(String, usize)],
) -> Vec<String> {
    let mut findings = Vec::new();
    for (fn_name, line) in index_clones {
        let loc = crate::ast_helpers::format_file_line(path_str, *line);
        let scope = fn_name
            .as_deref()
            .map_or_else(|| "<module-scope>".to_string(), |f| format!("fn `{f}`"));
        findings.push(format!(
            "[!] Code Clone Detected: 2D Grid Indexing pattern at {loc} ({scope})\n  Hint: Replace manual `row * width + col` indexing with `math_commons::math_kernel::types::flatten_2d_index`."
        ));
    }
    for (fn_name, line) in color_clones {
        let loc = crate::ast_helpers::format_file_line(path_str, *line);
        findings.push(format!(
            "[!] Code Clone Detected: Scalar-to-Color RGB conversion pattern at {loc} (fn `{fn_name}`)\n  Hint: Delegate RGB channel conversion to `math_commons::math_kernel::colormap` helpers instead of duplicating `* 255` and `as u8` casts."
        ));
    }
    findings
}

pub(crate) fn detect_clones_in_ast(path: &Path, ast: &syn::File) -> Vec<String> {
    let mut clone_visitor = CloneDetectorVisitor::default();
    clone_visitor.visit_file(ast);
    let path_str = crate::whitelist::lossy_slash_path(path);
    format_clone_findings(
        &path_str,
        &clone_visitor.index_clones,
        &clone_visitor.color_clones,
    )
}

pub fn run_clone_detector() -> bool {
    let mut cache = oxidize_core::traceability::AstCache::new();
    run_clone_detector_with_cache(&mut cache)
}

pub fn run_clone_detector_with_cache(cache: &mut oxidize_core::traceability::AstCache) -> bool {
    println!("Running Clone Detector on first-party repository...");
    let mut flags_found = 0;
    let first_party_dirs = ["crates", "apps", "math_explorer", "math_explorer_gui"];
    for path in crate::utils::collect_rs_files_filtered(&first_party_dirs, &["target", "egui_plot"])
    {
        let path_str = crate::whitelist::lossy_slash_path(&path);
        if path_str.contains("unified_verification")
            || path_str.contains("colormap.rs")
            || path_str.contains("types.rs")
        {
            continue;
        }

        if let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) {
            let findings = detect_clones_in_ast(&path, ast);
            for msg in &findings {
                println!("{msg}");
            }
            flags_found += findings.len();
        }
    }

    println!("Clone detector finished. Flagged {} patterns.", flags_found);
    flags_found == 0
}

const MALICIOUS_PATH_PAIRS: &[(&str, &str)] = &[
    ("Command", "new"),
    ("TcpStream", "connect"),
    ("UdpSocket", "bind"),
];

fn resolve_target_triple(
    cargo_build_target: Option<&str>,
    rustc_vv: Option<&str>,
) -> Option<String> {
    if let Some(target) = cargo_build_target.map(str::trim).filter(|t| !t.is_empty()) {
        return Some(target.to_string());
    }
    rustc_vv?
        .lines()
        .find_map(|line| line.strip_prefix("host:"))
        .map(str::trim)
        .filter(|triple| !triple.is_empty())
        .map(ToString::to_string)
}

fn detect_active_target_triple() -> Option<String> {
    let env_target = env::var("CARGO_BUILD_TARGET").ok();
    if let Some(target) = resolve_target_triple(env_target.as_deref(), None) {
        return Some(target);
    }
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(rustc).arg("-vV").output().ok()?;
    if !out.status.success() {
        return None;
    }
    resolve_target_triple(None, Some(&String::from_utf8_lossy(&out.stdout)))
}

fn collect_crates_io_dep_paths() -> Vec<PathBuf> {
    let meta = detect_active_target_triple()
        .and_then(|triple| {
            Command::new("cargo")
                .args([
                    "metadata",
                    "--format-version",
                    "1",
                    "--filter-platform",
                    &triple,
                ])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .and_then(|out| serde_json::from_slice(&out.stdout).ok())
        })
        .or_else(|| {
            Command::new("cargo")
                .args(["metadata", "--format-version", "1"])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .and_then(|out| serde_json::from_slice(&out.stdout).ok())
        })
        .unwrap_or_default();
    extract_crates_io_dep_paths(&meta)
}

fn extract_crates_io_dep_paths(meta: &serde_json::Value) -> Vec<PathBuf> {
    let mut deps_paths = Vec::new();
    let Some(packages) = meta.get("packages").and_then(|p| p.as_array()) else {
        return deps_paths;
    };
    for pkg in packages {
        if let Some(manifest_path) = pkg.get("manifest_path").and_then(|m| m.as_str())
            && let Some(source) = pkg.get("source").and_then(|s| s.as_str())
            && source.contains("crates.io")
            && let Some(parent) = PathBuf::from(manifest_path).parent()
        {
            deps_paths.push(parent.to_path_buf());
        }
    }
    deps_paths
}

fn has_candidate_path_pair(content: &str, lhs: &str, rhs: &str) -> bool {
    for (idx, _) in content.match_indices(lhs) {
        let rest = content[idx + lhs.len()..].trim_start();
        if rest.starts_with("//")
            || rest.starts_with("/*")
            || (rest.starts_with('<') && !rest.starts_with("</"))
        {
            return true;
        }
        let Some(after_sep) = rest.strip_prefix("::") else {
            continue;
        };
        let after_sep = after_sep.trim_start();
        if after_sep.starts_with("//")
            || after_sep.starts_with("/*")
            || after_sep.starts_with('<')
            || after_sep
                .strip_prefix("r#")
                .unwrap_or(after_sep)
                .starts_with(rhs)
        {
            return true;
        }
    }
    false
}

fn may_contain_malicious_call(content: &str) -> bool {
    content.contains("remove_dir_all")
        || MALICIOUS_PATH_PAIRS
            .iter()
            .any(|&(lhs, rhs)| has_candidate_path_pair(content, lhs, rhs))
}

fn scan_rs_file_for_malicious_calls(path: &Path) -> usize {
    let Ok(content) = fs::read_to_string(path) else {
        return 0;
    };
    if !may_contain_malicious_call(&content) {
        return 0;
    }
    let Ok(ast) = syn::parse_file(&content) else {
        return 0;
    };
    let mut visitor = MaliciousCodeVisitor::default();
    visitor.visit_file(&ast);

    let mut flags = 0;
    for (matched, label) in [
        (visitor.has_shell_execution, "1: Shell execution found"),
        (
            visitor.has_network_socket,
            "2: Network socket pattern found",
        ),
        (visitor.has_suspicious_fs, "3: Suspicious FS access found"),
    ] {
        if matched {
            println!("[!] Malicious pattern {} in {}", label, path.display());
            flags += 1;
        }
    }
    flags
}

/// Scans active-target `crates.io` dependency sources for suspicious shell, network,
/// or filesystem AST call patterns after a zero-false-negative text pre-filter.
pub fn run_ast_visitor() -> bool {
    println!("Running AST visitor on 100% of third-party dependencies...");

    let mut flags_found = 0;
    let dep_paths = collect_crates_io_dep_paths();
    for rs_file in crate::utils::collect_rs_files_filtered(&dep_paths, &[]) {
        flags_found += scan_rs_file_for_malicious_calls(&rs_file);
    }

    println!("AST visitor finished. Flagged {} patterns.", flags_found);
    true // Non-blocking for third-party crates
}

#[cfg(test)]
mod tests;
