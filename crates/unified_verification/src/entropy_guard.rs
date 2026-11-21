use crate::whitelist::PathMatchOptions;
pub use crate::whitelist::{EntropyGuardConfig, find_whitelist_path};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use syn::spanned::Spanned;
use syn::visit::Visit;
use walkdir::WalkDir;

pub const DEFAULT_SCAN_PATHS: &[&str] = &[
    "crates",
    "apps",
    "math_explorer",
    "math_explorer_gui",
    "egui_plot",
    ".github",
    "scripts",
    "Cargo.toml",
    "traceability.toml",
    "verification_whitelist.toml",
];

pub const DEFAULT_TARGET_EXTENSIONS: &[&str] =
    &["rs", "toml", "yml", "yaml", "json", "sh", "js", "html"];

struct SecretScanner {
    aws_re: Regex,
    gh_re: Regex,
    slack_re: Regex,
    b64_re: Regex,
    b64url_re: Regex,
}

impl SecretScanner {
    fn new() -> Self {
        Self {
            aws_re: Regex::new(r"AKIA[0-9A-Z]{16}").unwrap(),
            gh_re: Regex::new(r"ghp_[a-zA-Z0-9]{36,40}|github_pat_[a-zA-Z0-9_]{82}").unwrap(),
            slack_re: Regex::new(r"xoxb-[a-zA-Z0-9-]{10,}").unwrap(),
            b64_re: Regex::new(r"^[A-Za-z0-9+/=]{33,}$").unwrap(),
            b64url_re: Regex::new(r"^[A-Za-z0-9_-]{33,}$").unwrap(),
        }
    }

    fn classify(&self, text: &str) -> Option<&'static str> {
        if (text.contains("-----BEGIN ") && text.contains("PRIVATE KEY-----"))
            || (text.contains("-----BEGIN RSA ") && text.contains("PRIVATE KEY-----"))
        {
            Some("Private Key")
        } else if self.aws_re.is_match(text) {
            Some("AWS Access Key")
        } else if self.gh_re.is_match(text) {
            Some("GitHub API Token")
        } else if self.slack_re.is_match(text) {
            Some("Slack API Token")
        } else if self.has_high_entropy_token(text) {
            Some("High-Entropy Secret")
        } else {
            None
        }
    }

    fn has_high_entropy_token(&self, text: &str) -> bool {
        text.split(is_token_delimiter)
            .filter(|raw| !raw.is_empty())
            .any(|raw| self.is_high_entropy_raw_token(raw))
    }

    fn is_high_entropy_raw_token(&self, raw: &str) -> bool {
        if is_url_or_path_token(raw) {
            return false;
        }
        let candidate = strip_key_prefix(raw).trim_end_matches('.');
        if is_excluded_token(candidate) {
            return false;
        }
        if self.matches_secret_entropy(candidate) {
            return true;
        }
        candidate.contains('.')
            && candidate
                .split('.')
                .all(|seg| !seg.is_empty() && seg.chars().all(is_b64url_char))
            && candidate
                .split('.')
                .any(|seg| !is_excluded_token(seg) && self.matches_secret_entropy(seg))
    }

    fn matches_secret_entropy(&self, token: &str) -> bool {
        (self.b64_re.is_match(token) || self.b64url_re.is_match(token))
            && calculate_entropy(token) > 4.5
    }
}

fn is_token_delimiter(c: char) -> bool {
    c.is_whitespace() || "\"'`,;()[]{}<>?&@".contains(c)
}

fn is_b64url_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

fn strip_key_prefix(tok: &str) -> &str {
    let without_pad = tok.trim_end_matches('=');
    if let Some((_, rhs)) = without_pad.rsplit_once('=') {
        let pad_len = tok.len() - without_pad.len();
        let rhs_start = tok.len() - rhs.len() - pad_len;
        return &tok[rhs_start..];
    }
    if let Some((_, rhs)) = tok.rsplit_once(':') {
        return rhs;
    }
    tok
}

fn is_pure_hex(s: &str) -> bool {
    let hex = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit())
}

fn is_integrity_digest(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower.starts_with("sha256-") || lower.starts_with("sha384-") || lower.starts_with("sha512-")
}

fn is_url_or_path_token(raw: &str) -> bool {
    raw.contains("://")
        || raw.starts_with('/')
        || raw.starts_with("./")
        || raw.starts_with("../")
        || raw.ends_with('/')
        || raw.contains("//")
}

fn is_excluded_token(candidate: &str) -> bool {
    candidate.len() < 33
        || is_pure_hex(candidate)
        || is_integrity_digest(candidate)
        || is_url_or_path_token(candidate)
        || (candidate.contains('/') && !candidate.contains('+') && !candidate.ends_with('='))
}

struct SecurityVisitor<'a> {
    aliases: HashMap<String, String>,
    entropy_violations: Vec<(String, usize)>,
    secret_warnings: Vec<(String, usize)>,
    scanner: &'a SecretScanner,
}

fn flatten_use_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    mappings: &mut HashMap<String, String>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            flatten_use_tree(&p.tree, prefix, mappings);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let mut full_path = prefix.clone();
            full_path.push(n.ident.to_string());
            mappings.insert(n.ident.to_string(), full_path.join("::"));
        }
        syn::UseTree::Rename(r) => {
            let mut full_path = prefix.clone();
            full_path.push(r.ident.to_string());
            mappings.insert(r.rename.to_string(), full_path.join("::"));
        }
        syn::UseTree::Group(g) => {
            for item in &g.items {
                flatten_use_tree(item, prefix, mappings);
            }
        }
        syn::UseTree::Glob(_) => {}
    }
}

fn resolve_path(p: &syn::Path, mappings: &HashMap<String, String>) -> String {
    let mut parts = Vec::new();
    for (i, seg) in p.segments.iter().enumerate() {
        if i == 0 && p.leading_colon.is_none() {
            let first = seg.ident.to_string();
            if let Some(mapped) = mappings.get(&first) {
                parts.push(mapped.clone());
            } else {
                parts.push(first);
            }
        } else {
            parts.push(seg.ident.to_string());
        }
    }
    parts.join("::")
}

fn calculate_entropy(s: &str) -> f64 {
    let mut counts = HashMap::new();
    for c in s.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let len = s.len() as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / len;
        entropy -= p * p.log2();
    }
    entropy
}

impl<'a, 'ast> Visit<'ast> for SecurityVisitor<'a> {
    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        let mut prefix = Vec::new();
        flatten_use_tree(&i.tree, &mut prefix, &mut self.aliases);
        syn::visit::visit_item_use(self, i);
    }

    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        if let syn::Expr::Path(p) = &*i.func {
            let resolved = resolve_path(&p.path, &self.aliases);
            let line = i.func.span().start().line;

            if resolved == "thread_rng" || resolved.ends_with("::thread_rng") {
                self.entropy_violations
                    .push(("thread_rng()".to_string(), line));
            } else if resolved == "random" || resolved.ends_with("::random") {
                self.entropy_violations.push(("random()".to_string(), line));
            } else if resolved == "SystemTime::now" || resolved.ends_with("::SystemTime::now") {
                self.entropy_violations
                    .push(("SystemTime::now()".to_string(), line));
            }
        }
        syn::visit::visit_expr_call(self, i);
    }

    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        let value = lit.value();
        let line = lit.span().start().line;
        if let Some(kind) = self.scanner.classify(&value) {
            self.secret_warnings.push((kind.to_string(), line));
        }
        syn::visit::visit_lit_str(self, lit);
    }
}

pub fn load_whitelist_config(path: Option<&Path>) -> Result<EntropyGuardConfig, String> {
    let resolved = path.map(Path::to_path_buf).or_else(find_whitelist_path);
    crate::whitelist::load_whitelist_strict(resolved.as_deref(), &[] as &[&Path])
        .map(|cfg| cfg.entropy_guard.unwrap_or_default())
}

pub fn is_file_allowlisted(path: &Path, allowlist: &[String]) -> bool {
    let opts = PathMatchOptions {
        trim_dot_slash: true,
        require_non_empty: false,
        match_file_name: true,
    };
    allowlist
        .iter()
        .any(|allowed| crate::whitelist::path_matches_lossy(path, allowed, opts))
}

fn is_ignored_dir(entry: &walkdir::DirEntry) -> bool {
    if entry.file_type().is_dir() {
        let name = entry.file_name().to_string_lossy();
        if name == "target" || name == ".git" {
            return true;
        }
    }
    false
}

fn scan_rust_file(
    path: &Path,
    scanner: &SecretScanner,
    violations: &mut Vec<String>,
    cache: &mut oxidize_core::traceability::AstCache,
) {
    let Some(ast) = cache.get_or_load_fs(path).and_then(|e| e.ast.as_ref()) else {
        return;
    };
    let mut visitor = SecurityVisitor {
        aliases: HashMap::new(),
        entropy_violations: Vec::new(),
        secret_warnings: Vec::new(),
        scanner,
    };
    visitor.visit_file(ast);

    for (name, line_num) in visitor.entropy_violations {
        let msg = format!(
            "Entropy Guard Violation: Prohibited pattern '{}' found in {} at line {}",
            name,
            path.display(),
            line_num
        );
        violations.push(msg.clone());
        println!("[WARNING] {}", msg);
    }

    for (kind, line_num) in visitor.secret_warnings {
        let msg = format!(
            "Raw credential ({}) found in {} at line {}",
            kind,
            path.display(),
            line_num
        );
        violations.push(msg.clone());
        println!("[WARNING] {}", msg);
    }
}

fn scan_text_file(path: &Path, scanner: &SecretScanner, violations: &mut Vec<String>) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    for (idx, line) in content.lines().enumerate() {
        let line_num = idx + 1;
        if let Some(kind_str) = scanner.classify(line) {
            let msg = format!(
                "Raw credential ({}) found in {} at line {}",
                kind_str,
                path.display(),
                line_num
            );
            violations.push(msg.clone());
            println!("[WARNING] {}", msg);
        }
    }
}

#[cfg(test)]
pub fn scan_paths_internal(
    scan_targets: &[String],
    target_extensions: &[String],
    allowlist: &[String],
) -> Vec<String> {
    let mut cache = oxidize_core::traceability::AstCache::new();
    scan_paths_internal_with_cache(scan_targets, target_extensions, allowlist, &mut cache)
}

pub fn scan_paths_internal_with_cache(
    scan_targets: &[String],
    target_extensions: &[String],
    allowlist: &[String],
    cache: &mut oxidize_core::traceability::AstCache,
) -> Vec<String> {
    let mut all_violations = Vec::new();
    let scanner = SecretScanner::new();
    let mut visited = HashSet::new();

    for target in scan_targets {
        let p = Path::new(target);
        if !p.exists() {
            continue;
        }
        for entry in WalkDir::new(target)
            .into_iter()
            .filter_entry(|e| !is_ignored_dir(e))
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path_str = entry.path().to_string_lossy().replace('\\', "/");
            if path_str
                .split('/')
                .any(|seg| seg == "target" || seg == ".git")
            {
                continue;
            }
            let can = entry
                .path()
                .canonicalize()
                .unwrap_or_else(|_| entry.path().to_path_buf());
            if !visited.insert(can) {
                continue;
            }
            if is_file_allowlisted(entry.path(), allowlist) {
                continue;
            }
            let ext = entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let is_matched_ext = target_extensions
                .iter()
                .any(|e| e.trim_start_matches('.').eq_ignore_ascii_case(ext));
            if !is_matched_ext {
                continue;
            }

            if ext.eq_ignore_ascii_case("rs") {
                scan_rust_file(entry.path(), &scanner, &mut all_violations, cache);
            } else {
                scan_text_file(entry.path(), &scanner, &mut all_violations);
            }
        }
    }

    all_violations
}

pub fn check_entropy(members: &[String]) -> Vec<String> {
    let mut cache = oxidize_core::traceability::AstCache::new();
    check_entropy_with_cache(members, &mut cache)
}

pub fn check_entropy_with_cache(
    members: &[String],
    cache: &mut oxidize_core::traceability::AstCache,
) -> Vec<String> {
    println!("Running Entropy Guard and Secrets Scanner...");
    let config = match load_whitelist_config(None) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[!] {}", err);
            return vec![format!("Entropy Guard Configuration Error: {}", err)];
        }
    };
    let scan_targets: Vec<String> = if !members.is_empty() {
        members.to_vec()
    } else {
        config
            .scan_paths
            .as_ref()
            .filter(|p| !p.is_empty())
            .cloned()
            .unwrap_or_else(|| DEFAULT_SCAN_PATHS.iter().map(|s| s.to_string()).collect())
    };

    let target_extensions: Vec<String> = config
        .target_extensions
        .as_ref()
        .filter(|exts| !exts.is_empty())
        .cloned()
        .unwrap_or_else(|| {
            DEFAULT_TARGET_EXTENSIONS
                .iter()
                .map(|s| s.to_string())
                .collect()
        });

    let allowlist: Vec<String> = config.allowlist.unwrap_or_default();
    scan_paths_internal_with_cache(&scan_targets, &target_extensions, &allowlist, cache)
}

#[cfg(test)]
mod tests;
