use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::visit::Visit;
use walkdir::WalkDir;

#[derive(Deserialize, Debug, Clone, Default)]
pub struct WhitelistConfig {
    #[serde(default)]
    pub entropy_guard: Option<EntropyGuardConfig>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct EntropyGuardConfig {
    #[serde(default, alias = "targets", alias = "paths")]
    pub scan_paths: Option<Vec<String>>,
    #[serde(default, alias = "extensions", alias = "file_extensions")]
    pub target_extensions: Option<Vec<String>>,
    #[serde(default, alias = "exemptions", alias = "path_exemptions")]
    pub allowlist: Option<Vec<String>>,
}

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

pub const DEFAULT_TARGET_EXTENSIONS: &[&str] = &["rs", "toml", "yml", "yaml", "json", "sh"];

struct SecurityVisitor<'a> {
    aliases: HashMap<String, String>,
    entropy_violations: Vec<(String, usize)>,
    secret_warnings: Vec<(String, usize)>,
    aws_re: &'a Regex,
    gh_re: &'a Regex,
    slack_re: &'a Regex,
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
        if i == 0 {
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

            if resolved.contains("thread_rng")
                || resolved.ends_with("::thread_rng")
                || resolved == "thread_rng"
            {
                self.entropy_violations
                    .push(("thread_rng()".to_string(), line));
            } else if resolved.contains("random")
                || resolved.ends_with("::random")
                || resolved == "random"
            {
                self.entropy_violations.push(("random()".to_string(), line));
            } else if resolved.contains("SystemTime::now")
                || (resolved.ends_with("::now") && resolved.contains("SystemTime"))
            {
                self.entropy_violations
                    .push(("SystemTime::now()".to_string(), line));
            }
        }
        syn::visit::visit_expr_call(self, i);
    }

    fn visit_lit_str(&mut self, lit: &'ast syn::LitStr) {
        let value = lit.value();
        let line = lit.span().start().line;

        if (value.contains("-----BEGIN ") && value.contains("PRIVATE KEY-----"))
            || (value.contains("-----BEGIN RSA ") && value.contains("PRIVATE KEY-----"))
        {
            self.secret_warnings.push(("Private Key".to_string(), line));
        } else if self.aws_re.is_match(&value) {
            self.secret_warnings
                .push(("AWS Access Key".to_string(), line));
        } else if self.gh_re.is_match(&value) {
            self.secret_warnings
                .push(("GitHub API Token".to_string(), line));
        } else if self.slack_re.is_match(&value) {
            self.secret_warnings
                .push(("Slack API Token".to_string(), line));
        } else if value.len() > 32 {
            let is_hex_or_b64 = value
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=');
            if is_hex_or_b64 {
                let entropy = calculate_entropy(&value);
                if entropy > 4.5 {
                    self.secret_warnings
                        .push(("High-Entropy Secret".to_string(), line));
                }
            }
        }

        syn::visit::visit_lit_str(self, lit);
    }
}

pub fn find_whitelist_path() -> Option<PathBuf> {
    let candidates = [
        Path::new("verification_whitelist.toml"),
        Path::new("../../verification_whitelist.toml"),
    ];
    for p in &candidates {
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

pub fn load_whitelist_config(path: Option<&Path>) -> EntropyGuardConfig {
    let toml_path = match path {
        Some(p) => Some(p.to_path_buf()),
        None => find_whitelist_path(),
    };
    if let Some(p) = toml_path {
        if let Ok(content) = fs::read_to_string(&p) {
            if let Ok(cfg) = toml::from_str::<WhitelistConfig>(&content) {
                return cfg.entropy_guard.unwrap_or_default();
            }
        }
    }
    EntropyGuardConfig::default()
}

pub fn is_file_allowlisted(path: &Path, allowlist: &[String]) -> bool {
    if allowlist.is_empty() {
        return false;
    }
    let path_str = path.to_string_lossy().replace('\\', "/");
    let norm_path = path_str.trim_start_matches("./");
    for allowed in allowlist {
        let norm_allowed = allowed.replace('\\', "/");
        let norm_allowed = norm_allowed.trim_start_matches("./");
        if norm_path == norm_allowed
            || norm_path.ends_with(&format!("/{}", norm_allowed))
            || path.file_name().and_then(|f| f.to_str()) == Some(norm_allowed)
        {
            return true;
        }
    }
    false
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

fn is_line_ignored(line_num: usize, lines: &[&str]) -> bool {
    if line_num > 0 && line_num <= lines.len() {
        if lines[line_num - 1].contains("allow(entropy_guard)") {
            return true;
        }
        if line_num > 1 && lines[line_num - 2].contains("allow(entropy_guard)") {
            return true;
        }
    }
    false
}

fn scan_rust_file(
    path: &Path,
    aws_re: &Regex,
    gh_re: &Regex,
    slack_re: &Regex,
    violations: &mut Vec<String>,
) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let Ok(ast) = syn::parse_file(&content) else {
        return;
    };
    let mut visitor = SecurityVisitor {
        aliases: HashMap::new(),
        entropy_violations: Vec::new(),
        secret_warnings: Vec::new(),
        aws_re,
        gh_re,
        slack_re,
    };
    visitor.visit_file(&ast);

    let lines: Vec<&str> = content.lines().collect();

    for (name, line_num) in visitor.entropy_violations {
        if !is_line_ignored(line_num, &lines) {
            let msg = format!(
                "Entropy Guard Violation: Prohibited pattern '{}' found in {} at line {}",
                name,
                path.display(),
                line_num
            );
            violations.push(msg.clone());
            println!("[WARNING] {}", msg);
        }
    }

    for (kind, line_num) in visitor.secret_warnings {
        if !is_line_ignored(line_num, &lines) {
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
}

fn scan_text_file(
    path: &Path,
    aws_re: &Regex,
    gh_re: &Regex,
    slack_re: &Regex,
    violations: &mut Vec<String>,
) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };
    let lines: Vec<&str> = content.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        let line_num = idx + 1;
        if is_line_ignored(line_num, &lines) {
            continue;
        }
        let kind = if (line.contains("-----BEGIN ") && line.contains("PRIVATE KEY-----"))
            || (line.contains("-----BEGIN RSA ") && line.contains("PRIVATE KEY-----"))
        {
            Some("Private Key")
        } else if aws_re.is_match(line) {
            Some("AWS Access Key")
        } else if gh_re.is_match(line) {
            Some("GitHub API Token")
        } else if slack_re.is_match(line) {
            Some("Slack API Token")
        } else {
            None
        };

        if let Some(kind_str) = kind {
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

pub fn scan_paths_internal(
    scan_targets: &[String],
    target_extensions: &[String],
    allowlist: &[String],
) -> Vec<String> {
    let mut all_violations = Vec::new();
    let aws_re = Regex::new(r"AKIA[0-9A-Z]{16}").unwrap();
    let gh_re = Regex::new(r"ghp_[a-zA-Z0-9]{36,40}|github_pat_[a-zA-Z0-9_]{82}").unwrap();
    let slack_re = Regex::new(r"xoxb-[a-zA-Z0-9-]{10,}").unwrap();
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
                scan_rust_file(
                    entry.path(),
                    &aws_re,
                    &gh_re,
                    &slack_re,
                    &mut all_violations,
                );
            } else {
                scan_text_file(
                    entry.path(),
                    &aws_re,
                    &gh_re,
                    &slack_re,
                    &mut all_violations,
                );
            }
        }
    }

    all_violations
}

pub fn check_entropy(members: &[String]) -> Vec<String> {
    println!("Running Entropy Guard and Secrets Scanner...");
    let config = load_whitelist_config(None);
    let scan_targets: Vec<String> = if !members.is_empty() {
        members.to_vec()
    } else if let Some(ref paths) = config.scan_paths {
        if !paths.is_empty() {
            paths.clone()
        } else {
            DEFAULT_SCAN_PATHS.iter().map(|s| s.to_string()).collect()
        }
    } else {
        DEFAULT_SCAN_PATHS.iter().map(|s| s.to_string()).collect()
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
    scan_paths_internal(&scan_targets, &target_extensions, &allowlist)
}

#[cfg(test)]
mod tests;
