use regex::Regex;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::spanned::Spanned;
use syn::visit::Visit;
use walkdir::WalkDir;

pub const DEFAULT_SCAN_PATHS: &[&str] = &[
    ".github",
    "apps",
    "crates",
    "egui_plot",
    "math_explorer",
    "math_explorer_gui",
    "scripts",
    ".",
];

pub const DEFAULT_EXTENSIONS: &[&str] = &["rs", "toml", "yml", "yaml", "sh"];

#[derive(Deserialize, Debug, Clone, Default)]
pub struct WhitelistConfig {
    pub entropy_guard: Option<EntropyGuardConfig>,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct EntropyGuardConfig {
    #[serde(alias = "paths", alias = "scan_targets", alias = "targets")]
    pub scan_paths: Option<Vec<String>>,
    #[serde(alias = "file_extensions", alias = "extensions")]
    pub target_extensions: Option<Vec<String>>,
    #[serde(alias = "exemptions", alias = "path_exemptions")]
    pub allowlist: Option<Vec<String>>,
}

pub fn load_config(whitelist_path: &Path) -> EntropyGuardConfig {
    if let Ok(content) = fs::read_to_string(whitelist_path) {
        if let Ok(config) = toml::from_str::<WhitelistConfig>(&content) {
            return config.entropy_guard.unwrap_or_default();
        }
    }
    EntropyGuardConfig::default()
}

pub fn is_allowlisted(path: &Path, allowlist: &[String]) -> bool {
    let path_str = path.to_string_lossy().replace('\\', "/");
    let norm_path = path_str.trim_start_matches("./");
    for item in allowlist {
        let norm_item = item.replace('\\', "/");
        let norm_item = norm_item.trim_start_matches("./");
        if norm_path == norm_item
            || norm_path.ends_with(&format!("/{}", norm_item))
            || norm_path.starts_with(&format!("{}/", norm_item))
        {
            return true;
        }
    }
    false
}

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

pub fn calculate_entropy(s: &str) -> f64 {
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
                || resolved.ends_with("::now") && resolved.contains("SystemTime")
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

fn is_line_ignored(lines: &[&str], line_num: usize) -> bool {
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

fn scan_text_file(
    path: &Path,
    content: &str,
    aws_re: &Regex,
    gh_re: &Regex,
    slack_re: &Regex,
    all_violations: &mut Vec<String>,
) {
    let lines: Vec<&str> = content.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        let line_num = idx + 1;
        if is_line_ignored(&lines, line_num) {
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

        if let Some(secret_kind) = kind {
            let msg = format!(
                "Raw credential ({}) found in {} at line {}",
                secret_kind,
                path.display(),
                line_num
            );
            println!("[WARNING] {}", msg);
            all_violations.push(msg);
        }
    }
}

fn scan_rust_file(
    path: &Path,
    content: &str,
    aws_re: &Regex,
    gh_re: &Regex,
    slack_re: &Regex,
    all_violations: &mut Vec<String>,
) {
    if let Ok(ast) = syn::parse_file(content) {
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
            if !is_line_ignored(&lines, line_num) {
                let msg = format!(
                    "Entropy Guard Violation: Prohibited pattern '{}' found in {} at line {}",
                    name,
                    path.display(),
                    line_num
                );
                println!("[WARNING] {}", msg);
                all_violations.push(msg);
            }
        }

        for (kind, line_num) in visitor.secret_warnings {
            if !is_line_ignored(&lines, line_num) {
                let msg = format!(
                    "Raw credential ({}) found in {} at line {}",
                    kind,
                    path.display(),
                    line_num
                );
                println!("[WARNING] {}", msg);
                all_violations.push(msg);
            }
        }
    }
}

fn process_file(
    path: &Path,
    ext: &str,
    aws_re: &Regex,
    gh_re: &Regex,
    slack_re: &Regex,
    violations: &mut Vec<String>,
) {
    if let Ok(content) = fs::read_to_string(path) {
        if ext == "rs" {
            scan_rust_file(path, &content, aws_re, gh_re, slack_re, violations);
        } else {
            scan_text_file(path, &content, aws_re, gh_re, slack_re, violations);
        }
    }
}

pub fn check_entropy(members: &[String]) -> Vec<String> {
    check_entropy_with_whitelist(members, Path::new("verification_whitelist.toml"))
}

pub fn check_entropy_with_whitelist(targets: &[String], whitelist_path: &Path) -> Vec<String> {
    println!("Running Entropy Guard and Secrets Scanner...");
    let mut all_violations = Vec::new();

    let aws_re = Regex::new(r"AKIA[0-9A-Z]{16}").unwrap();
    let gh_re = Regex::new(r"ghp_[a-zA-Z0-9]{36,40}|github_pat_[a-zA-Z0-9_]{82}").unwrap();
    let slack_re = Regex::new(r"xoxb-[a-zA-Z0-9-]{10,}").unwrap();

    let config = load_config(whitelist_path);
    let scan_paths: Vec<String> = if !targets.is_empty() {
        targets.to_vec()
    } else {
        config.scan_paths.unwrap_or_else(|| {
            DEFAULT_SCAN_PATHS.iter().map(|s| s.to_string()).collect()
        })
    };

    let target_extensions: HashSet<String> = config
        .target_extensions
        .unwrap_or_else(|| DEFAULT_EXTENSIONS.iter().map(|s| s.to_string()).collect())
        .into_iter()
        .map(|ext| ext.trim_start_matches('.').to_string())
        .collect();

    let allowlist = config.allowlist.unwrap_or_default();
    let mut visited_paths: HashSet<PathBuf> = HashSet::new();

    for dir in &scan_paths {
        let root_path = Path::new(dir);
        if !root_path.exists() {
            continue;
        }

        for entry in WalkDir::new(root_path)
            .into_iter()
            .filter_entry(|e| {
                if e.file_type().is_dir() {
                    let name = e.file_name().to_string_lossy();
                    if name == "target" || name == ".git" {
                        return false;
                    }
                }
                true
            })
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let path_str = path.to_string_lossy().replace('\\', "/");
            if path_str.contains("/target/") || path_str.contains("/.git/") {
                continue;
            }

            if is_allowlisted(path, &allowlist) {
                continue;
            }

            let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
            if !target_extensions.contains(ext) {
                continue;
            }

            let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if !visited_paths.insert(key) {
                continue;
            }

            process_file(path, ext, &aws_re, &gh_re, &slack_re, &mut all_violations);
        }
    }

    all_violations
}

#[cfg(test)]
mod tests;
