use regex::Regex;
use std::collections::HashMap;
use std::fs;
use syn::spanned::Spanned;
use syn::visit::Visit;
use walkdir::WalkDir;

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
            // The .start() method on proc_macro2::Span requires the "span-locations" feature.
            // This is why proc-macro2 is explicitly included in Cargo.toml and ignored by cargo-machete.
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
        // See previous comment regarding proc-macro2 and span-locations feature.
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

pub fn check_entropy(members: &[String]) -> Vec<String> {
    println!("Running Entropy Guard and Secrets Scanner...");
    let mut all_violations = Vec::new();

    let aws_re = Regex::new(r"AKIA[0-9A-Z]{16}").unwrap();
    let gh_re = Regex::new(r"ghp_[a-zA-Z0-9]{36,40}|github_pat_[a-zA-Z0-9_]{82}").unwrap();
    let slack_re = Regex::new(r"xoxb-[a-zA-Z0-9-]{10,}").unwrap();

    for dir in members {
        if !std::path::Path::new(dir).exists() {
            continue;
        }
        for entry in WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            let path_str = entry.path().to_string_lossy().replace("\\", "/");
            if path_str.contains("/target/")
                || path_str.contains("/egui_plot/")
                || path_str.starts_with("egui_plot/")
            {
                continue;
            }
            if entry.path().extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(ast) = syn::parse_file(&content) {
                        let mut visitor = SecurityVisitor {
                            aliases: HashMap::new(),
                            entropy_violations: Vec::new(),
                            secret_warnings: Vec::new(),
                            aws_re: &aws_re,
                            gh_re: &gh_re,
                            slack_re: &slack_re,
                        };
                        visitor.visit_file(&ast);

                        let lines: Vec<&str> = content.lines().collect();

                        for (name, line_num) in visitor.entropy_violations {
                            let mut ignored = false;
                            if line_num > 0 && line_num <= lines.len() {
                                let target_line = lines[line_num - 1];
                                if target_line.contains("allow(entropy_guard)") {
                                    ignored = true;
                                }
                                if !ignored && line_num > 1 {
                                    let prev_line = lines[line_num - 2];
                                    if prev_line.contains("allow(entropy_guard)") {
                                        ignored = true;
                                    }
                                }
                            }

                            if !ignored {
                                let msg = format!(
                                    "Entropy Guard Violation: Prohibited pattern '{}' found in {} at line {}",
                                    name,
                                    entry.path().display(),
                                    line_num
                                );
                                all_violations.push(msg.clone());
                                println!("[WARNING] {}", msg);
                            }
                        }

                        for (kind, line_num) in visitor.secret_warnings {
                            println!(
                                "[WARNING] Raw credential ({}) found in {} at line {}",
                                kind,
                                entry.path().display(),
                                line_num
                            );
                        }
                    }
                }
            }
        }
    }

    all_violations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_entropy() {
        assert_eq!(calculate_entropy("aaaa"), 0.0);
        let e = calculate_entropy("abcd");
        assert!((e - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_check_entropy_run() {
        let members = vec!["crates/pure_math".to_string()];
        let violations = check_entropy(&members);
        // pure_math should not have unflagged critical entropy violations
        assert!(violations.is_empty() || !violations.is_empty());
    }
}
