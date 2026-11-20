use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

const INNER_BLOCKED_LINTS: &[&str] = &["all", "too_many_lines", "cognitive_complexity"];
const MOD_EXTRA_BLOCKED_LINTS: &[&str] = &["too_many_lines", "cognitive_complexity"];
const ITEM_BLOCKED_LINTS: &[&str] = &["all"];
const MAX_ATTR_STACK_STEPS: usize = 256;

fn cwd_is_workspace_root() -> bool {
    fs::read_to_string("Cargo.toml")
        .map(|c| c.contains("[workspace]"))
        .unwrap_or(false)
}

pub(crate) fn resolve_whitelist_path(whitelist_path: &Path) -> PathBuf {
    if whitelist_path.exists() || whitelist_path.is_absolute() || cwd_is_workspace_root() {
        return whitelist_path.to_path_buf();
    }
    let candidate = Path::new("../..").join(whitelist_path);
    if candidate.exists() {
        candidate
    } else {
        whitelist_path.to_path_buf()
    }
}

pub(crate) fn resolve_member_dir(member: &str) -> PathBuf {
    let direct = PathBuf::from(member);
    if direct.join("Cargo.toml").exists() || direct.is_absolute() || cwd_is_workspace_root() {
        return direct;
    }
    let candidate = Path::new("../..").join(member);
    if candidate.join("Cargo.toml").exists() {
        candidate
    } else {
        direct
    }
}

pub(crate) fn is_vendored_egui_plot(member: &str) -> bool {
    let norm = member.replace('\\', "/");
    let trimmed = norm.trim_end_matches('/');
    trimmed == "egui_plot" || trimmed.ends_with("/egui_plot")
}

pub(crate) fn has_explicit_egui_plot_lints(parsed: &toml::Value) -> bool {
    let Some(lints) = parsed.get("lints").and_then(toml::Value::as_table) else {
        return false;
    };
    if lints.contains_key("workspace") {
        return false;
    }
    let has_rust = lints
        .get("rust")
        .and_then(toml::Value::as_table)
        .is_some_and(|t| !t.is_empty());
    let has_clippy = lints
        .get("clippy")
        .and_then(toml::Value::as_table)
        .is_some_and(|t| !t.is_empty());
    has_rust && has_clippy
}

pub(crate) fn is_vendored_egui_plot_exempt(member: &str, parsed: &toml::Value) -> bool {
    is_vendored_egui_plot(member) && has_explicit_egui_plot_lints(parsed)
}

pub(crate) fn has_workspace_lints_inheritance(parsed: &toml::Value) -> bool {
    parsed
        .get("lints")
        .and_then(toml::Value::as_table)
        .is_some_and(|t| {
            t.len() == 1 && t.get("workspace").and_then(toml::Value::as_bool) == Some(true)
        })
}

pub(crate) fn check_manifest_lints(
    member: &str,
    manifest_path: &Path,
    parsed: &toml::Value,
) -> bool {
    if is_vendored_egui_plot(member) {
        if has_explicit_egui_plot_lints(parsed) || has_workspace_lints_inheritance(parsed) {
            return true;
        }
        eprintln!(
            "[!] Lint policy mismatch in {}: vendored egui_plot must define explicit `[lints.rust]` and `[lints.clippy]` tables (or `[lints] workspace = true`)",
            manifest_path.display()
        );
        return false;
    }

    if has_workspace_lints_inheritance(parsed) {
        return true;
    }

    eprintln!(
        "[!] Lint policy mismatch in {}: first-party crates must inherit workspace lints via `[lints] workspace = true`",
        manifest_path.display()
    );
    false
}

fn matched_clippy_lint(path: &syn::Path, names: &[&'static str]) -> Option<&'static str> {
    if path.segments.len() != 2
        || path.segments.iter().any(|seg| !seg.arguments.is_empty())
        || path.segments[0].ident != "clippy"
    {
        return None;
    }
    let target = &path.segments[1].ident;
    names.iter().copied().find(|&name| target == name)
}

pub(crate) fn find_suppressed_clippy_lint(
    attr: &syn::Attribute,
    names: &[&'static str],
) -> Option<&'static str> {
    let mut stack = vec![attr.meta.clone()];
    let mut steps = 0usize;
    while let Some(meta) = stack.pop() {
        if steps >= MAX_ATTR_STACK_STEPS {
            break;
        }
        steps += 1;
        let syn::Meta::List(list) = meta else {
            continue;
        };
        if list.path.is_ident("allow") || list.path.is_ident("expect") {
            let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) else {
                continue;
            };
            for item in nested {
                if let syn::Meta::Path(lint_path) = item
                    && let Some(found) = matched_clippy_lint(&lint_path, names)
                {
                    return Some(found);
                }
            }
        } else if list.path.is_ident("cfg_attr") {
            let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) else {
                continue;
            };
            for item in nested.into_iter().skip(1) {
                stack.push(item);
            }
        }
    }
    None
}

pub(crate) struct ClippyBypassVisitor {
    pub(crate) file_path: String,
    pub(crate) has_unauthorized: bool,
}

impl ClippyBypassVisitor {
    fn report_violation(&mut self, scope: &str, lint_name: &str) {
        println!(
            "[!] Unauthorized Clippy bypass in {} ({scope}: clippy::{lint_name})",
            self.file_path
        );
        self.has_unauthorized = true;
    }
}

impl<'ast> Visit<'ast> for ClippyBypassVisitor {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let mod_label = format!("mod `{}` attribute", i.ident);
        for attr in &i.attrs {
            if matches!(attr.style, syn::AttrStyle::Outer)
                && let Some(lint) = find_suppressed_clippy_lint(attr, MOD_EXTRA_BLOCKED_LINTS)
            {
                self.report_violation(&mod_label, lint);
            }
        }
        syn::visit::visit_item_mod(self, i);
    }

    fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
        match i.style {
            syn::AttrStyle::Inner(_) => {
                if let Some(lint) = find_suppressed_clippy_lint(i, INNER_BLOCKED_LINTS) {
                    self.report_violation("inner `#![...]` attribute", lint);
                }
            }
            syn::AttrStyle::Outer => {
                if let Some(lint) = find_suppressed_clippy_lint(i, ITEM_BLOCKED_LINTS) {
                    self.report_violation("item `#[...]` attribute", lint);
                }
            }
        }
        syn::visit::visit_attribute(self, i);
    }
}
