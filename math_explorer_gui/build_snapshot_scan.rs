//! Active Rust module tree and citation scanner for `build_snapshot.rs`.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::path_utils;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const CORE_DEP_CRATES: &[&str] = &[
    "diagnostics",
    "federated_registry",
    "math_commons",
    "oxidize_core",
    "scientific_metadata",
];

const FEATURE_DEP_CRATES: &[(&str, &str)] = &[
    ("AI", "crates/domain_ai/src"),
    ("APPLIED", "crates/domain_applied/src"),
    ("BIOLOGY", "crates/domain_biology/src"),
    ("CLIMATE", "crates/domain_climate/src"),
    ("EPIDEMIOLOGY", "crates/domain_epidemiology/src"),
    ("MATH_EXPLORER", "math_explorer/src"),
    ("PHYSICS", "crates/domain_physics/src"),
    ("PURE_MATH", "crates/pure_math/src"),
];

#[derive(Debug)]
struct SubmoduleDecl {
    ident: String,
    custom_path: Option<String>,
    feature_gate: Option<String>,
}

/// Emits a normalized forward-slash `cargo:rerun-if-changed` directive when non-empty.
pub(super) fn emit_rerun(path: &Path) {
    let normalized = path_utils::normalize_path(path);
    if !normalized.trim().is_empty() {
        println!("cargo:rerun-if-changed={normalized}");
    }
}

fn is_cargo_feature_enabled(feature: &str) -> bool {
    let key = format!(
        "CARGO_FEATURE_{}",
        feature.replace('-', "_").to_ascii_uppercase()
    );
    env::var_os(key).is_some()
}

/// Discovers active `.rs` files across Cargo-feature-enabled dependency crates and GUI tabs.
pub(super) fn discover_workspace_active_files(ws_root: &Path) -> BTreeMap<PathBuf, String> {
    let mut roots: Vec<(PathBuf, String, bool)> = Vec::new();
    for crate_name in CORE_DEP_CRATES {
        let repo_src = format!("crates/{crate_name}/src");
        roots.push((ws_root.join(&repo_src), repo_src, true));
    }
    for (feat, repo_src) in FEATURE_DEP_CRATES {
        if env::var_os(format!("CARGO_FEATURE_{feat}")).is_some() {
            roots.push((ws_root.join(repo_src), (*repo_src).to_string(), true));
        }
    }
    roots.push((
        Path::new("src").join("tabs"),
        "math_explorer_gui/src/tabs".to_string(),
        false,
    ));

    let mut active = BTreeMap::new();
    for (disk_dir, repo_dir, watch_root_dir) in roots {
        if disk_dir.is_dir() {
            if watch_root_dir {
                emit_rerun(&disk_dir);
            }
            gather_dir_files(&disk_dir, &repo_dir, &mut active);
        }
    }
    active
}

fn gather_dir_files(disk_dir: &Path, repo_dir: &str, active: &mut BTreeMap<PathBuf, String>) {
    for entry_name in ["lib.rs", "main.rs", "mod.rs"] {
        let entry_disk = disk_dir.join(entry_name);
        if entry_disk.is_file() {
            parse_module_tree(&entry_disk, &format!("{repo_dir}/{entry_name}"), active);
        }
    }
}

fn parse_module_tree(disk_file: &Path, repo_file: &str, active: &mut BTreeMap<PathBuf, String>) {
    let key = PathBuf::from(path_utils::normalize_path(disk_file));
    if active.contains_key(&key) {
        return;
    }
    let Ok(content) = fs::read_to_string(disk_file) else {
        return;
    };
    emit_rerun(disk_file);
    active.insert(key, repo_file.to_string());

    let parent_disk = disk_file.parent().unwrap_or_else(|| Path::new("."));
    let parent_repo = repo_file.rsplit_once('/').map_or("", |(p, _)| p);
    let (child_disk, child_repo) = module_search_dir(disk_file, repo_file);

    for decl in extract_external_submodules(&content) {
        if let Some(feat) = &decl.feature_gate {
            if !is_cargo_feature_enabled(feat) {
                continue;
            }
        }
        if let Some(rel_path) = &decl.custom_path {
            let clean = rel_path.trim_start_matches("./");
            let cand_disk = parent_disk.join(clean);
            let cand_repo = join_repo_path(parent_repo, clean);
            if cand_disk.is_file() {
                parse_module_tree(&cand_disk, &cand_repo, active);
            }
            continue;
        }
        let cand_rs = child_disk.join(format!("{}.rs", decl.ident));
        let cand_mod = child_disk.join(&decl.ident).join("mod.rs");
        if cand_rs.is_file() {
            let repo_rs = join_repo_path(&child_repo, &format!("{}.rs", decl.ident));
            parse_module_tree(&cand_rs, &repo_rs, active);
        } else if cand_mod.is_file() {
            let repo_mod = join_repo_path(&child_repo, &format!("{}/mod.rs", decl.ident));
            parse_module_tree(&cand_mod, &repo_mod, active);
        }
    }
}

fn join_repo_path(base: &str, rel: &str) -> String {
    let norm_rel = rel.replace('\\', "/");
    if base.is_empty() {
        norm_rel
    } else {
        format!("{base}/{norm_rel}")
    }
}

fn module_search_dir(disk_file: &Path, repo_file: &str) -> (PathBuf, String) {
    let file_name = disk_file.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let parent_disk = disk_file.parent().unwrap_or_else(|| Path::new("."));
    let repo_parent_base = repo_file
        .rsplit_once('/')
        .map_or("", |(prefix, _)| prefix)
        .to_string();

    if matches!(file_name, "mod.rs" | "lib.rs" | "main.rs") {
        (parent_disk.to_path_buf(), repo_parent_base)
    } else {
        let stem = file_name.strip_suffix(".rs").unwrap_or(file_name);
        let child_repo = join_repo_path(&repo_parent_base, stem);
        (parent_disk.join(stem), child_repo)
    }
}

fn extract_external_submodules(content: &str) -> Vec<SubmoduleDecl> {
    let stripped = strip_comments_preserving_attrs(content);
    let mut mods = Vec::new();
    let bytes = stripped.as_bytes();
    let mut idx = 0;
    let mut pending_attrs = String::new();

    while idx < bytes.len() {
        if bytes[idx].is_ascii_whitespace() {
            idx += 1;
            continue;
        }
        if stripped[idx..].starts_with("#![") {
            if let Some(end_attr) = find_attr_end(&stripped[idx..]) {
                idx += end_attr;
                continue;
            }
        }
        if stripped[idx..].starts_with("#[") {
            if let Some(end_attr) = find_attr_end(&stripped[idx..]) {
                pending_attrs.push_str(&stripped[idx..idx + end_attr]);
                pending_attrs.push('\n');
                idx += end_attr;
                continue;
            }
        }
        if let Some((ident, consumed)) = try_parse_mod_item(&stripped[idx..]) {
            mods.push(SubmoduleDecl {
                ident,
                custom_path: extract_attr_value(&pending_attrs, "path"),
                feature_gate: extract_cfg_feature(&pending_attrs),
            });
            pending_attrs.clear();
            idx += consumed;
            continue;
        }
        pending_attrs.clear();
        idx = advance_statement(&stripped, idx);
    }
    mods
}

fn find_attr_end(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_quotes {
            escaped = true;
        } else if ch == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes && ch == '[' {
            depth += 1;
        } else if !in_quotes && ch == ']' {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(i + 1);
            }
        }
    }
    None
}

fn try_parse_mod_item(s: &str) -> Option<(String, usize)> {
    let mut rest = s;
    if let Some(after_pub) = rest.strip_prefix("pub") {
        let trimmed = after_pub.trim_start();
        if let Some(in_vis) = trimmed.strip_prefix('(') {
            let close = in_vis.find(')')?;
            rest = in_vis[close + 1..].trim_start();
        } else if after_pub.starts_with(char::is_whitespace) {
            rest = trimmed;
        } else {
            return None;
        }
    }
    let after_mod = rest.strip_prefix("mod")?;
    if !after_mod.starts_with(char::is_whitespace) {
        return None;
    }
    let trimmed = after_mod.trim_start();
    let ident_src = trimmed.strip_prefix("r#").unwrap_or(trimmed);
    let end = ident_src
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(ident_src.len());
    if end == 0 {
        return None;
    }
    let tail = ident_src[end..].trim_start();
    if !tail.starts_with(';') {
        return None;
    }
    let consumed = s.len() - tail.len() + 1;
    Some((ident_src[..end].to_string(), consumed))
}

fn advance_statement(s: &str, start: usize) -> usize {
    if s[start..].starts_with([';', '{', '}']) {
        return start + 1;
    }
    for (off, ch) in s[start..].char_indices() {
        if off > 0 && matches!(ch, ';' | '{' | '}' | '#') {
            return start + off + usize::from(ch != '#');
        }
    }
    s.len()
}

fn extract_attr_value(attrs: &str, key: &str) -> Option<String> {
    let compact: String = attrs.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let needle = format!("#[{key}=\"");
    let pos = compact.find(&needle)?;
    let rest = &compact[pos + needle.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_cfg_feature(attrs: &str) -> Option<String> {
    let compact: String = attrs.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let needle = "#[cfg(feature=\"";
    let pos = compact.find(needle)?;
    let rest = &compact[pos + needle.len()..];
    let end = rest.find("\")]")?;
    (!rest[..end].contains('"')).then(|| rest[..end].to_string())
}

fn strip_comments_preserving_attrs(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut attr_depth = 0usize;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
            i = skip_block_comment(&chars, i);
            out.push(' ');
        } else if i + 2 < chars.len()
            && chars[i] == '#'
            && chars[i + 1] == '!'
            && chars[i + 2] == '['
        {
            attr_depth = 1;
            out.push_str("#![");
            i += 3;
        } else if i + 1 < chars.len() && chars[i] == '#' && chars[i + 1] == '[' {
            attr_depth = 1;
            out.push_str("#[");
            i += 2;
        } else if attr_depth > 0 {
            if chars[i] == '"' {
                let end = skip_quoted_string(&chars, i);
                out.extend(&chars[i..end]);
                i = end;
            } else {
                if chars[i] == '[' {
                    attr_depth += 1;
                } else if chars[i] == ']' {
                    attr_depth -= 1;
                }
                out.push(chars[i]);
                i += 1;
            }
        } else if let Some(next_i) = try_skip_raw_string(&chars, i) {
            i = next_i;
            out.push(' ');
        } else if let Some(next_i) = try_skip_char_literal(&chars, i) {
            i = next_i;
            out.push(' ');
        } else if chars[i] == '"' {
            i = skip_quoted_string(&chars, i);
            out.push(' ');
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn try_skip_raw_string(chars: &[char], start: usize) -> Option<usize> {
    let r_pos = if chars[start] == 'r' {
        start
    } else if chars[start] == 'b' && start + 1 < chars.len() && chars[start + 1] == 'r' {
        start + 1
    } else {
        return None;
    };
    let mut k = r_pos + 1;
    let mut hashes = 0usize;
    while k < chars.len() && chars[k] == '#' {
        hashes += 1;
        k += 1;
    }
    if k >= chars.len() || chars[k] != '"' {
        return None;
    }
    let mut i = k + 1;
    while i < chars.len() {
        if chars[i] == '"'
            && i + hashes < chars.len()
            && (0..hashes).all(|h| chars[i + 1 + h] == '#')
        {
            return Some(i + 1 + hashes);
        }
        i += 1;
    }
    Some(chars.len())
}

fn try_skip_char_literal(chars: &[char], start: usize) -> Option<usize> {
    if chars[start] != '\'' || start + 2 >= chars.len() {
        return None;
    }
    if chars[start + 1] != '\\' && chars[start + 2] == '\'' {
        return Some(start + 3);
    }
    if chars[start + 1] == '\\' {
        let mut end = start + 2;
        while end < chars.len() && end <= start + 9 {
            if chars[end] == '\'' {
                return Some(end + 1);
            }
            end += 1;
        }
    }
    None
}

fn skip_block_comment(chars: &[char], start: usize) -> usize {
    let mut i = start + 2;
    let mut depth = 1usize;
    while i + 1 < chars.len() && depth > 0 {
        if chars[i] == '/' && chars[i + 1] == '*' {
            depth += 1;
            i += 2;
        } else if chars[i] == '*' && chars[i + 1] == '/' {
            depth -= 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    i
}

fn skip_quoted_string(chars: &[char], start: usize) -> usize {
    let mut i = start + 1;
    while i < chars.len() {
        if chars[i] == '\\' {
            i += 2;
        } else if chars[i] == '"' {
            return i + 1;
        } else {
            i += 1;
        }
    }
    i
}

/// Extracts `[cite:module_name]` citations matching `TraceabilityEngine::extract_citations`.
pub(super) fn extract_citations(content: &str) -> Vec<String> {
    let mut cites = Vec::new();
    let mut search_idx = 0;
    while let Some(start) = content[search_idx..].find("[cite:") {
        let real_start = search_idx + start + 6;
        if let Some(end) = content[real_start..].find(']') {
            let paper_name = &content[real_start..real_start + end];
            if !paper_name.is_empty()
                && paper_name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
            {
                cites.push(paper_name.to_string());
            }
            search_idx = real_start + end + 1;
        } else {
            break;
        }
    }
    cites
}

/// Extracts verification macro module names (`module = "..."` and `embed_theory("...")`).
pub(super) fn extract_macro_modules(content: &str) -> Vec<String> {
    let mut modules = Vec::new();
    let compact: String = content
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();

    let mut idx = 0;
    while let Some(pos) = compact[idx..].find("module=\"") {
        let start = idx + pos + "module=\"".len();
        let Some(end) = compact[start..].find('"') else {
            break;
        };
        let name = &compact[start..start + end];
        if !name.is_empty() {
            modules.push(name.to_string());
        }
        idx = start + end + 1;
    }

    let mut eidx = 0;
    while let Some(pos) = compact[eidx..].find("embed_theory(\"") {
        let start = eidx + pos + "embed_theory(\"".len();
        let Some(end) = compact[start..].find('"') else {
            break;
        };
        let raw = compact[start..start + end].replace('\\', "/");
        let clean = raw
            .trim_start_matches("papers/")
            .trim_start_matches("spec:")
            .trim_start_matches("registry:")
            .trim_end_matches(".tex");
        if !clean.is_empty() {
            modules.push(clean.to_string());
        }
        eidx = start + end + 1;
    }
    modules
}
