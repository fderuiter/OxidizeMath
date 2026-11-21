//! Active Rust module tree and citation scanner for `build_snapshot.rs`.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::path_utils;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Emits a normalized forward-slash `cargo:rerun-if-changed` directive when non-empty.
pub(super) fn emit_rerun(path: &Path) {
    let normalized = path_utils::normalize_path(path);
    if !normalized.trim().is_empty() {
        println!("cargo:rerun-if-changed={normalized}");
    }
}

/// Discovers all active `.rs` files across workspace crates, `math_explorer`, and GUI tabs.
pub(super) fn discover_workspace_active_files(ws_root: &Path) -> BTreeMap<PathBuf, String> {
    let mut roots: Vec<(PathBuf, String)> = vec![
        (
            ws_root.join("math_explorer").join("src"),
            "math_explorer/src".to_string(),
        ),
        (
            Path::new("src").join("tabs"),
            "math_explorer_gui/src/tabs".to_string(),
        ),
    ];

    let crates_dir = ws_root.join("crates");
    if let Ok(entries) = fs::read_dir(&crates_dir) {
        let mut crate_names: Vec<String> = entries
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        crate_names.sort();
        for crate_name in crate_names {
            let disk_src = crates_dir.join(&crate_name).join("src");
            if disk_src.is_dir() {
                emit_rerun(&disk_src);
                roots.push((disk_src, format!("crates/{crate_name}/src")));
            }
        }
    }

    let mut active = BTreeMap::new();
    for (disk_dir, repo_dir) in roots {
        if disk_dir.is_dir() {
            gather_dir_files(&disk_dir, &repo_dir, &mut active);
        }
    }
    active
}

fn gather_dir_files(disk_dir: &Path, repo_dir: &str, active: &mut BTreeMap<PathBuf, String>) {
    let lib_disk = disk_dir.join("lib.rs");
    let main_disk = disk_dir.join("main.rs");
    let mut has_entry = false;

    if lib_disk.is_file() {
        parse_module_tree(&lib_disk, &format!("{repo_dir}/lib.rs"), active);
        has_entry = true;
    }
    if main_disk.is_file() {
        parse_module_tree(&main_disk, &format!("{repo_dir}/main.rs"), active);
        has_entry = true;
    }
    if !has_entry {
        gather_fallback_files(disk_dir, repo_dir, active);
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

    let (disk_parent, repo_parent) = module_search_dir(disk_file, repo_file);
    for submod in extract_external_submodules(&content) {
        let cand_rs = disk_parent.join(format!("{submod}.rs"));
        let cand_mod = disk_parent.join(&submod).join("mod.rs");
        if cand_rs.is_file() {
            parse_module_tree(&cand_rs, &format!("{repo_parent}/{submod}.rs"), active);
        } else if cand_mod.is_file() {
            parse_module_tree(&cand_mod, &format!("{repo_parent}/{submod}/mod.rs"), active);
        }
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
        let child_repo = if repo_parent_base.is_empty() {
            stem.to_string()
        } else {
            format!("{repo_parent_base}/{stem}")
        };
        (parent_disk.join(stem), child_repo)
    }
}

fn gather_fallback_files(
    start_disk: &Path,
    start_repo: &str,
    active: &mut BTreeMap<PathBuf, String>,
) {
    let mut stack = vec![(start_disk.to_path_buf(), start_repo.to_string())];
    while let Some((curr_disk, curr_repo)) = stack.pop() {
        emit_rerun(&curr_disk);
        let Ok(entries) = fs::read_dir(&curr_disk) else {
            continue;
        };
        let mut items: Vec<_> = entries.flatten().collect();
        items.sort_by_key(|e| e.file_name());
        for entry in items {
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            let child_disk = entry.path();
            let child_repo = format!("{curr_repo}/{name}");
            if name.ends_with(".rs") && child_disk.is_file() {
                emit_rerun(&child_disk);
                let key = PathBuf::from(path_utils::normalize_path(&child_disk));
                active.insert(key, child_repo);
            } else if !name.contains('.') && child_disk.is_dir() {
                stack.push((child_disk, child_repo));
            }
        }
    }
}

fn extract_external_submodules(content: &str) -> Vec<String> {
    let stripped = strip_comments_and_strings(content);
    let mut mods = Vec::new();
    let bytes = stripped.as_bytes();
    let mut idx = 0;
    while idx + 4 <= bytes.len() {
        let prev_ok =
            idx == 0 || !(bytes[idx - 1].is_ascii_alphanumeric() || bytes[idx - 1] == b'_');
        if prev_ok && stripped[idx..].starts_with("mod") {
            let after_mod = idx + 3;
            if after_mod < bytes.len() && bytes[after_mod].is_ascii_whitespace() {
                if let Some((ident, consumed)) = parse_mod_decl_ident(&stripped[after_mod..]) {
                    mods.push(ident);
                    idx = after_mod + consumed;
                    continue;
                }
            }
        }
        idx += 1;
    }
    mods
}

fn parse_mod_decl_ident(rest: &str) -> Option<(String, usize)> {
    let trimmed = rest.trim_start();
    let ws_len = rest.len() - trimmed.len();
    let ident_str = trimmed.strip_prefix("r#").unwrap_or(trimmed);
    let raw_prefix_len = trimmed.len() - ident_str.len();
    let end = ident_str
        .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .unwrap_or(ident_str.len());
    if end == 0 {
        return None;
    }
    let after_ident = ident_str[end..].trim_start();
    after_ident.starts_with(';').then(|| {
        (
            ident_str[..end].to_string(),
            ws_len + raw_prefix_len + end + 1,
        )
    })
}

fn strip_comments_and_strings(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
        } else if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '*' {
            i = skip_block_comment(&chars, i);
            out.push(' ');
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
