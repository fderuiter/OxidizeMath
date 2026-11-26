//! Zero-dependency build script staging shared pure-`std` `oxidize_core` workspace helpers
//! into `OUT_DIR` so `apps/xtask` avoids cross-crate `#[path = "..."]` attributes.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn strip_comment(line: &str) -> &str {
    let mut in_double = false;
    let mut in_single = false;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_double {
            escaped = true;
        } else if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '#' && !in_double && !in_single {
            return &line[..idx];
        }
    }
    line
}

fn has_workspace_section(manifest_path: &Path) -> bool {
    let Ok(content) = fs::read_to_string(manifest_path) else {
        return false;
    };
    content
        .lines()
        .any(|raw| strip_comment(raw).trim() == "[workspace]")
}

fn find_workspace_root(start: &Path) -> PathBuf {
    let resolved = fs::canonicalize(start).unwrap_or_else(|_| start.to_path_buf());
    for ancestor in resolved.ancestors() {
        if has_workspace_section(&ancestor.join("Cargo.toml")) {
            return ancestor.to_path_buf();
        }
    }
    for ancestor in start.ancestors() {
        if has_workspace_section(&ancestor.join("Cargo.toml")) {
            return ancestor.to_path_buf();
        }
    }
    panic!(
        "Unable to locate workspace root Cargo.toml from {}",
        start.display()
    );
}

fn parse_workspace_members(root_toml: &str) -> Vec<String> {
    let mut in_ws = false;
    let mut body = String::new();
    for raw in root_toml.lines() {
        let line = strip_comment(raw).trim();
        if line.starts_with('[') && line.ends_with(']') {
            if in_ws {
                break;
            }
            in_ws = line == "[workspace]";
            continue;
        }
        if in_ws {
            body.push_str(line);
            body.push('\n');
        }
    }
    for (idx, line) in body.lines().enumerate() {
        if let Some((k, rest)) = line.split_once('=')
            && k.trim() == "members"
        {
            let tail: String = std::iter::once(rest)
                .chain(body.lines().skip(idx + 1))
                .collect::<Vec<_>>()
                .join("\n");
            let Some(open) = tail.find('[') else {
                break;
            };
            let Some(close) = tail[open + 1..].find(']') else {
                break;
            };
            let mut out = Vec::new();
            let mut chars = tail[open + 1..open + 1 + close].chars();
            while let Some(ch) = chars.next() {
                if ch == '"' || ch == '\'' {
                    let q = ch;
                    let mut s = String::new();
                    for next in chars.by_ref() {
                        if next == q {
                            out.push(s);
                            break;
                        }
                        s.push(next);
                    }
                }
            }
            return out;
        }
    }
    Vec::new()
}

fn is_oxidize_core_manifest(manifest_path: &Path) -> bool {
    let Ok(content) = fs::read_to_string(manifest_path) else {
        return false;
    };
    let mut in_pkg = false;
    for raw in content.lines() {
        let line = strip_comment(raw).trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_pkg = line == "[package]";
            continue;
        }
        if in_pkg
            && let Some((k, v)) = line.split_once('=')
            && k.trim() == "name"
            && v.trim().trim_matches('"').trim_matches('\'') == "oxidize_core"
        {
            return true;
        }
    }
    false
}

fn sanitize_inner_docs_for_include(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("//!") {
            out.push_str("//");
            out.push_str(rest);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

fn write_if_changed(path: &Path, content: &[u8]) {
    if let Ok(existing) = fs::read(path)
        && existing == content
    {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create OUT_DIR directory");
    }
    fs::write(path, content).expect("Failed to stage oxidize_core source into OUT_DIR");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let ws_root = find_workspace_root(&manifest_dir);
    let root_manifest = ws_root.join("Cargo.toml");
    println!("cargo:rerun-if-changed={}", root_manifest.display());

    let root_toml = fs::read_to_string(&root_manifest).expect("Failed to read root Cargo.toml");
    let members = parse_workspace_members(&root_toml);
    let oxidize_core_dir = members
        .into_iter()
        .map(|m| ws_root.join(m))
        .find(|dir| is_oxidize_core_manifest(&dir.join("Cargo.toml")))
        .expect("Failed to locate oxidize_core member in workspace Cargo.toml");

    let src_path = oxidize_core_dir.join("src").join("workspace_wasm.rs");
    println!("cargo:rerun-if-changed={}", src_path.display());
    let raw = fs::read_to_string(&src_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", src_path.display()));
    let staged = sanitize_inner_docs_for_include(&raw);
    write_if_changed(&out_dir.join("workspace_wasm.rs"), staged.as_bytes());
}
