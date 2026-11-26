//! Stages workspace `traceability.toml` into `OUT_DIR` via ancestor root discovery.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn has_workspace_section(manifest_path: &Path) -> bool {
    let Ok(content) = fs::read_to_string(manifest_path) else {
        return false;
    };
    content.lines().any(|raw| {
        let line = raw.split('#').next().unwrap_or("").trim();
        line == "[workspace]"
    })
}

fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    let resolved = fs::canonicalize(start).unwrap_or_else(|_| start.to_path_buf());
    for ancestor in resolved.ancestors() {
        if has_workspace_section(&ancestor.join("Cargo.toml")) {
            return Some(ancestor.to_path_buf());
        }
    }
    for ancestor in start.ancestors() {
        if has_workspace_section(&ancestor.join("Cargo.toml")) {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn write_if_changed(path: &Path, content: &[u8]) {
    if let Ok(existing) = fs::read(path)
        && existing == content
    {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create OUT_DIR parent directory");
    }
    fs::write(path, content).expect("Failed to write staged traceability.toml");
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let dest = out_dir.join("traceability.toml");

    if let Some(ws_root) = find_workspace_root(&manifest_dir) {
        let registry_path = ws_root.join("traceability.toml");
        println!("cargo:rerun-if-changed={}", registry_path.display());
        let bytes = fs::read(&registry_path).unwrap_or_default();
        write_if_changed(&dest, &bytes);
    } else {
        write_if_changed(&dest, b"[links]\n[crates]\n[modules]\n");
    }
}
