//! Deterministic `OUT_DIR` staging helpers for `math_explorer_gui` build script.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use oxidize_core::path_utils;
use oxidize_core::workspace::find_workspace_root_from;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Computes `(ws_root_abs, rel_ws_root, gui_member_rel)` dynamically from `CARGO_MANIFEST_DIR`.
pub fn resolve_gui_workspace_paths() -> (PathBuf, PathBuf, String) {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let ws_root_abs = find_workspace_root_from(&manifest_dir)
        .expect("Failed to locate Cargo workspace root from CARGO_MANIFEST_DIR");
    let canon_manifest = fs::canonicalize(&manifest_dir).unwrap_or_else(|_| manifest_dir.clone());
    let canon_ws = fs::canonicalize(&ws_root_abs).unwrap_or_else(|_| ws_root_abs.clone());
    let rel_from_ws = canon_manifest
        .strip_prefix(&canon_ws)
        .expect("CARGO_MANIFEST_DIR must reside inside the workspace root");
    let gui_member_rel = path_utils::normalize_path(rel_from_ws);
    let depth = rel_from_ws.components().count();
    let rel_ws_root = if depth == 0 {
        PathBuf::from(".")
    } else {
        let mut parent = PathBuf::new();
        for _ in 0..depth {
            parent.push("..");
        }
        parent
    };
    (ws_root_abs, rel_ws_root, gui_member_rel)
}

/// Writes `content` to `path` only when the file does not exist or its bytes differ,
/// preserving mtimes across warm builds and snapshot syncs.
pub fn write_if_changed(path: &Path, content: &[u8]) {
    if fs::read(path).ok().as_deref() == Some(content) {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn stage_single_file(src_rel: &str, dst_path: &Path) {
    let src = Path::new(src_rel);
    if src.is_file() {
        println!("cargo:rerun-if-changed={src_rel}");
        if let Ok(bytes) = fs::read(src) {
            write_if_changed(dst_path, &bytes);
        }
    } else {
        write_if_changed(dst_path, b"");
    }
}

/// Stages compile-time embedded assets into `out_dir` so `rustc` never depends on
/// `env!("CARGO_MANIFEST_DIR")`.
pub fn stage_gui_inputs_to_out_dir(out_dir: &Path) {
    println!("cargo:rerun-if-changed=build_stage.rs");
    let (_, rel_ws_root, _) = resolve_gui_workspace_paths();
    let cert_rel = path_utils::normalize_path(rel_ws_root.join("Verification_Certificate.json"));
    let bib_rel = path_utils::normalize_path(rel_ws_root.join("papers/references.bib"));

    stage_single_file("sw.js", &out_dir.join("sw.js"));
    stage_single_file(&cert_rel, &out_dir.join("Verification_Certificate.json"));
    stage_single_file(&bib_rel, &out_dir.join("references.bib"));
}
