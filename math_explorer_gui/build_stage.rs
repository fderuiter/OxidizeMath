//! Deterministic `OUT_DIR` staging helpers for `math_explorer_gui` build script.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use oxidize_core::path_utils;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

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

fn mirror_dir_recursive(src_dir: &Path, dst_dir: &Path, expected_files: &mut HashSet<PathBuf>) {
    if !src_dir.is_dir() {
        return;
    }
    println!(
        "cargo:rerun-if-changed={}",
        path_utils::normalize_path(src_dir)
    );
    let Ok(read_dir) = fs::read_dir(src_dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = read_dir.flatten().map(|e| e.path()).collect();
    entries.sort();

    for src_path in entries {
        let Some(name) = src_path.file_name() else {
            continue;
        };
        let dst_path = dst_dir.join(name);
        if src_path.is_dir() {
            mirror_dir_recursive(&src_path, &dst_path, expected_files);
        } else if src_path.is_file() {
            println!(
                "cargo:rerun-if-changed={}",
                path_utils::normalize_path(&src_path)
            );
            if let Ok(bytes) = fs::read(&src_path) {
                write_if_changed(&dst_path, &bytes);
                expected_files.insert(dst_path);
            }
        }
    }
}

fn prune_untracked_files(dir: &Path, expected_files: &HashSet<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            prune_untracked_files(&path, expected_files);
            if fs::read_dir(&path)
                .map(|mut d| d.next().is_none())
                .unwrap_or(false)
            {
                let _ = fs::remove_dir(&path);
            }
        } else if !expected_files.contains(&path) {
            let _ = fs::remove_file(&path);
        }
    }
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

/// Stages `src/tabs/` and compile-time embedded assets into `out_dir` so `rustc` never
/// depends on `env!("CARGO_MANIFEST_DIR")`.
pub fn stage_gui_inputs_to_out_dir(out_dir: &Path) {
    println!("cargo:rerun-if-changed=build_stage.rs");

    let mut expected_tab_files = HashSet::new();
    let dst_tabs = out_dir.join("src").join("tabs");
    mirror_dir_recursive(Path::new("src/tabs"), &dst_tabs, &mut expected_tab_files);
    prune_untracked_files(&dst_tabs, &expected_tab_files);

    stage_single_file("sw.js", &out_dir.join("sw.js"));
    stage_single_file(
        "../Verification_Certificate.json",
        &out_dir.join("Verification_Certificate.json"),
    );
    stage_single_file("../papers/references.bib", &out_dir.join("references.bib"));
}
