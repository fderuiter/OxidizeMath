//! Build script for workspace markdown documentation tests.
#[path = "../oxidize_core/src/path_utils.rs"]
/// Path utilities.
pub mod path_utils;

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const EXCLUDED_DIRS: &[&str] = &[
    "target",
    ".git",
    "node_modules",
    ".cargo",
    ".rustup",
    "dist",
    "artifacts",
];

fn write_if_changed(path: &Path, content: &[u8]) {
    if fs::read(path).ok().as_deref() == Some(content) {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn get_markdown_files(dir: &Path, files: &mut Vec<PathBuf>, dirs: &mut Vec<PathBuf>) {
    if !dir.is_dir() {
        return;
    }
    dirs.push(dir.to_path_buf());
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = read_dir.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if EXCLUDED_DIRS.contains(&name.as_ref()) || name.starts_with("cargo-machete") {
                continue;
            }
            get_markdown_files(&path, files, dirs);
        } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md") {
            files.push(path);
        }
    }
}

fn prune_stale_staged_files(staged_root: &Path, expected_files: &HashSet<PathBuf>) {
    if !staged_root.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(staged_root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            prune_stale_staged_files(&path, expected_files);
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

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let staged_root = out_dir.join("staged_md");
    let dest_path = out_dir.join("generated_tests.rs");

    println!("cargo:rerun-if-changed=build.rs");

    let workspace_dir = Path::new("../..");
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    get_markdown_files(workspace_dir, &mut files, &mut dirs);

    dirs.sort_by_key(|d| path_utils::normalize_path(d));
    files.sort_by_key(|f| path_utils::normalize_path(f));

    for dir in &dirs {
        if dir != workspace_dir {
            println!("cargo:rerun-if-changed={}", path_utils::normalize_path(dir));
        }
    }

    let mut expected_staged = HashSet::with_capacity(files.len());
    let mut generated_code = String::new();
    for (i, path) in files.iter().enumerate() {
        let rel_path_str = path_utils::normalize_path(path);
        println!("cargo:rerun-if-changed={rel_path_str}");

        let ws_rel = rel_path_str
            .strip_prefix("../../")
            .unwrap_or(&rel_path_str)
            .trim_start_matches('/');
        let staged_path = staged_root.join(ws_rel);
        let bytes = fs::read(path).unwrap();
        write_if_changed(&staged_path, &bytes);
        expected_staged.insert(staged_path);

        generated_code.push_str(&format!(
            "#[doc = include_str!(concat!(env!(\"OUT_DIR\"), \"/staged_md/\", \"{ws_rel}\"))]\npub mod md_test_{i} {{}}\n"
        ));
    }

    prune_stale_staged_files(&staged_root, &expected_staged);
    write_if_changed(&dest_path, generated_code.as_bytes());
}
