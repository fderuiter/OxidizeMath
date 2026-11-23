//! Deterministic build script generating path-named workspace Markdown doctest modules.
#[path = "../oxidize_core/src/path_utils.rs"]
/// Path utilities.
pub mod path_utils;

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const IGNORED_DIRS: &[&str] = &[
    ".cargo",
    ".git",
    ".playwright-mcp",
    ".rustup",
    "artifacts",
    "dist",
    "node_modules",
    "target",
];

fn should_skip_dir(name: &str) -> bool {
    IGNORED_DIRS.contains(&name) || name.starts_with("cargo-machete")
}

fn has_crate_readme_doctest(crate_dir: &Path) -> bool {
    if !crate_dir.join("Cargo.toml").is_file() {
        return false;
    }
    let Ok(lib_src) = fs::read_to_string(crate_dir.join("src").join("lib.rs")) else {
        return false;
    };
    lib_src.lines().map(str::trim).any(|line| {
        !line.starts_with("//")
            && !line.starts_with("/*")
            && !line.starts_with('*')
            && line.starts_with("#![doc")
            && line.contains("include_str!")
            && line.contains("README.md")
    })
}

fn is_already_doctested_crate_readme(md_path: &Path) -> bool {
    md_path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n == "README.md")
        && md_path.parent().is_some_and(has_crate_readme_doctest)
}

fn collect_markdown_tree(dir: &Path, files: &mut Vec<PathBuf>, dirs: &mut Vec<PathBuf>) {
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
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !should_skip_dir(name) {
                collect_markdown_tree(&path, files, dirs);
            }
        } else if path.is_file()
            && path.extension().and_then(|s| s.to_str()) == Some("md")
            && !is_already_doctested_crate_readme(&path)
        {
            files.push(path);
        }
    }
}

fn module_ident_from_rel_path(rel_path_str: &str, used: &mut BTreeSet<String>) -> String {
    let repo_rel = rel_path_str.trim_start_matches("../../");
    let mut ident = String::with_capacity(repo_rel.len() + 4);
    let mut prev_underscore = false;
    for ch in repo_rel.chars() {
        if ch.is_ascii_alphanumeric() {
            ident.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore && !ident.is_empty() {
            ident.push('_');
            prev_underscore = true;
        }
    }
    while ident.ends_with('_') {
        ident.pop();
    }
    if ident.is_empty() || ident.as_bytes()[0].is_ascii_digit() {
        ident.insert_str(0, "md_");
    }
    if !ident.ends_with("_md") {
        ident.push_str("_md");
    }
    let base = ident.clone();
    let mut suffix = 2usize;
    while !used.insert(ident.clone()) {
        ident = format!("{base}_{suffix}");
        suffix += 1;
    }
    ident
}

fn write_if_changed(path: &Path, content: &str) {
    if let Ok(existing) = fs::read(path)
        && existing == content.as_bytes()
    {
        return;
    }
    fs::write(path, content).expect("Failed to write OUT_DIR/generated_tests.rs");
}

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    let dest_path = Path::new(&out_dir).join("generated_tests.rs");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../oxidize_core/src/path_utils.rs");
    println!("cargo:rerun-if-changed=../../Cargo.toml");

    let workspace_dir = Path::new("../..");
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    collect_markdown_tree(workspace_dir, &mut files, &mut dirs);

    let mut dir_reruns: Vec<String> = dirs
        .iter()
        .map(path_utils::normalize_path)
        .filter(|s| !s.is_empty() && s != "../..")
        .collect();
    dir_reruns.sort();
    dir_reruns.dedup();
    for rel_dir in dir_reruns {
        println!("cargo:rerun-if-changed={rel_dir}");
    }

    let mut normalized_files: Vec<String> = files
        .iter()
        .map(path_utils::normalize_path)
        .filter(|s| !s.is_empty())
        .collect();
    normalized_files.sort();
    normalized_files.dedup();

    let mut used_idents = BTreeSet::new();
    let mut generated_code = String::new();
    for rel_path_str in normalized_files {
        println!("cargo:rerun-if-changed={rel_path_str}");
        let mod_ident = module_ident_from_rel_path(&rel_path_str, &mut used_idents);
        generated_code.push_str(&format!(
            "#[doc = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", \"{rel_path_str}\"))]\npub mod {mod_ident} {{}}\n"
        ));
    }

    write_if_changed(&dest_path, &generated_code);
}
