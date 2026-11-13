//! Legacy crate.
#[path = "../oxidize_core/src/path_utils.rs"]
mod path_utils;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn get_markdown_files(dir: &Path, files: &mut Vec<PathBuf>, dirs: &mut Vec<PathBuf>) {
    if dir.is_dir() {
        dirs.push(dir.to_path_buf());
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name().unwrap().to_string_lossy();
                    if name == "target"
                        || name == ".git"
                        || name == "node_modules"
                        || name == ".cargo"
                        || name == ".rustup"
                        || name.starts_with("cargo-machete")
                        || name == "artifacts"
                        || name == "dist"
                    {
                        continue;
                    }
                    get_markdown_files(&path, files, dirs);
                } else if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md")
                {
                    files.push(path);
                }
            }
        }
    }
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("generated_tests.rs");

    let workspace_dir = Path::new("../..");
    let mut files = Vec::new();
    let mut dirs = Vec::new();
    get_markdown_files(workspace_dir, &mut files, &mut dirs);

    for dir in dirs {
        let rel_dir_str = path_utils::normalize_path(&dir);
        println!("cargo:rerun-if-changed={}", rel_dir_str);
    }

    let mut generated_code = String::new();
    for (i, path) in files.iter().enumerate() {
        let rel_path_str = path_utils::normalize_path(path);

        println!("cargo:rerun-if-changed={}", rel_path_str);

        generated_code.push_str(&format!(
            "#[doc = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/\", \"{}\"))]\npub mod md_test_{} {{}}\n",
            rel_path_str, i
        ));
    }

    fs::write(&dest_path, generated_code).unwrap();
}
