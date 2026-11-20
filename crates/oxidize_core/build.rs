//! Build script for virtual file system code generation in oxidize_core.
#[path = "src/path_utils.rs"]
/// Path utilities.
pub mod path_utils;

use std::env;
use std::fs;
use std::path::Path;

fn scan_vfs_root(root: &str, dirs_map: &mut String) {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return;
    }

    let mut stack = vec![root_path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let normalized_dir = path_utils::normalize_path(&current);
        if !normalized_dir.is_empty() {
            println!("cargo:rerun-if-changed={}", normalized_dir);
        }

        let relative_dir = path_utils::strip_and_normalize(&current, "../../").unwrap();
        if let Ok(entries) = fs::read_dir(&current) {
            let mut children = Vec::new();
            let mut subdirs = Vec::new();
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_str().unwrap().to_string();
                children.push(name);

                if path.is_dir() {
                    subdirs.push(path);
                }
            }
            children.sort();
            subdirs.sort();
            for subdir in subdirs.into_iter().rev() {
                stack.push(subdir);
            }
            let child_str = children
                .iter()
                .map(|c| format!("{:?}", c))
                .collect::<Vec<_>>()
                .join(", ");
            dirs_map.push_str(&format!(
                "{:?} => Some(&[{}]),\n        ",
                relative_dir, child_str
            ));
        }
    }
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("vfs_data.rs");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/path_utils.rs");

    let mut dirs_map = String::new();
    let root_dirs = [
        "../../papers",
        "../../math_explorer/src",
        "../../math_explorer_gui/src/tabs",
    ];

    for root in root_dirs {
        scan_vfs_root(root, &mut dirs_map);
    }

    let code = format!(
        "pub fn get_dir_children(path: &str) -> Option<&'static [&'static str]> {{
    match path {{
        {}
        _ => None,
    }}
}}
",
        dirs_map
    );

    fs::write(dest_path, code).unwrap();
}
