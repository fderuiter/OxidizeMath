//! Target-scoped build script generating virtual file system tables for `wasm32` in `oxidize_core`.
#[path = "src/path_utils.rs"]
/// Path utilities.
pub mod path_utils;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

const VFS_ROOTS: &[&str] = &[
    "../../papers",
    "../../math_explorer/src",
    "../../math_explorer_gui/src/tabs",
];

fn write_if_changed(path: &Path, content: &str) {
    if let Ok(existing) = fs::read(path)
        && existing == content.as_bytes()
    {
        return;
    }
    fs::write(path, content).expect("Failed to write OUT_DIR/vfs_data.rs");
}

fn scan_vfs_root(root: &str, dirs_map: &mut BTreeMap<String, Vec<String>>) {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return;
    }

    let mut stack = vec![root_path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let normalized_dir = path_utils::normalize_path(&current);
        if !normalized_dir.is_empty() {
            println!("cargo:rerun-if-changed={normalized_dir}");
        }

        let Some(relative_dir) = path_utils::strip_and_normalize(&current, "../../") else {
            continue;
        };
        let Ok(entries) = fs::read_dir(&current) else {
            continue;
        };
        let mut children = Vec::new();
        let mut subdirs = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = entry.file_name().to_str() {
                children.push(name.to_string());
            }
            if path.is_dir() {
                subdirs.push(path);
            }
        }
        children.sort();
        subdirs.sort();
        for subdir in subdirs.into_iter().rev() {
            stack.push(subdir);
        }
        dirs_map.insert(relative_dir, children);
    }
}

fn render_wasm_vfs(dirs_map: &BTreeMap<String, Vec<String>>) -> String {
    let mut arms = String::new();
    for (dir, children) in dirs_map {
        let child_str = children
            .iter()
            .map(|c| format!("{c:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        arms.push_str(&format!("        {dir:?} => Some(&[{child_str}]),\n"));
    }
    format!(
        "pub fn get_dir_children(path: &str) -> Option<&'static [&'static str]> {{\n    match path {{\n{arms}        _ => None,\n    }}\n}}\n"
    )
}

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    let dest_path = Path::new(&out_dir).join("vfs_data.rs");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/path_utils.rs");

    let is_wasm = env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32");
    let code = if is_wasm {
        let mut dirs_map = BTreeMap::new();
        for root in VFS_ROOTS {
            scan_vfs_root(root, &mut dirs_map);
        }
        render_wasm_vfs(&dirs_map)
    } else {
        "pub fn get_dir_children(_path: &str) -> Option<&'static [&'static str]> {\n    None\n}\n"
            .to_string()
    };

    write_if_changed(&dest_path, &code);
}
