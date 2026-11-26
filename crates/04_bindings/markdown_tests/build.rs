//! Deterministic build script generating path-named workspace Markdown doctest modules
//! via ancestor `[workspace]` root discovery.

use std::collections::{BTreeSet, HashSet};
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

fn normalize_slash_path(path: &Path) -> String {
    let mut path_str = path.to_str().unwrap_or("");
    if let Some(stripped) = path_str.strip_prefix(r"\\?\") {
        path_str = stripped;
    }
    path_str.replace('\\', "/")
}

fn has_workspace_section(manifest_path: &Path) -> bool {
    let Ok(content) = fs::read_to_string(manifest_path) else {
        return false;
    };
    content
        .lines()
        .any(|raw| raw.split('#').next().unwrap_or("").trim() == "[workspace]")
}

fn find_workspace_root(manifest_dir: &Path) -> PathBuf {
    let resolved = fs::canonicalize(manifest_dir).unwrap_or_else(|_| manifest_dir.to_path_buf());
    for ancestor in resolved.ancestors() {
        if has_workspace_section(&ancestor.join("Cargo.toml")) {
            return ancestor.to_path_buf();
        }
    }
    for ancestor in manifest_dir.ancestors() {
        if has_workspace_section(&ancestor.join("Cargo.toml")) {
            return ancestor.to_path_buf();
        }
    }
    panic!(
        "Unable to locate workspace root Cargo.toml from {}",
        manifest_dir.display()
    );
}

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

fn module_ident_from_ws_rel(ws_rel: &str, used: &mut BTreeSet<String>) -> String {
    let mut ident = String::with_capacity(ws_rel.len() + 4);
    let mut prev_underscore = false;
    for ch in ws_rel.chars() {
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

fn write_if_changed(path: &Path, content: &[u8]) {
    if let Ok(existing) = fs::read(path)
        && existing == content
    {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create staged_md directory");
    }
    fs::write(path, content).expect("Failed to write staged markdown_tests artifact");
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
            if fs::read_dir(&path).is_ok_and(|mut d| d.next().is_none()) {
                let _ = fs::remove_dir(&path);
            }
        } else if !expected_files.contains(&path) {
            let _ = fs::remove_file(&path);
        }
    }
}

fn relative_workspace_root(manifest_dir: &Path, workspace_dir: &Path) -> PathBuf {
    let manifest_abs =
        fs::canonicalize(manifest_dir).unwrap_or_else(|_| manifest_dir.to_path_buf());
    let depth = manifest_abs
        .strip_prefix(workspace_dir)
        .map(|rel| rel.components().count())
        .unwrap_or(2);
    let mut rel_root = PathBuf::new();
    for _ in 0..depth {
        rel_root.push("..");
    }
    rel_root
}

fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo"));
    let staged_root = out_dir.join("staged_md");
    let dest_path = out_dir.join("generated_tests.rs");

    let workspace_dir = find_workspace_root(&manifest_dir);
    let rel_ws_root = relative_workspace_root(&manifest_dir, &workspace_dir);
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        normalize_slash_path(&rel_ws_root.join("Cargo.toml"))
    );

    let mut files = Vec::new();
    let mut dirs = Vec::new();
    collect_markdown_tree(&workspace_dir, &mut files, &mut dirs);

    let mut dir_reruns: Vec<String> = dirs
        .iter()
        .filter(|d| d.as_path() != workspace_dir.as_path())
        .filter_map(|d| d.strip_prefix(&workspace_dir).ok())
        .map(|rel| normalize_slash_path(&rel_ws_root.join(rel)))
        .filter(|s| !s.is_empty())
        .collect();
    dir_reruns.sort();
    dir_reruns.dedup();
    for dir_path in dir_reruns {
        println!("cargo:rerun-if-changed={dir_path}");
    }

    let mut md_entries: Vec<(String, PathBuf)> = files
        .into_iter()
        .filter_map(|abs_path| {
            let ws_rel = abs_path.strip_prefix(&workspace_dir).ok()?;
            let ws_rel_str = normalize_slash_path(ws_rel);
            (!ws_rel_str.is_empty()).then_some((ws_rel_str, abs_path))
        })
        .collect();
    md_entries.sort_by(|a, b| a.0.cmp(&b.0));
    md_entries.dedup_by(|a, b| a.0 == b.0);

    let mut expected_staged = HashSet::with_capacity(md_entries.len());
    let mut used_idents = BTreeSet::new();
    let mut generated_code = String::new();
    for (ws_rel, abs_path) in md_entries {
        println!(
            "cargo:rerun-if-changed={}",
            normalize_slash_path(&rel_ws_root.join(&ws_rel))
        );
        let staged_path = staged_root.join(&ws_rel);
        let bytes = fs::read(&abs_path).expect("Failed to read workspace Markdown file");
        write_if_changed(&staged_path, &bytes);
        expected_staged.insert(staged_path);

        let mod_ident = module_ident_from_ws_rel(&ws_rel, &mut used_idents);
        generated_code.push_str(&format!(
            "#[doc = include_str!(concat!(env!(\"OUT_DIR\"), \"/staged_md/\", \"{ws_rel}\"))]\npub mod {mod_ident} {{}}\n"
        ));
    }

    prune_stale_staged_files(&staged_root, &expected_staged);
    write_if_changed(&dest_path, generated_code.as_bytes());
}
