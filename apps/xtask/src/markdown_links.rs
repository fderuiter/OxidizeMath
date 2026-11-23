use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;

pub(crate) fn check_markdown_links() {
    let root = if Path::new("Cargo.toml").is_file()
        && fs::read_to_string("Cargo.toml").is_ok_and(|c| c.contains("[workspace]"))
    {
        PathBuf::from(".")
    } else {
        crate::gates::workspace_root()
    };
    check_markdown_links_in_dir(&root);
}

pub(crate) fn check_markdown_links_in_dir(root: &Path) {
    println!("=== Checking Markdown Documentation Links ===");
    let mut md_files = Vec::new();
    collect_md_files(root, &mut md_files);
    check_markdown_links_for_files_in_dir(root, &md_files);
}

pub(crate) fn check_markdown_links_for_files_in_dir(root: &Path, md_files: &[PathBuf]) {
    if md_files.is_empty() {
        return;
    }

    let mut broken = Vec::new();
    for file in md_files {
        let full_file = if file.starts_with(root) {
            file.clone()
        } else {
            root.join(file)
        };
        if let Ok(content) = fs::read_to_string(&full_file) {
            let parent = full_file.parent().unwrap_or(root);
            let display_path = full_file.strip_prefix(root).unwrap_or(file.as_path());
            for link in extract_md_links(&content) {
                let target = link.split_whitespace().next().unwrap_or(&link);
                if is_external_or_anchor_link(target) {
                    continue;
                }
                let path_part = target.split('#').next().unwrap_or("");
                if path_part.is_empty() {
                    continue;
                }
                let rel_dest = parent.join(path_part);
                let root_dest = root.join(path_part.trim_start_matches('/'));
                if !rel_dest.exists() && !root_dest.exists() {
                    broken.push((display_path.display().to_string(), target.to_string()));
                }
            }
        }
    }

    if !broken.is_empty() {
        eprintln!("Found {} broken markdown link(s):", broken.len());
        for (f, l) in &broken {
            eprintln!("  In {}: broken link target -> {}", f, l);
        }
        exit(1);
    }
    println!(
        "All markdown documentation links in {} files resolved successfully.",
        md_files.len()
    );
}

fn is_external_or_anchor_link(target: &str) -> bool {
    target.starts_with("http://")
        || target.starts_with("https://")
        || target.starts_with("mailto:")
        || target.starts_with('#')
        || target.contains("::")
}

pub(crate) fn collect_md_files(dir: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name == "target"
                    || name == ".git"
                    || name == "node_modules"
                    || name.starts_with("cargo-machete")
                    || name == "artifacts"
                    || name == "dist"
                {
                    continue;
                }
                collect_md_files(&path, out);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
                out.push(path);
            }
        }
    }
}

pub(crate) fn extract_md_links(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut rest = content;
    while let Some(start_bracket) = rest.find('[') {
        rest = &rest[start_bracket + 1..];
        if let Some(end_bracket) = rest.find(']') {
            let after_bracket = &rest[end_bracket + 1..];
            if after_bracket.starts_with('(') {
                if let Some(end_paren) = after_bracket.find(')') {
                    let link = &after_bracket[1..end_paren];
                    links.push(link.trim().to_string());
                    rest = &after_bracket[end_paren + 1..];
                } else {
                    rest = after_bracket;
                }
            } else {
                rest = after_bracket;
            }
        }
    }
    links
}
