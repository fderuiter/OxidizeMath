use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;

const SKIPPED_DIRS: &[&str] = &[
    "target",
    ".git",
    "node_modules",
    "artifacts",
    "dist",
    ".cargo",
    ".rustup",
];

const DOC_FILE_EXTENSIONS: &[&str] = &[
    ".md", ".bib", ".tex", ".pdf", ".toml", ".json", ".rs", ".html", ".svg", ".png",
];

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
    println!("=== Checking Markdown & Docstring Documentation Links ===");
    let mut files = Vec::new();
    collect_md_files(root, &mut files);
    collect_rs_files(root, &mut files);
    check_markdown_links_for_files_in_dir(root, &files);
}

pub(crate) fn check_markdown_links_for_files_in_dir(root: &Path, files: &[PathBuf]) {
    if files.is_empty() {
        return;
    }
    let broken = find_broken_links_in_dir(root, files);
    if !broken.is_empty() {
        eprintln!("Found {} broken documentation link(s):", broken.len());
        for (f, l) in &broken {
            eprintln!("  In {}: broken link target -> {}", f, l);
        }
        exit(1);
    }
    println!(
        "All documentation links in {} files resolved successfully.",
        files.len()
    );
}

pub(crate) fn find_broken_links_in_dir(root: &Path, files: &[PathBuf]) -> Vec<(String, String)> {
    let bib_keys = load_bibtex_keys(&root.join("papers/references.bib"));
    let mut broken = Vec::new();

    for file in files {
        let full_file = if file.starts_with(root) {
            file.clone()
        } else {
            root.join(file)
        };
        let Ok(content) = fs::read_to_string(&full_file) else {
            continue;
        };
        let parent = full_file.parent().unwrap_or(root);
        let display_path = full_file
            .strip_prefix(root)
            .unwrap_or(file.as_path())
            .display()
            .to_string();
        let is_rs = full_file.extension().and_then(|e| e.to_str()) == Some("rs");
        if is_rs {
            check_rs_file_links(
                root,
                parent,
                &display_path,
                &content,
                bib_keys.as_ref(),
                &mut broken,
            );
        } else {
            let links = extract_md_links(&content);
            validate_targets(
                root,
                parent,
                &display_path,
                &links,
                false,
                bib_keys.as_ref(),
                &mut broken,
            );
        }
    }
    broken
}

fn check_rs_file_links(
    root: &Path,
    parent: &Path,
    display_path: &str,
    content: &str,
    bib_keys: Option<&HashSet<String>>,
    broken: &mut Vec<(String, String)>,
) {
    for inc in extract_rs_doc_include_paths(content) {
        if !parent.join(&inc).is_file() {
            broken.push((display_path.to_string(), format!("include_str!(\"{inc}\")")));
        }
    }
    let doc_markdown = extract_rs_doc_comments(content);
    if !doc_markdown.is_empty() {
        let links = extract_md_links(&doc_markdown);
        validate_targets(root, parent, display_path, &links, true, bib_keys, broken);
    }
}

fn validate_targets(
    root: &Path,
    parent: &Path,
    display_path: &str,
    links: &[String],
    rs_file_filter: bool,
    bib_keys: Option<&HashSet<String>>,
    broken: &mut Vec<(String, String)>,
) {
    for link in links {
        let target = link.split_whitespace().next().unwrap_or(link);
        if is_external_or_anchor_link(target) {
            continue;
        }
        let (path_part, anchor_opt) = match target.split_once('#') {
            Some((p, a)) => (p, Some(a)),
            None => (target, None),
        };
        if path_part.is_empty() || (rs_file_filter && !looks_like_relative_file_path(path_part)) {
            continue;
        }
        let rel_dest = parent.join(path_part);
        let root_dest = root.join(path_part.trim_start_matches('/'));
        if !rel_dest.exists() && !root_dest.exists() {
            broken.push((display_path.to_string(), target.to_string()));
            continue;
        }
        if path_part.ends_with("references.bib")
            && let (Some(anchor), Some(keys)) = (anchor_opt, bib_keys)
            && !anchor.is_empty()
            && !keys.contains(anchor)
        {
            broken.push((display_path.to_string(), target.to_string()));
        }
    }
}

fn looks_like_relative_file_path(path_part: &str) -> bool {
    path_part.contains('/')
        || DOC_FILE_EXTENSIONS
            .iter()
            .any(|ext| path_part.ends_with(ext))
}

fn is_external_or_anchor_link(target: &str) -> bool {
    target.starts_with("http://")
        || target.starts_with("https://")
        || target.starts_with("mailto:")
        || target.starts_with('#')
        || target.contains("::")
}

fn load_bibtex_keys(bib_path: &Path) -> Option<HashSet<String>> {
    let content = fs::read_to_string(bib_path).ok()?;
    let mut keys = HashSet::new();
    for line in content.lines() {
        let trimmed = line.trim_start();
        if let Some(after_at) = trimmed.strip_prefix('@')
            && let Some((_, after_brace)) = after_at.split_once('{')
            && let Some((key, _)) = after_brace.split_once(',')
        {
            let clean = key.trim();
            if !clean.is_empty() {
                keys.insert(clean.to_string());
            }
        }
    }
    Some(keys)
}

fn extract_rs_doc_comments(content: &str) -> String {
    let mut out = String::new();
    let mut in_raw_str = false;
    for line in content.lines() {
        if update_raw_str_state(line, &mut in_raw_str) {
            continue;
        }
        let trimmed = line.trim_start();
        let doc_line = if let Some(rest) = trimmed.strip_prefix("///") {
            if rest.starts_with('/') {
                None
            } else {
                Some(rest)
            }
        } else {
            trimmed.strip_prefix("//!")
        };
        if let Some(text) = doc_line {
            out.push_str(text.strip_prefix(' ').unwrap_or(text));
            out.push('\n');
        }
    }
    out
}

fn extract_rs_doc_include_paths(content: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut in_raw_str = false;
    for line in content.lines() {
        if update_raw_str_state(line, &mut in_raw_str) {
            continue;
        }
        let trimmed = line.trim();
        if (trimmed.starts_with("#[doc") || trimmed.starts_with("#![doc"))
            && let Some(idx) = trimmed.find("include_str!(\"")
        {
            let after = &trimmed[idx + "include_str!(\"".len()..];
            if let Some(end_q) = after.find('"') {
                paths.push(after[..end_q].to_string());
            }
        }
    }
    paths
}

fn update_raw_str_state(line: &str, in_raw_str: &mut bool) -> bool {
    if *in_raw_str {
        if line.contains("\"#") {
            *in_raw_str = false;
        }
        return true;
    }
    if line.contains("r#\"") && !line.ends_with("\"#") {
        *in_raw_str = true;
    }
    false
}

pub(crate) fn collect_md_files(dir: &Path, out: &mut Vec<PathBuf>) {
    collect_files_by_ext(dir, "md", out);
}

pub(crate) fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    collect_files_by_ext(dir, "rs", out);
}

fn collect_files_by_ext(dir: &Path, target_ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if SKIPPED_DIRS.contains(&name) || name.starts_with("cargo-machete") {
                continue;
            }
            collect_files_by_ext(&path, target_ext, out);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some(target_ext) {
            out.push(path);
        }
    }
}

fn strip_inline_code_and_math(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '`' {
            i += 1;
            while i < chars.len() && chars[i] != '`' {
                i += 1;
            }
            i += usize::from(i < chars.len());
        } else if chars[i] == '$' {
            let is_double = i + 1 < chars.len() && chars[i + 1] == '$';
            i += if is_double { 2 } else { 1 };
            while i < chars.len() {
                if is_double && i + 1 < chars.len() && chars[i] == '$' && chars[i + 1] == '$' {
                    i += 2;
                    break;
                } else if !is_double && chars[i] == '$' {
                    i += 1;
                    break;
                }
                i += 1;
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

pub(crate) fn extract_md_links(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let mut in_fence = false;
    let mut in_display_math = false;

    for raw_line in content.lines() {
        let trimmed = raw_line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if trimmed == "$$" {
            in_display_math = !in_display_math;
            continue;
        }
        if in_display_math {
            continue;
        }
        let cleaned = strip_inline_code_and_math(raw_line);
        extract_links_from_line(&cleaned, &mut links);
    }
    links
}

fn extract_links_from_line(line: &str, links: &mut Vec<String>) {
    let mut rest = line;
    while let Some(start_bracket) = rest.find('[') {
        rest = &rest[start_bracket + 1..];
        let Some(end_bracket) = rest.find(']') else {
            break;
        };
        let after_bracket = &rest[end_bracket + 1..];
        if let Some(inside_paren) = after_bracket.strip_prefix('(')
            && let Some(end_paren) = inside_paren.find(')')
        {
            links.push(inside_paren[..end_paren].trim().to_string());
            rest = &inside_paren[end_paren + 1..];
        } else {
            rest = after_bracket;
        }
    }
}
