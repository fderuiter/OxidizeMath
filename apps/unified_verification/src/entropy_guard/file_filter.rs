use super::is_file_allowlisted;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ScanFileKind {
    Skip,
    RustAst,
    Text,
}

fn is_ignored_license_file(file_name: &str) -> bool {
    let upper = file_name.to_ascii_uppercase();
    matches!(upper.as_str(), "LICENSE" | "NOTICE" | "COPYING")
        || upper.starts_with("LICENSE-")
        || upper.starts_with("NOTICE-")
        || upper.starts_with("COPYING-")
        || upper.starts_with("LICENSE.")
        || upper.starts_with("NOTICE.")
        || upper.starts_with("COPYING.")
}

pub(super) fn classify_scan_file(path: &Path, target_extensions: &[String]) -> ScanFileKind {
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if file_name.is_empty() || is_ignored_license_file(file_name) {
        return ScanFileKind::Skip;
    }
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if !ext.is_empty()
        && target_extensions
            .iter()
            .any(|e| e.trim_start_matches('.').eq_ignore_ascii_case(ext))
    {
        return if ext.eq_ignore_ascii_case("rs") {
            ScanFileKind::RustAst
        } else {
            ScanFileKind::Text
        };
    }
    if ext.is_empty() || file_name.starts_with(".env.") {
        return ScanFileKind::Text;
    }
    ScanFileKind::Skip
}

pub(super) fn is_ignored_dir(entry: &walkdir::DirEntry) -> bool {
    if entry.file_type().is_dir() {
        let name = entry.file_name().to_string_lossy();
        if name == "target" || name == ".git" {
            return true;
        }
    }
    false
}

pub(super) fn is_scannable_entry(entry: &walkdir::DirEntry) -> Option<bool> {
    let is_symlink = entry.path_is_symlink();
    if entry.file_type().is_file() || (is_symlink && entry.path().is_file()) {
        Some(is_symlink)
    } else {
        None
    }
}

pub(super) fn should_visit_file(
    path: &Path,
    is_symlink: bool,
    root_pair: Option<(&Path, &Path)>,
    allowlist: &[String],
    visited: &mut HashSet<PathBuf>,
) -> bool {
    let path_str = path.to_string_lossy();
    if path_str
        .split(['/', '\\'])
        .any(|seg| seg == "target" || seg == ".git")
    {
        return false;
    }
    if is_file_allowlisted(path, allowlist) {
        return false;
    }
    let dedup_key = if is_symlink {
        let can = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if is_file_allowlisted(&can, allowlist) {
            return false;
        }
        can
    } else if let Some((raw_root, can_root)) = root_pair
        && let Ok(rel) = path.strip_prefix(raw_root)
    {
        if rel.as_os_str().is_empty() {
            can_root.to_path_buf()
        } else {
            can_root.join(rel)
        }
    } else {
        path.to_path_buf()
    };
    visited.insert(dedup_key)
}
