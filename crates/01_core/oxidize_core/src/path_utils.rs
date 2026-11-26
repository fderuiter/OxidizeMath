use std::path::Path;

/// Normalizes a path to use forward slashes consistently.
/// It handles converting backslashes to forward slashes for cross-platform compatibility.
pub fn normalize_path<P: AsRef<Path>>(path: P) -> String {
    let mut path_str = path.as_ref().to_str().unwrap_or("");
    if path_str.starts_with(r"\\?\") {
        path_str = &path_str[4..];
    }
    path_str.replace('\\', "/")
}

/// Strips a prefix from a path and returns the normalized string representation.
pub fn strip_and_normalize<P: AsRef<Path>, B: AsRef<Path>>(path: P, prefix: B) -> Option<String> {
    path.as_ref().strip_prefix(prefix).ok().map(normalize_path)
}

/// Joins a suffix to a base path and returns the normalized string representation.
pub fn join_and_normalize<P: AsRef<Path>, B: AsRef<Path>>(base: P, suffix: B) -> String {
    normalize_path(base.as_ref().join(suffix))
}
