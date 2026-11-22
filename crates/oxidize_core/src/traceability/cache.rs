//! Shared in-memory cache of Rust source files and parsed `syn::File` ASTs.

use std::collections::HashMap;
use std::path::Path;

/// Cached UTF-8 source text and optional parsed syntax tree for a Rust file.
pub struct CachedRustFile {
    /// Raw UTF-8 source text of the Rust file.
    pub content: String,
    /// Parsed `syn::File` syntax tree, or `None` when syntax parsing failed.
    pub ast: Option<syn::File>,
}

/// In-memory cache ensuring each unique Rust file is parsed with `syn::parse_file` at most once.
#[derive(Default)]
pub struct AstCache {
    entries: HashMap<String, CachedRustFile>,
    parse_calls: usize,
}

impl AstCache {
    /// Creates an empty [`AstCache`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Normalizes a file path to a canonical forward-slash cache key without leading `./` prefixes
    /// and with internal `.` and `..` segments collapsed lexically.
    pub fn normalize_key<P: AsRef<Path>>(path: P) -> String {
        let norm = crate::path_utils::normalize_path(path);
        let leading_slash = norm.starts_with('/');
        let mut stack: Vec<&str> = Vec::new();
        for seg in norm.split('/') {
            if seg.is_empty() || seg == "." {
                continue;
            }
            if seg == ".." {
                if let Some(&last) = stack.last()
                    && last != ".."
                {
                    stack.pop();
                    continue;
                }
                if !leading_slash {
                    stack.push("..");
                }
                continue;
            }
            stack.push(seg);
        }
        let joined = stack.join("/");
        if leading_slash {
            format!("/{joined}")
        } else {
            joined
        }
    }

    /// Returns the total number of `syn::parse_file` invocations performed by this cache.
    pub fn parse_calls(&self) -> usize {
        self.parse_calls
    }

    /// Returns the number of cached Rust files stored in this cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no Rust files have been cached yet.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over all normalized file keys currently stored in this cache.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Looks up a cached Rust file by normalized key.
    pub fn get(&self, key: &str) -> Option<&CachedRustFile> {
        self.entries.get(key)
    }

    /// Parses `content` once with `syn::parse_file`, stores the entry at `key`, and returns a reference to it.
    pub fn insert_source(&mut self, key: String, content: String) -> &CachedRustFile {
        let ast = syn::parse_file(&content).ok();
        self.parse_calls += 1;
        self.entries
            .insert(key.clone(), CachedRustFile { content, ast });
        self.entries
            .get(&key)
            .expect("inserted cache entry must exist")
    }

    /// Returns the cached entry for `path`, reading from the filesystem and parsing once on a cache miss.
    pub fn get_or_load_fs(&mut self, path: &Path) -> Option<&CachedRustFile> {
        let key = Self::normalize_key(path);
        if !self.entries.contains_key(&key) {
            let content = std::fs::read_to_string(path).ok()?;
            let ast = syn::parse_file(&content).ok();
            self.parse_calls += 1;
            self.entries
                .insert(key.clone(), CachedRustFile { content, ast });
        }
        self.entries.get(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::AstCache;

    #[test]
    fn normalize_key_collapses_dot_and_parent_segments() {
        assert_eq!(
            AstCache::normalize_key("./crates/pure_math/src/../src/./lib.rs"),
            "crates/pure_math/src/lib.rs"
        );
        assert_eq!(
            AstCache::normalize_key("/app/OxidizeMath/crates/../crates/oxidize_core/src/lib.rs"),
            "/app/OxidizeMath/crates/oxidize_core/src/lib.rs"
        );
        assert_eq!(
            AstCache::normalize_key("crates\\domain_ai\\src\\.\\mod.rs"),
            "crates/domain_ai/src/mod.rs"
        );
    }

    #[test]
    fn get_or_load_fs_reuses_parsed_entry_across_equivalent_paths() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "oxidize_ast_cache_test_{}_{}",
            std::process::id(),
            id
        ));
        let sub = root.join("sub");
        std::fs::create_dir_all(&sub).expect("mkdir");
        let file = sub.join("sample.rs");
        std::fs::write(&file, "pub fn answer() -> i32 { 42 }\n").expect("write");

        let mut cache = AstCache::new();
        assert!(cache.get_or_load_fs(&file).is_some());
        let alias = sub.join("..").join("sub").join(".").join("sample.rs");
        assert!(cache.get_or_load_fs(&alias).is_some());
        assert_eq!(cache.parse_calls(), 1);
        assert_eq!(cache.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }
}
