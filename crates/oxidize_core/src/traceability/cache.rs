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

    /// Normalizes a file path to a canonical forward-slash cache key without leading `./` prefixes.
    pub fn normalize_key<P: AsRef<Path>>(path: P) -> String {
        let norm = crate::path_utils::normalize_path(path);
        let mut trimmed = norm.as_str();
        while let Some(rest) = trimmed.strip_prefix("./") {
            trimmed = rest;
        }
        trimmed.to_string()
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
