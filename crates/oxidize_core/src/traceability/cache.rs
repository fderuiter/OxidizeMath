//! Shared in-memory cache of Rust source files and parsed `syn::File` ASTs.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn read_sources_sequential(items: &[(String, &Path)]) -> Vec<(String, String)> {
    let mut out = Vec::with_capacity(items.len());
    for (key, path) in items {
        if let Ok(content) = std::fs::read_to_string(path) {
            out.push((key.clone(), content));
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn read_sources_batch(pending: &[(String, &Path)]) -> Vec<(String, String)> {
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(4)
        .min(8)
        .min(pending.len());
    if workers <= 1 {
        return read_sources_sequential(pending);
    }
    let chunk_size = pending.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let handles: Vec<_> = pending
            .chunks(chunk_size)
            .map(|chunk| scope.spawn(move || read_sources_sequential(chunk)))
            .collect();
        let mut loaded = Vec::with_capacity(pending.len());
        for handle in handles {
            if let Ok(part) = handle.join() {
                loaded.extend(part);
            }
        }
        loaded
    })
}

#[cfg(target_arch = "wasm32")]
fn read_sources_batch(pending: &[(String, &Path)]) -> Vec<(String, String)> {
    read_sources_sequential(pending)
}

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
    prefetched: HashMap<String, String>,
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

    /// Prefetches uncached UTF-8 Rust sources from `paths` into memory without invoking `syn::parse_file`.
    pub fn preload_sources<P: AsRef<Path>>(&mut self, paths: &[P]) {
        let mut pending: Vec<(String, &Path)> = Vec::new();
        let mut queued = HashSet::new();
        for raw_path in paths {
            let path = raw_path.as_ref();
            let key = Self::normalize_key(path);
            if !self.entries.contains_key(&key)
                && !self.prefetched.contains_key(&key)
                && queued.insert(key.clone())
            {
                pending.push((key, path));
            }
        }
        if pending.is_empty() {
            return;
        }
        for (key, content) in read_sources_batch(&pending) {
            self.prefetched.insert(key, content);
        }
    }

    /// Parses `content` once with `syn::parse_file`, stores the entry at `key`, and returns a reference to it.
    pub fn insert_source(&mut self, key: String, content: String) -> &CachedRustFile {
        self.prefetched.remove(&key);
        let ast = syn::parse_file(&content).ok();
        self.parse_calls += 1;
        match self.entries.entry(key) {
            Entry::Occupied(mut occupied) => {
                occupied.insert(CachedRustFile { content, ast });
                occupied.into_mut()
            }
            Entry::Vacant(vacant) => vacant.insert(CachedRustFile { content, ast }),
        }
    }

    /// Returns the cached entry for `path`, reading from prefetched memory or the filesystem and parsing once on a cache miss.
    pub fn get_or_load_fs(&mut self, path: &Path) -> Option<&CachedRustFile> {
        let key = Self::normalize_key(path);
        match self.entries.entry(key) {
            Entry::Occupied(occupied) => Some(occupied.into_mut()),
            Entry::Vacant(vacant) => {
                let content = match self.prefetched.remove(vacant.key()) {
                    Some(prefetched) => prefetched,
                    None => std::fs::read_to_string(path).ok()?,
                };
                let ast = syn::parse_file(&content).ok();
                self.parse_calls += 1;
                Some(vacant.insert(CachedRustFile { content, ast }))
            }
        }
    }
}
