use oxidize_core::traceability::AstCache;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Discovers the workspace root path and member list via
/// [`oxidize_core::workspace::get_workspace_members`].
pub(crate) fn discover_workspace() -> (PathBuf, Vec<String>) {
    let (is_root, members) = oxidize_core::workspace::get_workspace_members();
    let root = if is_root {
        PathBuf::from(".")
    } else {
        oxidize_core::workspace::find_workspace_root().unwrap_or_else(|| PathBuf::from("../.."))
    };
    (root, members)
}

/// Resolves the workspace root directory relative to the current directory.
pub(crate) fn workspace_root_dir() -> PathBuf {
    let (is_root, _) = oxidize_core::workspace::get_workspace_members();
    if is_root {
        PathBuf::from(".")
    } else {
        oxidize_core::workspace::find_workspace_root().unwrap_or_else(|| PathBuf::from("../.."))
    }
}

/// Resolves a workspace-relative path against an already-discovered `workspace_root`.
pub(crate) fn resolve_workspace_path_from(workspace_root: &Path, rel: &str) -> PathBuf {
    if workspace_root == Path::new(".") {
        PathBuf::from(rel)
    } else {
        workspace_root.join(rel)
    }
}

/// Resolves a workspace-relative path against [`workspace_root_dir`].
pub(crate) fn resolve_workspace_path(rel: &str) -> PathBuf {
    resolve_workspace_path_from(&workspace_root_dir(), rel)
}

/// RAII guard that temporarily switches the working directory to [`workspace_root_dir`]
/// when invoked from a workspace member crate directory, restoring the original directory on drop.
pub(crate) struct WorkspaceRootGuard {
    prev_dir: Option<PathBuf>,
}

impl WorkspaceRootGuard {
    pub(crate) fn enter() -> Self {
        let root = workspace_root_dir();
        if root == Path::new(".") {
            return Self { prev_dir: None };
        }
        let prev_dir = std::env::current_dir().ok();
        if std::env::set_current_dir(&root).is_err() {
            return Self { prev_dir: None };
        }
        Self { prev_dir }
    }
}

impl Drop for WorkspaceRootGuard {
    fn drop(&mut self) {
        if let Some(prev) = &self.prev_dir {
            let _ = std::env::set_current_dir(prev);
        }
    }
}

/// Cached UTF-8 source text and parsed `toml::Value` for a `Cargo.toml` manifest.
pub(crate) struct CachedManifest {
    pub(crate) content: String,
    pub(crate) parsed: toml::Value,
}

/// In-memory cache ensuring each `Cargo.toml` manifest is parsed once and `[lints]`
/// inheritance is validated at most once per crate during batched verification runs.
#[derive(Default)]
pub(crate) struct ManifestCache {
    entries: HashMap<String, CachedManifest>,
    checked_lints: HashMap<String, bool>,
    parse_calls: usize,
    lint_check_calls: usize,
    #[cfg(test)]
    per_key_parses: HashMap<String, usize>,
}

impl ManifestCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn normalize_key(path: &Path) -> String {
        AstCache::normalize_key(path)
    }

    /// Reads and parses `path` on a cache miss, returning a reference to the cached manifest
    /// or an error message if reading or TOML parsing fails.
    pub(crate) fn get_or_load(&mut self, path: &Path) -> Result<&CachedManifest, String> {
        let key = Self::normalize_key(path);
        if !self.entries.contains_key(&key) {
            let content = fs::read_to_string(path)
                .map_err(|err| format!("[!] Failed to read {}: {}", path.display(), err))?;
            let parsed = toml::from_str::<toml::Value>(&content)
                .map_err(|err| format!("[!] Failed to parse {}: {}", path.display(), err))?;
            self.parse_calls += 1;
            #[cfg(test)]
            {
                *self.per_key_parses.entry(key.clone()).or_insert(0) += 1;
            }
            self.entries
                .insert(key.clone(), CachedManifest { content, parsed });
        }
        Ok(self
            .entries
            .get(&key)
            .expect("inserted manifest cache entry must exist"))
    }

    /// Runs `lint_policy::check_manifest_lints` at most once per normalized manifest key.
    pub(crate) fn check_manifest_lints_once(
        &mut self,
        member: &str,
        manifest_path: &Path,
        parsed: &toml::Value,
    ) -> bool {
        let key = Self::normalize_key(manifest_path);
        if let Some(&passed) = self.checked_lints.get(&key) {
            return passed;
        }
        self.lint_check_calls += 1;
        let passed =
            crate::policy_audit::lint_policy::check_manifest_lints(member, manifest_path, parsed);
        self.checked_lints.insert(key, passed);
        passed
    }

    /// Drops any cached TOML and lint-validation state for `path` after an `auto_fix` write.
    pub(crate) fn invalidate(&mut self, path: &Path) {
        let key = Self::normalize_key(path);
        self.entries.remove(&key);
        self.checked_lints.remove(&key);
    }

    #[cfg(test)]
    pub(crate) fn parse_calls(&self) -> usize {
        self.parse_calls
    }

    #[cfg(test)]
    pub(crate) fn lint_check_calls(&self) -> usize {
        self.lint_check_calls
    }

    #[cfg(test)]
    pub(crate) fn parse_count_for(&self, path: &Path) -> usize {
        let key = Self::normalize_key(path);
        self.per_key_parses.get(&key).copied().unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, path: &Path) -> bool {
        let key = Self::normalize_key(path);
        self.entries.contains_key(&key)
    }
}
