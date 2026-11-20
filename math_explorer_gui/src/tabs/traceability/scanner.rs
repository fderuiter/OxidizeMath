#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// theory_verification!

#[cfg(not(target_arch = "wasm32"))]
use oxidize_core::vfs::VirtualFileSystem;
#[cfg(not(target_arch = "wasm32"))]
use std::future::Future;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::pin::Pin;

pub(super) fn has_traversal_sequence(raw: &str) -> bool {
    raw.contains('\0')
        || raw.split(['/', '\\']).any(|seg| seg == "..")
        || std::path::Path::new(raw)
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn has_workspace_header(manifest: &str) -> bool {
    manifest.lines().any(|line| {
        let h = line.split('#').next().unwrap_or("").trim();
        let k = h.strip_prefix('[').and_then(|s| s.strip_suffix(']'));
        k.is_some_and(|s| {
            let key = s.trim();
            !s.starts_with('[')
                && !s.ends_with(']')
                && (key == "workspace"
                    || key
                        .strip_prefix("workspace")
                        .is_some_and(|rest| rest.trim_start().starts_with('.')))
        })
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn canonical_workspace_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?.canonicalize().ok()?;
    cwd.ancestors()
        .take(64)
        .find(|dir| {
            std::fs::read_to_string(dir.join("Cargo.toml"))
                .is_ok_and(|m| has_workspace_header(&m))
        })
        .map(Path::to_path_buf)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn join_relative_segments(base: &Path, rel: &str) -> PathBuf {
    let mut out = base.to_path_buf();
    for seg in rel.split(['/', '\\']) {
        if !seg.is_empty() && seg != "." {
            out.push(seg);
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn validate_session_repo_path(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || has_traversal_sequence(trimmed) {
        return None;
    }
    let canonical_root = canonical_workspace_root()?;
    let candidate = Path::new(trimmed);
    let resolved = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        join_relative_segments(&canonical_root, trimmed)
    };
    let canonical_candidate = resolved.canonicalize().ok()?;
    if !canonical_candidate.is_dir() || !canonical_candidate.starts_with(&canonical_root) {
        return None;
    }
    let normalized = oxidize_core::path_utils::normalize_path(&canonical_candidate);
    (!normalized.is_empty()).then_some(normalized)
}

#[cfg(target_arch = "wasm32")]
pub(super) fn validate_session_repo_path(raw: &str) -> Option<String> {
    let _ = has_traversal_sequence(raw);
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn has_confined_child(canonical_root: &Path, child_name: &str, expect_dir: bool) -> bool {
    let child = canonical_root.join(child_name);
    let Ok(meta) = std::fs::symlink_metadata(&child) else {
        return false;
    };
    if meta.file_type().is_symlink() {
        return false;
    }
    if expect_dir != meta.is_dir() || (!expect_dir && !meta.is_file()) {
        return false;
    }
    child
        .canonicalize()
        .is_ok_and(|resolved| resolved.starts_with(canonical_root))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn validate_picked_repo_path(path: &Path) -> Result<String, String> {
    let raw = path.to_string_lossy();
    let trimmed = raw.trim();
    if trimmed.is_empty() || has_traversal_sequence(trimmed) {
        return Err("Selected directory path is empty or contains traversal segments.".to_string());
    }
    let meta = std::fs::symlink_metadata(path)
        .map_err(|err| format!("Selected path cannot be accessed ({}): {}", path.display(), err))?;
    if meta.file_type().is_symlink() {
        return Err("Selected codebase path must not be a symbolic link.".to_string());
    }
    if !meta.is_dir() {
        return Err(format!("Selected path is not a directory: {}", path.display()));
    }
    let canonical = path
        .canonicalize()
        .map_err(|err| format!("Failed to resolve directory {}: {}", path.display(), err))?;
    let has_registry = has_confined_child(&canonical, "traceability.toml", false);
    let has_papers = has_confined_child(&canonical, "papers", true);
    match (has_registry, has_papers) {
        (true, true) => {
            let normalized = oxidize_core::path_utils::normalize_path(&canonical);
            if normalized.is_empty() {
                Err("Selected directory resolved to an empty path.".to_string())
            } else {
                Ok(normalized)
            }
        }
        (false, false) => Err(format!(
            "Invalid repository folder '{}': missing both traceability.toml and papers/ directory.",
            path.display()
        )),
        (false, true) => Err(format!(
            "Invalid repository folder '{}': missing traceability.toml.",
            path.display()
        )),
        (true, false) => Err(format!(
            "Invalid repository folder '{}': missing papers/ directory.",
            path.display()
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn revalidate_repo_path(raw: &str, user_picked: bool) -> Option<String> {
    if user_picked {
        validate_picked_repo_path(Path::new(raw.trim())).ok()
    } else {
        validate_session_repo_path(raw)
    }
}

#[cfg(target_arch = "wasm32")]
pub(super) fn revalidate_repo_path(raw: &str, _user_picked: bool) -> Option<String> {
    validate_session_repo_path(raw)
}

#[cfg(not(target_arch = "wasm32"))]
struct RootedVfs {
    root: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl RootedVfs {
    fn new(root: &Path) -> Result<Self, std::io::Error> {
        let canonical = root.canonicalize()?;
        Ok(Self { root: canonical })
    }

    fn resolve_confined(&self, rel: &str) -> Result<PathBuf, std::io::Error> {
        if has_traversal_sequence(rel) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "Path traversal rejected",
            ));
        }
        let joined = join_relative_segments(&self.root, rel);
        let canonical = joined.canonicalize()?;
        if !canonical.starts_with(&self.root) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "Path escaped repository root",
            ));
        }
        Ok(canonical)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl VirtualFileSystem for RootedVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move {
            let resolved = self.resolve_confined(&path)?;
            std::fs::read_to_string(resolved)
        })
    }

    fn write_to_file(
        &self,
        _path: &str,
        _content: &[u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>> {
        Box::pin(async {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "Read-only traceability scan",
            ))
        })
    }

    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
        let path = path.to_string();
        Box::pin(async move {
            let resolved = self.resolve_confined(&path)?;
            let mut files = Vec::new();
            for entry in std::fs::read_dir(resolved)?.flatten() {
                if let Some(s) = entry.file_name().to_str() {
                    files.push(s.to_string());
                }
            }
            files.sort();
            Ok(files)
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) async fn execute_workspace_scan<V: VirtualFileSystem>(
    vfs: V,
    base_dir: String,
) -> Result<oxidize_core::traceability::TraceabilityReport, String> {
    let code_dirs = [
        format!("{base_dir}math_explorer/src"),
        format!("{base_dir}math_explorer_gui/src/tabs"),
    ];
    let papers_dir = format!("{base_dir}papers");
    let crate_base = format!("{base_dir}crates");
    let engine = oxidize_core::traceability::TraceabilityEngine::new(vfs);

    let mut crate_dirs = Vec::new();
    if let Ok(crates) = engine.vfs.list_dir(&crate_base).await {
        for crate_name in crates {
            crate_dirs.push(format!("{crate_base}/{crate_name}/src"));
        }
    }

    let mut all_dirs: Vec<&str> = code_dirs.iter().map(String::as_str).collect();
    all_dirs.extend(crate_dirs.iter().map(String::as_str));
    engine
        .scan_repository(&all_dirs, &papers_dir, false)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn execute_rooted_scan_blocking(
    repo_dir: &Path,
) -> Result<oxidize_core::traceability::TraceabilityReport, String> {
    let vfs = RootedVfs::new(repo_dir).map_err(|e| e.to_string())?;
    futures::executor::block_on(execute_workspace_scan(vfs, String::new()))
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn dispatch_native_refresh(
    repo_path: Option<String>,
    tx: std::sync::mpsc::Sender<Result<oxidize_core::traceability::TraceabilityReport, String>>,
    cert_tx: std::sync::mpsc::Sender<Option<super::web_snapshot::VerificationCertificate>>,
) {
    if let Some(repo_dir) = repo_path {
        std::thread::spawn(move || {
            let cert = super::web_snapshot::load_local_certificate(&repo_dir);
            let res = execute_rooted_scan_blocking(Path::new(&repo_dir));
            let _ = cert_tx.send(cert);
            let _ = tx.send(res);
        });
        return;
    }

    if let Some(ws_root) = canonical_workspace_root() {
        #[cfg(test)]
        {
            static DEFAULT_SCAN_CACHE: std::sync::OnceLock<
                Result<oxidize_core::traceability::TraceabilityReport, String>,
            > = std::sync::OnceLock::new();
            let cached = DEFAULT_SCAN_CACHE
                .get_or_init(|| execute_rooted_scan_blocking(&ws_root))
                .clone();
            let cert = super::web_snapshot::load_local_certificate(&ws_root.to_string_lossy());
            let _ = cert_tx.send(cert);
            let _ = tx.send(cached);
            return;
        }
        #[cfg(not(test))]
        {
            std::thread::spawn(move || {
                let cert = super::web_snapshot::load_local_certificate(&ws_root.to_string_lossy());
                let res = execute_rooted_scan_blocking(&ws_root);
                let _ = cert_tx.send(cert);
                let _ = tx.send(res);
            });
            return;
        }
    }

    match super::web_snapshot::embedded_traceability() {
        Ok((report, cert)) => {
            let _ = cert_tx.send(cert);
            let _ = tx.send(Ok(report));
        }
        Err(err) => {
            let _ = tx.send(Err(err));
        }
    }
}
