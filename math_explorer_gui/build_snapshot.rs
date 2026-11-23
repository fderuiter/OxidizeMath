//! Deterministic build-time generator for `traceability_manifest.json`.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use oxidize_core::path_utils;
use oxidize_core::traceability::{
    TraceabilityEngine, TraceabilityManifestPayload, VerificationCertificate,
};
use oxidize_core::vfs::VirtualFileSystem;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

const MANIFEST_FILENAME: &str = "traceability_manifest.json";

/// Writes `contents` to `path` only when the file does not exist or its byte contents differ,
/// preserving filesystem modification times (`mtime`) across no-op builds.
pub(crate) fn write_if_changed(path: &Path, contents: &[u8]) -> std::io::Result<bool> {
    if let Ok(existing) = fs::read(path)
        && existing == contents
    {
        return Ok(false);
    }
    fs::write(path, contents)?;
    Ok(true)
}

fn should_emit_rerun_for_vfs_path(rel: &str) -> bool {
    let clean = rel.trim_start_matches("./").trim_start_matches(['/', '\\']);
    clean == "traceability.toml"
        || clean == "Verification_Certificate.json"
        || clean == "papers"
        || clean.starts_with("papers/")
        || clean == "math_explorer_gui/src/tabs"
        || clean.starts_with("math_explorer_gui/src/tabs/")
}

/// Read-only build-script [`VirtualFileSystem`] adapter rooted at `..` that emits
/// scoped forward-slash `cargo:rerun-if-changed` directives for traceability inputs
/// (`traceability.toml`, `Verification_Certificate.json`, `papers/`, and `src/tabs`)
/// without subscribing `math_explorer_gui` to sibling crate `.rs` source trees.
struct BuildSnapshotVfs {
    root: PathBuf,
    emitted: RefCell<BTreeSet<String>>,
}

impl BuildSnapshotVfs {
    fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
            emitted: RefCell::new(BTreeSet::new()),
        }
    }

    fn resolve_path(&self, rel: &str) -> PathBuf {
        let clean = rel.trim_start_matches("./").trim_start_matches(['/', '\\']);
        if clean.is_empty() {
            self.root.clone()
        } else if self.root == Path::new("..") && clean == "math_explorer_gui" {
            PathBuf::from(".")
        } else if self.root == Path::new("..") && clean.starts_with("math_explorer_gui/") {
            PathBuf::from(&clean["math_explorer_gui/".len()..])
        } else {
            self.root.join(clean)
        }
    }

    fn emit_if_exists(&self, path: &Path) {
        if !path.exists() {
            return;
        }
        let normalized = path_utils::normalize_path(path);
        if !normalized.trim().is_empty() && self.emitted.borrow_mut().insert(normalized.clone()) {
            println!("cargo:rerun-if-changed={normalized}");
        }
    }

    fn emit_scoped_vfs_path(&self, vfs_rel: &str, disk_path: &Path) {
        if should_emit_rerun_for_vfs_path(vfs_rel) {
            self.emit_if_exists(disk_path);
        }
    }
}

impl VirtualFileSystem for BuildSnapshotVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let vfs_rel = path.to_string();
        let disk_path = self.resolve_path(path);
        Box::pin(async move {
            let content = fs::read_to_string(&disk_path)?;
            self.emit_scoped_vfs_path(&vfs_rel, &disk_path);
            Ok(content)
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
                "Read-only build snapshot VFS",
            ))
        })
    }

    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
        let vfs_rel = path.to_string();
        let disk_path = self.resolve_path(path);
        Box::pin(async move {
            let entries = fs::read_dir(&disk_path)?;
            self.emit_scoped_vfs_path(&vfs_rel, &disk_path);
            let mut names = Vec::new();
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
            names.sort();
            Ok(names)
        })
    }
}

/// Generates `traceability_manifest.json` into `out_dir` and Cargo's target directory.
pub fn generate_traceability_snapshot(out_dir: &Path) {
    let ws_root = Path::new("..");
    let vfs = BuildSnapshotVfs::new(ws_root);
    vfs.emit_if_exists(Path::new("build_snapshot.rs"));
    emit_papers_rerun_directives(ws_root, &vfs);

    let cert_path = ws_root.join("Verification_Certificate.json");
    vfs.emit_if_exists(&cert_path);
    let certificate = read_optional_certificate(&cert_path);

    let engine = TraceabilityEngine::new(vfs);
    let report = futures::executor::block_on(engine.scan_workspace("", false))
        .expect("Failed to execute TraceabilityEngine workspace scan in build.rs");

    let payload = TraceabilityManifestPayload {
        report,
        certificate,
    };
    let json = serialize_sorted_manifest(&payload);
    write_manifest_outputs(out_dir, json.as_bytes());
}

fn emit_papers_rerun_directives(ws_root: &Path, vfs: &BuildSnapshotVfs) {
    let papers_dir = ws_root.join("papers");
    vfs.emit_if_exists(&papers_dir);
    let Ok(entries) = fs::read_dir(&papers_dir) else {
        return;
    };
    let mut tex_paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("tex"))
        .collect();
    tex_paths.sort();
    for tex_path in tex_paths {
        vfs.emit_if_exists(&tex_path);
    }
}

fn read_optional_certificate(cert_path: &Path) -> Option<VerificationCertificate> {
    let raw = fs::read_to_string(cert_path).ok()?;
    serde_json::from_str::<VerificationCertificate>(&raw).ok()
}

fn serialize_sorted_manifest(payload: &TraceabilityManifestPayload) -> String {
    let sorted_value = serde_json::to_value(payload)
        .expect("Failed to convert TraceabilityManifestPayload to JSON value");
    let mut json = serde_json::to_string(&sorted_value)
        .expect("Failed to serialize TraceabilityManifestPayload JSON");
    json.push('\n');
    json
}

fn write_manifest_outputs(out_dir: &Path, json_bytes: &[u8]) {
    write_if_changed(&out_dir.join(MANIFEST_FILENAME), json_bytes)
        .expect("Failed to write OUT_DIR/traceability_manifest.json");

    if let Some(profile_dir) = out_dir
        .ancestors()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("build"))
        .and_then(Path::parent)
    {
        let _ = write_if_changed(&profile_dir.join(MANIFEST_FILENAME), json_bytes);
        if let Some(parent_target) = profile_dir.parent() {
            let _ = write_if_changed(&parent_target.join(MANIFEST_FILENAME), json_bytes);
        }
    }

    if let Some(custom_target) = env::var_os("CARGO_TARGET_DIR") {
        let target_path = PathBuf::from(custom_target);
        if target_path.is_dir() {
            let _ = write_if_changed(&target_path.join(MANIFEST_FILENAME), json_bytes);
        }
    } else if Path::new("../target").is_dir() {
        let _ = write_if_changed(&Path::new("../target").join(MANIFEST_FILENAME), json_bytes);
    }
}
