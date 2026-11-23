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

/// Read-only build-script [`VirtualFileSystem`] adapter rooted at `..` that emits
/// deduplicated forward-slash `cargo:rerun-if-changed` directives for existing paths.
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
}

impl VirtualFileSystem for BuildSnapshotVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let disk_path = self.resolve_path(path);
        Box::pin(async move {
            let content = fs::read_to_string(&disk_path)?;
            self.emit_if_exists(&disk_path);
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
        let disk_path = self.resolve_path(path);
        Box::pin(async move {
            let entries = fs::read_dir(&disk_path)?;
            self.emit_if_exists(&disk_path);
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
    write_manifest_outputs(out_dir, &json);
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

fn write_manifest_outputs(out_dir: &Path, json: &str) {
    fs::write(out_dir.join(MANIFEST_FILENAME), json)
        .expect("Failed to write OUT_DIR/traceability_manifest.json");

    if let Some(profile_dir) = out_dir
        .ancestors()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("build"))
        .and_then(Path::parent)
    {
        let _ = fs::write(profile_dir.join(MANIFEST_FILENAME), json);
        if let Some(parent_target) = profile_dir.parent() {
            let _ = fs::write(parent_target.join(MANIFEST_FILENAME), json);
        }
    }

    if let Some(custom_target) = env::var_os("CARGO_TARGET_DIR") {
        let target_path = PathBuf::from(custom_target);
        if target_path.is_dir() {
            let _ = fs::write(target_path.join(MANIFEST_FILENAME), json);
        }
    } else if Path::new("../target").is_dir() {
        let _ = fs::write(Path::new("../target").join(MANIFEST_FILENAME), json);
    }
}
