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

const CORE_DEP_CRATES: &[&str] = &[
    "diagnostics",
    "federated_registry",
    "math_commons",
    "oxidize_core",
    "scientific_metadata",
];

const FEATURE_DEP_CRATES: &[(&str, &str)] = &[
    ("CARGO_FEATURE_AI", "crates/domain_ai/src"),
    ("CARGO_FEATURE_APPLIED", "crates/domain_applied/src"),
    ("CARGO_FEATURE_BIOLOGY", "crates/domain_biology/src"),
    ("CARGO_FEATURE_CLIMATE", "crates/domain_climate/src"),
    (
        "CARGO_FEATURE_EPIDEMIOLOGY",
        "crates/domain_epidemiology/src",
    ),
    ("CARGO_FEATURE_MATH_EXPLORER", "math_explorer/src"),
    ("CARGO_FEATURE_PHYSICS", "crates/domain_physics/src"),
    ("CARGO_FEATURE_PURE_MATH", "crates/pure_math/src"),
];

/// Read-only build-script [`VirtualFileSystem`] adapter rooted at `..` that emits
/// deduplicated forward-slash `cargo:rerun-if-changed` directives for existing paths.
struct BuildSnapshotVfs {
    root: PathBuf,
    emitted: RefCell<BTreeSet<String>>,
    enabled_features: BTreeSet<String>,
}

impl BuildSnapshotVfs {
    fn new(root: &Path) -> Self {
        let enabled_features = env::vars_os()
            .filter_map(|(k, _)| {
                k.to_str()?
                    .strip_prefix("CARGO_FEATURE_")
                    .map(|f| f.to_ascii_lowercase())
            })
            .collect();
        Self {
            root: root.to_path_buf(),
            emitted: RefCell::new(BTreeSet::new()),
            enabled_features,
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

fn filter_disabled_tab_mods(content: &str, enabled_features: &BTreeSet<String>) -> String {
    let mut out = String::with_capacity(content.len());
    let mut skip_next_mod = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(feat) = trimmed
            .strip_prefix("#[cfg(feature = \"")
            .and_then(|rest| rest.strip_suffix("\")]"))
        {
            if !enabled_features.contains(feat) {
                skip_next_mod = true;
                out.push('\n');
                continue;
            }
        }
        if skip_next_mod && trimmed.starts_with("pub mod ") && trimmed.ends_with(';') {
            skip_next_mod = false;
            out.push('\n');
            continue;
        }
        skip_next_mod = false;
        out.push_str(line);
        out.push('\n');
    }
    out
}

impl VirtualFileSystem for BuildSnapshotVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let clean = path
            .trim_start_matches("./")
            .trim_start_matches(['/', '\\'])
            .to_string();
        let disk_path = self.resolve_path(path);
        Box::pin(async move {
            let content = fs::read_to_string(&disk_path)?;
            self.emit_if_exists(&disk_path);
            if clean == "math_explorer_gui/src/tabs/mod.rs" {
                Ok(filter_disabled_tab_mods(&content, &self.enabled_features))
            } else {
                Ok(content)
            }
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

fn active_code_dirs(vfs: &BuildSnapshotVfs) -> Vec<String> {
    let mut dirs = Vec::new();
    for crate_name in CORE_DEP_CRATES {
        let dir = format!("crates/{crate_name}/src");
        vfs.emit_if_exists(&vfs.resolve_path(&dir));
        dirs.push(dir);
    }
    for &(feature_env, dir) in FEATURE_DEP_CRATES {
        if env::var_os(feature_env).is_some() {
            vfs.emit_if_exists(&vfs.resolve_path(dir));
            dirs.push(dir.to_string());
        }
    }
    dirs.push("math_explorer_gui/src/tabs".to_string());
    dirs.sort();
    dirs
}

/// Generates `traceability_manifest.json` into `out_dir` and Cargo's target directory.
pub fn generate_traceability_snapshot(out_dir: &Path) {
    let ws_root = Path::new("..");
    let vfs = BuildSnapshotVfs::new(ws_root);
    vfs.emit_if_exists(Path::new("build_snapshot.rs"));

    let cert_path = ws_root.join("Verification_Certificate.json");
    vfs.emit_if_exists(&cert_path);
    let certificate = read_optional_certificate(&cert_path);

    let code_dirs = active_code_dirs(&vfs);
    let code_dir_refs: Vec<&str> = code_dirs.iter().map(String::as_str).collect();
    let engine = TraceabilityEngine::new(vfs);
    let report =
        futures::executor::block_on(engine.scan_repository(&code_dir_refs, "papers", false))
            .expect("Failed to execute TraceabilityEngine scan in build.rs");

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

fn write_if_changed(path: &Path, content: &str) -> std::io::Result<()> {
    if fs::read(path).is_ok_and(|existing| existing == content.as_bytes()) {
        return Ok(());
    }
    fs::write(path, content)
}

fn write_manifest_outputs(out_dir: &Path, json: &str) {
    write_if_changed(&out_dir.join(MANIFEST_FILENAME), json)
        .expect("Failed to write OUT_DIR/traceability_manifest.json");

    if let Some(profile_dir) = out_dir
        .ancestors()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("build"))
        .and_then(Path::parent)
    {
        let _ = write_if_changed(&profile_dir.join(MANIFEST_FILENAME), json);
        if let Some(parent_target) = profile_dir.parent() {
            let _ = write_if_changed(&parent_target.join(MANIFEST_FILENAME), json);
        }
    }

    if let Some(custom_target) = env::var_os("CARGO_TARGET_DIR") {
        let target_path = PathBuf::from(custom_target);
        if target_path.is_dir() {
            let _ = write_if_changed(&target_path.join(MANIFEST_FILENAME), json);
        }
    } else if Path::new("../target").is_dir() {
        let _ = write_if_changed(&Path::new("../target").join(MANIFEST_FILENAME), json);
    }
}
