//! Deterministic `OUT_DIR` staging helpers for `math_explorer_gui` build script.
#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::fs;
use std::path::Path;

/// Writes `content` to `path` only when the file does not exist or its bytes differ,
/// preserving mtimes across warm builds and snapshot syncs.
pub fn write_if_changed(path: &Path, content: &[u8]) {
    if fs::read(path).ok().as_deref() == Some(content) {
        return;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn stage_single_file(src_rel: &str, dst_path: &Path) {
    let src = Path::new(src_rel);
    if src.is_file() {
        println!("cargo:rerun-if-changed={src_rel}");
        if let Ok(bytes) = fs::read(src) {
            write_if_changed(dst_path, &bytes);
        }
    } else {
        write_if_changed(dst_path, b"");
    }
}

/// Stages compile-time embedded assets into `out_dir` so `rustc` never depends on
/// `env!("CARGO_MANIFEST_DIR")`.
pub fn stage_gui_inputs_to_out_dir(out_dir: &Path) {
    println!("cargo:rerun-if-changed=build_stage.rs");

    stage_single_file("sw.js", &out_dir.join("sw.js"));
    stage_single_file(
        "../Verification_Certificate.json",
        &out_dir.join("Verification_Certificate.json"),
    );
    stage_single_file("../papers/references.bib", &out_dir.join("references.bib"));
}
