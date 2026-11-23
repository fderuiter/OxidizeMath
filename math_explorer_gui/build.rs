//! Build script generating the scoped traceability snapshot for `math_explorer_gui`.
mod build_snapshot;

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    let out_path = Path::new(&out_dir);
    let _ = fs::remove_file(out_path.join("generated_tabs.rs"));
    println!("cargo:rerun-if-changed=build.rs");
    build_snapshot::generate_traceability_snapshot(out_path);
}
