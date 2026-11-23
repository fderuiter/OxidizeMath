//! Build script generating the scoped traceability snapshot for `math_explorer_gui`.
mod build_snapshot;

use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    println!("cargo:rerun-if-changed=build.rs");
    build_snapshot::generate_traceability_snapshot(Path::new(&out_dir));
}
