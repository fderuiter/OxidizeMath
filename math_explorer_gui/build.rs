//! Build script generating the scoped traceability snapshot and BibTeX link index for `math_explorer_gui`.
mod build_bib;
mod build_snapshot;

use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    let out_path = Path::new(&out_dir);
    println!("cargo:rerun-if-changed=build.rs");
    build_snapshot::generate_traceability_snapshot(out_path);
    build_bib::generate_bib_links_tsv(out_path);
}
