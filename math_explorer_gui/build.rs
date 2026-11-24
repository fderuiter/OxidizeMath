//! Build script generating the scoped traceability snapshot, BibTeX link index, and staged `OUT_DIR` assets for `math_explorer_gui`.
mod build_bib;
mod build_snapshot;
mod build_stage;

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR must be set by Cargo");
    let out_path = Path::new(&out_dir);
    let _ = fs::remove_file(out_path.join("generated_tabs.rs"));
    println!("cargo:rerun-if-changed=build.rs");
    build_stage::stage_gui_inputs_to_out_dir(out_path);
    build_snapshot::generate_traceability_snapshot(out_path);
    build_bib::generate_bib_links_tsv(out_path);
}
