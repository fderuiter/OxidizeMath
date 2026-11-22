//! CLI tool for verifying codebase traceability and semantic integrity.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let auto_fix = args.iter().any(|arg| arg == "--auto-fix");

    match oxidize_core::traceability::run_workspace_traceability(auto_fix) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(err) => {
            eprintln!("Error scanning repository: {}", err);
            std::process::exit(1);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
