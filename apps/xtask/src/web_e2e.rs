//! Headless Chromium WebAssembly E2E test runner for `xtask web-e2e`.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, exit};

/// Parsed command-line options for `xtask web-e2e`.
#[derive(Debug, PartialEq, Eq)]
pub struct WebE2eOptions {
    /// Skip running `trunk build --release` when `math_explorer_gui/dist` is pre-built.
    pub skip_build: bool,
    /// Additional arguments forwarded directly to `pytest`.
    pub pytest_args: Vec<String>,
}

/// Parse CLI arguments passed after `xtask web-e2e`.
pub fn parse_web_e2e_args(args: &[String]) -> WebE2eOptions {
    let mut skip_build = false;
    let mut pytest_args = Vec::with_capacity(args.len());
    let mut forward_all = false;

    for arg in args {
        if forward_all {
            pytest_args.push(arg.clone());
        } else if arg == "--" {
            forward_all = true;
        } else if arg == "--skip-build" {
            skip_build = true;
        } else {
            pytest_args.push(arg.clone());
        }
    }

    WebE2eOptions {
        skip_build,
        pytest_args,
    }
}

/// Build `math_explorer_gui/dist` (unless `--skip-build` is specified) and execute
/// the Python Playwright Chromium E2E test suite in `tests/web_e2e/`.
pub fn run_web_e2e(args: &[String]) {
    println!("=== WebAssembly Headless Chromium E2E Suite ===");
    let opts = parse_web_e2e_args(args);
    let root = resolve_workspace_root();
    let gui_dir = root.join("math_explorer_gui");
    let dist_index = gui_dir.join("dist").join("index.html");
    let suite_dir = root.join("tests").join("web_e2e");

    assert!(
        gui_dir.is_dir(),
        "Expected math_explorer_gui directory at {}",
        gui_dir.display()
    );
    assert!(
        suite_dir.is_dir(),
        "Expected browser E2E suite directory at {}",
        suite_dir.display()
    );

    if opts.skip_build {
        println!("Skipping Trunk build (--skip-build); verifying existing dist artifacts...");
        if !dist_index.is_file() {
            eprintln!(
                "Error: {} not found. Run without --skip-build or build math_explorer_gui/dist first.",
                dist_index.display()
            );
            exit(1);
        }
    } else {
        build_web_dist(&gui_dir, &dist_index);
    }

    let python_bin = resolve_python_interpreter();
    verify_python_dependencies(&python_bin);
    execute_pytest_suite(&python_bin, &root, &suite_dir, &opts.pytest_args);
    println!("=== All WebAssembly Browser E2E Tests Passed ===");
}

fn resolve_workspace_root() -> PathBuf {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("math_explorer_gui").is_dir() {
        return cwd;
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map_or(cwd, Path::to_path_buf)
}

fn build_web_dist(gui_dir: &Path, dist_index: &Path) {
    let trunk_ok = Command::new("trunk")
        .arg("--version")
        .output()
        .is_ok_and(|out| out.status.success());
    if !trunk_ok {
        eprintln!("Error: 'trunk' is not installed or not in PATH.");
        eprintln!(
            "Install Trunk via: cargo install trunk (and rustup target add wasm32-unknown-unknown)"
        );
        exit(1);
    }

    println!("Building math_explorer_gui/dist with 'trunk build --release'...");
    let status = Command::new("trunk")
        .current_dir(gui_dir)
        .args(["build", "--release"])
        .status();

    match status {
        Ok(s) if s.success() && dist_index.is_file() => {
            println!("Successfully built {}", dist_index.display());
        }
        _ => {
            eprintln!(
                "Error: 'trunk build --release' failed in {}",
                gui_dir.display()
            );
            exit(1);
        }
    }
}

fn resolve_python_interpreter() -> String {
    if let Ok(explicit) = env::var("PYTHON")
        && !explicit.trim().is_empty()
    {
        return explicit;
    }

    for candidate in ["python3", "python"] {
        let ok = Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success());
        if ok {
            return candidate.to_string();
        }
    }

    eprintln!("Error: Neither 'python3' nor 'python' is available in PATH.");
    exit(1);
}

fn verify_python_dependencies(python_bin: &str) {
    let check = Command::new(python_bin)
        .args(["-c", "import pytest, playwright"])
        .output();

    if !check.is_ok_and(|out| out.status.success()) {
        eprintln!(
            "Error: Python interpreter '{}' is missing 'pytest' or 'playwright'.",
            python_bin
        );
        eprintln!(
            "Install via: {} -m pip install pytest playwright && {} -m playwright install chromium",
            python_bin, python_bin
        );
        exit(1);
    }
}

fn execute_pytest_suite(
    python_bin: &str,
    workspace_root: &Path,
    suite_dir: &Path,
    extra_args: &[String],
) {
    let mut cmd_args = vec![
        "-m".to_string(),
        "pytest".to_string(),
        "-p".to_string(),
        "no:playwright".to_string(),
        suite_dir.display().to_string(),
        "-v".to_string(),
    ];
    cmd_args.extend(extra_args.iter().cloned());

    println!("Running: {} {}", python_bin, cmd_args.join(" "));
    let status = Command::new(python_bin)
        .current_dir(workspace_root)
        .args(&cmd_args)
        .status();

    match status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("Error: Headless Chromium browser E2E suite failed.");
            exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_web_e2e_args_defaults() {
        let opts = parse_web_e2e_args(&[]);
        assert!(!opts.skip_build);
        assert!(opts.pytest_args.is_empty());
    }

    #[test]
    fn test_parse_web_e2e_args_skip_build_and_passthrough() {
        let args = vec![
            "--skip-build".to_string(),
            "-k".to_string(),
            "cold_load".to_string(),
            "--".to_string(),
            "--maxfail=1".to_string(),
        ];
        let opts = parse_web_e2e_args(&args);
        assert!(opts.skip_build);
        assert_eq!(opts.pytest_args, vec!["-k", "cold_load", "--maxfail=1"]);
    }

    #[test]
    fn test_resolve_workspace_root_contains_gui_and_suite() {
        let root = resolve_workspace_root();
        assert!(root.join("math_explorer_gui").is_dir());
        assert!(root.join("tests").join("web_e2e").is_dir());
    }
}
