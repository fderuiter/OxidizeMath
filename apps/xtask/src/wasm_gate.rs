//! Hybrid Chromium + Node.js `wasm32-unknown-unknown` test artifact builder and runner for `xtask`.

use crate::gates::workspace_wasm::{
    self, WasmCrateSpec, WasmExecutionRuntime, build_wasm_no_run_json_invocations,
    extract_wasm_test_executables_from_cargo_json,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};

fn compile_wasm_artifacts(ws_root: &Path, cargo_args: &[String]) -> Vec<PathBuf> {
    println!(
        "Running (in {}): cargo {}",
        ws_root.display(),
        cargo_args.join(" ")
    );
    let output = Command::new("cargo")
        .args(cargo_args)
        .current_dir(ws_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .unwrap_or_else(|err| {
            eprintln!("Failed to spawn cargo test --no-run: {err}");
            exit(1);
        });
    if !output.status.success() {
        eprintln!("Cargo wasm32 test compilation failed!");
        exit(1);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    extract_wasm_test_executables_from_cargo_json(&stdout).unwrap_or_else(|err| {
        eprintln!("Failed to extract wasm32 test executables: {err}");
        exit(1);
    })
}

/// Builds argv for invoking `scripts/wasm_test_runner.py` on a compiled `.wasm` artifact without shell splitting.
#[must_use]
pub(crate) fn build_runner_command_args(runner_script: &Path, wasm_binary: &Path) -> Vec<PathBuf> {
    vec![
        PathBuf::from("-B"),
        PathBuf::from("-u"),
        runner_script.to_path_buf(),
        wasm_binary.to_path_buf(),
    ]
}

fn execute_wasm_artifact(
    python_bin: &str,
    runner_script: &Path,
    wasm_binary: &Path,
    ws_root: &Path,
) {
    let argv = build_runner_command_args(runner_script, wasm_binary);
    println!(
        "Executing WASM test binary: {} {}",
        runner_script.display(),
        wasm_binary.display()
    );
    let status = Command::new(python_bin)
        .args(&argv)
        .current_dir(ws_root)
        .status();
    match status {
        Ok(s) if s.success() => (),
        _ => {
            eprintln!("WASM test binary failed: {}", wasm_binary.display());
            exit(1);
        }
    }
}

fn run_runtime_group(
    ws_root: &Path,
    wasm_crates: &[WasmCrateSpec],
    runtime: WasmExecutionRuntime,
    label: &str,
) {
    let count = wasm_crates
        .iter()
        .filter(|c| c.execution_runtime() == runtime)
        .count();
    let invocations = build_wasm_no_run_json_invocations(wasm_crates, runtime);
    let python_bin = crate::setup::resolve_python_bin();
    let runner_script = ws_root.join("scripts/wasm_test_runner.py");

    println!("\n--- {label} ({count} crates) ---");
    for cargo_args in &invocations {
        let artifacts = compile_wasm_artifacts(ws_root, cargo_args);
        for artifact in &artifacts {
            execute_wasm_artifact(python_bin, &runner_script, artifact, ws_root);
        }
    }
}

pub(crate) fn run_hybrid_wasm_suites(ws_root: &Path) {
    let wasm_crates = workspace_wasm::discover_wasm_crates(ws_root).unwrap_or_else(|err| {
        eprintln!("Failed to discover wasm32 workspace crates: {err}");
        exit(1);
    });
    run_runtime_group(
        ws_root,
        &wasm_crates,
        WasmExecutionRuntime::BrowserChromium,
        "Step 1a: Browser #[wasm_bindgen_test] Suites in Headless Chromium",
    );
    run_runtime_group(
        ws_root,
        &wasm_crates,
        WasmExecutionRuntime::NodeJs,
        "Step 1b: Pure-Rust libtest Suites in Node.js",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_runner_command_args_preserve_spaces_without_splitting() {
        let root = Path::new("/tmp/oxidize with spaces");
        let script = root.join("scripts/wasm_test_runner.py");
        let wasm = root
            .join("target/wasm32-unknown-unknown/debug/deps/math_commons-1234567890abcdef.wasm");
        let argv = build_runner_command_args(&script, &wasm);
        assert_eq!(argv.len(), 4);
        assert_eq!(argv[2], script);
        assert_eq!(argv[3], wasm);
    }

    #[test]
    fn test_wasm_test_and_smoke_ci_parity() {
        let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let discovered = workspace_wasm::discover_wasm_crates(&ws_root)
            .expect("must discover wasm32 workspace crates");
        let browser_invocations =
            build_wasm_no_run_json_invocations(&discovered, WasmExecutionRuntime::BrowserChromium);
        assert_eq!(browser_invocations.len(), 2);
        assert!(browser_invocations[0].contains(&"oxidize_core".to_string()));
        assert!(browser_invocations[0].contains(&"egui_plot".to_string()));
        assert!(browser_invocations[0].contains(&"--lib".to_string()));
        assert!(browser_invocations[0].contains(&"--no-run".to_string()));
        assert!(browser_invocations[0].contains(&"--message-format=json".to_string()));
        assert_eq!(
            browser_invocations[1],
            vec![
                "test",
                "--target",
                "wasm32-unknown-unknown",
                "-p",
                "math_explorer_gui",
                "--bins",
                "--no-run",
                "--message-format=json",
            ]
        );
        let node_invocations =
            build_wasm_no_run_json_invocations(&discovered, WasmExecutionRuntime::NodeJs);
        assert_eq!(node_invocations.len(), 1);
        assert!(node_invocations[0].contains(&"math_explorer".to_string()));
        assert!(node_invocations[0].contains(&"math_commons".to_string()));
        assert!(node_invocations[0].contains(&"--no-run".to_string()));
        assert!(node_invocations[0].contains(&"--message-format=json".to_string()));

        let ci_path = ws_root.join(".github/workflows/ci.yml");
        let ci_yaml = fs::read_to_string(&ci_path)
            .expect("failed to read ci.yml")
            .replace("\r\n", "\n");
        assert!(
            !ci_yaml.contains("cargo check --workspace --target wasm32-unknown-unknown"),
            "ci.yml must not retain duplicate wasm32 cargo check steps"
        );
        let build_pos = ci_yaml
            .find("trunk build --release")
            .expect("ci.yml must build web app with trunk");
        let smoke_pos = ci_yaml
            .find("cargo run -p xtask -- test-wasm --skip-build")
            .expect("ci.yml must run xtask test-wasm gate");
        let upload_pos = ci_yaml
            .find("name: Upload Web Dist Artifact")
            .expect("ci.yml must upload wasm_web_dist artifact");
        assert!(smoke_pos > build_pos && smoke_pos < upload_pos);
    }
}
