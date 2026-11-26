//! Native benchmark compilation/test gate (`check-benches`) and isolated example smoke runner (`smoke-examples`).

use crate::gates::{
    ensure_verification_built_in, resolve_target_dir, resolve_workspace_root, run_cmd_in_path,
    workspace_wasm,
};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio, exit};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Cargo arguments executed by `cargo run -p xtask -- check-benches`.
pub const CHECK_BENCHES_ARGS: &[&str] = &["test", "--workspace", "--benches", "--all-features"];

/// Default per-example execution timeout in seconds for `smoke-examples`.
pub const DEFAULT_SMOKE_TIMEOUT_SECS: u64 = 30;

static SMOKE_SANDBOX_SEQ: AtomicU64 = AtomicU64::new(0);

/// RAII temporary sandbox isolating example working directories and captured output logs.
pub(crate) struct SmokeTempDir {
    base_dir: PathBuf,
    /// Empty working directory where the example process executes.
    pub(crate) work_dir: PathBuf,
    /// Captured stdout file path outside `work_dir`.
    pub(crate) stdout_log: PathBuf,
    /// Captured stderr file path outside `work_dir`.
    pub(crate) stderr_log: PathBuf,
}

impl SmokeTempDir {
    pub(crate) fn new(label: &str) -> Result<Self, String> {
        let seq = SMOKE_SANDBOX_SEQ.fetch_add(1, Ordering::Relaxed);
        let safe_label: String = label
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let base_dir = env::temp_dir().join(format!(
            "oxidize_math_smoke_{}_{}_{}",
            std::process::id(),
            seq,
            safe_label
        ));
        let _ = fs::remove_dir_all(&base_dir);
        let work_dir = base_dir.join("workdir");
        let log_dir = base_dir.join("logs");
        fs::create_dir_all(&work_dir)
            .map_err(|e| format!("Failed to create smoke workdir {}: {e}", work_dir.display()))?;
        fs::create_dir_all(&log_dir)
            .map_err(|e| format!("Failed to create smoke logdir {}: {e}", log_dir.display()))?;
        Ok(Self {
            stdout_log: log_dir.join("stdout.log"),
            stderr_log: log_dir.join("stderr.log"),
            work_dir,
            base_dir,
        })
    }
}

impl Drop for SmokeTempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base_dir);
    }
}

/// Runs `cargo test --workspace --benches --all-features` from the workspace root.
pub fn check_benches() {
    println!("=== Check Workspace Benchmarks ===");
    let ws_root = resolve_workspace_root();
    run_cmd_in_path("cargo", CHECK_BENCHES_ARGS, &ws_root);
}

pub(crate) fn discover_registered_examples(
    ws_root: &Path,
    target_dir: &Path,
) -> Result<Vec<String>, String> {
    let bin = ensure_verification_built_in(ws_root, target_dir);
    let output = Command::new(&bin)
        .current_dir(ws_root)
        .arg("list-examples")
        .output()
        .map_err(|e| format!("Failed to execute `{} list-examples`: {e}", bin.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "`unified_verification list-examples` failed ({}): {}",
            output.status,
            stderr.trim()
        ));
    }
    let examples: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if examples.is_empty() {
        return Err("No `[[example]]` targets discovered in workspace CLI manifest.".into());
    }
    Ok(examples)
}

fn discover_examples_package(ws_root: &Path) -> String {
    if workspace_wasm::find_package_member_path_std(ws_root, "oxidize-cli").is_some() {
        "oxidize-cli".to_string()
    } else {
        "math_explorer".to_string()
    }
}

fn resolve_example_binary(target_dir: &Path, name: &str) -> Option<PathBuf> {
    let mut file_name = PathBuf::from(name);
    if !env::consts::EXE_EXTENSION.is_empty() {
        file_name.set_extension(env::consts::EXE_EXTENSION);
    }
    let mut candidates = Vec::new();
    if let Ok(build_target) = env::var("CARGO_BUILD_TARGET")
        && !build_target.trim().is_empty()
    {
        candidates.push(
            target_dir
                .join(build_target.trim())
                .join("debug")
                .join("examples")
                .join(&file_name),
        );
    }
    candidates.push(target_dir.join("debug").join("examples").join(&file_name));
    candidates.into_iter().find(|p| p.is_file())
}

fn parse_smoke_timeout(raw: Option<&str>) -> Duration {
    let secs = raw
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(DEFAULT_SMOKE_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::disallowed_methods)]
fn wait_child_with_timeout(
    child: &mut Child,
    timeout: Duration,
) -> Result<Option<ExitStatus>, String> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(15));
            }
            Err(e) => return Err(format!("Failed polling child process status: {e}")),
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn wait_child_with_timeout(
    child: &mut Child,
    _timeout: Duration,
) -> Result<Option<ExitStatus>, String> {
    child
        .wait()
        .map(Some)
        .map_err(|e| format!("Failed waiting for child process: {e}"))
}

fn read_captured_logs(sandbox: &SmokeTempDir) -> String {
    let stdout = fs::read_to_string(&sandbox.stdout_log).unwrap_or_default();
    let stderr = fs::read_to_string(&sandbox.stderr_log).unwrap_or_default();
    format!(
        "--- stdout ---\n{}\n--- stderr ---\n{}",
        stdout.trim_end(),
        stderr.trim_end()
    )
}

pub(crate) fn run_command_in_smoke_sandbox(
    cmd: &mut Command,
    label: &str,
    timeout: Duration,
) -> Result<(), String> {
    let sandbox = SmokeTempDir::new(label)?;
    let stdout_file = File::create(&sandbox.stdout_log)
        .map_err(|e| format!("Failed creating stdout log for `{label}`: {e}"))?;
    let stderr_file = File::create(&sandbox.stderr_log)
        .map_err(|e| format!("Failed creating stderr log for `{label}`: {e}"))?;

    let mut child = cmd
        .current_dir(&sandbox.work_dir)
        .env("MATH_EXPLORER_SMOKE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .map_err(|e| format!("Failed spawning example `{label}`: {e}"))?;

    match wait_child_with_timeout(&mut child, timeout)? {
        Some(status) if status.success() => Ok(()),
        Some(status) => Err(format!(
            "Example `{label}` failed with status {status}.\n{}",
            read_captured_logs(&sandbox)
        )),
        None => Err(format!(
            "Example `{label}` timed out after {:.1}s.\n{}",
            timeout.as_secs_f64(),
            read_captured_logs(&sandbox)
        )),
    }
}

fn run_single_example_smoke(bin_path: &Path, name: &str, timeout: Duration) -> Result<(), String> {
    let mut cmd = Command::new(bin_path);
    run_command_in_smoke_sandbox(&mut cmd, name, timeout)
}

/// Builds and executes every registered workspace CLI example under `MATH_EXPLORER_SMOKE=1`.
pub fn smoke_examples() {
    println!("=== Smoke Test Registered CLI Examples ===");
    let ws_root = resolve_workspace_root();
    let target_dir = resolve_target_dir(&ws_root);
    let examples = discover_registered_examples(&ws_root, &target_dir).unwrap_or_else(|err| {
        eprintln!("Error discovering registered examples: {err}");
        exit(1);
    });
    let examples_pkg = discover_examples_package(&ws_root);

    run_cmd_in_path(
        "cargo",
        &["build", "-p", &examples_pkg, "--examples", "--all-features"],
        &ws_root,
    );

    let timeout = parse_smoke_timeout(env::var("MATH_EXPLORER_SMOKE_TIMEOUT_SECS").ok().as_deref());
    let total = examples.len();
    for (idx, name) in examples.iter().enumerate() {
        let Some(bin_path) = resolve_example_binary(&target_dir, name) else {
            eprintln!(
                "Error: Compiled binary not found for registered example `{name}` under {}",
                target_dir.display()
            );
            exit(1);
        };
        println!("  [{}/{}] Running example `{name}`...", idx + 1, total);
        if let Err(err) = run_single_example_smoke(&bin_path, name, timeout) {
            eprintln!("{err}");
            exit(1);
        }
    }
    println!("Smoke-tested {total}/{total} registered math_explorer examples successfully!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::setup::resolve_python_bin;

    #[test]
    fn test_discover_registered_examples_matches_manifest() {
        let ws_root = resolve_workspace_root();
        let target_dir = resolve_target_dir(&ws_root);
        let examples = discover_registered_examples(&ws_root, &target_dir)
            .expect("must list registered math_explorer examples");
        assert_eq!(examples.len(), 32);
        assert!(examples.iter().any(|name| name == "hello_world"));
        assert!(examples.iter().any(|name| name == "isosurface_torus"));
        assert!(!examples.iter().any(|name| name == "check_statrs"));
    }

    #[test]
    fn test_smoke_sandbox_cleans_up_written_files() {
        let work_dir_path;
        {
            let mut cmd = Command::new(resolve_python_bin());
            cmd.args([
                "-c",
                "import os, pathlib; assert os.environ.get('MATH_EXPLORER_SMOKE') == '1'; pathlib.Path('torus.obj').write_text('v 0 0 0\\n')",
            ]);
            let sandbox = SmokeTempDir::new("cleanup_probe").unwrap();
            work_dir_path = sandbox.work_dir.clone();
            assert!(work_dir_path.is_dir());
        }
        assert!(
            !work_dir_path.exists(),
            "SmokeTempDir Drop must remove isolated working directory"
        );
        let mut cmd = Command::new(resolve_python_bin());
        cmd.args([
            "-c",
            "import os, pathlib; assert os.environ.get('MATH_EXPLORER_SMOKE') == '1'; pathlib.Path('torus.obj').write_text('v 0 0 0\\n')",
        ]);
        run_command_in_smoke_sandbox(&mut cmd, "write_torus_probe", Duration::from_secs(5))
            .expect("sandboxed probe command must succeed");
        assert!(!Path::new("torus.obj").exists());
    }

    #[test]
    fn test_smoke_sandbox_reports_nonzero_exit() {
        let mut cmd = Command::new(resolve_python_bin());
        cmd.args(["-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(7)"]);
        let err = run_command_in_smoke_sandbox(&mut cmd, "failing_probe", Duration::from_secs(5))
            .expect_err("non-zero exit must fail smoke runner");
        assert!(err.contains("failing_probe"));
        assert!(err.contains("boom"));
    }

    #[test]
    fn test_smoke_sandbox_kills_timed_out_child() {
        let mut cmd = Command::new(resolve_python_bin());
        cmd.args(["-c", "import time; time.sleep(10)"]);
        let err =
            run_command_in_smoke_sandbox(&mut cmd, "hanging_probe", Duration::from_millis(120))
                .expect_err("hanging process must be terminated on timeout");
        assert!(err.contains("timed out"));
    }
}
