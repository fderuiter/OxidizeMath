use super::{ReportMode, VERIFY_SUITE_WRITE_ENV};
use std::process::Command;

/// Executes `cargo llvm-cov --all-features --workspace --json` and returns process output.
fn get_llvm_cov_output(mode: ReportMode) -> std::process::Output {
    let mut cmd = Command::new("cargo");
    cmd.args(["llvm-cov", "--all-features", "--workspace", "--json"]);
    if mode == ReportMode::Write {
        cmd.env(VERIFY_SUITE_WRITE_ENV, "1");
    }
    cmd.output().expect("Failed to run cargo llvm-cov")
}

/// Removes instrumented `cargo llvm-cov` build artifacts from the workspace target directory.
fn clean_llvm_cov_workspace() -> Result<(), String> {
    let output = Command::new("cargo")
        .args(["llvm-cov", "clean", "--workspace"])
        .output()
        .map_err(|err| format!("Failed to spawn `cargo llvm-cov clean --workspace`: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "Error running cargo llvm-cov clean --workspace:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Collects and parses workspace coverage JSON, then cleans `cargo llvm-cov` build artifacts.
pub fn collect_llvm_cov_json(mode: ReportMode) -> Result<serde_json::Value, String> {
    collect_llvm_cov_json_with(|| get_llvm_cov_output(mode), clean_llvm_cov_workspace)
}

pub(super) fn collect_llvm_cov_json_with<F, C>(
    run_cov: F,
    clean_cov: C,
) -> Result<serde_json::Value, String>
where
    F: FnOnce() -> std::process::Output,
    C: FnOnce() -> Result<(), String>,
{
    let output = run_cov();
    if !output.status.success() {
        return Err(format!(
            "Error running cargo llvm-cov:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let cov_json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|err| format!("Failed to parse coverage JSON: {err}"))?;
    clean_cov()?;
    Ok(cov_json)
}

pub fn parse_coverage(cov_json: &serde_json::Value) -> (f64, f64) {
    let mut native_lines_total = 0.0;
    let mut native_lines_covered = 0.0;

    if let Some(files) = cov_json
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|data| data.get("files"))
        .and_then(|f| f.as_array())
    {
        for f in files {
            if let Some(summary) = f.get("summary").and_then(|s| s.get("lines")) {
                native_lines_total += summary.get("count").and_then(|c| c.as_f64()).unwrap_or(0.0);
                native_lines_covered += summary
                    .get("covered")
                    .and_then(|c| c.as_f64())
                    .unwrap_or(0.0);
            }
        }
    }
    (native_lines_total, native_lines_covered)
}
