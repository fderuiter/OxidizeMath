//! Lean 4 (`elan`, `lean`, `lake`) toolchain probes for `xtask setup`.

use super::{ToolProbeResult, check_presence_probe, format_probe_report};
use std::fs;
use std::path::Path;
use std::process::{Command, exit};

/// Default pinned Lean 4 toolchain expected when `proofs/lean-toolchain` is absent.
pub const DEFAULT_EXPECTED_LEAN_TOOLCHAIN: &str = "leanprover/lean4:v4.14.0";

/// Parses the pinned Lean toolchain identifier from `proofs/lean-toolchain`.
#[must_use]
pub fn parse_expected_lean_toolchain(toolchain_file: &str) -> Option<String> {
    for raw_line in toolchain_file.lines() {
        let trimmed = raw_line.split('#').next().unwrap_or("").trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Extracts the bare version number (e.g. `4.14.0`) from a `lean-toolchain` identifier.
#[must_use]
pub fn extract_lean_semver_suffix(toolchain: &str) -> &str {
    let rhs = toolchain
        .trim()
        .rsplit(':')
        .next()
        .unwrap_or(toolchain.trim());
    rhs.strip_prefix('v').unwrap_or(rhs)
}

/// Evaluates `lean --version` or `lake --version` output against `expected_toolchain`.
#[must_use]
pub fn check_lean_tool_version(
    tool_name: &'static str,
    probe_stdout: Result<String, String>,
    expected_toolchain: &str,
) -> ToolProbeResult {
    let expected_ver = extract_lean_semver_suffix(expected_toolchain);
    let fix = format!(
        "curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain {expected_toolchain} && elan toolchain install {expected_toolchain}"
    );
    match probe_stdout {
        Err(err) => ToolProbeResult {
            tool_name,
            detected_summary: None,
            failure: Some((format!("`{tool_name}` CLI not available ({err})"), fix)),
        },
        Ok(raw) => {
            let first_line = raw.lines().next().unwrap_or("").trim();
            if first_line.contains(expected_ver) {
                ToolProbeResult {
                    tool_name,
                    detected_summary: Some(first_line.to_string()),
                    failure: None,
                }
            } else {
                ToolProbeResult {
                    tool_name,
                    detected_summary: None,
                    failure: Some((
                        format!(
                            "version mismatch: found `{first_line}`, expected Lean `{expected_ver}` ({expected_toolchain})"
                        ),
                        fix,
                    )),
                }
            }
        }
    }
}

fn run_stdout_probe_in_dir(program: &str, args: &[&str], cwd: &Path) -> Result<String, String> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if cwd.is_dir() {
        cmd.current_dir(cwd);
    }
    let output = cmd.output().map_err(|e| e.to_string())?;
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Ok(if stdout.is_empty() { stderr } else { stdout })
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        Err(if stderr.is_empty() {
            format!("exited with status {}", output.status)
        } else {
            stderr
        })
    }
}

/// Probes `elan`, `lean --version`, and `lake --version` against `proofs/lean-toolchain`.
#[must_use]
pub fn collect_lean_tool_probes(workspace_root: &Path) -> Vec<ToolProbeResult> {
    let proofs_dir = workspace_root.join("proofs");
    let expected_tc = fs::read_to_string(proofs_dir.join("lean-toolchain"))
        .ok()
        .and_then(|c| parse_expected_lean_toolchain(&c))
        .unwrap_or_else(|| DEFAULT_EXPECTED_LEAN_TOOLCHAIN.to_string());
    let probe_dir = if proofs_dir.is_dir() {
        proofs_dir.as_path()
    } else {
        workspace_root
    };
    let elan_fix = format!(
        "curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain {expected_tc}"
    );

    vec![
        check_presence_probe(
            "elan",
            run_stdout_probe_in_dir("elan", &["--version"], probe_dir),
            &elan_fix,
        ),
        check_lean_tool_version(
            "lean",
            run_stdout_probe_in_dir("lean", &["--version"], probe_dir),
            &expected_tc,
        ),
        check_lean_tool_version(
            "lake",
            run_stdout_probe_in_dir("lake", &["--version"], probe_dir),
            &expected_tc,
        ),
    ]
}

fn provision_lean_toolchain_if_needed(workspace_root: &Path, strict_lean: bool) {
    let proofs_dir = workspace_root.join("proofs");
    let expected_tc = fs::read_to_string(proofs_dir.join("lean-toolchain"))
        .ok()
        .and_then(|c| parse_expected_lean_toolchain(&c))
        .unwrap_or_else(|| DEFAULT_EXPECTED_LEAN_TOOLCHAIN.to_string());
    let probe_dir = if proofs_dir.is_dir() {
        proofs_dir.as_path()
    } else {
        workspace_root
    };
    let expected_ver = extract_lean_semver_suffix(&expected_tc);
    if run_stdout_probe_in_dir("lean", &["--version"], probe_dir)
        .is_ok_and(|out| out.contains(expected_ver))
    {
        return;
    }
    if run_stdout_probe_in_dir("elan", &["--version"], probe_dir).is_err() && strict_lean {
        println!("Provisioning `elan` with default toolchain `{expected_tc}`...");
        let cmd = format!(
            "curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain {expected_tc}"
        );
        let _ = Command::new("sh").args(["-c", &cmd]).status();
    }
    if run_stdout_probe_in_dir("elan", &["--version"], probe_dir).is_ok() {
        println!("Provisioning pinned Lean toolchain `{expected_tc}` via `elan`...");
        let _ = Command::new("elan")
            .args(["toolchain", "install", &expected_tc])
            .current_dir(probe_dir)
            .status();
    }
}

/// Verifies Lean 4 (`elan`, `lean`, `lake`) prerequisites; warns by default and exits `1` when `strict_lean` is `true`.
pub fn verify_lean_tools(workspace_root: &Path, strict_lean: bool) -> bool {
    println!("\n--- Verifying Lean 4 Formal Verification Prerequisites ---");
    provision_lean_toolchain_if_needed(workspace_root, strict_lean);
    let probes = collect_lean_tool_probes(workspace_root);
    let (all_ok, report) = format_probe_report(&probes);
    println!("{report}");
    if !all_ok {
        if strict_lean {
            eprintln!(
                "\nSetup failed (--lean): Lean 4 (`elan` / `lean` / `lake`) prerequisites are missing or mismatched."
            );
            exit(1);
        } else {
            println!(
                "\nWarning: Lean 4 (`elan` / `lean` / `lake`) tools are missing or mismatched (continuing; pass `--lean` to enforce)."
            );
        }
    }
    all_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lean_toolchain_parsing_and_version_probes() {
        assert_eq!(
            parse_expected_lean_toolchain("# comment\nleanprover/lean4:v4.14.0\n"),
            Some("leanprover/lean4:v4.14.0".to_string())
        );
        assert_eq!(
            extract_lean_semver_suffix("leanprover/lean4:v4.14.0"),
            "4.14.0"
        );

        let ok_lean = check_lean_tool_version(
            "lean",
            Ok("Lean (version 4.14.0, x86_64-unknown-linux-gnu, commit 4e67981)".to_string()),
            "leanprover/lean4:v4.14.0",
        );
        assert!(ok_lean.failure.is_none());

        let ok_lake = check_lean_tool_version(
            "lake",
            Ok("Lake version 5.0.0-4e67981 (Lean version 4.14.0)".to_string()),
            "leanprover/lean4:v4.14.0",
        );
        assert!(ok_lake.failure.is_none());

        let mismatch = check_lean_tool_version(
            "lean",
            Ok("Lean (version 4.13.0, x86_64-unknown-linux-gnu)".to_string()),
            "leanprover/lean4:v4.14.0",
        );
        assert!(mismatch.failure.is_some());

        let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let live_probes = collect_lean_tool_probes(&ws_root);
        assert_eq!(live_probes.len(), 3);
        assert!(
            live_probes.iter().all(|p| p.failure.is_none()),
            "Expected elan/lean/lake 4.14.0 probes to pass in dev workspace: {live_probes:?}"
        );
    }
}
