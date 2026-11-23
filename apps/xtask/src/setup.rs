//! Workspace setup verification for WebAssembly build and Playwright browser prerequisites.

use std::fs;
use std::path::Path;
use std::process::{Command, exit};

/// Default workspace `wasm-bindgen` version expected when parsing `Cargo.toml`.
pub const DEFAULT_EXPECTED_WASM_BINDGEN_VERSION: &str = "0.2.130";

/// Result of probing a single external CLI or browser runtime dependency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolProbeResult {
    /// Human-readable tool label.
    pub tool_name: &'static str,
    /// Detected version or path summary when valid.
    pub detected_summary: Option<String>,
    /// Error description and install remediation when missing or mismatched.
    pub failure: Option<(String, String)>,
}

/// Extracts the `[workspace.dependencies]` `wasm-bindgen` version from root `Cargo.toml`.
#[must_use]
pub fn parse_expected_wasm_bindgen_version(cargo_toml: &str) -> Option<String> {
    let mut in_ws_deps = false;
    for raw_line in cargo_toml.lines() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_ws_deps = line == "[workspace.dependencies]";
            continue;
        }
        if !in_ws_deps {
            continue;
        }
        let Some((key, val)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != "wasm-bindgen" {
            continue;
        }
        let rhs = val.trim();
        if let Some(stripped) = rhs.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            return Some(stripped.to_string());
        }
        if let Some(ver_idx) = rhs.find("version") {
            let after = &rhs[ver_idx..];
            if let Some((_, q_part)) = after.split_once('=') {
                let first_quote = q_part.find('"')?;
                let rest = &q_part[first_quote + 1..];
                let end_quote = rest.find('"')?;
                return Some(rest[..end_quote].to_string());
            }
        }
    }
    None
}

/// Evaluates `wasm-bindgen --version` output against `expected_version`.
#[must_use]
pub fn check_wasm_bindgen_version(
    probe_stdout: Result<String, String>,
    expected_version: &str,
) -> ToolProbeResult {
    let fix = format!(
        "cargo binstall --no-confirm wasm-bindgen-cli@{expected_version}  (or: cargo install --locked --force wasm-bindgen-cli --version {expected_version})"
    );
    match probe_stdout {
        Err(err) => ToolProbeResult {
            tool_name: "wasm-bindgen",
            detected_summary: None,
            failure: Some((format!("`wasm-bindgen` CLI not available ({err})"), fix)),
        },
        Ok(raw) => {
            let trimmed = raw.trim();
            let actual = trimmed
                .strip_prefix("wasm-bindgen")
                .map(str::trim)
                .unwrap_or(trimmed);
            if actual == expected_version {
                ToolProbeResult {
                    tool_name: "wasm-bindgen",
                    detected_summary: Some(actual.to_string()),
                    failure: None,
                }
            } else {
                ToolProbeResult {
                    tool_name: "wasm-bindgen",
                    detected_summary: None,
                    failure: Some((
                        format!(
                            "version mismatch: found `wasm-bindgen {actual}`, expected `{expected_version}`"
                        ),
                        fix,
                    )),
                }
            }
        }
    }
}

/// Evaluates a generic versioned CLI probe (`trunk` or `brotli`).
#[must_use]
pub fn check_presence_probe(
    tool_name: &'static str,
    probe_stdout: Result<String, String>,
    remediation: &str,
) -> ToolProbeResult {
    match probe_stdout {
        Ok(out) if !out.trim().is_empty() => ToolProbeResult {
            tool_name,
            detected_summary: Some(out.lines().next().unwrap_or("").trim().to_string()),
            failure: None,
        },
        Ok(_) => ToolProbeResult {
            tool_name,
            detected_summary: None,
            failure: Some((
                format!("`{tool_name}` returned empty version output"),
                remediation.to_string(),
            )),
        },
        Err(err) => ToolProbeResult {
            tool_name,
            detected_summary: None,
            failure: Some((
                format!("`{tool_name}` not available ({err})"),
                remediation.to_string(),
            )),
        },
    }
}

fn run_stdout_probe(program: &str, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
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

/// Selects a working Python 3 interpreter (`python3`, falling back to `python` on Windows).
#[must_use]
pub fn resolve_python_bin() -> &'static str {
    if run_stdout_probe(
        "python3",
        &["-c", "import sys; assert sys.version_info[0] >= 3"],
    )
    .is_ok()
    {
        "python3"
    } else {
        "python"
    }
}

fn probe_playwright_chromium() -> Result<String, String> {
    let script = concat!(
        "import pathlib, sys\n",
        "from playwright.sync_api import sync_playwright\n",
        "with sync_playwright() as p:\n",
        "    exe = pathlib.Path(p.chromium.executable_path)\n",
        "    if not exe.is_file():\n",
        "        sys.stderr.write(f'Chromium binary missing at {exe}\\n')\n",
        "        sys.exit(1)\n",
        "    print(exe)\n",
    );
    run_stdout_probe(resolve_python_bin(), &["-c", script])
}

/// Probes `wasm-bindgen`, `trunk`, `brotli`, and Playwright Chromium for `workspace_root`.
#[must_use]
pub fn collect_browser_tool_probes(workspace_root: &Path) -> Vec<ToolProbeResult> {
    let expected_wb = fs::read_to_string(workspace_root.join("Cargo.toml"))
        .ok()
        .and_then(|c| parse_expected_wasm_bindgen_version(&c))
        .unwrap_or_else(|| DEFAULT_EXPECTED_WASM_BINDGEN_VERSION.to_string());

    vec![
        check_wasm_bindgen_version(
            run_stdout_probe("wasm-bindgen", &["--version"]),
            &expected_wb,
        ),
        check_presence_probe(
            "trunk",
            run_stdout_probe("trunk", &["--version"]),
            "cargo binstall --no-confirm trunk  (or: cargo install --locked trunk)",
        ),
        check_presence_probe(
            "brotli",
            run_stdout_probe("brotli", &["--version"]),
            "sudo apt-get install -y brotli  (macOS: brew install brotli; Windows: winget install Google.Brotli)",
        ),
        check_presence_probe(
            "playwright-chromium",
            probe_playwright_chromium(),
            "python3 -m pip install --user playwright && python3 -m playwright install --with-deps chromium",
        ),
    ]
}

/// Formats a human-readable verification report and returns `true` iff all probes succeeded.
#[must_use]
pub fn format_probe_report(results: &[ToolProbeResult]) -> (bool, String) {
    let mut lines = Vec::new();
    let mut all_ok = true;
    for res in results {
        if let Some((problem, fix)) = &res.failure {
            all_ok = false;
            lines.push(format!("  [MISSING/MISMATCH] {}: {problem}", res.tool_name));
            lines.push(format!("    -> Fix: {fix}"));
        } else if let Some(summary) = &res.detected_summary {
            lines.push(format!("  [OK] {}: {summary}", res.tool_name));
        }
    }
    (all_ok, lines.join("\n"))
}

/// Verifies all required WebAssembly and Playwright browser tools, exiting non-zero on failure.
pub fn verify_browser_tools_or_exit(workspace_root: &Path) {
    println!("\n--- Verifying WebAssembly & Playwright Browser Prerequisites ---");
    let probes = collect_browser_tool_probes(workspace_root);
    let (all_ok, report) = format_probe_report(&probes);
    println!("{report}");
    if !all_ok {
        eprintln!(
            "\nSetup failed: one or more WebAssembly/Playwright prerequisites are missing or mismatched."
        );
        exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_runner_cli_flags_and_filtering() {
        let ws_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let script = ws_root.join("scripts/wasm_test_runner.py");
        let py = concat!(
            "import importlib.util, pathlib, subprocess, sys, tempfile\n",
            "sys.modules['playwright'] = None\n",
            "sys.modules['playwright.sync_api'] = None\n",
            "spec = importlib.util.spec_from_file_location('r', sys.argv[1])\n",
            "r = importlib.util.module_from_spec(spec)\n",
            "sys.modules['r'] = r\n",
            "spec.loader.exec_module(r)\n",
            "opts = r.parse_cargo_test_args(['--list', '--exact', '--skip', 'vfs::b', '--test-threads', '1', '-t', '2', '--no-capture', '--ignored', 'vfs::a', 'math::c'])\n",
            "assert opts.list_only and opts.exact and opts.only_ignored and opts.include_ignored\n",
            "assert opts.test_threads == 2 and opts.skips == ['vfs::b']\n",
            "assert opts.filters == ['vfs::a', 'math::c']\n",
            "pool = [\n",
            "    r.DiscoveredTest('vfs::a', 'oxidize_core::vfs::a', '__wbgt_vfs::a', False),\n",
            "    r.DiscoveredTest('vfs::ab', 'oxidize_core::vfs::ab', '__wbgt_vfs::ab', False),\n",
            "    r.DiscoveredTest('vfs::b', 'oxidize_core::vfs::b', '__wbgt_vfs::b', False),\n",
            "]\n",
            "exact_sel = r.filter_tests(pool, r.parse_cargo_test_args(['--exact', 'vfs::a']))\n",
            "assert [t.canonical_name for t in exact_sel] == ['vfs::a']\n",
            "qual_sel = r.filter_tests(pool, r.parse_cargo_test_args(['oxidize_core::vfs::ab']))\n",
            "assert [t.canonical_name for t in qual_sel] == ['vfs::ab']\n",
            "skip_sel = r.filter_tests(pool, r.parse_cargo_test_args(['vfs::', '--skip', 'ab']))\n",
            "assert [t.canonical_name for t in skip_sel] == ['vfs::a', 'vfs::b']\n",
            "no_pw = \"import runpy, sys; sys.modules['playwright'] = None; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')\"\n",
            "with tempfile.TemporaryDirectory() as d:\n",
            "    w = pathlib.Path(d) / 'sample-0123456789abcdef.wasm'\n",
            "    w.write_bytes(b'\\x00asm\\x01\\x00\\x00\\x00\\x07\\x15\\x01\\x11__wbgt_mod::alpha\\x00\\x00')\n",
            "    ls = subprocess.run([sys.executable, '-B', '-c', no_pw, sys.argv[1], str(w), '--list'], capture_output=True, text=True, check=True)\n",
            "    assert 'mod::alpha: test' in ls.stdout\n",
            "    miss = subprocess.run([sys.executable, '-B', '-c', no_pw, sys.argv[1], str(w), 'unmatched_filter'], capture_output=True, text=True)\n",
            "    assert miss.returncode != 0\n",
            "    st, _, _, _ = r._launch_chromium_suite(pathlib.Path(d))\n",
            "    assert st is None\n",
        );
        let python_bin = resolve_python_bin();
        let out = Command::new(python_bin)
            .args(["-B", "-c", py, &script.to_string_lossy()])
            .output()
            .expect("python3/python must run wasm_test_runner CLI test");
        assert!(
            out.status.success(),
            "runner CLI test failed: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
