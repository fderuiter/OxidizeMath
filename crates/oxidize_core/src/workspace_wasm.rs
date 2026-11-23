//! Pure-`std` workspace root resolution, `wasm32-unknown-unknown` crate discovery,
//! and Cargo `--message-format=json` test artifact extraction.
//!
//! Shared between `oxidize_core::workspace` and `xtask` so `apps/xtask` remains free
//! of external crate dependencies while dynamically discovering every WebAssembly-compatible
//! workspace crate.

use std::fs;
use std::path::{Path, PathBuf};

/// Host-only workspace packages excluded from `wasm32-unknown-unknown` test execution.
pub const HOST_ONLY_WORKSPACE_PACKAGES: &[&str] =
    &["xtask", "unified_verification", "markdown_tests"];

/// Workspace packages whose `wasm32-unknown-unknown` suites require `wasm-bindgen-test` in Chromium.
pub const BROWSER_WASM_PACKAGES: &[&str] = &["oxidize_core", "egui_plot", "math_explorer_gui"];

/// Target selection flag kind for testing a workspace crate on `wasm32-unknown-unknown`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmTestTargetKind {
    /// Library target exercised via `cargo test --lib`.
    Lib,
    /// Binary target exercised via `cargo test --bins`.
    Bin,
}

/// Runtime environment used to execute a workspace crate's `wasm32-unknown-unknown` test binary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmExecutionRuntime {
    /// Headless Chromium via `wasm-bindgen --target web` under `PRODUCTION_CSP`.
    BrowserChromium,
    /// Pure-Rust `libtest` (`exports.main(0, 0)`) instantiated directly in Node.js (`node`).
    NodeJs,
}

/// Metadata describing a `wasm32-unknown-unknown`-compatible workspace crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmCrateSpec {
    /// Workspace-relative member directory path from `[workspace].members`.
    pub member_path: String,
    /// Cargo `[package].name` identifier passed to `-p`.
    pub package_name: String,
    /// Whether the crate exposes a library or binary target for `wasm32` testing.
    pub target_kind: WasmTestTargetKind,
}

impl WasmCrateSpec {
    /// Returns the execution runtime (`BrowserChromium` vs `NodeJs`) for this crate.
    #[must_use]
    pub fn execution_runtime(&self) -> WasmExecutionRuntime {
        if BROWSER_WASM_PACKAGES.contains(&self.package_name.as_str()) {
            WasmExecutionRuntime::BrowserChromium
        } else {
            WasmExecutionRuntime::NodeJs
        }
    }
}

/// Walks `start` and its ancestors to find the directory whose `Cargo.toml` defines `[workspace]`.
#[must_use]
pub fn find_workspace_root(start: &Path) -> Option<PathBuf> {
    let base = if start.is_file() {
        start.parent()?
    } else {
        start
    };
    let resolved = fs::canonicalize(base).unwrap_or_else(|_| base.to_path_buf());
    for ancestor in resolved.ancestors() {
        let manifest = ancestor.join("Cargo.toml");
        if let Ok(content) = fs::read_to_string(&manifest)
            && has_toml_section(&content, "workspace")
        {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn strip_toml_line_comment(line: &str) -> &str {
    let mut in_double = false;
    let mut in_single = false;
    let mut escaped = false;
    for (idx, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_double {
            escaped = true;
            continue;
        }
        if ch == '"' && !in_single {
            in_double = !in_double;
        } else if ch == '\'' && !in_double {
            in_single = !in_single;
        } else if ch == '#' && !in_double && !in_single {
            return &line[..idx];
        }
    }
    line
}

fn has_toml_section(content: &str, section: &str) -> bool {
    let header = format!("[{section}]");
    content
        .lines()
        .map(|l| strip_toml_line_comment(l).trim())
        .any(|l| l == header)
}

fn extract_section_body(content: &str, section: &str) -> Option<String> {
    let header = format!("[{section}]");
    let mut in_section = false;
    let mut collected = String::new();
    for raw_line in content.lines() {
        let line = strip_toml_line_comment(raw_line).trim();
        if line.starts_with('[') && line.ends_with(']') {
            if in_section {
                break;
            }
            if line == header {
                in_section = true;
            }
            continue;
        }
        if in_section {
            collected.push_str(line);
            collected.push('\n');
        }
    }
    in_section.then_some(collected)
}

fn parse_quoted_array_items(array_body: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut chars = array_body.chars();
    while let Some(ch) = chars.next() {
        if ch == '"' || ch == '\'' {
            let quote = ch;
            let mut val = String::new();
            for next in chars.by_ref() {
                if next == quote {
                    items.push(val);
                    break;
                }
                val.push(next);
            }
        }
    }
    items
}

/// Parses `workspace.members` from a `Cargo.toml` string using only the Rust standard library.
pub fn parse_workspace_members_std(content: &str) -> Result<Vec<String>, String> {
    let ws_body = extract_section_body(content, "workspace")
        .ok_or_else(|| "Could not find [workspace] section in Cargo.toml".to_string())?;
    for (idx, line) in ws_body.lines().enumerate() {
        if let Some((key, rest)) = line.split_once('=')
            && key.trim() == "members"
        {
            let tail: String = std::iter::once(rest)
                .chain(ws_body.lines().skip(idx + 1))
                .collect::<Vec<_>>()
                .join("\n");
            let open = tail
                .find('[')
                .ok_or_else(|| "Malformed workspace.members array".to_string())?;
            let close = tail[open + 1..]
                .find(']')
                .ok_or_else(|| "Unterminated workspace.members array".to_string())?;
            return Ok(parse_quoted_array_items(&tail[open + 1..open + 1 + close]));
        }
    }
    Err("Could not find workspace.members array in Cargo.toml".to_string())
}

fn parse_key_string(section_body: &str, target_key: &str) -> Option<String> {
    for line in section_body.lines() {
        let Some((key, raw_val)) = line.split_once('=') else {
            continue;
        };
        if key.trim() != target_key {
            continue;
        }
        let trimmed = raw_val.trim();
        if (trimmed.starts_with('"') && trimmed.ends_with('"')
            || trimmed.starts_with('\'') && trimmed.ends_with('\''))
            && trimmed.len() >= 2
        {
            return Some(trimmed[1..trimmed.len() - 1].to_string());
        }
    }
    None
}

fn is_proc_macro_manifest(content: &str) -> bool {
    extract_section_body(content, "lib").is_some_and(|lib_body| {
        lib_body.lines().any(|line| {
            line.split_once('=')
                .is_some_and(|(k, v)| k.trim() == "proc-macro" && v.trim() == "true")
        })
    })
}

/// Discovers all `wasm32-unknown-unknown`-compatible crates under `workspace_root`.
pub fn discover_wasm_crates(workspace_root: &Path) -> Result<Vec<WasmCrateSpec>, String> {
    let root_manifest = workspace_root.join("Cargo.toml");
    let root_toml = fs::read_to_string(&root_manifest)
        .map_err(|e| format!("Failed to read {}: {e}", root_manifest.display()))?;
    let members = parse_workspace_members_std(&root_toml)?;

    let mut specs = Vec::with_capacity(members.len());
    for member in members {
        let member_dir = workspace_root.join(&member);
        let manifest_path = member_dir.join("Cargo.toml");
        let manifest = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read {}: {e}", manifest_path.display()))?;
        let pkg_body = extract_section_body(&manifest, "package")
            .ok_or_else(|| format!("Missing [package] in {}", manifest_path.display()))?;
        let package_name = parse_key_string(&pkg_body, "name")
            .ok_or_else(|| format!("Missing package.name in {}", manifest_path.display()))?;

        if HOST_ONLY_WORKSPACE_PACKAGES.contains(&package_name.as_str())
            || is_proc_macro_manifest(&manifest)
        {
            continue;
        }

        let has_lib = member_dir.join("src/lib.rs").is_file() || has_toml_section(&manifest, "lib");
        let has_bin = member_dir.join("src/main.rs").is_file() || manifest.contains("[[bin]]");
        let target_kind = if has_lib {
            WasmTestTargetKind::Lib
        } else if has_bin {
            WasmTestTargetKind::Bin
        } else {
            continue;
        };

        specs.push(WasmCrateSpec {
            member_path: member,
            package_name,
            target_kind,
        });
    }
    Ok(specs)
}

pub(crate) fn build_cargo_test_group_invocations(
    crates: &[&WasmCrateSpec],
    no_run_json: bool,
) -> Vec<Vec<String>> {
    let mut invocations = Vec::new();
    let lib_pkgs: Vec<&str> = crates
        .iter()
        .filter(|c| c.target_kind == WasmTestTargetKind::Lib)
        .map(|c| c.package_name.as_str())
        .collect();
    if !lib_pkgs.is_empty() {
        let mut args = vec![
            "test".to_string(),
            "--target".to_string(),
            "wasm32-unknown-unknown".to_string(),
        ];
        for pkg in lib_pkgs {
            args.push("-p".to_string());
            args.push(pkg.to_string());
        }
        args.push("--lib".to_string());
        if no_run_json {
            args.push("--no-run".to_string());
            args.push("--message-format=json".to_string());
        }
        invocations.push(args);
    }

    for bin_crate in crates
        .iter()
        .filter(|c| c.target_kind == WasmTestTargetKind::Bin)
    {
        let mut args = vec![
            "test".to_string(),
            "--target".to_string(),
            "wasm32-unknown-unknown".to_string(),
            "-p".to_string(),
            bin_crate.package_name.clone(),
            "--bins".to_string(),
        ];
        if no_run_json {
            args.push("--no-run".to_string());
            args.push("--message-format=json".to_string());
        }
        invocations.push(args);
    }
    invocations
}

/// Constructs Cargo `--no-run --message-format=json` argument vectors for a specific `runtime`.
#[must_use]
pub fn build_wasm_no_run_json_invocations(
    crates: &[WasmCrateSpec],
    runtime: WasmExecutionRuntime,
) -> Vec<Vec<String>> {
    let filtered: Vec<&WasmCrateSpec> = crates
        .iter()
        .filter(|c| c.execution_runtime() == runtime)
        .collect();
    build_cargo_test_group_invocations(&filtered, true)
}

fn decode_json_string_at(line: &str, start_quote: usize) -> Option<(String, usize)> {
    let bytes = line.as_bytes();
    if bytes.get(start_quote) != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = line[start_quote + 1..].char_indices();
    while let Some((rel_idx, ch)) = chars.next() {
        if ch == '"' {
            return Some((out, start_quote + 1 + rel_idx + 1));
        }
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        let (_, esc) = chars.next()?;
        match esc {
            '"' | '\\' | '/' => out.push(esc),
            'b' => out.push('\u{0008}'),
            'f' => out.push('\u{000C}'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'u' => {
                let mut hex = String::with_capacity(4);
                for _ in 0..4 {
                    hex.push(chars.next()?.1);
                }
                let code = u32::from_str_radix(&hex, 16).ok()?;
                out.push(char::from_u32(code)?);
            }
            _ => return None,
        }
    }
    None
}

fn extract_top_level_json_string_field(line: &str, target_key: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut depth = 0usize;
    let mut idx = 0usize;
    while idx < bytes.len() {
        match bytes[idx] {
            b'{' | b'[' => {
                depth += 1;
                idx += 1;
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                idx += 1;
            }
            b'"' => {
                let (decoded, next_idx) = decode_json_string_at(line, idx)?;
                idx = next_idx;
                if depth == 1 && decoded == target_key {
                    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
                        idx += 1;
                    }
                    if bytes.get(idx) == Some(&b':') {
                        idx += 1;
                        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
                            idx += 1;
                        }
                        if bytes.get(idx) == Some(&b'"') {
                            let (val, _) = decode_json_string_at(line, idx)?;
                            return Some(val);
                        }
                        return None;
                    }
                }
            }
            _ => idx += 1,
        }
    }
    None
}

fn line_has_test_profile_true(line: &str) -> bool {
    let Some(prof_pos) = line.find("\"profile\"") else {
        return false;
    };
    let tail = &line[prof_pos..];
    let Some(open) = tail.find('{') else {
        return false;
    };
    let Some(close) = tail[open..].find('}') else {
        return false;
    };
    let profile_obj = &tail[open..=open + close];
    let compact: String = profile_obj
        .chars()
        .filter(|c| !c.is_ascii_whitespace())
        .collect();
    compact.contains("\"test\":true")
}

/// Extracts compiled `.wasm` test executable paths from `cargo test --no-run --message-format=json` stdout.
pub fn extract_wasm_test_executables_from_cargo_json(stdout: &str) -> Result<Vec<PathBuf>, String> {
    let mut artifacts = Vec::new();
    for raw_line in stdout.lines() {
        let line = raw_line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Some(reason) = extract_top_level_json_string_field(line, "reason") else {
            continue;
        };
        if reason != "compiler-artifact" || !line_has_test_profile_true(line) {
            continue;
        }
        if let Some(exe) = extract_top_level_json_string_field(line, "executable")
            && exe.ends_with(".wasm")
        {
            let path = PathBuf::from(exe);
            if !artifacts.contains(&path) {
                artifacts.push(path);
            }
        }
    }
    if artifacts.is_empty() {
        return Err("No wasm32 test executable artifacts found in Cargo JSON output".to_string());
    }
    Ok(artifacts)
}
