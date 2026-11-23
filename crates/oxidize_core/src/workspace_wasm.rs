//! Pure-`std` workspace root resolution and `wasm32-unknown-unknown` crate discovery.
//!
//! Shared between `oxidize_core::workspace` and `xtask` so `apps/xtask` remains free
//! of external crate dependencies while dynamically discovering every WebAssembly-compatible
//! workspace crate.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// Typed error returned when discovering or parsing Cargo workspace manifests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    /// Manifest TOML or array syntax could not be parsed.
    InvalidManifestSyntax(String),
    /// Manifest does not contain a `[workspace]` table.
    MissingWorkspaceSection,
    /// `[workspace]` table does not contain a `members` array.
    MissingMembersArray,
    /// Failed to read a manifest file from disk.
    ManifestIo {
        /// Path to the manifest that failed to read.
        path: String,
        /// Underlying OS error description.
        reason: String,
    },
    /// Member `Cargo.toml` is missing `[package]` or `package.name`.
    MissingPackageField {
        /// Path to the member manifest.
        path: String,
        /// Name of the missing TOML key or table.
        field: &'static str,
    },
}

impl fmt::Display for WorkspaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifestSyntax(reason) => {
                write!(f, "Failed to parse Cargo.toml: {reason}")
            }
            Self::MissingWorkspaceSection => {
                write!(f, "Could not find [workspace] section in Cargo.toml")
            }
            Self::MissingMembersArray => {
                write!(f, "Could not find workspace.members array in Cargo.toml")
            }
            Self::ManifestIo { path, reason } => write!(f, "Failed to read {path}: {reason}"),
            Self::MissingPackageField { path, field } => {
                write!(f, "Missing {field} in {path}")
            }
        }
    }
}

impl std::error::Error for WorkspaceError {}

/// Host-only workspace packages excluded from `wasm32-unknown-unknown` test execution.
pub const HOST_ONLY_WORKSPACE_PACKAGES: &[&str] =
    &["xtask", "unified_verification", "markdown_tests"];

/// Stochastic workspace packages required to ship at least one `tests/*_deterministic.rs` suite
/// executed on both native and `wasm32-unknown-unknown` targets.
pub const STOCHASTIC_WORKSPACE_PACKAGES: &[&str] = &[
    "oxidize_core",
    "pure_math",
    "domain_ai",
    "domain_applied",
    "domain_biology",
    "domain_climate",
    "domain_epidemiology",
    "domain_physics",
];

/// Target selection flag kind for testing a workspace crate on `wasm32-unknown-unknown`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmTestTargetKind {
    /// Library target exercised via `cargo test --lib`.
    Lib,
    /// Binary target exercised via `cargo test --bins`.
    Bin,
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
pub fn parse_workspace_members_std(content: &str) -> Result<Vec<String>, WorkspaceError> {
    let ws_body = extract_section_body(content, "workspace")
        .ok_or(WorkspaceError::MissingWorkspaceSection)?;
    for (idx, line) in ws_body.lines().enumerate() {
        if let Some((key, rest)) = line.split_once('=')
            && key.trim() == "members"
        {
            let tail: String = std::iter::once(rest)
                .chain(ws_body.lines().skip(idx + 1))
                .collect::<Vec<_>>()
                .join("\n");
            let open = tail.find('[').ok_or_else(|| {
                WorkspaceError::InvalidManifestSyntax(
                    "Malformed workspace.members array".to_string(),
                )
            })?;
            let close = tail[open + 1..].find(']').ok_or_else(|| {
                WorkspaceError::InvalidManifestSyntax(
                    "Unterminated workspace.members array".to_string(),
                )
            })?;
            return Ok(parse_quoted_array_items(&tail[open + 1..open + 1 + close]));
        }
    }
    Err(WorkspaceError::MissingMembersArray)
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
pub fn discover_wasm_crates(workspace_root: &Path) -> Result<Vec<WasmCrateSpec>, WorkspaceError> {
    let root_manifest = workspace_root.join("Cargo.toml");
    let root_toml = fs::read_to_string(&root_manifest).map_err(|e| WorkspaceError::ManifestIo {
        path: root_manifest.display().to_string(),
        reason: e.to_string(),
    })?;
    let members = parse_workspace_members_std(&root_toml)?;

    let mut specs = Vec::with_capacity(members.len());
    for member in members {
        let member_dir = workspace_root.join(&member);
        let manifest_path = member_dir.join("Cargo.toml");
        let manifest =
            fs::read_to_string(&manifest_path).map_err(|e| WorkspaceError::ManifestIo {
                path: manifest_path.display().to_string(),
                reason: e.to_string(),
            })?;
        let pkg_body = extract_section_body(&manifest, "package").ok_or_else(|| {
            WorkspaceError::MissingPackageField {
                path: manifest_path.display().to_string(),
                field: "[package]",
            }
        })?;
        let package_name = parse_key_string(&pkg_body, "name").ok_or_else(|| {
            WorkspaceError::MissingPackageField {
                path: manifest_path.display().to_string(),
                field: "package.name",
            }
        })?;

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

/// Discovers sorted `tests/*_deterministic.rs` integration test target names for `member_path`.
#[must_use]
pub fn discover_deterministic_tests(workspace_root: &Path, member_path: &str) -> Vec<String> {
    let tests_dir = workspace_root.join(member_path).join("tests");
    let Ok(entries) = fs::read_dir(tests_dir) else {
        return Vec::new();
    };
    let mut targets = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() || path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            && stem.ends_with("_deterministic")
        {
            targets.push(stem.to_string());
        }
    }
    targets.sort();
    targets
}

fn contains_sha256_hex_literal(content: &str) -> bool {
    let bytes = content.as_bytes();
    if bytes.len() < 66 {
        return false;
    }
    for window in bytes.windows(66) {
        if window[0] == b'"'
            && window[65] == b'"'
            && window[1..65]
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
        {
            return true;
        }
    }
    false
}

fn check_suite_pinned_digests(
    workspace_root: &Path,
    spec: &WasmCrateSpec,
    suites: &[String],
    failures: &mut Vec<String>,
) {
    for suite in suites {
        let suite_path = workspace_root
            .join(&spec.member_path)
            .join("tests")
            .join(format!("{suite}.rs"));
        match fs::read_to_string(&suite_path) {
            Ok(src) if contains_sha256_hex_literal(&src) => {}
            Ok(_) => failures.push(format!(
                "{} (`{}/tests/{suite}.rs` missing 64-hex pinned SHA-256 digest)",
                spec.package_name, spec.member_path
            )),
            Err(err) => failures.push(format!(
                "{} (failed to read `{}`: {err})",
                spec.package_name,
                suite_path.display()
            )),
        }
    }
}

/// Verifies that every package in [`STOCHASTIC_WORKSPACE_PACKAGES`] is present in `workspace_root`,
/// defines at least one `tests/*_deterministic.rs` integration suite, and that every discovered
/// `tests/*_deterministic.rs` suite pins at least one 64-character SHA-256 hex digest literal.
pub fn verify_stochastic_deterministic_suites(
    workspace_root: &Path,
) -> Result<Vec<(String, Vec<String>)>, String> {
    let crates = discover_wasm_crates(workspace_root).map_err(|e| e.to_string())?;
    let mut report = Vec::with_capacity(STOCHASTIC_WORKSPACE_PACKAGES.len());
    let mut failures = Vec::new();
    for &pkg in STOCHASTIC_WORKSPACE_PACKAGES {
        let Some(spec) = crates.iter().find(|c| c.package_name == pkg) else {
            failures.push(format!("{pkg} (not discovered in workspace members)"));
            continue;
        };
        let suites = discover_deterministic_tests(workspace_root, &spec.member_path);
        if suites.is_empty() {
            failures.push(format!(
                "{pkg} (missing `{}/tests/*_deterministic.rs`)",
                spec.member_path
            ));
        } else {
            report.push((pkg.to_string(), suites));
        }
    }
    for spec in &crates {
        let suites = discover_deterministic_tests(workspace_root, &spec.member_path);
        check_suite_pinned_digests(workspace_root, spec, &suites, &mut failures);
    }
    if failures.is_empty() {
        Ok(report)
    } else {
        Err(format!(
            "Deterministic suite verification failed: {}",
            failures.join(", ")
        ))
    }
}

/// Constructs Cargo CLI argument vectors (`cargo test --target wasm32-unknown-unknown ...`)
/// covering library targets, binary targets, and `tests/*_deterministic.rs` suites under `workspace_root`.
#[must_use]
pub fn build_wasm_cargo_test_invocations_in(
    workspace_root: Option<&Path>,
    crates: &[WasmCrateSpec],
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
        invocations.push(args);
    }

    for bin_crate in crates
        .iter()
        .filter(|c| c.target_kind == WasmTestTargetKind::Bin)
    {
        invocations.push(vec![
            "test".to_string(),
            "--target".to_string(),
            "wasm32-unknown-unknown".to_string(),
            "-p".to_string(),
            bin_crate.package_name.clone(),
            "--bins".to_string(),
        ]);
    }

    if let Some(root) = workspace_root {
        for spec in crates {
            let suites = discover_deterministic_tests(root, &spec.member_path);
            if suites.is_empty() {
                continue;
            }
            let mut args = vec![
                "test".to_string(),
                "--target".to_string(),
                "wasm32-unknown-unknown".to_string(),
                "-p".to_string(),
                spec.package_name.clone(),
            ];
            if spec.package_name == "math_explorer" {
                args.push("--all-features".to_string());
            }
            for suite in suites {
                args.push("--test".to_string());
                args.push(suite);
            }
            invocations.push(args);
        }
    }
    invocations
}

/// Constructs Cargo CLI argument vectors (`cargo test --target wasm32-unknown-unknown ...`)
/// covering all discovered library, binary, and `tests/*_deterministic.rs` targets.
#[must_use]
pub fn build_wasm_cargo_test_invocations(crates: &[WasmCrateSpec]) -> Vec<Vec<String>> {
    let ws_root = std::env::current_dir()
        .ok()
        .and_then(|cwd| find_workspace_root(&cwd))
        .or_else(|| find_workspace_root(Path::new(env!("CARGO_MANIFEST_DIR"))));
    build_wasm_cargo_test_invocations_in(ws_root.as_deref(), crates)
}
