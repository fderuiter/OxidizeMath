//! Dynamic `#![no_std]` crate discovery, isolated feature-leg checks, and bare-metal verification.

use crate::gates::{resolve_workspace_root, run_cmd_in_dir_with_env, workspace_wasm};
use std::fs;
use std::path::Path;
use std::process::{Command, exit};

/// Canonical bare-metal ARM Cortex-M4F target without `std` or an OS allocator.
pub const BARE_METAL_TARGET: &str = "thumbv7em-none-eabihf";

/// Discovered metadata for a workspace library crate declaring `#![no_std]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoStdCrateSpec {
    /// Workspace-relative path from `[workspace].members`.
    pub member_path: String,
    /// Cargo `[package].name` identifier passed to `-p`.
    pub package_name: String,
    /// Whether `[features]` defines an `alloc` feature.
    pub has_alloc: bool,
    /// Whether `[features]` defines a `serde` feature.
    pub has_serde: bool,
    /// Whether `[features]` defines a `std` feature.
    pub has_std: bool,
    /// Whether `[features].default` enables `std`.
    pub default_has_std: bool,
}

/// Feature combination leg exercised by `xtask check-no-std`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureLeg {
    /// Human-readable label for diagnostic banners.
    pub label: &'static str,
    /// Extra Cargo CLI flags selecting the feature set.
    pub cargo_args: Vec<&'static str>,
    /// Whether this leg may compile on bare-metal `thumbv7em-none-eabihf`.
    pub bare_metal_compatible: bool,
}

#[derive(Default)]
struct ParsedManifest {
    package_name: Option<String>,
    is_proc_macro: bool,
    has_alloc: bool,
    has_serde: bool,
    has_std: bool,
    default_has_std: bool,
}

fn strip_unquoted_hash(raw: &str) -> &str {
    let mut quoted = false;
    for (idx, ch) in raw.char_indices() {
        if ch == '"' {
            quoted = !quoted;
        } else if ch == '#' && !quoted {
            return &raw[..idx];
        }
    }
    raw
}

fn collect_quoted_literals(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut inside = false;
    for ch in text.chars() {
        if ch == '"' {
            if inside {
                out.push(std::mem::take(&mut current));
            }
            inside = !inside;
        } else if inside {
            current.push(ch);
        }
    }
    out
}

fn record_feature_entry(entry: &str, meta: &mut ParsedManifest) {
    let Some((raw_key, raw_rhs)) = entry.split_once('=') else {
        return;
    };
    match raw_key.trim() {
        "alloc" => meta.has_alloc = true,
        "serde" => meta.has_serde = true,
        "std" => meta.has_std = true,
        "default" => {
            meta.default_has_std = collect_quoted_literals(raw_rhs)
                .iter()
                .any(|item| item == "std");
        }
        _ => {}
    }
}

fn parse_crate_manifest(manifest: &str) -> ParsedManifest {
    let mut meta = ParsedManifest::default();
    let mut section = String::new();
    let mut feature_buf = String::new();
    let mut bracket_depth: usize = 0;

    for raw_line in manifest.lines() {
        let line = strip_unquoted_hash(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        if bracket_depth == 0 && line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }
        match section.as_str() {
            "package" => {
                if let Some((k, rhs)) = line.split_once('=')
                    && k.trim() == "name"
                {
                    meta.package_name = collect_quoted_literals(rhs).into_iter().next();
                }
            }
            "lib" => {
                if let Some((k, rhs)) = line.split_once('=')
                    && k.trim() == "proc-macro"
                    && rhs.trim() == "true"
                {
                    meta.is_proc_macro = true;
                }
            }
            "features" => {
                bracket_depth = bracket_depth
                    .saturating_add(line.bytes().filter(|&b| b == b'[').count())
                    .saturating_sub(line.bytes().filter(|&b| b == b']').count());
                feature_buf.push_str(line);
                feature_buf.push(' ');
                if bracket_depth == 0 {
                    record_feature_entry(&feature_buf, &mut meta);
                    feature_buf.clear();
                }
            }
            _ => {}
        }
    }
    meta
}

/// Returns `true` when `lib_source` declares crate-level `#![no_std]` or
/// `#![cfg_attr(not(feature = "std"), no_std)]`.
#[must_use]
pub fn declares_no_std_attribute(lib_source: &str) -> bool {
    for raw_line in lib_source.lines() {
        let trimmed = raw_line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        let code_head = trimmed.split("//").next().unwrap_or("").trim();
        if !code_head.starts_with("#![") {
            continue;
        }
        let compact: String = code_head.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.contains("#![no_std]")
            || compact.contains("#![cfg_attr(not(feature=\"std\"),no_std)]")
        {
            return true;
        }
    }
    false
}

/// Dynamically discovers all `#![no_std]` library crates registered in `[workspace].members`.
pub fn discover_no_std_crates(workspace_root: &Path) -> Result<Vec<NoStdCrateSpec>, String> {
    let root_manifest = workspace_root.join("Cargo.toml");
    let root_toml = fs::read_to_string(&root_manifest)
        .map_err(|e| format!("Failed to read {}: {e}", root_manifest.display()))?;
    let members = workspace_wasm::parse_workspace_members_std(&root_toml)
        .map_err(|e| format!("Failed to parse workspace members: {e}"))?;

    let mut discovered = Vec::new();
    for member in members {
        let member_dir = workspace_root.join(&member);
        let lib_path = member_dir.join("src/lib.rs");
        if !lib_path.is_file() {
            continue;
        }
        let manifest_path = member_dir.join("Cargo.toml");
        let manifest_src = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read {}: {e}", manifest_path.display()))?;
        let meta = parse_crate_manifest(&manifest_src);
        if meta.is_proc_macro {
            continue;
        }
        let package_name = meta
            .package_name
            .ok_or_else(|| format!("Missing [package].name in {}", manifest_path.display()))?;
        let lib_src = fs::read_to_string(&lib_path)
            .map_err(|e| format!("Failed to read {}: {e}", lib_path.display()))?;
        if !declares_no_std_attribute(&lib_src) {
            continue;
        }
        discovered.push(NoStdCrateSpec {
            member_path: member,
            package_name,
            has_alloc: meta.has_alloc,
            has_serde: meta.has_serde,
            has_std: meta.has_std,
            default_has_std: meta.default_has_std,
        });
    }
    Ok(discovered)
}

/// Builds the ordered feature legs exercised for a discovered `#![no_std]` crate.
#[must_use]
pub fn feature_legs_for_crate(spec: &NoStdCrateSpec) -> Vec<FeatureLeg> {
    let mut legs = vec![
        FeatureLeg {
            label: "default",
            cargo_args: Vec::new(),
            bare_metal_compatible: false,
        },
        FeatureLeg {
            label: "no-default-features",
            cargo_args: vec!["--no-default-features"],
            bare_metal_compatible: true,
        },
    ];
    if spec.has_alloc {
        legs.push(FeatureLeg {
            label: "alloc",
            cargo_args: vec!["--no-default-features", "--features", "alloc"],
            bare_metal_compatible: true,
        });
    }
    if spec.has_serde {
        legs.push(FeatureLeg {
            label: "serde",
            cargo_args: vec!["--no-default-features", "--features", "serde"],
            bare_metal_compatible: true,
        });
    }
    if spec.has_alloc && spec.has_serde {
        legs.push(FeatureLeg {
            label: "alloc,serde",
            cargo_args: vec!["--no-default-features", "--features", "alloc,serde"],
            bare_metal_compatible: true,
        });
    }
    legs
}

/// Scans `cargo tree -e normal --prefix none -f "{p} [{f}]"` output for `std` or `alloc` leaks.
#[must_use]
pub fn find_serde_feature_leaks(cargo_tree_output: &str) -> Vec<String> {
    let mut leaks = Vec::new();
    for raw_line in cargo_tree_output.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.contains("(proc-macro)") {
            continue;
        }
        let Some((pkg_part, rest)) = line.rsplit_once('[') else {
            continue;
        };
        let Some(features_body) = rest.strip_suffix(']') else {
            continue;
        };
        for feat in features_body.split(',').map(str::trim) {
            if matches!(feat, "std" | "alloc") {
                leaks.push(format!("{} activates `{feat}`", pkg_part.trim()));
            }
        }
    }
    leaks
}

fn parse_package_filters(args: &[String]) -> Result<Vec<String>, String> {
    let mut filters = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "-p" | "--package" => {
                let Some(pkg) = args.get(idx + 1) else {
                    return Err(format!("Missing package name after `{}`", args[idx]));
                };
                filters.push(pkg.clone());
                idx += 2;
            }
            other => {
                return Err(format!(
                    "Unknown check-no-std flag `{other}` (expected `-p <pkg>` or `--package <pkg>`)"
                ));
            }
        }
    }
    Ok(filters)
}

fn verify_serde_feature_graph(ws_root: &Path, package_name: &str) {
    let tree_args = [
        "tree",
        "-e",
        "normal",
        "--prefix",
        "none",
        "-p",
        package_name,
        "--no-default-features",
        "--features",
        "serde",
        "-f",
        "{p} [{f}]",
    ];
    println!(
        "Running (in {}): cargo {}",
        ws_root.display(),
        tree_args.join(" ")
    );
    let output = Command::new("cargo")
        .current_dir(ws_root)
        .args(tree_args)
        .output()
        .unwrap_or_else(|e| {
            eprintln!("Failed to execute cargo tree for {package_name}: {e}");
            exit(1);
        });
    if !output.status.success() {
        eprintln!(
            "cargo tree failed for {package_name}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        exit(1);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let leaks = find_serde_feature_leaks(&stdout);
    if !leaks.is_empty() {
        eprintln!(
            "Error: `{package_name}` leaked std/alloc under `--no-default-features --features serde`:"
        );
        for leak in leaks {
            eprintln!("  - {leak}");
        }
        exit(1);
    }
}

fn run_crate_checks(ws_root: &Path, spec: &NoStdCrateSpec) {
    println!(
        "\n--- Checking #![no_std] crate `{}` ({}) ---",
        spec.package_name, spec.member_path
    );
    if spec.has_serde {
        verify_serde_feature_graph(ws_root, &spec.package_name);
    }

    for leg in feature_legs_for_crate(spec) {
        if leg.bare_metal_compatible {
            let mut bm_args = vec![
                "check",
                "--lib",
                "--target",
                BARE_METAL_TARGET,
                "-p",
                spec.package_name.as_str(),
            ];
            bm_args.extend_from_slice(&leg.cargo_args);
            run_cmd_in_dir_with_env("cargo", &bm_args, ws_root, &[]);
        }

        let mut doc_args = vec!["doc", "--no-deps", "-p", spec.package_name.as_str()];
        doc_args.extend_from_slice(&leg.cargo_args);
        run_cmd_in_dir_with_env(
            "cargo",
            &doc_args,
            ws_root,
            &[("RUSTDOCFLAGS", "-D warnings")],
        );

        let mut test_args = vec!["test", "-p", spec.package_name.as_str(), "--lib", "--tests"];
        test_args.extend_from_slice(&leg.cargo_args);
        run_cmd_in_dir_with_env("cargo", &test_args, ws_root, &[]);

        let mut doctest_args = vec!["test", "--doc", "-p", spec.package_name.as_str()];
        doctest_args.extend_from_slice(&leg.cargo_args);
        run_cmd_in_dir_with_env("cargo", &doctest_args, ws_root, &[]);
    }
}

/// Discovers all `#![no_std]` workspace crates and verifies bare-metal, rustdoc, test, doctest,
/// and `serde` feature-graph invariants.
pub fn check_no_std(args: &[String]) {
    println!("=== Check #![no_std] Crates & Bare-Metal Target ({BARE_METAL_TARGET}) ===");
    let filters = parse_package_filters(args).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        exit(1);
    });
    let ws_root = resolve_workspace_root();
    let discovered = discover_no_std_crates(&ws_root).unwrap_or_else(|err| {
        eprintln!("Error: {err}");
        exit(1);
    });
    if discovered.is_empty() {
        eprintln!("Error: No #![no_std] crates discovered in workspace");
        exit(1);
    }

    let selected: Vec<&NoStdCrateSpec> = if filters.is_empty() {
        discovered.iter().collect()
    } else {
        for wanted in &filters {
            if !discovered.iter().any(|c| &c.package_name == wanted) {
                eprintln!("Error: Package `{wanted}` is not a discovered #![no_std] crate");
                exit(1);
            }
        }
        discovered
            .iter()
            .filter(|c| filters.contains(&c.package_name))
            .collect()
    };

    let names: Vec<&str> = selected.iter().map(|c| c.package_name.as_str()).collect();
    println!(
        "Discovered {} #![no_std] crate(s): {}",
        selected.len(),
        names.join(", ")
    );
    for spec in selected {
        run_crate_checks(&ws_root, spec);
    }
    println!("\n=== All #![no_std] & Bare-Metal Checks Passed ===");
}

#[cfg(test)]
mod tests;
