use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashSet;

#[test]
fn test_cargo_cache_action_pins_v4_node20_and_hashes_cargo_toml() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let root = crate::manifest_cache::workspace_root_dir();
    let action_path = root.join(".github/actions/cargo-cache/action.yml");
    let content =
        fs::read_to_string(&action_path).expect("cargo-cache action.yml should be readable");

    assert!(
        !content.contains("6f8efc29b200d32929f49075959781ed54ec270c"),
        "Deprecated Node 16 actions/cache v3.5.0 SHA must not remain in cargo-cache"
    );

    let yaml: Value =
        serde_saphyr::from_str(&content).expect("cargo-cache action.yml must be valid YAML");
    let steps = yaml
        .get("runs")
        .and_then(|r| r.get("steps"))
        .and_then(Value::as_array)
        .expect("composite action must define runs.steps");

    let lock_hash_expr = steps[0]
        .get("env")
        .and_then(|e| e.get("LOCK_AND_CONFIG_HASH"))
        .and_then(Value::as_str)
        .expect("cache-meta step must bind LOCK_AND_CONFIG_HASH in env");
    for pattern in [
        "'Cargo.lock'",
        "'**/Cargo.lock'",
        "'Cargo.toml'",
        "'**/Cargo.toml'",
        "'.cargo/config.toml'",
        "'math_explorer_gui/Trunk.toml'",
    ] {
        assert!(
            lock_hash_expr.contains(pattern),
            "LOCK_AND_CONFIG_HASH must hash {pattern}, got: {lock_hash_expr}"
        );
    }

    assert!(
        yaml.get("inputs")
            .and_then(|i| i.get("restore-only"))
            .is_some(),
        "cargo-cache action.yml must define a restore-only input"
    );

    assert_cargo_cache_uses_steps(steps);
}

fn assert_cargo_cache_uses_steps(steps: &[Value]) {
    let uses_steps: Vec<&Value> = steps.iter().filter(|s| s.get("uses").is_some()).collect();
    assert_eq!(
        uses_steps.len(),
        4,
        "Expected read-write and restore-only steps for registry and target caches"
    );
    let mut target_step_count = 0usize;
    for step in uses_steps {
        let uses_str = step.get("uses").and_then(Value::as_str).unwrap_or_default();
        assert!(
            matches!(
                uses_str,
                "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830"
                    | "actions/cache/restore@0057852bfaa89a56745cba8c7296529d2fc39830"
            ),
            "Unexpected action pin in cargo-cache: {uses_str}"
        );
        let cache_paths = step
            .get("with")
            .and_then(|w| w.get("path"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        assert!(
            !cache_paths.contains(".cargo/bin"),
            "cargo-cache must never cache mutable ~/.cargo/bin binaries"
        );
        if cache_paths.contains("target/") {
            target_step_count += 1;
            for excluded in [
                "incremental/",
                "doc/",
                "llvm-cov-target/",
                "package/",
                "tmp/",
            ] {
                assert!(
                    cache_paths.contains(excluded) && cache_paths.contains('!'),
                    "target cache path must exclude {excluded}, got: {cache_paths}"
                );
            }
        }
    }
    assert_eq!(target_step_count, 2, "Expected two target cache steps");
}

#[test]
fn test_ci_tool_cache_action_isolates_binaries_outside_cargo_bin() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let root = crate::manifest_cache::workspace_root_dir();
    let action_path = root.join(".github/actions/ci-tool-cache/action.yml");
    let content =
        fs::read_to_string(&action_path).expect("ci-tool-cache action.yml should be readable");
    let yaml: Value =
        serde_saphyr::from_str(&content).expect("ci-tool-cache action.yml must be valid YAML");

    let steps = yaml
        .get("runs")
        .and_then(|r| r.get("steps"))
        .and_then(Value::as_array)
        .expect("ci-tool-cache must define runs.steps");
    let uses_steps: Vec<&Value> = steps.iter().filter(|s| s.get("uses").is_some()).collect();
    assert_eq!(uses_steps.len(), 2, "Expected cache and binstall steps");
    assert_eq!(
        uses_steps[0].get("uses").and_then(Value::as_str),
        Some("actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830")
    );
    let tool_cache_path = uses_steps[0]
        .get("with")
        .and_then(|w| w.get("path"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        tool_cache_path.contains(".cache/ci-tools/") && !tool_cache_path.contains(".cargo/bin"),
        "ci-tool-cache must store binaries outside ~/.cargo/bin, got: {tool_cache_path}"
    );
    assert_eq!(
        uses_steps[1].get("uses").and_then(Value::as_str),
        Some("cargo-bins/cargo-binstall@5885714d0a40a14b2535c6ffc051fd8e49aa9665")
    );
    assert_eq!(
        uses_steps[1].get("if").and_then(Value::as_str),
        Some("steps.tool-cache.outputs.cache-hit != 'true'"),
        "cargo-binstall must be skipped on tool cache hits"
    );
}

#[test]
fn test_ci_workflow_shares_feature_matrix_cache_and_caches_pinned_tools() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let root = crate::manifest_cache::workspace_root_dir();
    let ci_content = fs::read_to_string(root.join(".github/workflows/ci.yml"))
        .expect("ci.yml should be readable");
    let yaml: Value = serde_saphyr::from_str(&ci_content).expect("ci.yml must be valid YAML");

    assert_eq!(
        yaml.get("env")
            .and_then(|e| e.get("CARGO_INCREMENTAL"))
            .and_then(Value::as_str),
        Some("0"),
        "ci.yml must set CARGO_INCREMENTAL: \"0\" globally"
    );

    let fm_steps = yaml["jobs"]["feature-matrix"]["steps"]
        .as_array()
        .expect("feature-matrix must define steps");
    let fm_cache = fm_steps
        .iter()
        .find(|s| s.get("uses").and_then(Value::as_str) == Some("./.github/actions/cargo-cache"))
        .expect("feature-matrix must use cargo-cache");
    assert_eq!(
        fm_cache["with"]["scope"].as_str(),
        Some("features-shared"),
        "feature-matrix must share one features-shared scope"
    );
    assert!(
        fm_cache["with"]["restore-only"]
            .as_str()
            .is_some_and(|expr| expr.contains("matrix.feature !=")),
        "feature-matrix must gate 7 of 8 legs with restore-only"
    );

    for expected in [
        "tool: cargo-audit",
        "tool: cargo-llvm-cov",
        "tool: trunk",
        "tool: wasm-bindgen-cli",
        "tool: cargo-packager",
        "path: ~/.cache/trunk",
    ] {
        assert!(
            ci_content.contains(expected),
            "ci.yml missing expected cached tool entry: {expected}"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn locate_build_root(exe: &Path) -> PathBuf {
    if let Some(build_dir) = exe
        .ancestors()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("build") && p.is_dir())
    {
        return build_dir.to_path_buf();
    }
    let deps_dir = exe.parent().expect("test binary must have parent dir");
    let profile_dir = if deps_dir.file_name().and_then(|n| n.to_str()) == Some("deps") {
        deps_dir
            .parent()
            .expect("deps dir must have profile parent")
    } else {
        deps_dir
    };
    profile_dir.join("build")
}

#[cfg(not(target_arch = "wasm32"))]
fn find_latest_build_output(
    build_root: &Path,
    crate_name: &str,
    artifact: &str,
) -> Option<PathBuf> {
    let mut unit_dirs = Vec::new();
    let v2_crate_dir = build_root.join(crate_name);
    if let Ok(entries) = fs::read_dir(&v2_crate_dir) {
        unit_dirs.extend(entries.flatten().map(|e| e.path()));
    }
    let v1_prefix = format!("{crate_name}-");
    if let Ok(entries) = fs::read_dir(build_root) {
        unit_dirs.extend(entries.flatten().map(|e| e.path()).filter(|dir| {
            dir.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&v1_prefix))
        }));
    }

    unit_dirs
        .into_iter()
        .filter(|dir| dir.join("out").join(artifact).is_file())
        .filter_map(|dir| {
            [dir.join("output"), dir.join("run").join("stdout")]
                .into_iter()
                .find(|p| p.is_file())
        })
        .max_by_key(|out_file| fs::metadata(out_file).and_then(|m| m.modified()).ok())
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_rerun_directives(output_path: &Path) -> HashSet<String> {
    let raw = fs::read_to_string(output_path).expect("build script output file should be readable");
    let paths: HashSet<String> = raw
        .lines()
        .filter_map(|line| line.strip_prefix("cargo:rerun-if-changed="))
        .map(ToString::to_string)
        .collect();
    assert!(
        !paths.is_empty(),
        "Expected non-empty rerun directives in {}",
        output_path.display()
    );
    for p in &paths {
        assert!(
            !p.trim().is_empty(),
            "Empty rerun path in {}",
            output_path.display()
        );
        assert!(
            !p.contains('\\'),
            "Unnormalized backslash in {}: {p}",
            output_path.display()
        );
    }
    paths
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_workspace_build_script_outputs_emit_rerun_directives() {
    let exe = std::env::current_exe().expect("current_exe should resolve");
    let build_root = locate_build_root(&exe);

    if let Some(md_output) =
        find_latest_build_output(&build_root, "markdown_tests", "generated_tests.rs")
    {
        let md_rerun = parse_rerun_directives(&md_output);
        for expected in ["build.rs", "../../Cargo.toml"] {
            assert!(
                md_rerun.contains(expected),
                "markdown_tests output missing cargo:rerun-if-changed={expected}"
            );
        }
    }

    if let Some(gui_output) = find_latest_build_output(
        &build_root,
        "math_explorer_gui",
        "traceability_manifest.json",
    ) {
        let gui_rerun = parse_rerun_directives(&gui_output);
        for expected in [
            "build.rs",
            "build_snapshot.rs",
            "src/tabs/mod.rs",
            "../traceability.toml",
            "../papers",
            "../crates/federated_registry/src",
            "../crates/oxidize_core/src",
            "../crates/math_commons/src",
            "../crates/scientific_metadata/src",
            "../crates/diagnostics/src",
        ] {
            assert!(
                gui_rerun.contains(expected),
                "math_explorer_gui output missing cargo:rerun-if-changed={expected}"
            );
        }
        for excluded in [
            "Cargo.toml",
            "../crates/unified_verification/src",
            "../crates/markdown_tests/src",
            "../crates/verified_engine/src",
            "../crates/verified_engine_macros/src",
            "../apps/xtask/src",
        ] {
            assert!(
                !gui_rerun.contains(excluded),
                "math_explorer_gui output must not watch unrelated crate {excluded}"
            );
        }
    }
}
