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

    let uses_steps: Vec<&Value> = steps.iter().filter(|s| s.get("uses").is_some()).collect();
    assert_eq!(uses_steps.len(), 2, "Expected two actions/cache steps");
    for step in uses_steps {
        assert_eq!(
            step.get("uses").and_then(Value::as_str),
            Some("actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830")
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

    let ox_output = find_latest_build_output(&build_root, "oxidize_core", "vfs_data.rs")
        .expect("oxidize_core build output must exist because unified_verification depends on it");
    let ox_rerun = parse_rerun_directives(&ox_output);
    for expected in ["build.rs", "src/path_utils.rs"] {
        assert!(
            ox_rerun.contains(expected),
            "oxidize_core output missing cargo:rerun-if-changed={expected}"
        );
    }
    for excluded in [
        "../../papers",
        "../../math_explorer/src",
        "../../math_explorer_gui/src/tabs",
    ] {
        assert!(
            !ox_rerun.contains(excluded),
            "native oxidize_core output must not watch {excluded}"
        );
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
            "build_snapshot_scan.rs",
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
