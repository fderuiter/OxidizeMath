use super::*;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashSet;

#[test]
fn test_cargo_cache_action_pins_v5_node24_and_hashes_cargo_toml() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let root = crate::manifest_cache::workspace_root_dir();
    let content = fs::read_to_string(root.join(".github/actions/cargo-cache/action.yml"))
        .expect("cargo-cache action.yml should be readable");
    for sha in [
        "6f8efc29b200d32929f49075959781ed54ec270c",
        "0057852bfaa89a56745cba8c7296529d2fc39830",
    ] {
        assert!(!content.contains(sha), "Deprecated cache SHA {sha}");
    }
    let yaml: Value =
        serde_saphyr::from_str(&content).expect("cargo-cache action.yml must be valid YAML");
    let steps = yaml["runs"]["steps"]
        .as_array()
        .expect("composite action must define runs.steps");
    assert_cargo_cache_inputs_and_env(&yaml, steps);
    assert_cargo_cache_uses_steps(steps);
}

fn assert_cargo_cache_inputs_and_env(yaml: &Value, steps: &[Value]) {
    let inputs = yaml
        .get("inputs")
        .expect("cargo-cache action.yml must define inputs");
    assert!(
        inputs.get("restore-only").is_some(),
        "cargo-cache action.yml must define a restore-only input"
    );
    let default_lines: Vec<&str> = inputs["manifest-files"]["default"]
        .as_str()
        .expect("manifest-files input must define a default")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(default_lines, vec!["Cargo.toml", "**/Cargo.toml"]);

    let env_map = steps[0]
        .get("env")
        .expect("cache-meta step must define env");
    let lock_hash_expr = env_map["LOCK_AND_CONFIG_HASH"]
        .as_str()
        .expect("cache-meta step must bind LOCK_AND_CONFIG_HASH in env");
    for pattern in ["'Cargo.lock'", "'**/Cargo.lock'", "'.cargo/config.toml'"] {
        assert!(
            lock_hash_expr.contains(pattern),
            "LOCK_AND_CONFIG_HASH must hash {pattern}, got: {lock_hash_expr}"
        );
    }
    assert!(
        !lock_hash_expr.contains("Trunk.toml"),
        "LOCK_AND_CONFIG_HASH must not hash Trunk.toml globally: {lock_hash_expr}"
    );

    let manifest_expr = env_map["MANIFEST_FILES_HASH"]
        .as_str()
        .expect("cache-meta step must bind MANIFEST_FILES_HASH in env");
    assert!(
        manifest_expr.contains("hashFiles(inputs.manifest-files)"),
        "MANIFEST_FILES_HASH must hash inputs.manifest-files, got: {manifest_expr}"
    );

    let gui_dir = oxidize_core::workspace::find_package_member_path_std(
        &crate::manifest_cache::workspace_root_dir(),
        "math_explorer_gui",
    )
    .unwrap_or_else(|| "apps/math_explorer_gui".to_string());
    let trunk_hash = format!("hashFiles('{gui_dir}/Trunk.toml')");
    let trunk_expr = env_map["TRUNK_CONFIG_HASH"]
        .as_str()
        .expect("cache-meta step must bind TRUNK_CONFIG_HASH in env");
    assert!(
        trunk_expr.contains("inputs.target == 'wasm32-unknown-unknown'")
            && trunk_expr.contains(&trunk_hash),
        "TRUNK_CONFIG_HASH must hash {gui_dir}/Trunk.toml only for wasm32-unknown-unknown, got: {trunk_expr}"
    );
}

fn assert_target_restore_keys_fallback(step: &Value) {
    let restore_keys = step["with"]["restore-keys"]
        .as_str()
        .expect("target cache step must define restore-keys");
    let prefixes: Vec<&str> = restore_keys
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        prefixes.len(),
        2,
        "target cache restore-keys must include compiler-hash and compiler-agnostic fallback prefixes, got: {prefixes:?}"
    );
    assert!(
        prefixes[0].ends_with("-${{ steps.cache-meta.outputs.compiler-hash }}-"),
        "primary restore-key must end with compiler-hash prefix, got: {}",
        prefixes[0]
    );
    assert!(
        prefixes[1].ends_with("-${{ steps.cache-meta.outputs.target }}-")
            && !prefixes[1].contains("compiler-hash"),
        "fallback restore-key must omit compiler-hash and end with target prefix, got: {}",
        prefixes[1]
    );
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
                "actions/cache@caa296126883cff596d87d8935842f9db880ef25"
                    | "actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25"
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
            assert_target_restore_keys_fallback(step);
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
        Some("actions/cache@caa296126883cff596d87d8935842f9db880ef25")
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

fn assert_rust_toolchain_pin_parity(root: &Path, ci_yaml: &Value) {
    let toolchain_raw = fs::read_to_string(root.join("rust-toolchain.toml"))
        .expect("rust-toolchain.toml should be readable");
    let toolchain_toml: toml::Value =
        toml::from_str(&toolchain_raw).expect("rust-toolchain.toml must be valid TOML");
    let channel = toolchain_toml["toolchain"]["channel"]
        .as_str()
        .expect("rust-toolchain.toml must define toolchain.channel");
    let parts: Vec<&str> = channel.split('.').collect();
    assert!(
        parts.len() == 3
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())),
        "rust-toolchain.toml channel must be an explicit MAJOR.MINOR.PATCH version, got: {channel}"
    );

    let jobs = ci_yaml["jobs"]
        .as_object()
        .expect("ci.yml must define jobs");
    let mut toolchain_step_count = 0usize;
    for (job_name, job_val) in jobs {
        let Some(steps) = job_val.get("steps").and_then(Value::as_array) else {
            continue;
        };
        for step in steps {
            let Some(uses) = step.get("uses").and_then(Value::as_str) else {
                continue;
            };
            if uses.starts_with("dtolnay/rust-toolchain@") {
                toolchain_step_count += 1;
                assert_eq!(
                    uses, "dtolnay/rust-toolchain@b725eda525362e7cf34af9a7243cac1e3f01a7ec",
                    "dtolnay/rust-toolchain must retain its 40-char SHA pin in job '{job_name}'"
                );
                assert_eq!(
                    step["with"]["toolchain"].as_str(),
                    Some(channel),
                    "job '{job_name}' dtolnay/rust-toolchain must pin toolchain '{channel}'"
                );
            }
        }
    }
    assert_eq!(
        toolchain_step_count, 6,
        "Expected six dtolnay/rust-toolchain steps across ci.yml Rust jobs"
    );
}

fn find_job_cargo_cache_step<'a>(ci_yaml: &'a Value, job: &str) -> &'a Value {
    ci_yaml["jobs"][job]["steps"]
        .as_array()
        .unwrap_or_else(|| panic!("{job} must define steps"))
        .iter()
        .find(|s| s.get("uses").and_then(Value::as_str) == Some("./.github/actions/cargo-cache"))
        .unwrap_or_else(|| panic!("{job} must use ./.github/actions/cargo-cache"))
}

fn assert_ci_cache_scoping_and_seeding(ci_yaml: &Value) {
    for job in ["pre-flight", "feature-matrix", "package", "publish-release"] {
        let cache_step = find_job_cargo_cache_step(ci_yaml, job);
        assert!(
            cache_step["with"].get("extra-files").is_none(),
            "{job} cargo-cache step must not set redundant extra-files"
        );
    }

    let papers_cache = find_job_cargo_cache_step(ci_yaml, "compile-papers");
    assert!(
        papers_cache["with"].get("extra-files").is_none(),
        "compile-papers cargo-cache step must not set extra-files"
    );
    let papers_manifests: Vec<&str> = papers_cache["with"]["manifest-files"]
        .as_str()
        .expect("compile-papers must narrow manifest-files")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(
        papers_manifests,
        vec!["Cargo.toml", "apps/xtask/Cargo.toml"]
    );

    let fm_cache = find_job_cargo_cache_step(ci_yaml, "feature-matrix");
    assert_eq!(fm_cache["with"]["scope"].as_str(), Some("features-shared"));
    assert_eq!(
        fm_cache["with"]["restore-only"].as_str(),
        Some("${{ matrix.feature != 'climate' }}")
    );

    let fm_steps = ci_yaml["jobs"]["feature-matrix"]["steps"]
        .as_array()
        .expect("feature-matrix must define steps");
    let has_all_features_seed = fm_steps.iter().any(|s| {
        s.get("if").and_then(Value::as_str) == Some("matrix.feature == 'climate'")
            && s.get("run")
                .and_then(Value::as_str)
                .is_some_and(|r| r.contains("cargo test -p math_explorer --no-run --all-features"))
    });
    assert!(
        has_all_features_seed,
        "climate writer leg in feature-matrix must run `cargo test -p math_explorer --no-run --all-features`"
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

    assert_rust_toolchain_pin_parity(&root, &yaml);
    assert_ci_cache_scoping_and_seeding(&yaml);

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
            !p.trim().is_empty() && !p.contains('\\'),
            "Invalid rerun path: {p}"
        );
    }
    paths
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_gui_build_rerun_directives(
    ws_root: &Path,
    gui_rerun: &std::collections::HashSet<String>,
) {
    let member_dir = |pkg: &str| {
        oxidize_core::workspace::find_package_member_path_std(ws_root, pkg)
            .unwrap_or_else(|| pkg.to_string())
    };
    let gp = std::iter::repeat_n(
        "..",
        Path::new(&member_dir("math_explorer_gui"))
            .components()
            .count(),
    )
    .collect::<Vec<_>>()
    .join("/");
    for exp in [
        "build.rs".to_string(),
        "build_snapshot.rs".to_string(),
        "src/tabs/mod.rs".to_string(),
        format!("{gp}/traceability.toml"),
        format!("{gp}/papers"),
    ] {
        assert!(gui_rerun.contains(&exp), "missing rerun {exp}");
    }
    for pkg in [
        "federated_registry",
        "oxidize_core",
        "math_commons",
        "scientific_metadata",
        "diagnostics",
    ] {
        let exp = format!("{gp}/{}/src", member_dir(pkg));
        assert!(gui_rerun.contains(&exp), "missing rerun {exp}");
    }
    for pkg in [
        "unified_verification",
        "markdown_tests",
        "verified_engine",
        "verified_engine_macros",
        "xtask",
    ] {
        let excluded = format!("{gp}/{}/src", member_dir(pkg));
        assert!(
            !gui_rerun.contains(&excluded),
            "watched unrelated {excluded}"
        );
    }
    assert!(!gui_rerun.contains("Cargo.toml"));
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_workspace_build_script_outputs_emit_rerun_directives() {
    let build_root = locate_build_root(&std::env::current_exe().expect("current_exe"));
    let ws_root = crate::manifest_cache::workspace_root_dir();

    if let Some(md_out) =
        find_latest_build_output(&build_root, "markdown_tests", "generated_tests.rs")
    {
        let md_rerun = parse_rerun_directives(&md_out);
        let md_dir =
            oxidize_core::workspace::find_package_member_path_std(&ws_root, "markdown_tests")
                .unwrap_or_else(|| "markdown_tests".to_string());
        let md_prefix = std::iter::repeat_n("..", Path::new(&md_dir).components().count())
            .collect::<Vec<_>>()
            .join("/");
        for exp in ["build.rs".to_string(), format!("{md_prefix}/Cargo.toml")] {
            assert!(
                md_rerun.contains(&exp),
                "missing markdown_tests rerun {exp}"
            );
        }
    }

    if let Some(gui_out) = find_latest_build_output(
        &build_root,
        "math_explorer_gui",
        "traceability_manifest.json",
    ) {
        let gui_rerun = parse_rerun_directives(&gui_out);
        assert_gui_build_rerun_directives(&ws_root, &gui_rerun);
    }
}
