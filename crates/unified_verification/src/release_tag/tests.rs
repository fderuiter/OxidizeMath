use super::*;
use crate::utils::test_utils::TempTestDir;

#[test]
fn test_stable_and_named_prerelease_wix_mappings() {
    let stable = evaluate_release_tag("0.5.0", Some("v0.5.0")).unwrap();
    assert!(!stable.is_prerelease);
    assert_eq!(stable.wix_version, "0.5.0");
    assert_eq!(stable.wix_product_version, "0.5.0.0");
    assert_eq!(convert_version(&stable.wix_version).unwrap(), "0.5.0.0");

    let alpha = evaluate_release_tag("0.5.0-alpha.1", Some("v0.5.0-alpha.1")).unwrap();
    assert!(alpha.is_prerelease);
    assert_eq!(alpha.wix_version, "0.5.0-1001");
    assert_eq!(convert_version(&alpha.wix_version).unwrap(), "0.5.0.1001");

    let beta = evaluate_release_tag("0.5.0-beta.3", Some("v0.5.0-beta.3")).unwrap();
    assert!(beta.is_prerelease);
    assert_eq!(beta.wix_version, "0.5.0-10003");
    assert_eq!(convert_version(&beta.wix_version).unwrap(), "0.5.0.10003");

    let rc = evaluate_release_tag("0.5.0-rc.1", Some("v0.5.0-rc.1")).unwrap();
    assert!(rc.is_prerelease);
    assert_eq!(rc.wix_version, "0.5.0-20001");
    assert_eq!(rc.wix_product_version, "0.5.0.20001");
    assert_eq!(convert_version(&rc.wix_version).unwrap(), "0.5.0.20001");

    let rc_max = evaluate_release_tag("255.255.65535-rc.45535", Some("v255.255.65535-rc.45535"))
        .expect("max WiX boundary should succeed");
    assert_eq!(rc_max.wix_version, "255.255.65535-65535");
    assert_eq!(
        convert_version(&rc_max.wix_version).unwrap(),
        "255.255.65535.65535"
    );
}

#[test]
fn test_rejects_tag_parity_mismatch_build_metadata_and_wix_overflows() {
    let mismatch = evaluate_release_tag("0.4.0", Some("v0.5.0")).unwrap_err();
    assert!(mismatch.contains("parity mismatch"));

    let missing_v = evaluate_release_tag("0.4.0", Some("0.4.0")).unwrap_err();
    assert!(missing_v.contains("start with 'v'"));

    for bad_version in [
        "",
        " 0.4.0",
        "0.4",
        "0.4.0.1",
        "01.4.0",
        "0.04.0",
        "0.4.00",
        "256.0.0",
        "0.256.0",
        "0.4.65536",
        "0.4.0+build.1",
        "0.4.0+123",
        "0.5.0-rc.1+20260101",
        "0.5.0-dev",
        "0.5.0-rc",
        "0.5.0-rc.01",
        "0.5.0-rc.1.2",
        "0.5.0-rc.45536",
        "0.5.0-alpha.64536",
        "0.5.0-beta.55536",
        "0.5.0-65536",
    ] {
        assert!(
            parse_release_version(bad_version).is_err(),
            "expected version '{bad_version}' to fail validation"
        );
    }

    let tag_with_build = evaluate_release_tag("0.5.0", Some("v0.5.0+sha.abcdef")).unwrap_err();
    assert!(tag_with_build.contains("+build"));
    assert!(convert_version("0.5.0-rc.1").is_err());
}

#[test]
fn test_resolve_effective_tag_handles_branches_prs_tags_and_require_tag() {
    assert_eq!(
        resolve_effective_tag(None, Some("main"), Some("refs/heads/main"), false).unwrap(),
        None
    );
    assert_eq!(
        resolve_effective_tag(None, Some("42/merge"), Some("refs/pull/42/merge"), false).unwrap(),
        None
    );
    assert!(resolve_effective_tag(None, Some("main"), Some("refs/heads/main"), true).is_err());

    assert_eq!(
        resolve_effective_tag(None, Some("v0.5.0"), Some("refs/tags/v0.5.0"), true).unwrap(),
        Some("v0.5.0".to_string())
    );
    assert!(resolve_effective_tag(None, Some("v0.4.0"), Some("refs/tags/v0.5.0"), true).is_err());

    // Explicit RELEASE_TAG override without GIT_REF or overriding branch ref
    assert_eq!(
        resolve_effective_tag(None, Some("v0.5.0"), None, false).unwrap(),
        Some("v0.5.0".to_string())
    );
    assert_eq!(
        resolve_effective_tag(None, Some("v0.5.0"), Some("refs/heads/main"), false).unwrap(),
        Some("v0.5.0".to_string())
    );
    assert!(resolve_effective_tag(None, None, None, true).is_err());
}

#[test]
fn test_transient_wix_packager_override_preserves_workspace_version() {
    let root = crate::manifest_cache::workspace_root_dir();
    let gui_toml = fs::read_to_string(root.join("math_explorer_gui/Cargo.toml"))
        .expect("math_explorer_gui/Cargo.toml must be readable");
    let overridden =
        inject_transient_wix_packager_version(&gui_toml, "0.5.0-20001").expect("override valid");
    let parsed: toml::Value = toml::from_str(&overridden).unwrap();
    assert_eq!(
        parsed["package"]["version"]["workspace"].as_bool(),
        Some(true)
    );
    assert_eq!(
        parsed["package"]["metadata"]["packager"]["version"].as_str(),
        Some("0.5.0-20001")
    );
    assert_eq!(convert_version("0.5.0-20001").unwrap(), "0.5.0.20001");

    let temp = TempTestDir::new("release_tag_gh_out");
    let out_file = temp.join("github_output.txt");
    let outcome = evaluate_release_tag("0.5.0-rc.1", Some("v0.5.0-rc.1")).unwrap();
    write_github_outputs(&out_file, &outcome).unwrap();
    let written = fs::read_to_string(&out_file).unwrap();
    assert!(written.contains("is_prerelease=true\n"));
    assert!(written.contains("wix_version=0.5.0-20001\n"));
}

#[cfg(unix)]
fn extract_ci_job_step_script(job_id: &str, step_name: &str) -> String {
    let root = crate::manifest_cache::workspace_root_dir();
    let ci_raw = fs::read_to_string(root.join(".github/workflows/ci.yml")).unwrap();
    let yaml: serde_json::Value = serde_saphyr::from_str(&ci_raw).unwrap();
    let steps = yaml["jobs"][job_id]["steps"]
        .as_array()
        .expect("job steps array");
    steps
        .iter()
        .find(|s| s["name"].as_str() == Some(step_name))
        .and_then(|s| s["run"].as_str())
        .expect("matching CI step run script")
        .to_string()
}

#[cfg(unix)]
fn setup_mock_publish_repo() -> (TempTestDir, String) {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempTestDir::new("ci_publish_release_exec");
    let bin_dir = temp.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::create_dir_all(temp.join("release-assets")).unwrap();
    fs::write(temp.join("release-assets/math-explorer.deb"), "deb-bytes").unwrap();
    fs::write(temp.join("release-assets/SHA256SUMS"), "sum-bytes").unwrap();

    let gh_mock = bin_dir.join("gh");
    fs::write(
        &gh_mock,
        "#!/usr/bin/env bash\nset -euo pipefail\necho \"$*\" >> \"$GH_LOG\"\nif [ \"$1\" = \"release\" ] && [ \"$2\" = \"view\" ]; then\n  [ \"${MOCK_RELEASE_EXISTS:-0}\" = \"1\" ]\n  exit $?\nfi\n",
    )
    .unwrap();
    let mut perms = fs::metadata(&gh_mock).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&gh_mock, perms).unwrap();
    let path_env = format!(
        "{}:{}",
        bin_dir.display(),
        env::var("PATH").unwrap_or_default()
    );
    (temp, path_env)
}

#[test]
#[cfg(unix)]
fn test_ci_publish_release_step_draft_prerelease_and_rerun_clobber() {
    use std::process::Command;

    let script = extract_ci_job_step_script("publish-release", "Publish GitHub Release");
    let (temp, path_env) = setup_mock_publish_repo();
    let run_publish = |tag: &str, is_pre: &str, exists: &str| -> (bool, String) {
        let log_path = temp.join("gh.log");
        let _ = fs::remove_file(&log_path);
        let status = Command::new("bash")
            .arg("-c")
            .arg(&script)
            .current_dir(&*temp)
            .env("PATH", &path_env)
            .env("GH_LOG", &log_path)
            .env("RELEASE_TAG", tag)
            .env("IS_PRERELEASE", is_pre)
            .env("MOCK_RELEASE_EXISTS", exists)
            .status()
            .unwrap();
        let log = fs::read_to_string(&log_path).unwrap_or_default();
        (status.success(), log)
    };

    let (ok_stable, log_stable) = run_publish("v0.5.0", "false", "0");
    assert!(ok_stable);
    assert!(
        log_stable.contains(
            "release create --draft --verify-tag --title v0.5.0 --generate-notes -- v0.5.0"
        )
    );
    assert!(!log_stable.contains(
        "release create --draft --verify-tag --title v0.5.0 --generate-notes --prerelease"
    ));
    assert!(
        log_stable.contains("release edit --draft=false --verify-tag --prerelease=false -- v0.5.0")
    );

    let (ok_rc, log_rc) = run_publish("v0.5.0-rc.1", "true", "0");
    assert!(ok_rc);
    assert!(log_rc.contains("release create --draft --verify-tag --title v0.5.0-rc.1 --generate-notes --prerelease -- v0.5.0-rc.1"));
    assert!(log_rc.contains("release edit --draft=false --verify-tag --prerelease -- v0.5.0-rc.1"));

    let (ok_rerun, log_rerun) = run_publish("v0.5.0-rc.1", "true", "1");
    assert!(ok_rerun);
    assert!(!log_rerun.contains("release create"));
    assert!(log_rerun.contains("release upload --clobber -- v0.5.0-rc.1"));
    assert!(
        log_rerun.contains("release edit --draft=false --verify-tag --prerelease -- v0.5.0-rc.1")
    );

    let (ok_build_meta, _) = run_publish("v0.5.0+20260101", "false", "0");
    assert!(!ok_build_meta);
}

#[cfg(unix)]
fn setup_mock_package_repo() -> (TempTestDir, String, String) {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    let root = crate::manifest_cache::workspace_root_dir();
    let original_gui = fs::read_to_string(root.join("math_explorer_gui/Cargo.toml")).unwrap();
    let temp = TempTestDir::new("ci_package_wix_exec");
    fs::create_dir_all(temp.join("math_explorer_gui")).unwrap();
    fs::write(temp.join("math_explorer_gui/Cargo.toml"), &original_gui).unwrap();
    for args in [
        &["init", "-q"][..],
        &["add", "math_explorer_gui/Cargo.toml"][..],
    ] {
        assert!(
            Command::new("git")
                .args(args)
                .current_dir(&*temp)
                .status()
                .unwrap()
                .success()
        );
    }

    let bin_dir = temp.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let cargo_mock = bin_dir.join("cargo");
    fs::write(
        &cargo_mock,
        "#!/usr/bin/env bash\nset -euo pipefail\ncp math_explorer_gui/Cargo.toml \"$CAPTURED_MANIFEST\"\n",
    )
    .unwrap();
    let mut perms = fs::metadata(&cargo_mock).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&cargo_mock, perms).unwrap();
    let path_env = format!(
        "{}:{}",
        bin_dir.display(),
        env::var("PATH").unwrap_or_default()
    );
    (temp, original_gui, path_env)
}

#[test]
#[cfg(unix)]
fn test_ci_package_step_transient_wix_override_and_restore() {
    use std::process::Command;

    let script = extract_ci_job_step_script("package", "Build and Package");
    let (temp, original_gui, path_env) = setup_mock_package_repo();

    let captured = temp.join("captured_wix.toml");
    let status = Command::new("bash")
        .arg("-c")
        .arg(&script)
        .current_dir(&*temp)
        .env("PATH", &path_env)
        .env("CAPTURED_MANIFEST", &captured)
        .env("MATRIX_TARGET", "wix")
        .env("IS_PRERELEASE", "true")
        .env("WIX_VERSION", "0.5.0-20001")
        .status()
        .unwrap();
    assert!(status.success());
    let during_wix: toml::Value = toml::from_str(&fs::read_to_string(&captured).unwrap()).unwrap();
    assert_eq!(
        during_wix["package"]["version"]["workspace"].as_bool(),
        Some(true)
    );
    assert_eq!(
        during_wix["package"]["metadata"]["packager"]["version"].as_str(),
        Some("0.5.0-20001")
    );
    assert_eq!(
        fs::read_to_string(temp.join("math_explorer_gui/Cargo.toml")).unwrap(),
        original_gui
    );

    let captured_deb = temp.join("captured_deb.toml");
    let deb_status = Command::new("bash")
        .arg("-c")
        .arg(&script)
        .current_dir(&*temp)
        .env("PATH", &path_env)
        .env("CAPTURED_MANIFEST", &captured_deb)
        .env("MATRIX_TARGET", "deb")
        .env("IS_PRERELEASE", "true")
        .env("WIX_VERSION", "0.5.0-20001")
        .status()
        .unwrap();
    assert!(deb_status.success());
    assert_eq!(fs::read_to_string(&captured_deb).unwrap(), original_gui);
}

#[test]
fn test_cli_runner_parity_mismatch_and_rc_output_emission() {
    let root = crate::manifest_cache::workspace_root_dir();

    // Mismatched tag v0.5.0 against workspace 0.4.0 fails in pre-flight & publish-release modes
    for args in [Vec::<String>::new(), vec!["--require-tag".to_string()]] {
        let err =
            run_check_release_tag_in_dir(&root, &args, Some("v0.5.0"), Some("refs/tags/v0.5.0"))
                .unwrap_err();
        assert!(err.contains("parity mismatch"));
    }

    // Matching pre-release v0.5.0-rc.1 against 0.5.0-rc.1 emits is_prerelease=true and wix_version=0.5.0-20001
    let temp = TempTestDir::new("cli_rc_parity");
    fs::write(
        temp.join("Cargo.toml"),
        "[workspace]\nmembers = []\n[workspace.package]\nversion = \"0.5.0-rc.1\"\n",
    )
    .unwrap();
    let gh_out = temp.join("gh_out.txt");
    let args = vec![
        "--require-tag".to_string(),
        "--github-output".to_string(),
        gh_out.to_string_lossy().into_owned(),
    ];
    let stdout = run_check_release_tag_in_dir(
        &temp,
        &args,
        Some("v0.5.0-rc.1"),
        Some("refs/tags/v0.5.0-rc.1"),
    )
    .unwrap();
    assert!(stdout.contains("is_prerelease=true\n"));
    assert!(stdout.contains("wix_version=0.5.0-20001\n"));
    assert_eq!(convert_version("0.5.0-20001").unwrap(), "0.5.0.20001");
    let gh_written = fs::read_to_string(&gh_out).unwrap();
    assert_eq!(gh_written, "is_prerelease=true\nwix_version=0.5.0-20001\n");
}
