use super::*;
use crate::report::ReportMode;
use crate::utils::test_utils::{CurrentDirGuard, TempTestDir};
use std::fs;

fn parse(args: &[&str]) -> Result<VerifySuiteOptions, String> {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    parse_verify_suite_args(&owned)
}

fn sample_cov_json(count: u64, covered: u64) -> serde_json::Value {
    serde_json::json!({
        "data": [{
            "files": [{
                "summary": {
                    "lines": {
                        "count": count,
                        "covered": covered
                    }
                }
            }]
        }]
    })
}

fn seed_temp_verify_suite_workspace(dir: &TempTestDir) {
    fs::create_dir_all(dir.join("crates/demo/src")).unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/demo\"]\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/demo/Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/demo/src/lib.rs"),
        concat!(
            "#[verified_engine::verified]\n",
            "pub fn step(x: f64) -> f64 { assert!(x.is_finite()); x }\n",
            "theory_verification! { module = \"demo\", test = { assert_eq!(step(1.0), 1.0); } }\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("verification_whitelist.toml"),
        concat!(
            "[verify_suite]\n",
            "min_verified_assertion_density = 0.40\n",
            "min_total_production_density = 0.20\n",
            "min_verified_production_density = 0.20\n",
            "coverage_tolerance_pct = 1.5\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("README.md"),
        "# Demo\n![Coverage](https://img.shields.io/badge/coverage-10.0%25-yellow)\n",
    )
    .unwrap();
}

#[test]
fn verify_suite_defaults_to_read_only_check_and_validates_flags() {
    assert_eq!(
        parse(&[]).unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Check,
            auto_fix: false,
            coverage_only: false,
        }
    );
    assert_eq!(
        parse(&["--check", "--coverage-only"]).unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Check,
            auto_fix: false,
            coverage_only: true,
        }
    );
    assert_eq!(
        parse(&["--coverage-only"]).unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Check,
            auto_fix: false,
            coverage_only: true,
        }
    );
    assert_eq!(
        parse(&["--write", "--auto-fix", "--coverage-only"]).unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Write,
            auto_fix: true,
            coverage_only: true,
        }
    );
    assert!(parse(&["--check", "--write"]).is_err());
    assert!(parse(&["--auto-fix"]).is_err());
    assert!(parse(&["--check", "--auto-fix"]).is_err());
    assert!(parse(&["--coverage-only", "--auto-fix"]).is_err());
    assert!(parse(&["--invalid-flag"]).is_err());
    assert_eq!(parse_check_certificate_mode(&[]), Ok(ReportMode::Check));
    assert_eq!(
        parse_check_certificate_mode(&["--check".to_string()]),
        Ok(ReportMode::Check)
    );
    assert_eq!(
        parse_check_certificate_mode(&["--write".to_string()]),
        Ok(ReportMode::Write)
    );
    assert!(parse_check_certificate_mode(&["--bad".to_string()]).is_err());
}

#[test]
fn test_verify_suite_coverage_only_skips_static_gates_and_enforces_thresholds() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let temp = TempTestDir::new("verify_suite_cov_only");
    seed_temp_verify_suite_workspace(&temp);
    env::set_current_dir(&*temp).unwrap();

    let members = vec!["crates/demo".to_string()];
    let mut static_calls = 0usize;

    let write_cov = parse(&["--write", "--coverage-only"]).unwrap();
    let mut write_cache = oxidize_core::traceability::AstCache::new();
    let seeded = run_verify_suite_with(
        write_cov,
        &members,
        &mut write_cache,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        |mode| {
            assert_eq!(mode, ReportMode::Write);
            Ok(sample_cov_json(1000, 500))
        },
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(seeded && static_calls == 0);
    assert_eq!(write_cache.parse_calls(), 1);

    let cov_only = parse(&["--coverage-only"]).unwrap();
    let mut low_cache = oxidize_core::traceability::AstCache::new();
    let low = run_verify_suite_with(
        cov_only,
        &members,
        &mut low_cache,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        |_| Ok(sample_cov_json(1000, 499)),
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(!low && static_calls == 0, "49.9% coverage must fail 50.0%");

    let mut ok_cache = oxidize_core::traceability::AstCache::new();
    let ok_cov = run_verify_suite_with(
        cov_only,
        &members,
        &mut ok_cache,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        |_| Ok(sample_cov_json(1000, 500)),
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(ok_cov && static_calls == 0, "50.0% coverage must pass");
}

#[test]
fn test_verify_suite_shares_ast_cache_and_propagates_coverage_errors() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let temp = TempTestDir::new("verify_suite_shared_ast");
    seed_temp_verify_suite_workspace(&temp);
    env::set_current_dir(&*temp).unwrap();

    let members = vec!["crates/demo".to_string()];
    let mut seed_cache = oxidize_core::traceability::AstCache::new();
    assert!(
        run_verify_suite_with(
            parse(&["--write", "--coverage-only"]).unwrap(),
            &members,
            &mut seed_cache,
            |_, _, _| Vec::new(),
            |_| Ok(sample_cov_json(1000, 500)),
        )
        .unwrap()
    );

    let mut static_calls = 0usize;
    let mut shared_cache = oxidize_core::traceability::AstCache::new();
    let full = run_verify_suite_with(
        parse(&[]).unwrap(),
        &members,
        &mut shared_cache,
        |_, _, cache| {
            static_calls += 1;
            let _ = cache.get_or_load_fs(std::path::Path::new("crates/demo/src/lib.rs"));
            Vec::new()
        },
        |_| Ok(sample_cov_json(1000, 510)),
    )
    .expect("default verify-suite should evaluate cleanly");
    assert!(full && static_calls == 1);
    assert_eq!(shared_cache.parse_calls(), 1);
    assert!(
        run_verify_suite_with(
            parse(&["--coverage-only"]).unwrap(),
            &members,
            &mut shared_cache,
            |_, _, _| Vec::new(),
            |_| Err("llvm-cov failed".to_string()),
        )
        .is_err()
    );
}

#[test]
fn test_verify_suite_security_checks_parse_each_member_manifest_once() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let (_, members) = manifest_cache::discover_workspace();
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut ast_cache = oxidize_core::traceability::AstCache::new();
    let mut manifests = manifest_cache::ManifestCache::new();

    let (passed, entropy_debt) = run_security_checks_with_hooks(
        &member_refs,
        false,
        &mut ast_cache,
        &mut manifests,
        || true,
        || true,
    );
    assert!(passed);
    assert!(entropy_debt.is_empty());
    assert_eq!(manifests.lint_check_calls(), members.len());
    assert_eq!(manifests.parse_calls(), members.len() + 1);
    for member in &members {
        let manifest = std::path::Path::new(member).join("Cargo.toml");
        assert_eq!(manifests.parse_count_for(&manifest), 1);
    }
}
