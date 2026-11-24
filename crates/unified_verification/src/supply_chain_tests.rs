use super::*;
use crate::utils::test_utils::TempTestDir;
use crate::whitelist::IgnoredAdvisoryEntry;

fn sample_supply_chain_config() -> SupplyChainConfig {
    SupplyChainConfig {
        max_duplicate_families: Some(1),
        allowed_registries: vec![
            "registry+https://github.com/rust-lang/crates.io-index".to_string(),
        ],
        allowed_licenses: vec![
            "MIT".to_string(),
            "Apache-2.0".to_string(),
            "Apache-2.0 WITH LLVM-exception".to_string(),
            "BSD-3-Clause".to_string(),
            "OFL-1.1".to_string(),
        ],
        ignored_advisories: vec![IgnoredAdvisoryEntry {
            id: "RUSTSEC-2024-0436".to_string(),
            package: "paste".to_string(),
            reason: "Transitive proc-macro notice".to_string(),
        }],
    }
}

#[test]
fn spdx_evaluator_handles_conjunctive_disjunctive_slash_and_with_clauses() {
    let cfg = sample_supply_chain_config();
    assert_eq!(
        evaluate_spdx_expression(
            "MIT OR Apache-2.0 OR LGPL-2.1-or-later",
            &cfg.allowed_licenses
        ),
        Ok(true)
    );
    assert_eq!(
        evaluate_spdx_expression(
            "(MIT OR Apache-2.0) AND BSD-3-Clause AND OFL-1.1",
            &cfg.allowed_licenses
        ),
        Ok(true)
    );
    assert_eq!(
        evaluate_spdx_expression(
            "Apache-2.0 WITH LLVM-exception OR MIT",
            &cfg.allowed_licenses
        ),
        Ok(true)
    );
    assert_eq!(
        evaluate_spdx_expression("MIT/Apache-2.0", &cfg.allowed_licenses),
        Ok(true)
    );
    assert_eq!(
        evaluate_spdx_expression("GPL-3.0-only", &cfg.allowed_licenses),
        Ok(false)
    );
    assert_eq!(
        evaluate_spdx_expression(
            "(MIT OR Apache-2.0) AND GPL-3.0-only",
            &cfg.allowed_licenses
        ),
        Ok(false)
    );
    assert!(evaluate_spdx_expression("(MIT OR", &cfg.allowed_licenses).is_err());
}

#[test]
fn cargo_lock_audit_enforces_allowed_registries_and_duplicate_family_cap() {
    let cfg = sample_supply_chain_config();
    let ok_lock = r#"
[[package]]
name = "first_party"
version = "0.4.0"

[[package]]
name = "syn"
version = "1.0.109"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "syn"
version = "2.0.90"
source = "registry+https://github.com/rust-lang/crates.io-index"
"#;
    assert!(audit_cargo_lock(ok_lock, &cfg).is_empty());

    let bad_lock = r#"
[[package]]
name = "git_dep"
version = "0.1.0"
source = "git+https://github.com/example/rogue.git#deadbeef"

[[package]]
name = "syn"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "syn"
version = "2.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "hashbrown"
version = "0.14.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "hashbrown"
version = "0.15.0"
source = "registry+https://github.com/rust-lang/crates.io-index"
"#;
    let errs = audit_cargo_lock(bad_lock, &cfg);
    assert_eq!(errs.len(), 2);
    assert!(
        errs.iter()
            .any(|e| e.contains("Unauthorized registry/git source"))
    );
    assert!(errs.iter().any(|e| e.contains("exceed ceiling")));
}

#[test]
fn metadata_license_and_tier5_suite_audits_flag_missing_allocators_and_copyleft() {
    let cfg = sample_supply_chain_config();
    let meta = serde_json::json!({
        "workspace_members": ["path+file:///ws/crates/demo#0.1.0"],
        "packages": [
            {
                "id": "path+file:///ws/crates/demo#0.1.0",
                "name": "demo",
                "version": "0.1.0",
                "license": null
            },
            {
                "id": "registry+https://github.com/rust-lang/crates.io-index#good@1.0.0",
                "name": "good",
                "version": "1.0.0",
                "license": "MIT OR Apache-2.0"
            },
            {
                "id": "registry+https://github.com/rust-lang/crates.io-index#copyleft@1.0.0",
                "name": "copyleft",
                "version": "1.0.0",
                "license": "AGPL-3.0-only"
            }
        ]
    });
    let lic_errs = audit_metadata_licenses(&meta, &cfg);
    assert_eq!(lic_errs.len(), 1);
    assert!(lic_errs[0].contains("copyleft"));

    let dir = TempTestDir::new("tier5_audit");
    fs::create_dir_all(dir.join("crates/demo_math/src")).unwrap();
    fs::create_dir_all(dir.join("crates/demo_math/tests")).unwrap();
    fs::write(dir.join("crates/demo_math/src/lib.rs"), "pub fn f() {}\n").unwrap();
    let missing_errs = audit_tier5_security_suites(&dir, &["crates/demo_math"]);
    assert_eq!(missing_errs.len(), 2);

    fs::write(
        dir.join("crates/demo_math/src/lib.rs"),
        "#[cfg(test)]\n#[global_allocator]\nstatic A: verified_engine::allocator::VerifiedAllocator = verified_engine::allocator::VerifiedAllocator;\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/demo_math/tests/security_demo.rs"),
        "#[global_allocator]\nstatic A: verified_engine::allocator::VerifiedAllocator = verified_engine::allocator::VerifiedAllocator;\n#[test]\nfn ok() {}\n",
    )
    .unwrap();
    assert!(audit_tier5_security_suites(&dir, &["crates/demo_math"]).is_empty());
}

#[test]
fn workspace_offline_check_security_passes() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    assert!(run_check_security(&["--offline".to_string()]));
}
