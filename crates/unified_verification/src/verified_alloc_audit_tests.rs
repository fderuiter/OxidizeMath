use super::*;
use crate::utils::test_utils::TempTestDir;
use std::fs;

#[test]
fn test_rejects_missing_global_allocator_on_run_verified_and_lock_allocations() {
    let dir = TempTestDir::new("verified_alloc_missing_hook");
    let crate_dir = dir.join("crates/demo");
    fs::create_dir_all(crate_dir.join("src")).unwrap();
    fs::create_dir_all(crate_dir.join("tests")).unwrap();
    fs::write(
        crate_dir.join("src/lib.rs"),
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() { verified_engine::engine::VerifiedEngine::run_verified(|| 1); }\n}\n",
    )
    .unwrap();
    fs::write(
        crate_dir.join("tests/integ.rs"),
        "#[test]\nfn it() { verified_engine::allocator::lock_allocations(); }\n",
    )
    .unwrap();

    let mut cache = AstCache::new();
    let member = crate_dir.to_string_lossy().into_owned();
    let violations = collect_verified_alloc_violations(&[member.as_str()], &mut cache);
    assert_eq!(
        violations.len(),
        2,
        "Expected 2 missing-hook violations: {violations:?}"
    );
}

#[test]
fn test_flags_unannotated_heap_allocs_and_accepts_alloc_ok() {
    let dir = TempTestDir::new("verified_alloc_role_audit");
    let crate_dir = dir.join("crates/demo");
    fs::create_dir_all(crate_dir.join("src")).unwrap();
    fs::create_dir_all(crate_dir.join("tests")).unwrap();
    fs::write(
        crate_dir.join("src/lib.rs"),
        concat!(
            "#[cfg(test)]\n#[global_allocator]\nstatic A: std::alloc::System = std::alloc::System;\n",
            "#[verified_engine::verified]\npub fn hot_step(buf: &mut [f64]) { buf[0] += 1.0; }\n",
            "#[verified_engine::verified(alloc_ok = \"constructor pre-allocates buffer\")]\npub fn make_buf(n: usize) -> Vec<f64> { vec![0.0; n] }\n",
            "#[verified_engine::verified]\npub fn leaky_hot_step(n: usize) -> Vec<f64> { Vec::with_capacity(n) }\n",
        ),
    )
    .unwrap();
    fs::write(
        crate_dir.join("tests/integ.rs"),
        "#[global_allocator]\nstatic A: std::alloc::System = std::alloc::System;\n#[test]\nfn it() { verified_engine::engine::VerifiedEngine::run_verified(|| 1); }\n",
    )
    .unwrap();

    let mut cache = AstCache::new();
    let member = crate_dir.to_string_lossy().into_owned();
    let violations = collect_verified_alloc_violations(&[member.as_str()], &mut cache);
    assert_eq!(
        violations.len(),
        1,
        "Expected only leaky_hot_step flagged: {violations:?}"
    );
    assert!(violations[0].contains("leaky_hot_step"));
}
