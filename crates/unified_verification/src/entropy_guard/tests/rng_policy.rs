use super::*;

#[test]
fn test_rejects_stdrng_imports_types_calls_and_macros() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_rejects_stdrng");
    let file = write_fixture(
        &dir,
        "bad_stdrng.rs",
        concat!(
            "use rand::rngs::StdRng;\n",
            "pub fn make(s: u64) -> rand::rngs::StdRng {\n",
            "    let a = StdRng::seed_from_u64(s);\n",
            "    let _ = vec![rand::rngs::StdRng::seed_from_u64(1)];\n",
            "    a\n",
            "}\n",
        ),
    );
    let violations = check_entropy(&[file.to_str().unwrap().to_string()]);
    assert!(
        violations.len() >= 4,
        "Expected StdRng violations across use, type, call, and macro: {violations:?}"
    );
    assert!(has_match(&violations, "'StdRng'", "at line 1"));
    assert!(has_match(&violations, "'StdRng'", "at line 2"));
    assert!(has_match(&violations, "'StdRng'", "at line 3"));
    assert!(has_match(&violations, "'StdRng'", "at line 4"));
}

#[test]
fn test_rejects_unallowlisted_oxidize_rng_default_and_honors_default_rng_allowlist() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = create_test_dir("test_oxidize_rng_default_policy");
    let bad = write_fixture(
        &dir,
        "crates/demo/src/lib.rs",
        concat!(
            "use oxidize_core::rng::OxidizeRng;\n",
            "pub fn bad(seed: Option<u64>) {\n",
            "    let _a = OxidizeRng::default();\n",
            "    let _b = seed.map_or_else(OxidizeRng::default, OxidizeRng::new);\n",
            "    let _c = vec![oxidize_core::rng::OxidizeRng::default()];\n",
            "}\n",
        ),
    );
    let allowed = write_fixture(
        &dir,
        "crates/oxidize_core/src/rng.rs",
        concat!(
            "pub fn ok_in_rng_impl() {\n",
            "    let _r = OxidizeRng::default();\n",
            "}\n",
        ),
    );

    let mut cache = oxidize_core::traceability::AstCache::new();
    let targets = vec![
        bad.to_str().unwrap().to_string(),
        allowed.to_str().unwrap().to_string(),
    ];
    let exts = vec!["rs".to_string()];
    let default_allow = vec!["crates/oxidize_core/src/rng.rs".to_string()];
    let violations =
        scan_paths_internal_with_cache(&targets, &exts, &[], &default_allow, &mut cache);
    assert_eq!(
        violations.len(),
        3,
        "Expected 3 violations in bad file and 0 in allowlisted rng.rs: {violations:?}"
    );
    assert!(has_match(
        &violations,
        "'OxidizeRng::default()'",
        "at line 3"
    ));
    assert!(has_match(
        &violations,
        "'OxidizeRng::default()'",
        "at line 4"
    ));
    assert!(has_match(
        &violations,
        "'OxidizeRng::default()'",
        "at line 5"
    ));
    assert!(
        !violations
            .iter()
            .any(|v| v.contains("oxidize_core/src/rng.rs"))
    );
}
