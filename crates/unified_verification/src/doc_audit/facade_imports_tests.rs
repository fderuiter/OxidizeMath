use super::*;

#[test]
fn markdown_facade_import_check_allows_hidden_shims_and_rejects_visible_internal_imports() {
    let clean = "```toml\ndomain_applied = \"0.4\"\n```\n\n```rust\n# use domain_applied::__doctest_facade as math_explorer;\nuse math_explorer::applied::battery_degradation::PowerLawModel;\nuse math_explorer::rng::OxidizeRng;\nlet _rng = OxidizeRng::new(42);\n```\n";
    assert!(check_markdown_facade_imports(Path::new("guide.md"), clean).is_empty());

    let dirty = "```rust\nuse domain_applied::applied::battery_degradation::PowerLawModel;\n```\n";
    let violations = check_markdown_facade_imports(Path::new("guide.md"), dirty);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("domain_applied"));

    let rustdoc_dirty = "//! ```rust\n//! use pure_math::pure_math::number_theory;\n//! ```\n";
    assert_eq!(
        check_rustdoc_facade_imports(Path::new("lib.rs"), rustdoc_dirty).len(),
        1
    );
    assert_eq!(
        check_example_facade_imports(Path::new("demo.rs"), "use nalgebra::DMatrix;\n").len(),
        1
    );
}

#[test]
fn rejects_std_rng_and_default_rng_in_guides_rustdoc_and_examples() {
    let md_std_rng = "```rust\nuse math_explorer::rand::rngs::StdRng;\n```\n";
    let v_md = check_markdown_facade_imports(Path::new("guide.md"), md_std_rng);
    assert_eq!(v_md.len(), 1);
    assert!(v_md[0].contains("StdRng"));

    let md_default_rng = "```rust\nlet mut rng = OxidizeRng::default();\n```\n";
    let v_md_def = check_markdown_facade_imports(Path::new("guide.md"), md_default_rng);
    assert_eq!(v_md_def.len(), 1);
    assert!(v_md_def[0].contains("OxidizeRng::default"));

    // Hidden `# ` lines in rustdoc must also be checked for StdRng and OxidizeRng::default
    let rustdoc_hidden = concat!(
        "/// ```\n",
        "/// # use math_explorer::rand::rngs::StdRng;\n",
        "/// # let mut rng = OxidizeRng::default();\n",
        "/// ```\n",
    );
    let v_doc = check_rustdoc_facade_imports(Path::new("solver.rs"), rustdoc_hidden);
    assert_eq!(v_doc.len(), 2, "Expected 2 violations, got: {v_doc:?}");
    assert!(v_doc[0].contains("StdRng"));
    assert!(v_doc[1].contains("OxidizeRng::default"));

    let ex_bad = concat!(
        "use math_explorer::rand::rngs::StdRng;\n",
        "fn main() { let _ = math_explorer::rng::OxidizeRng::default(); }\n",
    );
    let v_ex = check_example_facade_imports(Path::new("demo.rs"), ex_bad);
    assert_eq!(v_ex.len(), 2, "Expected 2 violations, got: {v_ex:?}");
}

#[test]
fn rustdoc_and_adr_exclusion_work_end_to_end() {
    let dir = crate::utils::test_utils::create_temp_test_dir("facade_import_hygiene");
    let adr_dir = dir.join("docs/adr");
    let arch_dir = dir.join("docs/architecture");
    let dom_src = dir.join("crates/domain_ai/src");
    let ex_dir = dir.join("math_explorer/examples");
    fs::create_dir_all(&adr_dir).unwrap();
    fs::create_dir_all(&arch_dir).unwrap();
    fs::create_dir_all(&dom_src).unwrap();
    fs::create_dir_all(&ex_dir).unwrap();

    // Historical ADR with legacy domain_ai import must be ignored
    fs::write(
        adr_dir.join("0001-legacy.md"),
        "```rust\nuse domain_ai::ai::transformer::Encoder;\n```\n",
    )
    .unwrap();
    // Active guide with direct domain_climate import must be flagged
    fs::write(
        arch_dir.join("guide.md"),
        "```rust\nuse domain_climate::climate::cera::Cera;\n```\n",
    )
    .unwrap();
    // Rustdoc with visible pure_math import must be flagged
    fs::write(
        dom_src.join("lib.rs"),
        "//! ```rust\n//! use pure_math::pure_math::number_theory;\n//! ```\n",
    )
    .unwrap();
    // Example bypassing math_explorer::nalgebra must be flagged
    fs::write(
        ex_dir.join("demo.rs"),
        "use nalgebra::DMatrix;\nfn main() {}\n",
    )
    .unwrap();

    let violations = audit_facade_import_hygiene(&dir);
    assert_eq!(violations.len(), 3, "Violations: {violations:?}");
    assert!(!violations.iter().any(|v| v.contains("0001-legacy.md")));
    let _ = fs::remove_dir_all(&dir);
}
