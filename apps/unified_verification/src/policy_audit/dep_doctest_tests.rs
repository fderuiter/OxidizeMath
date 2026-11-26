use super::run_policy_audit_with_whitelist;
use super::tests::write_member_crate;
use crate::utils::test_utils::TempTestDir;
use std::fs;

#[test]
fn test_unused_dependency_tables_derives_doctests_and_whitelist() {
    let dir = TempTestDir::new("policy_audit_dep_tables");
    let whitelist_file = dir.join("verification_whitelist.toml");

    for header in [
        "[dependencies]\nunused_dep = \"0.1\"",
        "[dev-dependencies]\nunused_dev = \"0.1\"",
        "[build-dependencies]\nunused_build = \"0.1\"",
        "[target.'cfg(target_arch = \"wasm32\")'.dependencies]\ngetrandom = \"0.2\"",
    ] {
        let member = write_member_crate(&dir, "crates/dep_case", "pub fn ok() {}\n");
        fs::write(
            dir.join("crates/dep_case/Cargo.toml"),
            format!(
                "[package]\nname = \"dep_case\"\nversion = \"0.1.0\"\n\n{header}\n\n[lints]\nworkspace = true\n"
            ),
        )
        .unwrap();
        assert!(!run_policy_audit_with_whitelist(
            &[member.as_str()],
            &whitelist_file
        ));
    }

    // Derive mapping + README doctest + doc-comment fence + whitelist authorization pass
    let valid = write_member_crate(
        &dir,
        "crates/rich",
        concat!(
            "#![doc = include_str!(\"../README.md\")]\n",
            "/// ```rust\n/// use approx::assert_relative_eq;\n/// assert_relative_eq!(1.0, 1.0);\n/// ```\n",
            "#[derive(Serialize, Deserialize)]\npub struct Data;\n",
        ),
    );
    fs::write(
        dir.join("crates/rich/README.md"),
        "# Rich\n```rust\nuse thiserror::Error;\n#[derive(Debug, Error)]\npub enum E { #[error(\"x\")] X }\nlet _ = E::X;\n```\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/rich/Cargo.toml"),
        concat!(
            "[package]\nname = \"rich\"\nversion = \"0.1.0\"\n\n",
            "[dependencies]\nserde = \"1.0\"\n\n",
            "[dev-dependencies]\napprox = \"0.5\"\nthiserror = \"2.0\"\n\n",
            "[target.'cfg(target_arch = \"wasm32\")'.dependencies]\ngetrandom = \"0.2\"\n\n",
            "[lints]\nworkspace = true\n",
        ),
    )
    .unwrap();
    fs::write(
        &whitelist_file,
        "[[unused_dependencies]]\ncrate = \"crates/rich\"\ndependency = \"getrandom\"\nreason = \"wasm js backend\"\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[valid.as_str()],
        &whitelist_file
    ));
}

#[test]
fn test_markdown_tests_audits_workspace_markdown_fences() {
    let dir = TempTestDir::new("policy_audit_md_tests");
    let whitelist_file = dir.join("verification_whitelist.toml");
    let md_member = write_member_crate(
        &dir,
        "crates/markdown_tests",
        "#![doc = include_str!(\"../README.md\")]\n",
    );
    fs::write(dir.join("crates/markdown_tests/README.md"), "# MD Tests\n").unwrap();
    fs::create_dir_all(dir.join("docs")).unwrap();
    fs::write(
        dir.join("docs/guide.md"),
        "```rust\nuse nalgebra::Vector2;\nlet _ = Vector2::new(1.0, 2.0);\n```\n```rust,ignore\nuse eframe::App;\n```\n",
    )
    .unwrap();

    fs::write(
        dir.join("crates/markdown_tests/Cargo.toml"),
        "[package]\nname = \"markdown_tests\"\nversion = \"0.1.0\"\n\n[dev-dependencies]\nnalgebra = \"0.34\"\neframe = \"0.33\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[md_member.as_str()],
        &whitelist_file
    ));

    fs::write(
        dir.join("crates/markdown_tests/Cargo.toml"),
        "[package]\nname = \"markdown_tests\"\nversion = \"0.1.0\"\n\n[dev-dependencies]\nnalgebra = \"0.34\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[md_member.as_str()],
        &whitelist_file
    ));
}

#[test]
fn test_dependency_audit_skips_doctests_once_all_dependencies_matched() {
    let dir = TempTestDir::new("policy_audit_short_circuit_doctests");
    let whitelist_file = dir.join("verification_whitelist.toml");
    let member = write_member_crate(
        &dir,
        "crates/fast_match",
        concat!(
            "#![doc = include_str!(\"../MISSING_DOC.md\")]\n",
            "#[derive(Serialize)]\n",
            "pub struct Payload {\n",
            "    pub v: nalgebra::Vector2<f64>,\n",
            "}\n",
        ),
    );
    fs::write(
        dir.join("crates/fast_match/Cargo.toml"),
        concat!(
            "[package]\nname = \"fast_match\"\nversion = \"0.1.0\"\n\n",
            "[dependencies]\nserde = \"1.0\"\nnalgebra = \"0.34\"\n\n",
            "[lints]\nworkspace = true\n",
        ),
    )
    .unwrap();
    // Because both `serde` and `nalgebra` match during AST traversal, doctest and
    // `include_str!` markdown scanning is skipped rather than failing on MISSING_DOC.md.
    assert!(run_policy_audit_with_whitelist(
        &[member.as_str()],
        &whitelist_file
    ));
}

#[test]
fn test_markdown_tests_missing_library_dev_dependency_fails_policy_audit() {
    let dir = TempTestDir::new("policy_audit_md_lib_coverage");
    let whitelist_file = dir.join("verification_whitelist.toml");
    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\n  \"crates/02_engines/engine_pde\",\n  \"crates/04_bindings/markdown_tests\",\n]\n\n[workspace.lints.rust]\nmissing_docs = \"warn\"\n",
    )
    .unwrap();
    let _eng = write_member_crate(
        &dir,
        "crates/02_engines/engine_pde",
        "pub struct Stencil;\n",
    );
    fs::write(
        dir.join("crates/02_engines/engine_pde/Cargo.toml"),
        "[package]\nname = \"engine_pde\"\nversion = \"0.4.0\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/02_engines/engine_pde/README.md"),
        "# Engine PDE\n```rust\nuse engine_pde::Stencil;\nlet _ = Stencil;\n```\n",
    )
    .unwrap();

    let md_member = write_member_crate(
        &dir,
        "crates/04_bindings/markdown_tests",
        "#![doc = include_str!(\"../README.md\")]\n",
    );
    fs::write(
        dir.join("crates/04_bindings/markdown_tests/README.md"),
        "# MD Tests\n",
    )
    .unwrap();

    // Missing `engine_pde` in `[dev-dependencies]` must fail policy audit
    fs::write(
        dir.join("crates/04_bindings/markdown_tests/Cargo.toml"),
        "[package]\nname = \"markdown_tests\"\nversion = \"0.1.0\"\n\n[dev-dependencies]\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[md_member.as_str()],
        &whitelist_file
    ));

    // Declaring `engine_pde` in `[dev-dependencies]` and using it in README fence passes
    fs::write(
        dir.join("crates/04_bindings/markdown_tests/Cargo.toml"),
        "[package]\nname = \"markdown_tests\"\nversion = \"0.1.0\"\n\n[dev-dependencies]\nengine_pde = { path = \"../../02_engines/engine_pde\" }\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[md_member.as_str()],
        &whitelist_file
    ));
}
