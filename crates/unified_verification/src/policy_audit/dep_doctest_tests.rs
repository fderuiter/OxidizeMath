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
