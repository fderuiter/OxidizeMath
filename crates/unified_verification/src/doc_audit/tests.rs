use super::*;
use std::fs;
use std::path::PathBuf;

fn create_test_dir() -> PathBuf {
    crate::utils::test_utils::create_temp_test_dir("test_doc_audit")
}

#[test]
fn test_check_prohibited_placeholders_finds_stub() {
    let test_dir = create_test_dir();
    let readme = test_dir.join("README.md");
    let content = "Legacy crate.\nSome other text\n";
    fs::write(&readme, content).unwrap();

    let violations = check_prohibited_placeholders(&readme, content);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("README.md:1: Prohibited placeholder string found"));

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_check_prohibited_placeholders_clean() {
    let test_dir = create_test_dir();
    let readme = test_dir.join("README.md");
    let content = "# Valid Crate\n\nDetailed documentation here.\n";
    fs::write(&readme, content).unwrap();

    let violations = check_prohibited_placeholders(&readme, content);
    assert!(violations.is_empty());

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_check_content_thresholds_missing_headings() {
    let readme = Path::new("test/README.md");
    let content = "Line 1\nLine 2\nLine 3\nLine 4\n";
    let violations = check_content_thresholds(readme, content, false);
    assert!(
        violations
            .iter()
            .any(|v| v.contains("below minimum threshold"))
    );
    assert!(violations.iter().any(|v| {
        v.contains("missing mandatory architectural heading '## 1. Overview & Architecture'")
    }));
    assert!(violations.iter().any(|v| {
        v.contains("missing mandatory architectural heading '## 2. Developer Instructions'")
    }));
}

#[test]
fn test_check_content_thresholds_allowlisted_skips() {
    let readme = Path::new("test/README.md");
    let content = "Short stub\n";
    let violations = check_content_thresholds(readme, content, true);
    assert!(violations.is_empty());
}

#[test]
fn test_check_entrypoint_attribute_valid() {
    let test_dir = create_test_dir();
    let lib_rs = test_dir.join("lib.rs");
    fs::write(
        &lib_rs,
        "#![doc = include_str!(\"../README.md\")]\npub fn foo() {}\n",
    )
    .unwrap();

    assert!(check_entrypoint_attribute(&lib_rs).is_ok());
    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_check_entrypoint_attribute_missing() {
    let test_dir = create_test_dir();
    let lib_rs = test_dir.join("lib.rs");
    fs::write(&lib_rs, "pub fn foo() {}\n").unwrap();

    let err = check_entrypoint_attribute(&lib_rs).unwrap_err();
    assert!(err.contains("Missing required crate-level documentation attribute"));

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_check_entrypoint_attribute_commented_out() {
    let test_dir = create_test_dir();
    let lib_rs = test_dir.join("lib.rs");
    fs::write(
        &lib_rs,
        "// #![doc = include_str!(\"../README.md\")]\npub fn foo() {}\n",
    )
    .unwrap();

    let err = check_entrypoint_attribute(&lib_rs).unwrap_err();
    assert!(err.contains("Missing required crate-level documentation attribute"));

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_audit_crate_end_to_end_success() {
    let test_dir = create_test_dir();
    let readme_content = format!(
        "# Demo Crate\n\n## 1. Overview & Architecture\n\nIntro\n\n## 2. Developer Instructions\n\nInstructions\n{}",
        "\nMore content...".repeat(25)
    );
    fs::write(test_dir.join("README.md"), readme_content).unwrap();

    let src_dir = test_dir.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(
        src_dir.join("lib.rs"),
        "#![doc = include_str!(\"../README.md\")]\npub fn hello() {}\n",
    )
    .unwrap();

    let report = audit_crate("test_crate", &test_dir, false);
    assert!(report.is_success());

    let _ = fs::remove_dir_all(&test_dir);
}

#[test]
fn test_load_allowlist_merges_and_falls_back() {
    let test_dir = create_test_dir();
    let missing = test_dir.join("missing.toml");
    let defaults = load_allowlist(&missing);
    assert!(defaults.contains("egui_plot"));
    assert!(!defaults.contains("crates/diagnostics"));
    assert!(!defaults.contains("math_explorer"));

    let valid = test_dir.join("valid.toml");
    fs::write(
        &valid,
        "[doc_verification]\nallowlist = [\"crates/extra\"]\n",
    )
    .unwrap();
    let merged = load_allowlist(&valid);
    assert!(merged.contains("egui_plot"));
    assert!(merged.contains("crates/extra"));

    let broken = test_dir.join("broken.toml");
    fs::write(&broken, "[doc_verification\nbad =").unwrap();
    assert_eq!(load_allowlist(&broken), defaults);

    let _ = fs::remove_dir_all(&test_dir);
}
