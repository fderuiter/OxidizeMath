use super::*;
use crate::utils::test_utils::{CurrentDirGuard, create_temp_test_dir};
use std::path::PathBuf;

fn create_temp_workspace(prefix: &str) -> PathBuf {
    create_temp_test_dir(prefix)
}

fn write_member_crate(root: &Path, member: &str, lib_source: &str) -> String {
    let member_dir = root.join(member);
    let src_dir = member_dir.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(
        member_dir.join("Cargo.toml"),
        "[package]\nname = \"sample_member\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(src_dir.join("lib.rs"), lib_source).unwrap();
    member_dir.to_string_lossy().replace('\\', "/")
}

fn scan_clippy_snippet(source: &str) -> bool {
    let ast = syn::parse_file(source).expect("valid Rust snippet");
    let mut visitor = lint_policy::ClippyBypassVisitor {
        file_path: "crates/sample/src/lib.rs".to_string(),
        has_unauthorized: false,
    };
    visitor.visit_file(&ast);
    visitor.has_unauthorized
}

fn scan_snippet(source: &str, allowed: Vec<DeadCodeEntry>) -> bool {
    let ast = syn::parse_file(source).expect("valid Rust snippet");
    let mut visitor = DeadCodeVisitor {
        file_path: "crates/sample/src/lib.rs".to_string(),
        allowed,
        has_unauthorized: false,
    };
    visitor.visit_file(&ast);
    visitor.has_unauthorized
}

#[test]
fn test_policy_audit_workspace_run() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let members = vec!["crates/pure_math"];
    let success = run_policy_audit(&members);
    assert!(success, "pure_math should pass policy audit");
}

#[test]
fn test_combined_allow_expect_and_prefixed_lints() {
    assert!(scan_snippet(
        "#[allow(missing_docs, dead_code)]\nfn hidden() {}",
        vec![],
    ));
    assert!(scan_snippet(
        "#[expect(dead_code, reason = \"staged\")]\nfn expected_unused() {}",
        vec![],
    ));
    assert!(scan_snippet("#[allow(unused)]\nstruct UnusedBox;", vec![]));
    assert!(scan_snippet(
        "#[allow(warnings)]\nenum SilentEnum { A }",
        vec![]
    ));
    assert!(scan_snippet(
        "#[allow(rust::dead_code)]\nconst SECRET: u32 = 1;",
        vec![],
    ));
    assert!(scan_snippet(
        "#[expect(rust::unused)]\nstatic COUNTER: u32 = 0;",
        vec![],
    ));
    assert!(scan_snippet(
        "#![allow(rust::warnings)]\npub fn top() {}",
        vec![],
    ));
}

#[test]
fn test_cfg_attr_and_nested_cfg_attr_detection() {
    assert!(scan_snippet(
        "#[cfg_attr(not(test), allow(dead_code))]\nfn native_helper() {}",
        vec![],
    ));
    assert!(scan_snippet(
        "#[cfg_attr(all(unix, not(test)), cfg_attr(debug_assertions, expect(missing_docs, rust::unused)))]\nfn nested_helper() {}",
        vec![],
    ));
    // Predicate name in position 0 must not be mistaken for an allow/expect attribute
    assert!(!scan_snippet(
        "#[cfg_attr(dead_code, derive(Debug))]\npub struct SafeStruct;",
        vec![],
    ));
}

#[test]
fn test_unrelated_unused_lints_are_ignored() {
    assert!(!scan_snippet(
        "#[allow(unused_imports)]\npub use core::fmt::Debug;\n#[allow(missing_docs, clippy::all)]\npub fn visible() {}",
        vec![],
    ));
}

#[test]
fn test_impl_type_union_trait_and_foreign_visitors() {
    assert!(scan_snippet(
        "struct Panel;\n#[allow(missing_docs, dead_code)]\nimpl Panel {\n    fn helper(&self) {}\n}",
        vec![],
    ));
    assert!(scan_snippet(
        "#[allow(dead_code)]\ntype Alias = u64;",
        vec![],
    ));
    assert!(scan_snippet(
        "#[allow(dead_code)]\nunion RawBits { i: u32, f: f32 }",
        vec![],
    ));
    assert!(scan_snippet(
        "trait Assoc { type Out; }\nstruct ImplTarget;\nimpl Assoc for ImplTarget { #[allow(dead_code)] type Out = u8; }",
        vec![],
    ));
    assert!(scan_snippet(
        "trait Spec { #[allow(dead_code)] fn hook(&self); #[allow(dead_code)] const ID: usize; #[allow(dead_code)] type Item; }",
        vec![],
    ));
    assert!(scan_snippet(
        "unsafe extern \"C\" { #[allow(dead_code)] fn ext_fn(); #[allow(dead_code)] static EXT_VAL: i32; }",
        vec![],
    ));
}

#[test]
fn test_whitelist_authorization_and_unrelated_tables() {
    let dir = create_temp_workspace("policy_audit_whitelist_ok");
    let member_path = write_member_crate(
        &dir,
        "crates/demo",
        "#[allow(missing_docs, dead_code)]\nimpl Demo {\n    fn helper() {}\n}\npub struct Demo;\n",
    );
    let whitelist_file = dir.join("verification_whitelist.toml");
    fs::write(
        &whitelist_file,
        r#"
[doc_verification]
allowlist = ["crates/diagnostics"]

[entropy_guard]
scan_paths = ["crates"]

[[dead_code]]
file = "crates/demo/src/lib.rs"
identifier = "Demo"
"#,
    )
    .unwrap();

    assert!(run_policy_audit_with_whitelist(
        &[member_path.as_str()],
        &whitelist_file
    ));

    // Mismatched identifier must fail closed
    fs::write(
        &whitelist_file,
        r#"
[[dead_code]]
file = "crates/demo/src/lib.rs"
identifier = "OtherType"
"#,
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[member_path.as_str()],
        &whitelist_file
    ));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_invalid_toml_and_unknown_dead_code_fields_fail_closed() {
    let dir = create_temp_workspace("policy_audit_bad_toml");
    let member_path = write_member_crate(&dir, "crates/clean", "pub fn ok() {}\n");
    let whitelist_file = dir.join("verification_whitelist.toml");

    // Malformed TOML syntax fails closed even on clean crates
    fs::write(&whitelist_file, "[doc_verification\nbroken = ").unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[member_path.as_str()],
        &whitelist_file
    ));

    // Unknown field on [[dead_code]] entry fails closed via deny_unknown_fields
    fs::write(
        &whitelist_file,
        "[[dead_code]]\nfile = \"crates/clean/src/lib.rs\"\nident = \"typo_field\"\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[member_path.as_str()],
        &whitelist_file
    ));

    // Unreadable directory at whitelist path fails closed
    let _ = fs::remove_file(&whitelist_file);
    fs::create_dir_all(&whitelist_file).unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[member_path.as_str()],
        &whitelist_file
    ));

    // Also verify run_policy_audit() directly in a directory with invalid verification_whitelist.toml
    {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let orig = std::env::current_dir().unwrap();
        let _guard = CurrentDirGuard(orig);
        let _ = fs::remove_dir_all(&whitelist_file);
        fs::write(&whitelist_file, "invalid = [unclosed").unwrap();
        std::env::set_current_dir(&dir).unwrap();
        assert!(!run_policy_audit(&["crates/clean"]));
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_inner_and_mod_clippy_bypasses_are_rejected() {
    assert!(scan_clippy_snippet("#![allow(clippy::all)]\npub fn f() {}"));
    assert!(scan_clippy_snippet(
        "#![allow(clippy::too_many_lines)]\npub fn f() {}"
    ));
    assert!(scan_clippy_snippet(
        "#![expect(clippy::cognitive_complexity, reason = \"legacy\")]\npub fn f() {}"
    ));
    assert!(scan_clippy_snippet(
        "#![cfg_attr(not(test), cfg_attr(unix, allow(clippy::too_many_lines)))]\npub fn f() {}"
    ));
    assert!(scan_clippy_snippet(
        "#[allow(clippy::too_many_lines)]\nmod sub { pub fn f() {} }"
    ));
    assert!(scan_clippy_snippet(
        "#[cfg_attr(unix, expect(clippy::cognitive_complexity))]\nmod sub;"
    ));
    assert!(scan_clippy_snippet(
        "mod sub {\n    #![allow(clippy::too_many_lines)]\n    pub fn f() {}\n}"
    ));
    assert!(scan_clippy_snippet(
        "pub fn sneaky() {\n    #![allow(clippy::too_many_lines)]\n}"
    ));
}

#[test]
fn test_item_clippy_all_rejected_while_fn_too_many_lines_allowed() {
    assert!(scan_clippy_snippet("#[allow(clippy::all)]\npub fn f() {}"));
    assert!(scan_clippy_snippet(
        "#[expect(missing_docs, clippy::all)]\npub struct S;"
    ));
    assert!(scan_clippy_snippet(
        "#[cfg_attr(test, allow(clippy::all))]\nimpl S {}\npub struct S;"
    ));
    assert!(!scan_clippy_snippet(
        "#[allow(clippy::too_many_lines)]\npub fn allowed_fn() {}"
    ));
    assert!(!scan_clippy_snippet(
        "#[allow(clippy::cognitive_complexity)]\npub fn complex_fn() {}"
    ));
    assert!(!scan_clippy_snippet(
        "#[cfg_attr(clippy::all, derive(Debug))]\npub struct PredicateOnly;"
    ));
}

#[test]
fn test_missing_or_custom_lints_table_rejected_for_first_party_crates() {
    let dir = create_temp_workspace("policy_audit_lints_manifest");
    let whitelist_file = dir.join("verification_whitelist.toml");

    let no_lints = write_member_crate(&dir, "crates/no_lints", "pub fn ok() {}\n");
    fs::write(
        dir.join("crates/no_lints/Cargo.toml"),
        "[package]\nname = \"no_lints\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[no_lints.as_str()],
        &whitelist_file
    ));

    let custom_lints = write_member_crate(&dir, "crates/custom_lints", "pub fn ok() {}\n");
    fs::write(
        dir.join("crates/custom_lints/Cargo.toml"),
        "[package]\nname = \"custom_lints\"\nversion = \"0.1.0\"\n\n[lints.rust]\nmissing_docs = \"warn\"\n[lints.clippy]\ntoo_many_lines = \"deny\"\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[custom_lints.as_str()],
        &whitelist_file
    ));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn test_vendored_egui_plot_lints_exemption_conditions() {
    let dir = create_temp_workspace("policy_audit_egui_plot");
    let whitelist_file = dir.join("verification_whitelist.toml");

    let vendored = write_member_crate(
        &dir,
        "egui_plot",
        "#![allow(clippy::too_many_lines, clippy::cognitive_complexity)]\npub fn plot() {}\n",
    );
    fs::write(
        dir.join("egui_plot/Cargo.toml"),
        "[package]\nname = \"egui_plot\"\nversion = \"0.34.0\"\n\n[lints.rust]\nmissing_docs = \"warn\"\n[lints.clippy]\ntoo_many_lines = \"deny\"\n",
    )
    .unwrap();
    assert!(run_policy_audit_with_whitelist(
        &[vendored.as_str()],
        &whitelist_file
    ));

    // Missing [lints.clippy] revokes the vendored egui_plot exemption
    fs::write(
        dir.join("egui_plot/Cargo.toml"),
        "[package]\nname = \"egui_plot\"\nversion = \"0.34.0\"\n\n[lints.rust]\nmissing_docs = \"warn\"\n",
    )
    .unwrap();
    assert!(!run_policy_audit_with_whitelist(
        &[vendored.as_str()],
        &whitelist_file
    ));

    let _ = fs::remove_dir_all(&dir);
}
