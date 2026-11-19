use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

static TEST_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

struct CurrentDirGuard(PathBuf);

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

fn create_temp_workspace(prefix: &str) -> PathBuf {
    let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), id));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_member_crate(root: &Path, member: &str, lib_source: &str) -> String {
    let member_dir = root.join(member);
    let src_dir = member_dir.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(
        member_dir.join("Cargo.toml"),
        "[package]\nname = \"sample_member\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::write(src_dir.join("lib.rs"), lib_source).unwrap();
    member_dir.to_string_lossy().replace('\\', "/")
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
