use super::*;
use crate::utils::test_utils::{CurrentDirGuard, TempTestDir};
use std::fs;

pub(super) fn write_member_crate(root: &Path, member: &str, lib_source: &str) -> String {
    let member_dir = root.join(member);
    let src_dir = member_dir.join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(
        member_dir.join("Cargo.toml"),
        "[package]\nname = \"sample_member\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[lints]\nworkspace = true\n",
    )
    .unwrap();
    fs::write(src_dir.join("lib.rs"), lib_source).unwrap();
    crate::whitelist::lossy_slash_path(&member_dir)
}

fn scan_clippy_snippet(source: &str) -> bool {
    let ast = syn::parse_file(source).expect("valid Rust snippet");
    let mut visitor = lint_policy::ClippyBypassVisitor::new(
        "crates/sample/src/lib.rs".to_string(),
        lint_policy::DEFAULT_TOO_MANY_LINES_THRESHOLD,
    );
    visitor.visit_file(&ast);
    visitor.has_unauthorized
}

fn scan_snippet(source: &str, allowed: Vec<DeadCodeEntry>) -> bool {
    let ast = syn::parse_file(source).expect("valid Rust snippet");
    let mut visitor = DeadCodeVisitor::new("crates/sample/src/lib.rs".to_string(), allowed);
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
    let dir = TempTestDir::new("policy_audit_whitelist_ok");
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
}

#[test]
fn test_invalid_toml_and_unknown_dead_code_fields_fail_closed() {
    let dir = TempTestDir::new("policy_audit_bad_toml");
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
        std::env::set_current_dir(&*dir).unwrap();
        assert!(!run_policy_audit(&["crates/clean"]));
    }
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
    let long_fn = format!(
        "#[allow(clippy::too_many_lines)]\npub fn allowed_fn() {{\n{}}}\n",
        "    let _ = 0;\n".repeat(60)
    );
    assert!(!scan_clippy_snippet(&long_fn));
    assert!(!scan_clippy_snippet(
        "#[allow(clippy::cognitive_complexity)]\npub fn complex_fn() {}"
    ));
    assert!(!scan_clippy_snippet(
        "#[cfg_attr(clippy::all, derive(Debug))]\npub struct PredicateOnly;"
    ));
}

#[test]
fn test_missing_or_custom_lints_table_rejected_for_first_party_crates() {
    let dir = TempTestDir::new("policy_audit_lints_manifest");
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
}

#[test]
fn test_vendored_egui_plot_lints_exemption_conditions() {
    let dir = TempTestDir::new("policy_audit_egui_plot");
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
}

#[test]
fn test_clippy_groups_impl_trait_macros_and_short_fn_rejected() {
    for group in [
        "pedantic",
        "nursery",
        "restriction",
        "complexity",
        "correctness",
        "suspicious",
        "style",
        "perf",
        "cargo",
    ] {
        assert!(scan_clippy_snippet(&format!(
            "#![allow(clippy::{group})]\npub fn f() {{}}"
        )));
        assert!(scan_clippy_snippet(&format!(
            "#[expect(clippy::{group})]\npub fn f() {{}}"
        )));
    }
    assert!(scan_clippy_snippet(
        "pub struct S;\n#[allow(clippy::too_many_lines)]\nimpl S { pub fn f() {} }"
    ));
    assert!(scan_clippy_snippet(
        "#[expect(clippy::cognitive_complexity)]\npub trait T { fn f(&self); }"
    ));
    assert!(scan_clippy_snippet(
        "macro_rules! m { () => { #[allow(unused_imports, clippy::all)] mod sub {} }; }"
    ));
    assert!(scan_clippy_snippet(
        "macro_rules! m { () => { #[allow(clippy::too_many_lines)] impl S {} }; }"
    ));
    assert!(scan_clippy_snippet(
        "#[allow(clippy::too_many_lines)]\npub fn one_liner() {}"
    ));
    let exact_sixty = format!(
        "#[allow(clippy::too_many_lines)]\npub fn sixty_lines() {{\n{}}}",
        "    let _ = 1;\n".repeat(58)
    );
    assert!(scan_clippy_snippet(&exact_sixty));
}

#[test]
fn test_dead_code_visitor_enforces_256_step_attr_bound() {
    let build_nested_cfg_attr = |wrappers: usize| -> String {
        let mut attr = "allow(dead_code)".to_string();
        for _ in 0..wrappers {
            attr = format!("cfg_attr(unix, {attr})");
        }
        format!("#[{attr}]\nfn bounded_target() {{}}\n")
    };

    // 255 cfg_attr wrappers + 1 allow(dead_code) = 256 total stack steps -> detected
    assert!(scan_snippet(&build_nested_cfg_attr(255), vec![]));
    // 256 cfg_attr wrappers + 1 allow(dead_code) = 257 total stack steps -> stops at 256 bound
    assert!(!scan_snippet(&build_nested_cfg_attr(256), vec![]));
}

#[test]
fn test_dead_code_visitor_nested_paren_and_qself_rejection() {
    let demo_entry = DeadCodeEntry {
        file: "crates/sample/src/lib.rs".to_string(),
        identifier: Some("Demo".to_string()),
    };

    let nested_paren =
        "pub struct Demo;\n#[allow(dead_code)]\nimpl ((Demo)) {\n    fn ok() {}\n}\n";
    assert!(scan_snippet(nested_paren, vec![]));
    assert!(!scan_snippet(nested_paren, vec![demo_entry.clone()]));

    let mut deep_ty = "Demo".to_string();
    for _ in 0..32 {
        deep_ty = format!("({deep_ty})");
    }
    let over_32_parens =
        format!("pub struct Demo;\n#[allow(dead_code)]\nimpl {deep_ty} {{\n    fn f() {{}}\n}}\n");
    assert!(scan_snippet(&over_32_parens, vec![demo_entry.clone()]));

    let qself_impl = concat!(
        "trait Assoc { type Demo; }\n",
        "struct Other;\n",
        "impl Assoc for Other { type Demo = u8; }\n",
        "#[allow(dead_code)]\n",
        "impl <Other as Assoc>::Demo {}\n",
    );
    assert!(scan_snippet(qself_impl, vec![demo_entry]));
}

#[test]
fn test_standalone_policy_audit_rejects_missing_lints_from_root_and_subdir() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let dir = TempTestDir::new("policy_audit_root_and_subdir");
    let _guard = CurrentDirGuard(std::env::current_dir().unwrap());

    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/good\", \"crates/bad\"]\n",
    )
    .unwrap();
    write_member_crate(&dir, "crates/good", "pub fn ok() {}\n");
    write_member_crate(&dir, "crates/bad", "pub fn bad() {}\n");
    fs::write(
        dir.join("crates/bad/Cargo.toml"),
        "[package]\nname = \"bad\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();

    std::env::set_current_dir(&*dir).unwrap();
    assert!(run_policy_audit(&["crates/good"]));
    assert!(!run_policy_audit(&["crates/good", "crates/bad"]));

    std::env::set_current_dir(dir.join("crates/good")).unwrap();
    assert!(run_policy_audit(&["crates/good"]));
    assert!(!run_policy_audit(&["crates/good", "crates/bad"]));
}

#[test]
fn test_dead_code_and_clippy_diagnostics_emit_spans_idents_and_guidance() {
    let dc_src = "pub fn ok() {\n    #[allow(dead_code)]\n    let local_buf = 1;\n}\n#[allow(dead_code)]\nfn hidden_fn() {}\n";
    let dc_ast = syn::parse_file(dc_src).unwrap();
    let mut dc = DeadCodeVisitor::new("crates/sample/src/lib.rs".to_string(), vec![]);
    dc.visit_file(&dc_ast);
    assert_eq!(dc.violations.len(), 2);
    assert!(dc.violations[0].contains("crates/sample/src/lib.rs:2 (identifier `local_buf`)"));
    assert!(dc.violations[1].contains("crates/sample/src/lib.rs:5 (identifier `hidden_fn`)"));
    assert!(dc.violations[1].contains("#[cfg(test)]"));
    assert!(dc.violations[1].contains("verification_whitelist.toml"));

    let mut syn_dc = DeadCodeVisitor::new("crates/sample/src/lib.rs".to_string(), vec![]);
    syn_dc.record_violation(0, Some("macro_fn"));
    assert!(syn_dc.violations[0].contains("at crates/sample/src/lib.rs (identifier `macro_fn`)"));
    assert!(!syn_dc.violations[0].contains(":0"));

    let clippy_src =
        "pub struct Widget;\n#[allow(clippy::all)]\nimpl Widget {\n    pub fn f() {}\n}\n";
    let clippy_ast = syn::parse_file(clippy_src).unwrap();
    let mut cv = lint_policy::ClippyBypassVisitor::new("crates/sample/src/lib.rs".to_string(), 60);
    cv.visit_file(&clippy_ast);
    assert_eq!(cv.violations.len(), 1);
    assert!(
        cv.violations[0]
            .contains("crates/sample/src/lib.rs:2 (impl `Widget` `#[...]` attribute: clippy::all)")
    );
    assert!(cv.violations[0].contains("#[cfg(test)]"));
    assert!(cv.violations[0].contains("verification_whitelist.toml"));
}
