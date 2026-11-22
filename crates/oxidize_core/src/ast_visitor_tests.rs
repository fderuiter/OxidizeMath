use super::{AstVisitor, is_dedicated_test_path};
use crate::traceability::{AstCache, TraceabilityEngine};
use crate::vfs::VirtualFileSystem;
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use syn::visit::Visit;

#[derive(Default)]
struct TestVfs {
    files: HashMap<String, String>,
    dirs: HashMap<String, Vec<String>>,
}

impl TestVfs {
    fn add_file(&mut self, path: &str, content: &str) {
        self.files.insert(path.to_string(), content.to_string());
        if let Some((parent, name)) = path.rsplit_once('/') {
            self.dirs
                .entry(parent.to_string())
                .or_default()
                .push(name.to_string());
        }
    }
}

impl VirtualFileSystem for TestVfs {
    fn read_to_string(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, std::io::Error>> + '_>> {
        let res = self.files.get(path).cloned().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing {path}"))
        });
        Box::pin(async move { res })
    }

    fn write_to_file(
        &self,
        _path: &str,
        _content: &[u8],
    ) -> Pin<Box<dyn Future<Output = Result<(), std::io::Error>> + '_>> {
        Box::pin(async { Ok(()) })
    }

    fn list_dir(
        &self,
        path: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, std::io::Error>> + '_>> {
        let res = self.dirs.get(path).cloned().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("missing dir {path}"))
        });
        Box::pin(async move { res })
    }
}

fn visit_src(src: &str) -> AstVisitor {
    let ast = syn::parse_file(src).expect("valid Rust syntax");
    let mut visitor = AstVisitor::new();
    visitor.visit_file(&ast);
    visitor
}

#[test]
fn parses_inner_outer_and_cfg_attr_opt_outs() {
    let inner_cfg = visit_src(
        r#"
        #![cfg_attr(any(), verified(opt_out = "infrastructure"))]
        pub fn helper() -> usize { 1 }
        "#,
    );
    assert!(inner_cfg.opted_out);
    assert_eq!(inner_cfg.opt_outs, vec!["infrastructure".to_string()]);
    assert_eq!(inner_cfg.opted_out_funcs, 1);
    assert_eq!(inner_cfg.verified_funcs, 0);

    let inner_direct = visit_src(
        r#"
        #![verified(opt_out = "gui_tool")]
        pub fn draw() {}
        "#,
    );
    assert!(inner_direct.opted_out);
    assert_eq!(inner_direct.opt_outs, vec!["gui_tool".to_string()]);

    let recursion = visit_src(
        r#"
        #[verified(allow_recursion)]
        pub fn fact(n: u64) -> u64 {
            assert!(n <= 20);
            if n <= 1 { 1 } else { n * fact(n - 1) }
        }
        "#,
    );
    assert!(!recursion.opted_out);
    assert!(recursion.opt_outs.is_empty());
    assert_eq!(recursion.verified_funcs, 1);
    assert_eq!(recursion.verified_prod_asserts, 1);
}

#[test]
fn item_opt_out_is_scoped_and_preserves_sibling_verified_counts() {
    let v = visit_src(
        r#"
        #[verified_engine::verified(opt_out = "Legacy missing assertions")]
        pub fn legacy_step(x: f64) -> f64 { x }

        #[verified]
        pub fn verified_step(x: f64) -> f64 {
            assert!(x.is_finite());
            assert_ne!(x, -0.0);
            x + 1.0
        }

        pub fn plain_helper() -> usize { 0 }
        "#,
    );
    assert!(!v.opted_out);
    assert_eq!(v.opt_outs, vec!["Legacy missing assertions".to_string()]);
    assert_eq!(v.total_funcs, 3);
    assert_eq!(v.opted_out_funcs, 1);
    assert_eq!(v.verified_funcs, 1);
    assert_eq!(v.verified_asserts, 2);
    assert_eq!(v.verified_prod_funcs, 1);
    assert_eq!(v.verified_prod_asserts, 2);
}

#[test]
fn traceability_flags_remaining_unverified_functions_after_item_opt_out() {
    let mut vfs = TestVfs::default();
    vfs.add_file(
        "traceability.toml",
        "[links]\ncalculus = \"calculus.tex\"\n",
    );
    vfs.add_file("papers/calculus.tex", "\\section{Calculus}");
    vfs.add_file(
        "crates/pure_math/src/lib.rs",
        "pub mod mixed;\npub mod only_opted;\npub mod good;\n",
    );
    vfs.add_file(
        "crates/pure_math/src/mixed.rs",
        "#[verified(opt_out = \"Legacy\")] pub fn skipped() {}\npub fn forgotten() -> i32 { 7 }\n",
    );
    vfs.add_file(
        "crates/pure_math/src/only_opted.rs",
        "#[verified(opt_out = \"Legacy\")] pub fn sole_opted_out_fn() {}\n",
    );
    vfs.add_file(
        "crates/pure_math/src/good.rs",
        "// [cite:calculus]\n#[verified(opt_out = \"Legacy\")] pub fn skipped_first() {}\n#[verified] pub fn checked_second(x: f64) -> f64 { assert!(x.is_finite()); x }\n",
    );

    let engine = TraceabilityEngine::new(vfs);
    let mut cache = AstCache::new();
    let src_dirs = ["crates/pure_math/src"];
    let fut = engine.scan_repository_with_cache(&src_dirs, "papers", false, &mut cache);
    let report = futures::executor::block_on(fut).expect("scan succeeds");
    let unverified = &report.unverified_modules;
    assert!(unverified.contains(&"crates/pure_math/src/mixed.rs".to_string()));
    assert!(!unverified.contains(&"crates/pure_math/src/only_opted.rs".to_string()));
    assert!(!unverified.contains(&"crates/pure_math/src/good.rs".to_string()));
    assert_eq!(report.verified_funcs, 1);
    assert_eq!(report.verified_asserts, 1);
    let status = report.semantic_integrity_status.get("calculus");
    assert_eq!(status.map(String::as_str), Some("Verified"));
}

#[test]
fn ignores_comments_and_string_literals_for_macros_and_attributes() {
    let spoof_src = concat!(
        "// theory_",
        "verification! { module = \"spoof\", test = { assert!(true); } }\n",
        "// #[verified(opt_out = \"comment_spoof\")]\n",
        "/// Regular documentation mentioning #[verified] without citations.\n",
        "pub fn utility() -> &'static str {\n",
        "    \"theory_",
        "verification! #[verified] opt_out = \\\"literal\\\"\"\n",
        "}\n"
    );
    let v = visit_src(spoof_src);
    assert!(!v.has_verification_macro);
    assert!(!v.opted_out);
    assert!(v.opt_outs.is_empty());
    assert!(v.verified_modules.is_empty());
    assert_eq!(v.verified_funcs, 0);
    assert_eq!(v.total_asserts, 0);
}

#[test]
fn counts_verification_macros_without_synthetic_floor() {
    let empty_macro = visit_src(
        r#"
        crate::theory_verification! {
            module = "algebra",
            theory_fn = "spec",
            test = {}
        }
        "#,
    );
    assert!(empty_macro.has_verification_macro);
    assert_eq!(empty_macro.verified_modules, vec!["algebra".to_string()]);
    assert_eq!(empty_macro.total_asserts, 0);
    assert_eq!(empty_macro.verified_asserts, 0);
    assert_eq!(empty_macro.test_asserts, 0);
    assert_eq!(empty_macro.prod_asserts, 0);

    let populated = visit_src(
        r#"
        theory_verification! {
            module = "analysis",
            test = {
                assert!(1.0_f64.is_finite());
                approx::assert_relative_eq!(1.0_f64, 1.0_f64);
            }
        }
        stochastic_signature_verification! {
            module = "sde",
            signatures = { mean: 0.0; var: 1.0; },
            statistical_bounds = { mean: (-0.1, 0.1); },
            stochastic_test = {
                assert_abs_diff_eq!(0.5_f64, 0.5_f64);
            }
        }
        "#,
    );
    assert_eq!(populated.verified_asserts, 6);
    assert_eq!(populated.verified_test_asserts, 6);
    assert_eq!(populated.prod_asserts, 0);
    assert_eq!(
        populated.module_tiers.get("sde").map(String::as_str),
        Some("Stochastic")
    );
}

#[test]
fn separates_production_and_test_assertions_and_detects_legacy_fns() {
    let v = visit_src(
        r#"
        #[verified]
        pub fn compute(x: f64) -> f64 {
            fn nested_helper() {
                assert!(true);
            }
            nested_helper();
            assert!(x.is_finite());
            x * 2.0
        }

        pub fn step_legacy() {}

        #[cfg(test)]
        mod tests {
            #[test]
            #[verified]
            fn verifies_compute() {
                assert_eq!(super::compute(2.0), 4.0);
                assert_ne!(super::compute(1.0), 0.0);
            }
        }
        "#,
    );
    assert_eq!(v.prod_funcs, 3);
    assert_eq!(v.prod_asserts, 2);
    assert_eq!(v.verified_prod_funcs, 1);
    assert_eq!(v.verified_prod_asserts, 1);
    assert_eq!(v.test_funcs, 1);
    assert_eq!(v.test_asserts, 2);
    assert_eq!(v.verified_test_funcs, 1);
    assert_eq!(v.verified_test_asserts, 2);
    assert_eq!(v.total_funcs, 4);
    assert_eq!(v.total_asserts, 4);
    assert_eq!(v.legacy_funcs, vec!["step_legacy".to_string()]);
    assert!(is_dedicated_test_path(Path::new(
        "crates/foo/tests/integration.rs"
    )));
    assert!(is_dedicated_test_path(Path::new(
        "crates/foo/src/bar_tests.rs"
    )));
    assert!(!is_dedicated_test_path(Path::new(
        "crates/foo/src/testing_utils.rs"
    )));
}
