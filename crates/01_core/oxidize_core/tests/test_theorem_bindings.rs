//! Integration tests for Lean 4 source hygiene, AST theorem bindings, and `Certified` qualification.

#![cfg(feature = "ast-analysis")]

use oxidize_core::ast_visitor::AstVisitor;
use oxidize_core::traceability::{
    AstCache, MaturityRegistry, MaturityTier, evaluate_maturity_with_registry_and_cache,
    extract_lean_theorem_names, validate_lean_proof_source,
};
use std::path::Path;

#[test]
fn proof_digest_rejects_sorry_ax_opaque_extern_and_implemented_by() {
    for kw in [
        "sorry",
        "sorryAx",
        "admit",
        "axiom",
        "opaque",
        "extern",
        "implemented_by",
    ] {
        let bad = format!("theorem demo (x : Int) : x = x := rfl\n-- body\n{kw} foo\n");
        let err = validate_lean_proof_source("Demo.lean", &bad)
            .expect_err("banned keyword must be rejected");
        assert!(err.to_string().contains(kw), "got: {err}");
    }

    let src = concat!(
        "/- theorem commented_out : True := trivial -/\n",
        "private theorem internal_helper (x : Int) : x = x := rfl\n",
        "theorem alpha_thm (x : Int) : x = x := rfl\n",
        "lemma beta_lemma (y : Int) : y = y := rfl\n",
    );
    assert!(validate_lean_proof_source("Valid.lean", src).is_ok());
    assert_eq!(
        extract_lean_theorem_names(src),
        vec!["alpha_thm".to_string(), "beta_lemma".to_string()]
    );
}

#[test]
fn ast_visitor_extracts_formal_proof_paths_and_theorem_lists() {
    let rust_src = r#"
        crate::formal_proof_verification!(
            module = "math_commons",
            proof = "IntervalSoundness.lean",
            theorems = ["add_outward_sound", "sub_outward_sound"],
            test = {
                assert!(1 + 1 == 2);
            }
        );
    "#;
    let parsed = syn::parse_file(rust_src).expect("valid Rust syntax");
    let mut visitor = AstVisitor::new();
    syn::visit::Visit::visit_file(&mut visitor, &parsed);
    assert_eq!(visitor.formal_proof_bindings.len(), 1);
    let fb = &visitor.formal_proof_bindings[0];
    assert_eq!(fb.module, "math_commons");
    assert_eq!(fb.proof, "IntervalSoundness.lean");
    assert_eq!(fb.theorems, vec!["add_outward_sound", "sub_outward_sound"]);
    assert!(fb.has_assertions);
}

#[test]
fn certified_qualification_rejects_unproven_demoted_targets() {
    let root =
        oxidize_core::workspace::find_workspace_root_from(Path::new(env!("CARGO_MANIFEST_DIR")))
            .expect("workspace root");
    let mut cache = AstCache::new();
    let baseline = MaturityRegistry::load_or_embedded(Some(&root));

    for demoted_mod in ["optics", "freesurfer", "engine_opt", "biology"] {
        assert_eq!(
            baseline.module_tier(demoted_mod),
            Some(MaturityTier::Validated)
        );
        let mut overclaim = baseline.clone();
        overclaim
            .modules
            .insert(demoted_mod.to_string(), MaturityTier::Certified);
        let report = evaluate_maturity_with_registry_and_cache(&root, overclaim, &mut cache);
        assert!(
            !report.passed(),
            "`{demoted_mod}` without Tier6_Formal theorem binding must fail Certified gate"
        );
        assert!(
            report
                .violations
                .iter()
                .any(|v| v.contains(demoted_mod) && v.contains("Tier6_Formal"))
        );
    }

    for proven_mod in [
        "clinical_trials",
        "epidemiology",
        "math_commons",
        "pure_math",
        "quantum",
    ] {
        assert_eq!(
            baseline.module_tier(proven_mod),
            Some(MaturityTier::Certified)
        );
    }
}
