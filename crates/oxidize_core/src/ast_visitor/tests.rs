use super::*;

#[test]
fn test_ast_visitor_item_scoped_opt_out_resets_and_order_independence() {
    let verified_first = format!(
        "#[verified]\nfn sample_fn() {{\n    assert!(true);\n    assert_eq!(1, 1);\n}}\n#[{}(opt_out = \"testing\")]\nfn opt_out_fn() {{}}\n",
        "verified"
    );
    let opt_out_first = format!(
        "#[{}(opt_out = \"testing\")]\nfn opt_out_fn() {{}}\n#[verified]\nfn sample_fn() {{\n    assert!(true);\n    assert_eq!(1, 1);\n}}\n",
        "verified"
    );

    for snippet in [&verified_first, &opt_out_first] {
        let file = syn::parse_file(snippet).unwrap();
        let mut visitor = AstVisitor::default();
        visitor.visit_file(&file);

        assert_eq!(visitor.total_funcs, 2);
        assert_eq!(visitor.verified_funcs, 1);
        assert_eq!(visitor.total_asserts, 2);
        assert_eq!(visitor.verified_asserts, 2);
        assert!(!visitor.opted_out, "item opt_out must reset after item");
    }

    // Also verify struct-level outer opt_out does NOT suppress subsequent functions
    let struct_then_fn = format!(
        "#[cfg_attr(any(), {}(opt_out = \"gui_tool\"))]\npub struct Panel;\npub fn unverified_after_struct() {{}}\n",
        "verified"
    );
    let parsed_struct = syn::parse_file(&struct_then_fn).unwrap();
    let mut struct_visitor = AstVisitor::default();
    struct_visitor.visit_file(&parsed_struct);
    assert!(!struct_visitor.opted_out);
    assert_eq!(
        struct_visitor.first_unverified_fn,
        Some(("unverified_after_struct".to_string(), 3))
    );

    let file_opt_out = format!(
        "#![cfg_attr(any(), {}(opt_out = \"gui_tool\"))]\nfn helper() {{}}\n",
        "verified"
    );
    let parsed_inner = syn::parse_file(&file_opt_out).unwrap();
    let mut inner_visitor = AstVisitor::default();
    inner_visitor.visit_file(&parsed_inner);
    assert!(
        inner_visitor.opted_out,
        "inner #![...] opt_out must persist"
    );
}

#[test]
fn test_stochastic_vacuous_bypass_allows_valid_bounds_and_flags_zero_inits() {
    let valid_stochastic = r#"
        stochastic_signature_verification!(
            module = "ai",
            external_dependencies = [],
            signatures = { output => (4, 4); },
            statistical_bounds = {
                mean in [-0.05, 0.05];
                spread in [0.0, 10.0];
            },
            stochastic_test = {
                let scale = 0.05;
                let cap = 10.0;
                let output = build_matrix(scale, cap);
                let mean = output.mean();
                let spread = output.max();
            }
        );
    "#;
    let valid_ast = syn::parse_file(valid_stochastic).unwrap();
    let mut valid_visitor = AstVisitor::default();
    valid_visitor.visit_file(&valid_ast);
    assert!(!valid_visitor.has_vacuous_bypass);
    assert_eq!(valid_visitor.verified_funcs, 1);

    for zero_body in [
        "let x = DMatrix::zeros(4, 4);",
        "let mean = 0.0;",
        "buf.fill(0);",
        "let arr = [0.0; 8];",
        "let v = vec![0.0; 8];",
    ] {
        let bypass_src = format!(
            "stochastic_signature_verification!(module = \"ai\", external_dependencies = [], signatures = {{}}, statistical_bounds = {{}}, stochastic_test = {{\n    {zero_body}\n}});\n"
        );
        let ast = syn::parse_file(&bypass_src).unwrap();
        let mut visitor = AstVisitor::default();
        visitor.visit_file(&ast);
        assert!(
            visitor.has_vacuous_bypass,
            "expected bypass on: {zero_body}"
        );
        assert!(visitor.vacuous_bypass_line.is_some());
    }
}

#[test]
fn test_ast_visitor_allow_recursion_preserves_verification_counts() {
    let code = r#"
        #[verified(allow_recursion = "delegates to inherent method")]
        fn local_delegate() {
            assert!(true);
        }

        #[verified_engine::verified(allow_recursion = "delegates to inherent method")]
        fn qualified_delegate() {
            assert_eq!(2, 2);
        }

        #[cfg_attr(feature = "std", verified_engine::verified)]
        fn nostd_feature_gated_verified() {
            assert_eq!(3, 3);
        }

        #[cfg_attr(any(), verified)]
        fn dead_any_not_verified() {}

        #[cfg_attr(test, verified)]
        fn test_only_attr_not_verified() {}
    "#;
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert_eq!(visitor.total_funcs, 5);
    assert_eq!(visitor.verified_funcs, 3);
    assert_eq!(visitor.semantic_integrity_funcs, 3);
    assert_eq!(visitor.verified_asserts, 3);
    assert!(!visitor.opted_out);
}

#[test]
fn test_ast_visitor_ignores_macro_examples_in_doc_comments() {
    let code = concat!(
        "/// Example:\n",
        "/// ```rust\n",
        "/// use verified_engine_macros::embed_theory;\n",
        "/// #[embed_theory(\"papers/mmwave_radiotherapy_setup.tex\", label = \"eq:range\")]\n",
        "/// #[verified]\n",
        "/// fn demo() {}\n",
        "/// ```\n",
        "pub fn macro_entrypoint() {}\n",
    );
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert_eq!(visitor.total_funcs, 1);
    assert_eq!(visitor.verified_funcs, 0);
    assert_eq!(visitor.semantic_integrity_funcs, 0);
    assert!(visitor.verified_modules.is_empty());
}

#[test]
fn test_ast_visitor_ignores_verified_and_embed_theory_in_rustdoc_while_keeping_citations() {
    let code = r#"
        /// Extracts equations via `use verified_engine_macros::embed_theory;`
        /// and `#[embed_theory("papers/essay.tex", label = "eq:one")]` or `#[verified]`.
        pub fn macro_entrypoint() {}

        /// Documented citation `[cite:essay]` still registers semantic module link.
        #[verified]
        pub fn cited_fn() {
            assert!(true);
        }
    "#;
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert_eq!(visitor.total_funcs, 2);
    assert_eq!(visitor.verified_funcs, 1);
    assert_eq!(visitor.semantic_integrity_funcs, 1);
    assert_eq!(visitor.verified_modules, vec!["essay".to_string()]);
    assert_eq!(
        visitor.first_unverified_fn,
        Some(("macro_entrypoint".to_string(), 4))
    );
}

#[test]
fn test_ast_visitor_preserves_doc_citation_semantic_link_under_file_opt_out() {
    let code = concat!(
        "#![cfg_attr(any(), verified(opt_out = \"gui_tool\"))]\n",
        "/// Interactive lab citing `[cite:essay]`.\n",
        "pub fn render_panel() {}\n",
    );
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert!(visitor.opted_out);
    assert_eq!(visitor.opted_out_funcs, 1);
    assert_eq!(visitor.verified_funcs, 0);
    assert_eq!(visitor.semantic_integrity_funcs, 1);
    assert_eq!(visitor.verified_modules, vec!["essay".to_string()]);
}

#[test]
fn test_ast_visitor_ignores_embed_theory_and_verified_tokens_inside_doc_comments() {
    let code = r#"
        /// Example macro usage:
        /// ```rust
        /// use verified_engine_macros::embed_theory;
        /// #[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
        /// #[verified]
        /// pub fn example() {}
        /// ```
        /// [cite:win_ratio]
        pub fn documented_macro_entrypoint() {}
    "#;
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert_eq!(visitor.total_funcs, 1);
    assert_eq!(visitor.verified_funcs, 0);
    assert_eq!(visitor.verified_modules, vec!["win_ratio".to_string()]);
}

#[test]
fn test_six_tier_pyramid_macro_classification_and_counts() {
    let code = r#"
        theory_verification!(module = "pure_math", constants = {}, test = {});
        mms_convergence_verification!(module = "epidemiology", test = {});
        property_invariant_verification!(module = "quantum", test = {});
        golden_fixture_verification!(module = "pure_math_golden", test = {});
        stochastic_signature_verification!(
            module = "ai",
            external_dependencies = [],
            signatures = {},
            statistical_bounds = {},
            stochastic_test = { let v = 1.5; let _ = v; }
        );
        interval_enclosure_verification!(module = "math_commons", test = {});
        formal_proof_verification!(module = "clinical_trials", proof = "SurvivalBounds.lean", test = {});
    "#;
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert_eq!(visitor.tier_counts.get("Tier1_Convergence"), Some(&2));
    assert_eq!(visitor.tier_counts.get("Tier2_Property"), Some(&1));
    assert_eq!(visitor.tier_counts.get("Tier3_Golden"), Some(&1));
    assert_eq!(visitor.tier_counts.get("Tier4_Stochastic"), Some(&1));
    assert_eq!(visitor.tier_counts.get("Tier5_Interval"), Some(&1));
    assert_eq!(visitor.tier_counts.get("Tier6_Formal"), Some(&1));
    assert_eq!(visitor.verified_funcs, 7);
}

#[test]
fn test_ast_visitor_counts_exact_macro_assertions_without_max3_padding() {
    let empty_macro = r#"
        crate::theory_verification!(module = "pure_math", constants = {}, test = {});
    "#;
    let empty_ast = syn::parse_file(empty_macro).unwrap();
    let mut empty_visitor = AstVisitor::default();
    empty_visitor.visit_file(&empty_ast);
    assert_eq!(empty_visitor.verified_funcs, 1);
    assert_eq!(empty_visitor.total_funcs, 1);
    assert_eq!(empty_visitor.verified_asserts, 0);
    assert_eq!(empty_visitor.total_asserts, 0);

    let real_macro = r#"
        theory_verification!(
            module = "algebra",
            constants = { TOL = 1e-9; },
            test = {
                // assert!(true);
                let _msg = "assert!(true); assert_eq!(1, 1);";
                assert!(1.0 > TOL);
                assert_ne!(2, 3);
            }
        );
        stochastic_signature_verification!(
            module = "ai",
            external_dependencies = [],
            signatures = {},
            statistical_bounds = {},
            stochastic_test = {
                let val = compute_sample(0.5);
                debug_assert!(val.is_finite());
            }
        );
    "#;
    let real_ast = syn::parse_file(real_macro).unwrap();
    let mut real_visitor = AstVisitor::default();
    real_visitor.visit_file(&real_ast);
    assert_eq!(real_visitor.verified_funcs, 2);
    assert_eq!(real_visitor.verified_asserts, 3);
    assert_eq!(real_visitor.total_asserts, 3);

    // Live workspace regression: crates/pure_math/src/pure_math/mod.rs has 1 empty theory_verification! (0)
    // plus 3 pyramid macros with 1 assert each (0 + 1 + 1 + 1 = 3 exact assertions, not 4 * 3 = 12)
    let ws_root = crate::workspace::find_workspace_root().expect("workspace root");
    let pure_math_mod =
        std::fs::read_to_string(ws_root.join("crates/pure_math/src/pure_math/mod.rs"))
            .expect("read pure_math/mod.rs");
    let pure_math_ast = syn::parse_file(&pure_math_mod).unwrap();
    let mut pure_math_visitor = AstVisitor::default();
    pure_math_visitor.visit_file(&pure_math_ast);
    assert_eq!(pure_math_visitor.verified_funcs, 4);
    assert_eq!(pure_math_visitor.verified_asserts, 3);
    assert_eq!(pure_math_visitor.total_asserts, 3);
}

#[test]
fn test_ast_visitor_separates_cfg_test_and_test_fns_while_keeping_wasm_or_test_in_prod() {
    let code = r#"
        #[cfg(any(target_arch = "wasm32", test))]
        #[verified]
        pub fn wasm_shared_prod() -> usize {
            assert_eq!(2 + 2, 4);
            4
        }

        #[cfg(test)]
        #[verified]
        fn helper_only_for_tests() {
            assert!(true);
            assert!(true);
        }

        #[test]
        #[verified]
        fn unit_test_case() {
            assert!(true);
        }

        #[wasm_bindgen_test]
        fn browser_test_case() {
            assert!(true);
        }

        #[cfg(test)]
        mod tests {
            mod nested_outline;
            #[verified]
            fn inside_test_mod() {
                assert_eq!(1, 1);
            }
        }
    "#;
    let ast = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&ast);
    assert_eq!(visitor.prod_funcs, 1);
    assert_eq!(visitor.verified_prod_funcs, 1);
    assert_eq!(visitor.prod_asserts, 1);
    assert_eq!(visitor.verified_prod_asserts, 1);
    assert_eq!(visitor.test_funcs, 4);
    assert_eq!(visitor.verified_test_funcs, 3);
    assert_eq!(visitor.test_asserts, 5);
    assert_eq!(visitor.verified_test_asserts, 4);
    assert_eq!(visitor.total_funcs, 5);
    assert_eq!(visitor.total_asserts, 6);
    assert!(visitor.active_submodules.is_empty());
}

#[test]
fn test_collect_ast_file_debt_detects_direct_and_cfg_attr_opt_outs_and_ignores_not_opt_out() {
    let ws_root = crate::workspace::find_workspace_root().expect("workspace root");
    let evolution_src = std::fs::read_to_string(
        ws_root.join("crates/pure_math/src/pure_math/analysis/evolution.rs"),
    )
    .expect("read evolution.rs");
    let evolution_ast = syn::parse_file(&evolution_src).unwrap();
    assert_eq!(
        collect_ast_file_debt(&evolution_ast),
        vec!["infrastructure".to_string()]
    );

    let synthetic = format!(
        concat!(
            "#![cfg_attr(any(), {0}(opt_out = \"gui_tool\"))]\n",
            "#[{0}(not_opt_out = \"ignored\")]\n",
            "#[verified_engine::{0}(opt_out = \"Legacy missing assertions\")]\n",
            "pub fn prod_solver_legacy() {{}}\n",
            "#[cfg(test)]\nmod tests {{\n    fn test_only_legacy() {{}}\n}}\n"
        ),
        "verified"
    );
    let ast = syn::parse_file(&synthetic).unwrap();
    assert_eq!(
        collect_ast_file_debt(&ast),
        vec![
            "gui_tool".to_string(),
            "Legacy function call: prod_solver_legacy".to_string(),
            "Legacy missing assertions".to_string(),
        ]
    );
}
