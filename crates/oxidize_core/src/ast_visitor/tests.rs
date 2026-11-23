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
    "#;
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert_eq!(visitor.total_funcs, 2);
    assert_eq!(visitor.verified_funcs, 2);
    assert_eq!(visitor.semantic_integrity_funcs, 2);
    assert_eq!(visitor.verified_asserts, 2);
    assert!(!visitor.opted_out);
}

#[test]
fn test_ast_visitor_ignores_macro_examples_in_doc_comments_while_keeping_citations() {
    let code = r#"
        /// Example usage:
        /// ```rust
        /// use verified_engine_macros::embed_theory;
        /// #[embed_theory("papers/mmwave_radiotherapy_setup.tex", label = "eq:range")]
        /// #[verified]
        /// fn demo() {}
        /// ```
        /// Reference: [cite:dijkstra]
        pub fn documented_helper() {
            assert!(true);
        }
    "#;
    let file = syn::parse_file(code).unwrap();
    let mut visitor = AstVisitor::default();
    visitor.visit_file(&file);

    assert_eq!(visitor.total_funcs, 1);
    assert_eq!(visitor.verified_funcs, 0);
    assert_eq!(visitor.semantic_integrity_funcs, 1);
    assert_eq!(visitor.verified_modules, vec!["dijkstra".to_string()]);
}
