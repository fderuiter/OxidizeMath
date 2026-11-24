// [cite:math_commons]

use super::{
    BinaryOp, EvalError, Expr, Func, MAX_AST_NODES, MAX_EXPR_BYTES, MAX_EXPR_TOKENS,
    MAX_LATEX_OUTPUT_BYTES, ParseError, parse, parse_equation, tokenize,
};

fn build_balanced_expr(var: &str, height: usize) -> String {
    let mut cur = var.to_string();
    for level in 1..=height {
        cur = match level % 3 {
            1 => format!("{cur}*{cur}"),
            2 => format!("{cur}+{cur}"),
            _ => format!("({cur})({cur})"),
        };
    }
    cur
}

#[test]
fn test_reproduction_harness_balanced_trees_rejected() {
    let expr_h10 = build_balanced_expr("z", 10);
    assert_eq!(expr_h10.len(), 2_485);
    assert!(
        parse_equation(&expr_h10).is_err(),
        "height-10 balanced binary tree (2,047 AST nodes) must be rejected"
    );

    let expr_h19 = build_balanced_expr("y", 19);
    assert_eq!(expr_h19.len(), 1_273_269);
    assert!(
        parse_equation(&expr_h19).is_err(),
        "height-19 balanced binary tree (1,048,575 AST nodes) must be rejected"
    );
}

#[test]
fn test_three_tier_lexer_and_parser_budget_boundaries() {
    // Height 7: 309 bytes, 309 tokens, 255 AST nodes (<= MAX_AST_NODES = 256) -> succeeds.
    let expr_h7 = build_balanced_expr("z", 7);
    assert!(expr_h7.len() <= MAX_EXPR_BYTES);
    let (_, parsed_h7) = parse_equation(&expr_h7).expect("255 AST nodes within 256 budget");
    assert!(parsed_h7.to_latex().len() <= MAX_LATEX_OUTPUT_BYTES);

    // Exact 256 vs 257 AST node boundary:
    // Height-7 balanced tree has 255 nodes; unary negation brings it to 256, another unary to 257.
    let exact_256 = format!("-({expr_h7})");
    assert!(parse(&exact_256).is_ok(), "exact 256 AST nodes should pass");
    let over_257 = format!("--({expr_h7})");
    assert!(tokenize(&over_257).expect("<= MAX_EXPR_TOKENS").len() <= MAX_EXPR_TOKENS + 1);
    assert_eq!(
        parse(&over_257),
        Err(ParseError::NodeLimitExceeded(MAX_AST_NODES))
    );

    // Three height-6 balanced branches: 470 tokens (<= 512), 383 AST nodes (> 256) -> NodeLimitExceeded.
    let expr_h6 = build_balanced_expr("z", 6);
    let wide_383 = format!("({expr_h6})+({expr_h6})+({expr_h6})");
    assert!(wide_383.len() <= MAX_EXPR_BYTES);
    assert!(tokenize(&wide_383).expect("470 tokens <= 512").len() <= MAX_EXPR_TOKENS + 1);
    assert_eq!(
        parse_equation(&wide_383),
        Err(ParseError::NodeLimitExceeded(MAX_AST_NODES))
    );

    // Height 8: 619 bytes (<= 2,048), 619 tokens (> MAX_EXPR_TOKENS = 512) -> LexError.
    let expr_h8 = build_balanced_expr("z", 8);
    assert!(expr_h8.len() <= MAX_EXPR_BYTES);
    let tok_err = tokenize(&expr_h8).expect_err("619 tokens must exceed MAX_EXPR_TOKENS");
    assert!(tok_err.user_message().contains("tokens"));

    // Byte cap exact boundary: 2,048 bytes succeeds, 2,049 bytes fails in O(1).
    let ok_bytes = format!("{}x", " ".repeat(MAX_EXPR_BYTES - 1));
    assert_eq!(ok_bytes.len(), MAX_EXPR_BYTES);
    assert!(tokenize(&ok_bytes).is_ok());

    let over_bytes = format!("{}x", " ".repeat(MAX_EXPR_BYTES));
    assert_eq!(over_bytes.len(), MAX_EXPR_BYTES + 1);
    let byte_err = tokenize(&over_bytes).expect_err("2,049 bytes must exceed MAX_EXPR_BYTES");
    assert!(byte_err.user_message().contains("maximum"));
}

#[test]
fn test_budget_error_messages_are_plain_language() {
    let expr_h7 = build_balanced_expr("z", 7);
    let cases = [
        parse(&format!("--({expr_h7})")).unwrap_err(),
        parse(&build_balanced_expr("z", 8)).unwrap_err(),
        parse(&build_balanced_expr("z", 10)).unwrap_err(),
    ];
    for err in &cases {
        let msg = err.user_message();
        let ctx_msg = err.user_message_for_input("z + z");
        assert!(!msg.is_empty());
        assert!(!msg.contains("Span"));
        assert!(!msg.contains("<EOF>"));
        assert!(!ctx_msg.contains("Span"));
        assert!(!ctx_msg.contains("<EOF>"));
    }
}

#[test]
fn test_to_latex_output_size_bounded_for_programmatic_trees() {
    // Programmatic balanced binary tree of height 14 (32,767 nodes)
    let mut balanced = Expr::Variable("z".to_string());
    for _ in 0..14 {
        balanced = Expr::binary(BinaryOp::Add, balanced.clone(), balanced);
    }
    let latex_balanced = balanced.to_latex();
    assert!(
        latex_balanced.len() <= MAX_LATEX_OUTPUT_BYTES,
        "balanced tree LaTeX length {} exceeded {}",
        latex_balanced.len(),
        MAX_LATEX_OUTPUT_BYTES
    );
    assert!(latex_balanced.contains(r"\dots"));

    // Programmatic function node with 2,000 arguments: try_to_latex fails at MAX_AST_NODES,
    // while infallible to_latex stays <= MAX_LATEX_OUTPUT_BYTES and closes \right).
    let many_args = Expr::function(Func::Max, vec![Expr::Number(1.25); 2_000]);
    assert_eq!(
        many_args.try_to_latex(),
        Err(EvalError::InstructionLimitExceeded(MAX_AST_NODES))
    );
    let latex_args = many_args.to_latex();
    assert!(latex_args.len() <= MAX_LATEX_OUTPUT_BYTES);
    assert!(latex_args.ends_with(r"\right)"));

    // Programmatic oversized variable identifier (> MAX_LATEX_OUTPUT_BYTES)
    let huge_ident = Expr::Variable("alpha_".repeat(3_500));
    assert_eq!(
        huge_ident.try_to_latex(),
        Err(EvalError::OutputLimitExceeded(MAX_LATEX_OUTPUT_BYTES))
    );
    let latex_ident = huge_ident.to_latex();
    assert!(latex_ident.len() <= MAX_LATEX_OUTPUT_BYTES);
}
