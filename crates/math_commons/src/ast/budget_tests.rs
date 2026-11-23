#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::latex::MAX_LATEX_OUTPUT_BYTES;
use super::lexer::{MAX_AST_INPUT_BYTES, MAX_AST_TOKENS};
use super::parser::MAX_AST_NODES;
use super::{BinaryOp, Expr, Func, ParseError, parse, parse_equation, tokenize};

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
    assert!(expr_h7.len() <= MAX_AST_INPUT_BYTES);
    let (_, parsed_h7) = parse_equation(&expr_h7).expect("255 AST nodes within 256 budget");
    assert!(parsed_h7.to_latex().len() <= MAX_LATEX_OUTPUT_BYTES);

    // Height 8: 619 bytes, 619 tokens, 511 AST nodes (> MAX_AST_NODES = 256, height 8 <= 64).
    let expr_h8 = build_balanced_expr("z", 8);
    assert!(expr_h8.len() <= MAX_AST_INPUT_BYTES);
    assert!(tokenize(&expr_h8).expect("619 tokens <= 1024").len() <= MAX_AST_TOKENS);
    assert_eq!(
        parse_equation(&expr_h8),
        Err(ParseError::NodeLimitExceeded(MAX_AST_NODES))
    );

    // Exact 256 vs 257 AST node boundary:
    // Height-7 balanced tree has 255 nodes; unary negation brings it to 256, another unary to 257.
    let exact_256 = format!("-({expr_h7})");
    assert!(parse(&exact_256).is_ok(), "exact 256 AST nodes should pass");
    let over_257 = format!("--({expr_h7})");
    assert_eq!(
        parse(&over_257),
        Err(ParseError::NodeLimitExceeded(MAX_AST_NODES))
    );

    // Height 9: 1,242 bytes (<= 2,048), 1,242 tokens (> MAX_AST_TOKENS = 1,024) -> LexError.
    let expr_h9 = build_balanced_expr("z", 9);
    assert!(expr_h9.len() <= MAX_AST_INPUT_BYTES);
    let tok_err = tokenize(&expr_h9).expect_err("1,242 tokens must exceed MAX_AST_TOKENS");
    assert!(tok_err.user_message().contains("tokens"));

    // Byte cap exact boundary: 2,048 bytes of whitespace + "x" > 2,048 bytes -> LexError.
    let ok_bytes = format!("{}x", " ".repeat(MAX_AST_INPUT_BYTES - 1));
    assert_eq!(ok_bytes.len(), MAX_AST_INPUT_BYTES);
    assert!(tokenize(&ok_bytes).is_ok());

    let over_bytes = format!("{}x", " ".repeat(MAX_AST_INPUT_BYTES));
    assert_eq!(over_bytes.len(), MAX_AST_INPUT_BYTES + 1);
    let byte_err = tokenize(&over_bytes).expect_err("2,049 bytes must exceed MAX_AST_INPUT_BYTES");
    assert!(byte_err.user_message().contains("maximum"));
}

#[test]
fn test_budget_error_messages_are_plain_language() {
    let cases = [
        parse(&build_balanced_expr("z", 8)).unwrap_err(),
        parse(&build_balanced_expr("z", 9)).unwrap_err(),
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

    // Programmatic function node with 1,000 arguments
    let many_args = Expr::function(Func::Max, vec![Expr::Number(1.25); 1_000]);
    let latex_args = many_args.to_latex();
    assert!(latex_args.len() <= MAX_LATEX_OUTPUT_BYTES);
    assert!(latex_args.ends_with(r"\right)"));

    // Programmatic oversized variable identifier
    let huge_ident = Expr::Variable("alpha_".repeat(2_000));
    let latex_ident = huge_ident.to_latex();
    assert!(latex_ident.len() <= MAX_LATEX_OUTPUT_BYTES);
}
