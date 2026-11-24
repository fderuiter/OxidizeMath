#![cfg_attr(any(), verified(opt_out = "tests"))]
//! Unit tests verifying byte, token, AST node, bytecode instruction, and output growth limits.
// [cite:math_commons]

use super::{
    BinaryOp, CompiledExpr, EvalError, Expr, GlslError, MAX_AST_NODES, MAX_COMPILED_INSTRUCTIONS,
    MAX_EXPR_BYTES, MAX_LATEX_OUTPUT_BYTES, ParseError, parse, parse_equation, tokenize,
};
use std::collections::HashMap;

fn build_balanced_binary_string(height: usize) -> String {
    if height == 0 {
        "x".to_string()
    } else {
        let sub = build_balanced_binary_string(height - 1);
        format!("({sub}+{sub})")
    }
}

fn build_balanced_programmatic_expr(height: usize) -> Expr {
    let mut node = Expr::Variable("x".to_string());
    for _ in 0..height {
        node = Expr::binary(BinaryOp::Add, node.clone(), node);
    }
    node
}

#[test]
fn test_tokenize_enforces_byte_and_token_caps() {
    let exact_bytes = format!("x + {}", " ".repeat(MAX_EXPR_BYTES - 5)) + "1";
    assert_eq!(exact_bytes.len(), MAX_EXPR_BYTES);
    assert!(tokenize(&exact_bytes).is_ok());

    let over_bytes = format!("{exact_bytes} ");
    assert_eq!(over_bytes.len(), MAX_EXPR_BYTES + 1);
    let byte_err = tokenize(&over_bytes).expect_err("oversized byte input must fail");
    assert!(byte_err.message.contains("exceeds maximum limit"));
    assert!(matches!(
        parse_equation(&over_bytes),
        Err(ParseError::Lex(_))
    ));

    // 256 identifiers separated by 255 '+' operators = 511 non-EOF tokens <= 512.
    let within_tokens = vec!["x"; 256].join("+");
    assert!(within_tokens.len() < MAX_EXPR_BYTES);
    let ok_tokens = tokenize(&within_tokens).expect("511 non-EOF tokens should succeed");
    assert_eq!(ok_tokens.len(), 512); // 511 + Eof

    // "(x)" repeated 171 times via implicit multiplication = 513 non-EOF tokens > 512, 513 bytes < 2048.
    let over_tokens = "(x)".repeat(171);
    assert!(over_tokens.len() < MAX_EXPR_BYTES);
    let tok_err = tokenize(&over_tokens).expect_err("513 non-EOF tokens must fail");
    assert!(tok_err.message.contains("tokens"));
    assert!(matches!(
        parse_equation(&over_tokens),
        Err(ParseError::Lex(_))
    ));
}

#[test]
fn test_parser_counts_cumulative_ast_nodes_across_balanced_branches() {
    // Balanced binary tree of height 7 has 2^8 - 1 = 255 AST nodes <= MAX_AST_NODES (256),
    // and string length is < MAX_EXPR_BYTES.
    let height_7 = build_balanced_binary_string(7);
    assert!(height_7.len() <= MAX_EXPR_BYTES);
    let expr_255 = parse(&height_7).expect("255-node balanced tree should parse");
    let compiled_255 = CompiledExpr::compile_with_vars(&expr_255, &["x"]).expect("255 insts ok");
    assert_eq!(compiled_255.instruction_count(), 255);
    assert!((compiled_255.eval_1d(1.0, &[]) - 128.0).abs() < 1e-12);

    // Adding "+x" brings total AST nodes to 257 > MAX_AST_NODES (256) at height 8 << 64!
    let nodes_257 = format!("{height_7}+x");
    assert_eq!(
        parse(&nodes_257),
        Err(ParseError::NodeLimitExceeded(MAX_AST_NODES))
    );

    // Depth-4 ternary clamp with implicit products: 283 AST nodes (> 256) in 362 tokens (<= 512).
    let mut c = "clamp(x x,x x,x x)".to_string(); // 10 AST nodes, 11 tokens
    for _ in 0..3 {
        c = format!("clamp({c},{c},{c})");
    }
    assert!(c.len() <= MAX_EXPR_BYTES);
    assert!(tokenize(&c).is_ok());
    assert_eq!(parse(&c), Err(ParseError::NodeLimitExceeded(MAX_AST_NODES)));
}

#[test]
fn test_compiled_expr_and_formatters_reject_oversized_programmatic_trees() {
    // Height-12 balanced programmatic tree: depth 12 <= 64, but 2^13 - 1 = 8,191 AST nodes!
    let wide_tree = build_balanced_programmatic_expr(12);

    assert_eq!(
        CompiledExpr::compile_with_vars(&wide_tree, &["x"]),
        Err(EvalError::InstructionLimitExceeded(
            MAX_COMPILED_INSTRUCTIONS
        ))
    );
    assert_eq!(
        wide_tree.try_to_latex(),
        Err(EvalError::InstructionLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide_tree.to_latex_with_params(&HashMap::new()),
        Err(EvalError::InstructionLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide_tree.to_rust_expr(&HashMap::new()),
        Err(EvalError::InstructionLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide_tree.to_python_expr(&HashMap::new()),
        Err(EvalError::InstructionLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide_tree.to_glsl_scalar(),
        Err(GlslError::NodeLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide_tree.to_glsl_complex(),
        Err(GlslError::NodeLimitExceeded(MAX_AST_NODES))
    );

    // Infallible Expr::to_latex() must remain bounded and elide excess subtrees with \dots.
    let bounded_latex = wide_tree.to_latex();
    assert!(bounded_latex.contains(r"\dots"));
    assert!(bounded_latex.len() <= MAX_LATEX_OUTPUT_BYTES * 2);
}

#[test]
fn test_output_byte_caps_on_long_identifier_programmatic_ast() {
    let huge_ident = "a".repeat(MAX_LATEX_OUTPUT_BYTES + 64);
    let expr = Expr::Variable(huge_ident.clone());
    assert_eq!(
        expr.try_to_latex(),
        Err(EvalError::OutputLimitExceeded(MAX_LATEX_OUTPUT_BYTES))
    );
    let params = HashMap::new();
    let allowed = [huge_ident.as_str()];
    assert!(matches!(
        expr.to_rust_expr_with_vars(&allowed, &params),
        Err(EvalError::OutputLimitExceeded(_))
    ));
    assert!(matches!(
        expr.to_glsl_scalar_with_vars(&allowed),
        Err(GlslError::OutputLimitExceeded(_))
    ));
}
