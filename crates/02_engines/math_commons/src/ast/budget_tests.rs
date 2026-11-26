//! Regression and boundary tests for AST input, node, instruction, LaTeX, and shader caps.
// [cite:math_commons]

use super::{
    BinaryOp, CompiledExpr, EvalError, Expr, Func, MAX_AST_NODES, MAX_COMPILED_INSTRUCTIONS,
    MAX_EXPR_BYTES, MAX_GLSL_OUTPUT_BYTES, MAX_LATEX_OUTPUT_BYTES, ParseError, ShaderEmitError,
    ShaderTarget, parse, parse_equation, tokenize,
};

fn build_balanced_expr_str(var: &str, height: usize) -> String {
    let mut s = var.to_string();
    for _ in 0..height {
        s = format!("({s}+{s})");
    }
    s
}

fn build_programmatic_balanced_expr(height: usize) -> Expr {
    let mut node = Expr::Variable("x".to_string());
    for _ in 0..height {
        node = Expr::binary(BinaryOp::Add, node.clone(), node);
    }
    node
}

#[test]
fn test_parser_rejects_balanced_exponential_trees_linearly() {
    // Height-10 balanced tree (< 16 KiB) exceeds MAX_EXPR_BYTES (2,048) and fails O(1).
    let h10 = build_balanced_expr_str("x", 10);
    assert!(h10.len() > MAX_EXPR_BYTES && h10.len() < 16_384);
    assert!(matches!(parse(&h10), Err(ParseError::Lex(_))));

    // Height-19 balanced tree (> 1 MiB, > 1,000,000 potential nodes) rejects O(1) at byte pre-check.
    let h19 = build_balanced_expr_str("x", 19);
    assert!(h19.len() > 1_000_000);
    let err_h19 = parse_equation(&h19).expect_err("height-19 balanced tree must fail");
    assert!(matches!(err_h19, ParseError::Lex(_)));
    assert!(err_h19.user_message().contains(&MAX_EXPR_BYTES.to_string()));

    // Height-8 balanced tree with compact variable: < 2,048 bytes, 619 tokens > MAX_EXPR_TOKENS (512).
    let mut compact_h8 = "x+x".to_string();
    for _ in 0..7 {
        compact_h8 = format!("({compact_h8})+({compact_h8})");
    }
    assert!(compact_h8.len() <= MAX_EXPR_BYTES);
    assert!(matches!(parse(&compact_h8), Err(ParseError::Lex(_))));

    // 3-branch sum of height-5 implicit-product trees: <= 2,048 bytes, 473 tokens <= 512, 383 AST nodes > 256.
    let h5_prod = build_balanced_expr_str("x x", 5);
    let three_branch_383 = format!("{h5_prod}+{h5_prod}+{h5_prod}");
    assert!(three_branch_383.len() <= MAX_EXPR_BYTES);
    assert!(tokenize(&three_branch_383).is_ok());
    assert_eq!(
        parse(&three_branch_383),
        Err(ParseError::NodeLimitExceeded(MAX_AST_NODES))
    );
}

#[test]
fn test_exact_byte_node_and_instruction_boundaries() {
    // Exact 2,048 bytes succeeds; 2,049 bytes fails.
    let pad_ok = " ".repeat(MAX_EXPR_BYTES - 3);
    let exact_2048 = format!("x+{pad_ok}1");
    assert_eq!(exact_2048.len(), MAX_EXPR_BYTES);
    assert!(parse(&exact_2048).is_ok());

    let over_2049 = format!("{exact_2048} ");
    assert_eq!(over_2049.len(), MAX_EXPR_BYTES + 1);
    assert!(matches!(parse(&over_2049), Err(ParseError::Lex(_))));

    // Exact 256 AST nodes (unary '-' wrapping a 255-node height-7 balanced binary tree) succeeds;
    // 257 AST nodes fails in both parser and bytecode compiler.
    let h7 = build_balanced_expr_str("x", 7);
    let exact_256 = format!("-{h7}");
    assert!(exact_256.len() <= MAX_EXPR_BYTES);
    let expr_256 = parse(&exact_256).expect("256-node expression must parse");
    let compiled_256 =
        CompiledExpr::compile_with_vars(&expr_256, &["x"]).expect("256 instructions must compile");
    assert_eq!(compiled_256.instruction_count(), MAX_COMPILED_INSTRUCTIONS);
    assert!((compiled_256.eval_1d(1.0, &[]) + 128.0).abs() < 1e-12);

    let over_257 = format!("{h7}+x");
    assert_eq!(
        parse(&over_257),
        Err(ParseError::NodeLimitExceeded(MAX_AST_NODES))
    );

    let prog_257 = Expr::binary(
        BinaryOp::Add,
        build_programmatic_balanced_expr(7),
        Expr::Number(1.0),
    );
    assert_eq!(
        CompiledExpr::compile_with_vars(&prog_257, &["x"]),
        Err(EvalError::InstructionLimitExceeded(
            MAX_COMPILED_INSTRUCTIONS
        ))
    );
}

#[test]
fn test_plain_language_diagnostics_have_one_indexed_positions_and_no_raw_spans() {
    let lex_err = tokenize("x + @").expect_err("invalid character '@'");
    let lex_msg = lex_err.user_message();
    assert!(lex_msg.contains("character 5"), "got: {lex_msg}");
    assert!(!lex_msg.contains("Span") && !lex_msg.contains("<EOF>"));

    let eof_input = "sin(x + ";
    let eof_err = parse(eof_input).expect_err("incomplete input must fail");
    let eof_msg = eof_err.user_message_for_input(eof_input);
    assert!(eof_msg.contains("character 8"), "got: {eof_msg}");
    assert!(!eof_msg.contains("Span") && !eof_msg.contains("<EOF>"));

    let bad_tok = parse("x + )").expect_err("stray ')' must fail");
    let tok_msg = bad_tok.user_message();
    assert!(tok_msg.contains("At character 5"), "got: {tok_msg}");
    assert!(!tok_msg.contains("Span") && !tok_msg.contains("<EOF>"));
}

#[test]
fn test_infallible_to_latex_strictly_obeys_max_latex_output_bytes() {
    // 1. Height-14 balanced binary tree (32,767 nodes) clamps strictly to <= MAX_LATEX_OUTPUT_BYTES.
    let tree_h14 = build_programmatic_balanced_expr(14);
    let latex_h14 = tree_h14.to_latex();
    assert!(latex_h14.contains(r"\dots"));
    assert!(
        latex_h14.len() <= MAX_LATEX_OUTPUT_BYTES,
        "height-14 to_latex() len {} exceeded {MAX_LATEX_OUTPUT_BYTES}",
        latex_h14.len()
    );

    // 2. Programmatic multi-argument function with 2,000 arguments caps frame expansion and bytes.
    let many_args = Expr::function(Func::Max, vec![Expr::Variable("x_1".to_string()); 2_000]);
    assert_eq!(
        many_args.try_to_latex(),
        Err(EvalError::InstructionLimitExceeded(MAX_AST_NODES))
    );
    let latex_args = many_args.to_latex();
    assert!(latex_args.contains(r"\dots"));
    assert!(latex_args.ends_with(r"\right)"));
    assert!(latex_args.len() <= MAX_LATEX_OUTPUT_BYTES);

    // 3. Oversized identifier with underscores streams safely without exceeding MAX_LATEX_OUTPUT_BYTES.
    let underscore_ident = "alpha_".repeat(4_000);
    let ident_expr = Expr::Variable(underscore_ident);
    assert_eq!(
        ident_expr.try_to_latex(),
        Err(EvalError::OutputLimitExceeded(MAX_LATEX_OUTPUT_BYTES))
    );
    let ident_latex = ident_expr.to_latex();
    assert!(ident_latex.len() <= MAX_LATEX_OUTPUT_BYTES);
}

#[test]
fn test_shader_emitters_enforce_node_and_output_caps() {
    let h7 = build_balanced_expr_str("z", 7);
    let ok_256 = parse(&format!("-{h7}")).expect("256-node expression");
    assert!(ok_256.to_wgsl().is_ok());
    assert!(ok_256.to_glsl().is_ok());
    assert!(ok_256.to_wgsl_complex().is_ok());
    assert!(ok_256.to_shader_complex(ShaderTarget::Glsl).is_ok());

    let wide = build_programmatic_balanced_expr(9);
    assert_eq!(
        wide.to_wgsl(),
        Err(ShaderEmitError::NodeLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide.to_glsl(),
        Err(ShaderEmitError::NodeLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide.to_wgsl_complex(),
        Err(ShaderEmitError::NodeLimitExceeded(MAX_AST_NODES))
    );
    assert_eq!(
        wide.to_shader_complex(ShaderTarget::Glsl),
        Err(ShaderEmitError::NodeLimitExceeded(MAX_AST_NODES))
    );

    let long_var = Expr::Variable("a".repeat(MAX_GLSL_OUTPUT_BYTES + 32));
    assert_eq!(
        long_var.to_wgsl(),
        Err(ShaderEmitError::OutputLimitExceeded(MAX_GLSL_OUTPUT_BYTES))
    );
}
