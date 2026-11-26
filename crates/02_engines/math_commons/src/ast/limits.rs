//! Explicit complexity and output bounds for the symbolic AST pipeline.
//!
//! These constants enforce linear-time, bounded-memory lexical scanning,
//! recursive-descent parsing, stack-machine bytecode compilation, and
//! iterative LaTeX / GLSL / source-code formatting.

/// Maximum UTF-8 input length accepted by [`super::lexer::tokenize`] (`2 KiB`,
/// matching `egui_plot`'s bounded math label cap).
pub const MAX_EXPR_BYTES: usize = 2_048;

/// Maximum number of non-EOF lexical tokens emitted by [`super::lexer::tokenize`].
pub const MAX_EXPR_TOKENS: usize = 512;

/// Maximum cumulative AST node count across all expression branches constructed
/// by [`super::parser::parse`] and [`super::parser::parse_equation`].
pub const MAX_AST_NODES: usize = 256;

/// Maximum stack-machine bytecode instruction count emitted by
/// [`super::compiled::CompiledExpr`].
pub const MAX_COMPILED_INSTRUCTIONS: usize = 256;

/// Maximum UTF-8 byte length permitted for fallible LaTeX formatting
/// ([`super::Expr::try_to_latex`] and [`super::Expr::to_latex_with_params`])
/// and infallible [`super::Expr::to_latex`] body emission.
pub const MAX_LATEX_OUTPUT_BYTES: usize = 16_384;

/// Maximum `Expr` nodes expanded by infallible [`super::Expr::to_latex`] on
/// programmatically constructed trees before eliding deeper subtrees with `\dots`.
pub const MAX_LATEX_FALLBACK_NODES: usize = 1_024;

/// Maximum UTF-8 byte length permitted for emitted GLSL shader expressions.
pub const MAX_GLSL_OUTPUT_BYTES: usize = 16_384;

/// Maximum UTF-8 byte length permitted for emitted Rust or Python source expressions.
pub const MAX_CODEGEN_OUTPUT_BYTES: usize = 16_384;
