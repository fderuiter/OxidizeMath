//! Native Abstract Syntax Tree (AST) expression parser and dynamic evaluator.
// [cite:math_commons]

#[cfg(feature = "std")]
pub mod codegen;
pub mod compiled;
pub mod eval;
pub mod expr;
pub mod glsl;
mod glsl_helpers;
#[cfg(feature = "std")]
pub mod latex;
pub mod lexer;
pub mod limits;
pub mod parser;
mod parser_diagnostics;
pub mod shader;
mod shader_diagnostics;
pub mod shader_preludes;

#[cfg(all(test, feature = "std"))]
mod arity_tests;
#[cfg(all(test, feature = "std"))]
mod budget_tests;
#[cfg(all(test, feature = "std"))]
mod codegen_tests;
#[cfg(all(test, feature = "std"))]
mod compiled_tests;
#[cfg(all(test, feature = "std"))]
mod glsl_sim_test_eval;
#[cfg(all(test, feature = "std"))]
mod glsl_tests;
#[cfg(all(test, feature = "std"))]
mod limit_tests;
#[cfg(all(test, feature = "std"))]
mod planar_tests;
#[cfg(all(test, feature = "std"))]
mod shader_tests;
#[cfg(all(test, feature = "std"))]
mod tests;

#[cfg(feature = "std")]
pub use codegen::{format_latex_scalar, format_python_float, format_rust_f64};
pub use compiled::{CompiledExpr, Instruction};
pub use eval::{EvalError, MAX_EVAL_DEPTH};
pub use expr::{BinaryOp, Expr, Func, UnaryOp, is_universal_constant};
pub use glsl::{GLSL_COMPLEX_HELPERS, GLSL_SCALAR_HELPERS, GlslError, format_glsl_float};
pub use lexer::{LexError, Span, Token, TokenKind, tokenize};
pub use limits::{
    MAX_AST_NODES, MAX_CODEGEN_OUTPUT_BYTES, MAX_COMPILED_INSTRUCTIONS, MAX_EXPR_BYTES,
    MAX_EXPR_TOKENS, MAX_GLSL_OUTPUT_BYTES, MAX_LATEX_FALLBACK_NODES, MAX_LATEX_OUTPUT_BYTES,
};
pub use num_complex::Complex64;
pub use parser::{MAX_PARSE_DEPTH, ParseError, parse, parse_equation};
pub use shader::{ShaderEmitError, format_shader_f32};
pub use shader_preludes::{
    ShaderTarget, complex_prelude_for, glsl_complex_prelude, glsl_real_prelude, real_prelude_for,
    wgsl_complex_prelude, wgsl_real_prelude,
};
