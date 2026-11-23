//! Native Abstract Syntax Tree (AST) expression parser and dynamic evaluator.

pub mod compiled;
pub mod eval;
pub mod expr;
pub mod glsl;
mod glsl_helpers;
pub mod latex;
pub mod lexer;
pub mod parser;

#[cfg(test)]
mod compiled_tests;
#[cfg(test)]
#[cfg(test)]
mod csg_tests;
#[cfg(test)]
mod glsl_sim_test_eval;
#[cfg(test)]
mod glsl_tests;
#[cfg(test)]
mod tests;

pub use compiled::{CompiledExpr, Instruction};
pub use eval::{EvalError, MAX_EVAL_DEPTH};
pub use expr::{BinaryOp, Expr, Func, UnaryOp, is_universal_constant};
pub use glsl::{GLSL_COMPLEX_HELPERS, GLSL_SCALAR_HELPERS, GlslError, format_glsl_float};
pub use lexer::{LexError, Span, Token, TokenKind, tokenize};
pub use num_complex::Complex64;
pub use parser::{MAX_PARSE_DEPTH, ParseError, parse, parse_equation};
