//! Native Abstract Syntax Tree (AST) expression parser and dynamic evaluator.

pub mod codegen;
pub mod eval;
pub mod expr;
pub mod latex;
pub mod lexer;
pub mod parser;

#[cfg(test)]
mod codegen_tests;
#[cfg(test)]
mod tests;

pub use codegen::{format_latex_scalar, format_python_float, format_rust_f64};
pub use eval::{EvalError, MAX_EVAL_DEPTH};
pub use expr::{BinaryOp, Expr, Func, UnaryOp};
pub use lexer::{LexError, Span, Token, TokenKind, tokenize};
pub use num_complex::Complex64;
pub use parser::{MAX_PARSE_DEPTH, ParseError, parse, parse_equation};
