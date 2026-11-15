//! Native Abstract Syntax Tree (AST) expression parser and dynamic evaluator.

pub mod eval;
pub mod expr;
pub mod latex;
pub mod lexer;
pub mod parser;

#[cfg(test)]
mod tests;

pub use eval::{EvalError, MAX_EVAL_DEPTH};
pub use expr::{BinaryOp, Expr, Func, UnaryOp};
pub use lexer::{LexError, Span, Token, TokenKind, tokenize};
pub use parser::{MAX_PARSE_DEPTH, ParseError, parse, parse_equation};
