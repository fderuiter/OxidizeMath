//! Native Abstract Syntax Tree (AST) expression parser and dynamic evaluator.

pub mod compiled;
pub mod eval;
pub mod expr;
pub mod latex;
pub mod lexer;
pub mod parser;
pub mod shader;
pub mod shader_preludes;

#[cfg(test)]
mod compiled_tests;
#[cfg(test)]
mod shader_tests;
#[cfg(test)]
mod tests;

pub use compiled::{CompiledExpr, Instruction};
pub use eval::{EvalError, MAX_EVAL_DEPTH};
pub use expr::{BinaryOp, Expr, Func, UnaryOp, is_universal_constant};
pub use lexer::{LexError, Span, Token, TokenKind, tokenize};
pub use num_complex::Complex64;
pub use parser::{MAX_PARSE_DEPTH, ParseError, parse, parse_equation};
pub use shader::{ShaderEmitError, format_shader_f32};
pub use shader_preludes::{
    ShaderTarget, complex_prelude_for, glsl_complex_prelude, glsl_real_prelude, real_prelude_for,
    wgsl_complex_prelude, wgsl_real_prelude,
};
