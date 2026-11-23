//! Function-call Rust and Python AST code generation helpers supporting unary, binary, and ternary [`Func`]s.
// [cite:math_commons]

use super::{CodeFrame, CodeTarget};
use crate::ast::eval::EvalError;
use crate::ast::expr::{Expr, Func};

pub(super) fn push_code_func<'a>(
    func: Func,
    args: &'a [Expr],
    next_depth: usize,
    target: CodeTarget<'a>,
    stack: &mut Vec<CodeFrame<'a>>,
    out: &mut String,
) -> Result<(), EvalError> {
    let expected = func.arity();
    if args.len() != expected {
        return Err(EvalError::WrongArgumentCount {
            func,
            expected,
            found: args.len(),
        });
    }
    match target {
        CodeTarget::Rust { .. } => push_rust_func(func, args, next_depth, stack, out),
        CodeTarget::Python { .. } => push_python_func(func, args, next_depth, stack, out),
    }
    Ok(())
}

fn push_rust_func<'a>(
    func: Func,
    args: &'a [Expr],
    next_depth: usize,
    stack: &mut Vec<CodeFrame<'a>>,
    out: &mut String,
) {
    match func {
        Func::Step => {
            out.push_str("{ let _v = ");
            stack.push(CodeFrame::Literal(
                "; if _v.is_nan() { f64::NAN } else if _v < 0.0_f64 { 0.0_f64 } else { 1.0_f64 } }",
            ));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
        Func::Rect => {
            out.push_str("{ let _v = ");
            stack.push(CodeFrame::Literal(
                "; if _v.is_nan() { f64::NAN } else if _v.abs() <= 0.5_f64 { 1.0_f64 } else { 0.0_f64 } }",
            ));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
        Func::Min | Func::Max | Func::Atan2 => {
            let method = match func {
                Func::Min => ".min(",
                Func::Max => ".max(",
                Func::Atan2 => ".atan2(",
                _ => unreachable!(),
            };
            stack.push(CodeFrame::Literal(")"));
            stack.push(CodeFrame::Eval(&args[1], 0, next_depth));
            stack.push(CodeFrame::Literal(method));
            stack.push(CodeFrame::Eval(&args[0], 2, next_depth));
        }
        Func::Clamp => {
            out.push_str("{ let (_x, _lo, _hi) = (");
            stack.push(CodeFrame::Literal(
                "); if _lo > _hi { f64::NAN } else { _x.max(_lo).min(_hi) } }",
            ));
            stack.push(CodeFrame::Eval(&args[2], 0, next_depth));
            stack.push(CodeFrame::Literal(", "));
            stack.push(CodeFrame::Eval(&args[1], 0, next_depth));
            stack.push(CodeFrame::Literal(", "));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
        _ => {
            stack.push(CodeFrame::Literal(rust_unary_method_suffix(func)));
            stack.push(CodeFrame::Eval(&args[0], 2, next_depth));
        }
    }
}

fn push_python_func<'a>(
    func: Func,
    args: &'a [Expr],
    next_depth: usize,
    stack: &mut Vec<CodeFrame<'a>>,
    out: &mut String,
) {
    match func {
        Func::Step => {
            out.push_str("np.heaviside(");
            stack.push(CodeFrame::Literal(", 1.0)"));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
        Func::Rect => {
            out.push_str("np.heaviside(0.5 - np.abs(");
            stack.push(CodeFrame::Literal("), 1.0)"));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
        Func::Min | Func::Max | Func::Atan2 => {
            let prefix = match func {
                Func::Min => "np.minimum(",
                Func::Max => "np.maximum(",
                Func::Atan2 => "np.arctan2(",
                _ => unreachable!(),
            };
            out.push_str(prefix);
            stack.push(CodeFrame::Literal(")"));
            stack.push(CodeFrame::Eval(&args[1], 0, next_depth));
            stack.push(CodeFrame::Literal(", "));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
        Func::Clamp => {
            out.push_str("(lambda _x, _lo, _hi: np.where(_lo > _hi, np.nan, np.minimum(np.maximum(_x, _lo), _hi)))(");
            stack.push(CodeFrame::Literal(")"));
            stack.push(CodeFrame::Eval(&args[2], 0, next_depth));
            stack.push(CodeFrame::Literal(", "));
            stack.push(CodeFrame::Eval(&args[1], 0, next_depth));
            stack.push(CodeFrame::Literal(", "));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
        _ => {
            out.push_str(python_unary_func_prefix(func));
            stack.push(CodeFrame::Literal(")"));
            stack.push(CodeFrame::Eval(&args[0], 0, next_depth));
        }
    }
}

const fn rust_unary_method_suffix(func: Func) -> &'static str {
    match func {
        Func::Sin => ".sin()",
        Func::Cos => ".cos()",
        Func::Tan => ".tan()",
        Func::Exp => ".exp()",
        Func::Ln => ".ln()",
        Func::Log10 => ".log10()",
        Func::Sqrt => ".sqrt()",
        Func::Abs => ".abs()",
        Func::Sinh => ".sinh()",
        Func::Cosh => ".cosh()",
        Func::Tanh => ".tanh()",
        Func::Asin => ".asin()",
        Func::Acos => ".acos()",
        Func::Atan => ".atan()",
        Func::Min | Func::Max | Func::Clamp | Func::Step | Func::Rect | Func::Atan2 => "",
    }
}

const fn python_unary_func_prefix(func: Func) -> &'static str {
    match func {
        Func::Sin => "np.sin(",
        Func::Cos => "np.cos(",
        Func::Tan => "np.tan(",
        Func::Exp => "np.exp(",
        Func::Ln => "np.log(",
        Func::Log10 => "np.log10(",
        Func::Sqrt => "np.sqrt(",
        Func::Abs => "np.abs(",
        Func::Sinh => "np.sinh(",
        Func::Cosh => "np.cosh(",
        Func::Tanh => "np.tanh(",
        Func::Asin => "np.arcsin(",
        Func::Acos => "np.arccos(",
        Func::Atan => "np.arctan(",
        Func::Min | Func::Max | Func::Clamp | Func::Step | Func::Rect | Func::Atan2 => "",
    }
}
