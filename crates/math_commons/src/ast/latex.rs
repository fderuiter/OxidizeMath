//! LaTeX representation generator for Abstract Syntax Tree expressions.

use super::eval::EvalError;
use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use std::collections::HashMap;
use std::fmt::Write;

enum LatexFrame<'a> {
    Eval {
        expr: &'a Expr,
        parent_prec: u8,
        depth: usize,
    },
    Literal(&'static str),
}

impl Expr {
    /// Converts the AST expression to a formatted symbolic LaTeX string.
    #[must_use]
    pub fn to_latex(&self) -> String {
        format_latex_iterative(self, 0, None).expect("unbounded symbolic LaTeX cannot fail")
    }
}

pub(super) fn format_latex_iterative(
    root: &Expr,
    initial_prec: u8,
    calibrated: Option<(&HashMap<String, f64>, usize)>,
) -> Result<String, EvalError> {
    let mut out = String::new();
    let mut stack = vec![LatexFrame::Eval {
        expr: root,
        parent_prec: initial_prec,
        depth: 0,
    }];
    while let Some(frame) = stack.pop() {
        match frame {
            LatexFrame::Literal(lit) => out.push_str(lit),
            LatexFrame::Eval {
                expr,
                parent_prec,
                depth,
            } => {
                if let Some((_, max_depth)) = calibrated
                    && depth > max_depth
                {
                    return Err(EvalError::RecursionLimitExceeded(max_depth));
                }
                eval_latex_frame(expr, parent_prec, depth, calibrated, &mut stack, &mut out)?;
            }
        }
    }
    Ok(out)
}

fn eval_latex_frame<'a>(
    expr: &'a Expr,
    parent_prec: u8,
    depth: usize,
    calibrated: Option<(&HashMap<String, f64>, usize)>,
    stack: &mut Vec<LatexFrame<'a>>,
    out: &mut String,
) -> Result<(), EvalError> {
    match expr {
        Expr::Number(n) => {
            if calibrated.is_some() && !n.is_finite() {
                return Err(EvalError::DomainError(format!(
                    "Numeric constant must be finite, got {n}"
                )));
            }
            write_signed_number(out, *n, calibrated.is_some() && parent_prec > 0);
        }
        Expr::Variable(name) => write_latex_symbol(out, name, parent_prec, calibrated),
        Expr::Binary { op, left, right } => {
            push_binary_frames(*op, left, right, parent_prec, depth + 1, stack, out);
        }
        Expr::Unary { op, expr } => {
            let wrap = calibrated.is_some() && parent_prec >= 5;
            if wrap {
                out.push_str(r"\left(");
                stack.push(LatexFrame::Literal(r"\right)"));
            }
            out.push(match op {
                UnaryOp::Pos => '+',
                UnaryOp::Neg => '-',
            });
            stack.push(LatexFrame::Eval {
                expr,
                parent_prec: 6,
                depth: depth + 1,
            });
        }
        Expr::Function { func, args } => {
            let expected = func.arity();
            if calibrated.is_some() && args.len() != expected {
                return Err(EvalError::WrongArgumentCount {
                    func: *func,
                    expected,
                    found: args.len(),
                });
            }
            push_function_frames(*func, args, depth + 1, stack, out);
        }
    }
    Ok(())
}

pub(super) fn write_number(out: &mut String, n: f64) {
    if n.fract() == 0.0 && n.abs() < 1e12 {
        write!(out, "{}", n as i64).expect("writing to String cannot fail");
    } else {
        write!(out, "{n}").expect("writing to String cannot fail");
    }
}

fn write_signed_number(out: &mut String, n: f64, wrap_negative: bool) {
    if wrap_negative && n < 0.0 {
        out.push_str(r"\left(");
        write_number(out, n);
        out.push_str(r"\right)");
    } else {
        write_number(out, n);
    }
}

fn write_latex_symbol(
    out: &mut String,
    name: &str,
    parent_prec: u8,
    calibrated: Option<(&HashMap<String, f64>, usize)>,
) {
    let is_reserved = name.eq_ignore_ascii_case("pi")
        || name.eq_ignore_ascii_case("e")
        || name.eq_ignore_ascii_case("tau")
        || name.eq_ignore_ascii_case("t")
        || name.eq_ignore_ascii_case("y")
        || name.eq_ignore_ascii_case("x")
        || name.eq_ignore_ascii_case("z");
    if !is_reserved
        && let Some((params, _)) = calibrated
        && let Some(val) = super::codegen::lookup_param(params, name)
        && val.is_finite()
    {
        write_signed_number(out, val, parent_prec > 0);
        return;
    }
    write_variable(out, name);
}

pub(super) fn write_variable(out: &mut String, name: &str) {
    if name.eq_ignore_ascii_case("pi") {
        out.push_str(r"\pi");
    } else if name.eq_ignore_ascii_case("tau") {
        out.push_str(r"\tau");
    } else if name.contains('_') {
        out.push_str(&name.replace('_', r"\_"));
    } else {
        out.push_str(name);
    }
}

fn push_binary_frames<'a>(
    op: BinaryOp,
    left: &'a Expr,
    right: &'a Expr,
    parent_prec: u8,
    next_depth: usize,
    stack: &mut Vec<LatexFrame<'a>>,
    out: &mut String,
) {
    let (prec, prefix, sep, suffix, left_prec, right_prec) = match op {
        BinaryOp::Add => (1, "", " + ", "", 1, 1),
        BinaryOp::Sub => (1, "", " - ", "", 1, 2),
        BinaryOp::Mul => (3, "", r" \cdot ", "", 3, 3),
        BinaryOp::Div => (4, r"\frac{", "}{", "}", 0, 0),
        BinaryOp::Pow => (5, "{", "}^{", "}", 5, 0),
    };

    if prec < parent_prec && op != BinaryOp::Div {
        out.push_str(r"\left(");
        stack.push(LatexFrame::Literal(r"\right)"));
    }
    if !suffix.is_empty() {
        stack.push(LatexFrame::Literal(suffix));
    }
    out.push_str(prefix);
    stack.push(LatexFrame::Eval {
        expr: right,
        parent_prec: right_prec,
        depth: next_depth,
    });
    stack.push(LatexFrame::Literal(sep));
    stack.push(LatexFrame::Eval {
        expr: left,
        parent_prec: left_prec,
        depth: next_depth,
    });
}

fn push_function_frames<'a>(
    func: Func,
    args: &'a [Expr],
    next_depth: usize,
    stack: &mut Vec<LatexFrame<'a>>,
    out: &mut String,
) {
    match func {
        Func::Sqrt if !args.is_empty() => {
            out.push_str(r"\sqrt{");
            stack.push(LatexFrame::Literal("}"));
            stack.push(LatexFrame::Eval {
                expr: &args[0],
                parent_prec: 0,
                depth: next_depth,
            });
        }
        Func::Abs if !args.is_empty() => {
            out.push_str(r"\left|");
            stack.push(LatexFrame::Literal(r"\right|"));
            stack.push(LatexFrame::Eval {
                expr: &args[0],
                parent_prec: 0,
                depth: next_depth,
            });
        }
        _ => {
            out.push_str(latex_func_name(func));
            out.push_str(r"\left(");
            stack.push(LatexFrame::Literal(r"\right)"));
            for (idx, arg) in args.iter().enumerate().rev() {
                stack.push(LatexFrame::Eval {
                    expr: arg,
                    parent_prec: 0,
                    depth: next_depth,
                });
                if idx > 0 {
                    stack.push(LatexFrame::Literal(", "));
                }
            }
        }
    }
}

fn latex_func_name(func: Func) -> &'static str {
    match func {
        Func::Sin => r"\sin",
        Func::Cos => r"\cos",
        Func::Tan => r"\tan",
        Func::Exp => r"\exp",
        Func::Ln => r"\ln",
        Func::Log10 => r"\log",
        Func::Sinh => r"\sinh",
        Func::Cosh => r"\cosh",
        Func::Tanh => r"\tanh",
        Func::Asin => r"\arcsin",
        Func::Acos => r"\arccos",
        Func::Atan => r"\arctan",
        Func::Min => r"\min",
        Func::Max => r"\max",
        Func::Clamp => r"\mathrm{clamp}",
        Func::Step => r"\mathrm{step}",
        Func::Rect => r"\mathrm{rect}",
        Func::Atan2 => r"\mathrm{atan2}",
        Func::Sqrt | Func::Abs => func.as_str(),
    }
}
