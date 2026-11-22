//! LaTeX representation generator for Abstract Syntax Tree expressions.

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use std::fmt::Write;

enum LatexFrame<'a> {
    Eval { expr: &'a Expr, parent_prec: u8 },
    Literal(&'static str),
}

impl Expr {
    /// Converts the AST expression to a formatted LaTeX string.
    #[must_use]
    pub fn to_latex(&self) -> String {
        self.to_latex_prec(0)
    }

    fn to_latex_prec(&self, parent_prec: u8) -> String {
        let mut out = String::new();
        let mut stack = vec![LatexFrame::Eval {
            expr: self,
            parent_prec,
        }];
        while let Some(frame) = stack.pop() {
            match frame {
                LatexFrame::Literal(lit) => out.push_str(lit),
                LatexFrame::Eval { expr, parent_prec } => {
                    eval_latex_frame(expr, parent_prec, &mut stack, &mut out);
                }
            }
        }
        out
    }
}

fn eval_latex_frame<'a>(
    expr: &'a Expr,
    parent_prec: u8,
    stack: &mut Vec<LatexFrame<'a>>,
    out: &mut String,
) {
    match expr {
        Expr::Number(n) => write_number(out, *n),
        Expr::Variable(name) => write_variable(out, name),
        Expr::Binary { op, left, right } => {
            push_binary_frames(*op, left, right, parent_prec, stack, out);
        }
        Expr::Unary { op, expr } => {
            push_unary_frames(*op, expr, stack, out);
        }
        Expr::Function { func, args } => {
            push_function_frames(*func, args, stack, out);
        }
    }
}

fn write_number(out: &mut String, n: f64) {
    if n.fract() == 0.0 && n.abs() < 1e12 {
        write!(out, "{}", n as i64).expect("writing to String cannot fail");
    } else {
        write!(out, "{n}").expect("writing to String cannot fail");
    }
}

fn write_variable(out: &mut String, name: &str) {
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

    let wrap_parens = prec < parent_prec && op != BinaryOp::Div;
    if wrap_parens {
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
    });
    stack.push(LatexFrame::Literal(sep));
    stack.push(LatexFrame::Eval {
        expr: left,
        parent_prec: left_prec,
    });
}

fn push_unary_frames<'a>(
    op: UnaryOp,
    expr: &'a Expr,
    stack: &mut Vec<LatexFrame<'a>>,
    out: &mut String,
) {
    match op {
        UnaryOp::Pos => out.push('+'),
        UnaryOp::Neg => out.push('-'),
    }
    stack.push(LatexFrame::Eval {
        expr,
        parent_prec: 6,
    });
}

fn push_function_frames<'a>(
    func: Func,
    args: &'a [Expr],
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
            });
        }
        Func::Abs if !args.is_empty() => {
            out.push_str(r"\left|");
            stack.push(LatexFrame::Literal(r"\right|"));
            stack.push(LatexFrame::Eval {
                expr: &args[0],
                parent_prec: 0,
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
        Func::Atan2 => r"\mathrm{atan2}",
        _ => func.as_str(),
    }
}
