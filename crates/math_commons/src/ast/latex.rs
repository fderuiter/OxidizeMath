//! LaTeX representation generator for Abstract Syntax Tree expressions.
// [cite:math_commons]

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use std::fmt::Write;

/// Maximum `Expr` nodes expanded during LaTeX serialization (matches `MAX_AST_NODES`).
const MAX_LATEX_EVAL_NODES: usize = 256;

/// Soft byte threshold above which further subtree expansions emit `\dots`.
const MAX_LATEX_EXPAND_BYTES: usize = 3_584;

/// Hard UTF-8 byte ceiling on strings returned by [`Expr::to_latex`].
pub(crate) const MAX_LATEX_OUTPUT_BYTES: usize = 4_096;

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
        let mut eval_nodes = 0usize;
        while let Some(frame) = stack.pop() {
            match frame {
                LatexFrame::Literal(lit) => push_bounded_str(&mut out, lit),
                LatexFrame::Eval { expr, parent_prec } => {
                    if eval_nodes >= MAX_LATEX_EVAL_NODES || out.len() >= MAX_LATEX_EXPAND_BYTES {
                        push_bounded_str(&mut out, r"\dots");
                    } else {
                        eval_nodes += 1;
                        eval_latex_frame(expr, parent_prec, &mut stack, &mut out);
                    }
                }
            }
        }
        out
    }
}

fn push_bounded_str(out: &mut String, text: &str) {
    let remaining = MAX_LATEX_OUTPUT_BYTES.saturating_sub(out.len());
    if remaining == 0 || text.is_empty() {
        return;
    }
    if text.len() <= remaining {
        out.push_str(text);
    } else {
        let mut end = remaining;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        out.push_str(&text[..end]);
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
    let mut buf = String::with_capacity(24);
    if n.fract() == 0.0 && n.abs() < 1e12 {
        write!(buf, "{}", n as i64).expect("writing to String cannot fail");
    } else {
        write!(buf, "{n}").expect("writing to String cannot fail");
    }
    push_bounded_str(out, &buf);
}

fn write_variable(out: &mut String, name: &str) {
    if name.eq_ignore_ascii_case("pi") {
        push_bounded_str(out, r"\pi");
    } else if name.eq_ignore_ascii_case("tau") {
        push_bounded_str(out, r"\tau");
    } else {
        for ch in name.chars() {
            if out.len() >= MAX_LATEX_EXPAND_BYTES {
                push_bounded_str(out, r"\dots");
                break;
            }
            if ch == '_' {
                push_bounded_str(out, r"\_");
            } else if out.len() + ch.len_utf8() <= MAX_LATEX_OUTPUT_BYTES {
                out.push(ch);
            } else {
                break;
            }
        }
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
        push_bounded_str(out, r"\left(");
        stack.push(LatexFrame::Literal(r"\right)"));
    }
    if !suffix.is_empty() {
        stack.push(LatexFrame::Literal(suffix));
    }
    push_bounded_str(out, prefix);
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
        UnaryOp::Pos => push_bounded_str(out, "+"),
        UnaryOp::Neg => push_bounded_str(out, "-"),
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
            push_bounded_str(out, r"\sqrt{");
            stack.push(LatexFrame::Literal("}"));
            stack.push(LatexFrame::Eval {
                expr: &args[0],
                parent_prec: 0,
            });
        }
        Func::Abs if !args.is_empty() => {
            push_bounded_str(out, r"\left|");
            stack.push(LatexFrame::Literal(r"\right|"));
            stack.push(LatexFrame::Eval {
                expr: &args[0],
                parent_prec: 0,
            });
        }
        _ => {
            push_bounded_str(out, latex_func_name(func));
            push_bounded_str(out, r"\left(");
            stack.push(LatexFrame::Literal(r"\right)"));
            let capped_len = args.len().min(MAX_LATEX_EVAL_NODES);
            if args.len() > capped_len {
                stack.push(LatexFrame::Literal(r", \dots"));
            }
            for (idx, arg) in args[..capped_len].iter().enumerate().rev() {
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
        Func::Clamp => r"\mathrm{clamp}",
        Func::Step => r"\mathrm{step}",
        Func::Rect => r"\mathrm{rect}",
        Func::Atan2 => r"\mathrm{atan2}",
        Func::Sqrt | Func::Abs => func.as_str(),
    }
}
