//! Iterative LaTeX, Rust, and Python code generators for [`Expr`] bounded by [`MAX_EVAL_DEPTH`].
// [cite:math_commons]

#[path = "codegen_funcs.rs"]
mod codegen_funcs;

use super::eval::{EvalError, MAX_EVAL_DEPTH};
use super::expr::{BinaryOp, Expr, UnaryOp};
use super::latex::{format_latex_iterative, write_number};
use codegen_funcs::push_code_func;
use std::collections::HashMap;

pub(super) enum CodeFrame<'a> {
    Eval(&'a Expr, u8, usize),
    Literal(&'static str),
}

#[derive(Clone, Copy)]
pub(super) enum CodeTarget<'a> {
    Rust {
        allowed_args: Option<&'a [&'a str]>,
    },
    Python {
        ode_state_index: bool,
        allowed_args: Option<&'a [&'a str]>,
    },
}

/// Formats a finite `f64` scalar using compact LaTeX integer/decimal notation.
pub fn format_latex_scalar(val: f64) -> Result<String, EvalError> {
    if !val.is_finite() {
        return Err(EvalError::DomainError(format!(
            "Scalar value must be finite, got {val}"
        )));
    }
    let mut out = String::new();
    write_number(&mut out, val);
    Ok(out)
}

/// Formats a finite `f64` scalar as an explicit typed Rust `_f64` literal.
pub fn format_rust_f64(val: f64) -> Result<String, EvalError> {
    let base = format_python_float(val)?;
    Ok(format!("{base}_f64"))
}

/// Formats a finite `f64` scalar as a round-trip Python floating-point literal.
pub fn format_python_float(val: f64) -> Result<String, EvalError> {
    if !val.is_finite() {
        return Err(EvalError::DomainError(format!(
            "Scalar value must be finite, got {val}"
        )));
    }
    let raw = format!("{val}");
    if !raw.contains('.') && !raw.contains('e') && !raw.contains('E') {
        Ok(format!("{raw}.0"))
    } else {
        Ok(raw)
    }
}

impl Expr {
    /// Converts the AST to a calibrated LaTeX string iteratively up to [`MAX_EVAL_DEPTH`].
    ///
    /// Substitutes finite values from `params` while keeping built-in constants
    /// (`pi`, `e`, `tau`) and bound variables (`t`, `y`, `x`, `z`) symbolic.
    pub fn to_latex_with_params(&self, params: &HashMap<String, f64>) -> Result<String, EvalError> {
        format_latex_iterative(self, 0, Some((params, MAX_EVAL_DEPTH)))
    }

    /// Converts the AST to a standalone Rust `f64` expression iteratively up to [`MAX_EVAL_DEPTH`].
    pub fn to_rust_expr(&self, params: &HashMap<String, f64>) -> Result<String, EvalError> {
        emit_code_iterative(self, params, CodeTarget::Rust { allowed_args: None })
    }

    /// Converts the AST to a standalone Rust `f64` expression over explicit bound `vars`.
    pub fn to_rust_expr_with_vars(
        &self,
        vars: &[&str],
        params: &HashMap<String, f64>,
    ) -> Result<String, EvalError> {
        for v in vars {
            validate_rust_ident(v)?;
        }
        emit_code_iterative(
            self,
            params,
            CodeTarget::Rust {
                allowed_args: Some(vars),
            },
        )
    }

    /// Converts the AST to a standalone public Rust `f64` function iteratively up to [`MAX_EVAL_DEPTH`].
    pub fn to_rust_fn(
        &self,
        fn_name: &str,
        args: &[&str],
        params: &HashMap<String, f64>,
    ) -> Result<String, EvalError> {
        validate_rust_ident(fn_name)?;
        for arg in args {
            validate_rust_ident(arg)?;
        }
        let body = emit_code_iterative(
            self,
            params,
            CodeTarget::Rust {
                allowed_args: Some(args),
            },
        )?;
        let sig_args = args
            .iter()
            .map(|a| {
                if a.starts_with('_') || expr_uses_arg(self, a) {
                    format!("{a}: f64")
                } else {
                    format!("_{a}: f64")
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!(
            "pub fn {fn_name}({sig_args}) -> f64 {{\n    {body}\n}}"
        ))
    }

    /// Converts the AST to a `numpy`-compatible Python expression iteratively up to [`MAX_EVAL_DEPTH`].
    pub fn to_python_expr(&self, params: &HashMap<String, f64>) -> Result<String, EvalError> {
        emit_code_iterative(
            self,
            params,
            CodeTarget::Python {
                ode_state_index: false,
                allowed_args: None,
            },
        )
    }

    /// Converts the AST to a `numpy`-compatible Python expression over explicit bound `vars`.
    pub fn to_python_expr_with_vars(
        &self,
        vars: &[&str],
        params: &HashMap<String, f64>,
    ) -> Result<String, EvalError> {
        for v in vars {
            validate_rust_ident(v)?;
        }
        emit_code_iterative(
            self,
            params,
            CodeTarget::Python {
                ode_state_index: false,
                allowed_args: Some(vars),
            },
        )
    }

    /// Converts the AST to a `scipy.integrate.solve_ivp` Python expression (`y` mapped to `y[0]`).
    pub fn to_python_ode_rhs(&self, params: &HashMap<String, f64>) -> Result<String, EvalError> {
        emit_code_iterative(
            self,
            params,
            CodeTarget::Python {
                ode_state_index: true,
                allowed_args: None,
            },
        )
    }
}

pub(super) fn lookup_param(params: &HashMap<String, f64>, name: &str) -> Option<f64> {
    if let Some(&val) = params.get(name) {
        return Some(val);
    }
    params
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, &v)| v)
}

fn expr_uses_arg(root: &Expr, target_arg: &str) -> bool {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node {
            Expr::Number(_) => {}
            Expr::Variable(v) if v.eq_ignore_ascii_case(target_arg) => return true,
            Expr::Variable(_) => {}
            Expr::Unary { expr, .. } => stack.push(expr),
            Expr::Binary { left, right, .. } => {
                stack.push(right);
                stack.push(left);
            }
            Expr::Function { args, .. } => stack.extend(args.iter()),
        }
    }
    false
}

fn validate_rust_ident(ident: &str) -> Result<(), EvalError> {
    let mut chars = ident.chars();
    let Some(first) = chars.next() else {
        return Err(EvalError::DomainError(
            "Rust identifier cannot be empty".to_string(),
        ));
    };
    if !(first.is_ascii_alphabetic() || first == '_')
        || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(EvalError::DomainError(format!(
            "Invalid Rust identifier '{ident}'"
        )));
    }
    Ok(())
}

fn emit_code_iterative<'a>(
    root: &'a Expr,
    params: &HashMap<String, f64>,
    target: CodeTarget<'a>,
) -> Result<String, EvalError> {
    let mut out = String::new();
    let mut stack = vec![CodeFrame::Eval(root, 0, 0)];
    while let Some(frame) = stack.pop() {
        match frame {
            CodeFrame::Literal(lit) => out.push_str(lit),
            CodeFrame::Eval(expr, parent_prec, depth) => {
                if depth > MAX_EVAL_DEPTH {
                    return Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
                }
                step_code_frame(
                    expr,
                    parent_prec,
                    depth,
                    params,
                    target,
                    &mut stack,
                    &mut out,
                )?;
            }
        }
    }
    Ok(out)
}

fn step_code_frame<'a>(
    expr: &'a Expr,
    parent_prec: u8,
    depth: usize,
    params: &HashMap<String, f64>,
    target: CodeTarget<'a>,
    stack: &mut Vec<CodeFrame<'a>>,
    out: &mut String,
) -> Result<(), EvalError> {
    match expr {
        Expr::Number(n) => write_code_scalar(*n, parent_prec, target, out),
        Expr::Variable(name) => write_code_symbol(name, parent_prec, params, target, out),
        Expr::Binary { op, left, right } => {
            push_code_binary(
                *op,
                (left, right),
                parent_prec,
                depth + 1,
                target,
                stack,
                out,
            );
            Ok(())
        }
        Expr::Unary { op, expr } => {
            push_code_unary(*op, expr, parent_prec, depth + 1, target, stack, out);
            Ok(())
        }
        Expr::Function { func, args } => push_code_func(*func, args, depth + 1, target, stack, out),
    }
}

fn write_code_scalar(
    val: f64,
    parent_prec: u8,
    target: CodeTarget<'_>,
    out: &mut String,
) -> Result<(), EvalError> {
    let wrap = val < 0.0 && parent_prec >= 2;
    if wrap {
        out.push('(');
    }
    let formatted = match target {
        CodeTarget::Rust { .. } => format_rust_f64(val)?,
        CodeTarget::Python { .. } => format_python_float(val)?,
    };
    out.push_str(&formatted);
    if wrap {
        out.push(')');
    }
    Ok(())
}

fn write_builtin_code_const(name: &str, target: CodeTarget<'_>, out: &mut String) -> bool {
    if name.eq_ignore_ascii_case("pi") {
        out.push_str(match target {
            CodeTarget::Rust { .. } => "std::f64::consts::PI",
            CodeTarget::Python { .. } => "np.pi",
        });
        true
    } else if name.eq_ignore_ascii_case("e") {
        out.push_str(match target {
            CodeTarget::Rust { .. } => "std::f64::consts::E",
            CodeTarget::Python { .. } => "np.e",
        });
        true
    } else if name.eq_ignore_ascii_case("tau") {
        out.push_str(match target {
            CodeTarget::Rust { .. } => "std::f64::consts::TAU",
            CodeTarget::Python { .. } => "(2.0 * np.pi)",
        });
        true
    } else {
        false
    }
}

fn is_bound_var(name: &str) -> bool {
    name.eq_ignore_ascii_case("t")
        || name.eq_ignore_ascii_case("y")
        || name.eq_ignore_ascii_case("x")
        || name.eq_ignore_ascii_case("z")
}

fn write_code_symbol(
    name: &str,
    parent_prec: u8,
    params: &HashMap<String, f64>,
    target: CodeTarget<'_>,
    out: &mut String,
) -> Result<(), EvalError> {
    if write_builtin_code_const(name, target, out) {
        return Ok(());
    }
    if is_bound_var(name) {
        return write_bound_code_symbol(name, target, out);
    }
    if let Some(val) = lookup_param(params, name) {
        if !val.is_finite() {
            return Err(EvalError::DomainError(format!(
                "Parameter '{name}' must be finite, got {val}"
            )));
        }
        return write_code_scalar(val, parent_prec, target, out);
    }
    let allowed_opt = match target {
        CodeTarget::Rust { allowed_args } | CodeTarget::Python { allowed_args, .. } => allowed_args,
    };
    if let Some(args) = allowed_opt
        && let Some(&arg) = args.iter().find(|a| a.eq_ignore_ascii_case(name))
    {
        out.push_str(arg);
        return Ok(());
    }
    Err(EvalError::UnknownVariable(name.to_string()))
}

fn write_bound_code_symbol(
    name: &str,
    target: CodeTarget<'_>,
    out: &mut String,
) -> Result<(), EvalError> {
    match target {
        CodeTarget::Rust {
            allowed_args: Some(args),
        }
        | CodeTarget::Python {
            ode_state_index: false,
            allowed_args: Some(args),
        } => {
            if let Some(&arg) = args.iter().find(|a| a.eq_ignore_ascii_case(name)) {
                out.push_str(arg);
                Ok(())
            } else {
                Err(EvalError::UnknownVariable(name.to_string()))
            }
        }
        CodeTarget::Rust { allowed_args: None } => {
            out.push_str(&name.to_ascii_lowercase());
            Ok(())
        }
        CodeTarget::Python {
            ode_state_index, ..
        } => {
            if ode_state_index && name.eq_ignore_ascii_case("y") {
                out.push_str("y[0]");
            } else {
                out.push_str(&name.to_ascii_lowercase());
            }
            Ok(())
        }
    }
}

fn push_code_binary<'a>(
    op: BinaryOp,
    (left, right): (&'a Expr, &'a Expr),
    parent_prec: u8,
    next_depth: usize,
    target: CodeTarget<'a>,
    stack: &mut Vec<CodeFrame<'a>>,
    out: &mut String,
) {
    if matches!(target, CodeTarget::Rust { .. }) && op == BinaryOp::Pow {
        stack.push(CodeFrame::Literal(")"));
        stack.push(CodeFrame::Eval(right, 0, next_depth));
        stack.push(CodeFrame::Literal(".powf("));
        stack.push(CodeFrame::Eval(left, 2, next_depth));
        return;
    }
    let (sep, left_prec) = match op {
        BinaryOp::Add => (" + ", 1),
        BinaryOp::Sub => (" - ", 1),
        BinaryOp::Mul => (" * ", 1),
        BinaryOp::Div => (" / ", 1),
        BinaryOp::Pow => (" ** ", 2),
    };
    if parent_prec >= 1 {
        out.push('(');
        stack.push(CodeFrame::Literal(")"));
    }
    stack.push(CodeFrame::Eval(right, 2, next_depth));
    stack.push(CodeFrame::Literal(sep));
    stack.push(CodeFrame::Eval(left, left_prec, next_depth));
}

fn push_code_unary<'a>(
    op: UnaryOp,
    inner: &'a Expr,
    parent_prec: u8,
    next_depth: usize,
    target: CodeTarget<'a>,
    stack: &mut Vec<CodeFrame<'a>>,
    out: &mut String,
) {
    if matches!(target, CodeTarget::Rust { .. }) && op == UnaryOp::Pos {
        stack.push(CodeFrame::Eval(inner, parent_prec, next_depth));
        return;
    }
    if parent_prec >= 2 {
        out.push('(');
        stack.push(CodeFrame::Literal(")"));
    }
    out.push(match op {
        UnaryOp::Pos => '+',
        UnaryOp::Neg => '-',
    });
    stack.push(CodeFrame::Eval(inner, 2, next_depth));
}
