//! Parameter binding and variable validation for symbolic complex WGSL/GLSL shaders.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_commons::ast::{
    is_universal_constant, BinaryOp, Expr, Func, ShaderEmitError, UnaryOp, MAX_EVAL_DEPTH,
};
use std::collections::BTreeMap;

enum BindFrame<'a> {
    Visit { expr: &'a Expr, depth: usize },
    Unary(UnaryOp),
    Binary(BinaryOp),
    Func { func: Func, arity: usize },
}

fn visit_bind_node<'a>(
    expr: &'a Expr,
    depth: usize,
    params: &BTreeMap<String, f64>,
    axes: &[&str],
    work: &mut Vec<BindFrame<'a>>,
    out: &mut Vec<Expr>,
) -> Result<(), String> {
    if depth > MAX_EVAL_DEPTH {
        return Err(format!(
            "Custom complex expression exceeds depth limit {MAX_EVAL_DEPTH}"
        ));
    }
    match expr {
        Expr::Number(v) => out.push(Expr::Number(*v)),
        Expr::Variable(name) => {
            let is_axis = axes.iter().any(|ax| name.eq_ignore_ascii_case(ax));
            let is_reserved =
                is_axis || name.eq_ignore_ascii_case("i") || is_universal_constant(name);
            if !is_reserved {
                if let Some(&val) = params.get(name) {
                    if !val.is_finite() {
                        return Err(format!("Non-finite parameter '{name}'"));
                    }
                    out.push(Expr::Number(val));
                    return Ok(());
                }
            }
            out.push(Expr::Variable(name.clone()));
        }
        Expr::Unary { op, expr: inner } => {
            work.push(BindFrame::Unary(*op));
            work.push(BindFrame::Visit {
                expr: inner,
                depth: depth + 1,
            });
        }
        Expr::Binary { op, left, right } => {
            work.push(BindFrame::Binary(*op));
            work.push(BindFrame::Visit {
                expr: right,
                depth: depth + 1,
            });
            work.push(BindFrame::Visit {
                expr: left,
                depth: depth + 1,
            });
        }
        Expr::Function { func, args } => {
            work.push(BindFrame::Func {
                func: *func,
                arity: args.len(),
            });
            for arg in args.iter().rev() {
                work.push(BindFrame::Visit {
                    expr: arg,
                    depth: depth + 1,
                });
            }
        }
    }
    Ok(())
}

/// Iteratively substitutes synthesized scalar slider parameters into `root` while preserving `axes` and universal constants.
pub fn bind_complex_shader_params(
    root: &Expr,
    params: &BTreeMap<String, f64>,
    axes: &[&str],
) -> Result<Expr, String> {
    let mut work = vec![BindFrame::Visit {
        expr: root,
        depth: 0,
    }];
    let mut out = Vec::new();
    while let Some(frame) = work.pop() {
        match frame {
            BindFrame::Visit { expr, depth } => {
                visit_bind_node(expr, depth, params, axes, &mut work, &mut out)?;
            }
            BindFrame::Unary(op) => {
                let inner = out.pop().ok_or("Shader AST stack underflow")?;
                out.push(Expr::unary(op, inner));
            }
            BindFrame::Binary(op) => {
                let rhs = out.pop().ok_or("Shader AST stack underflow")?;
                let lhs = out.pop().ok_or("Shader AST stack underflow")?;
                out.push(Expr::binary(op, lhs, rhs));
            }
            BindFrame::Func { func, arity } => {
                let split = out
                    .len()
                    .checked_sub(arity)
                    .ok_or("Shader AST stack underflow")?;
                let args = out.split_off(split);
                out.push(Expr::function(func, args));
            }
        }
    }
    out.pop().ok_or_else(|| "Empty shader AST".to_string())
}

/// Iteratively verifies that every variable in `root` is bound by the complex shader harness.
pub fn validate_complex_shader_vars(root: &Expr) -> Result<(), String> {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        match node {
            Expr::Number(v) => {
                if !v.is_finite() {
                    return Err(ShaderEmitError::NonFiniteLiteral.to_string());
                }
            }
            Expr::Variable(name) => {
                let allowed = name.eq_ignore_ascii_case("z")
                    || name.eq_ignore_ascii_case("c")
                    || name == "t"
                    || name.eq_ignore_ascii_case("pi")
                    || name.eq_ignore_ascii_case("e")
                    || name.eq_ignore_ascii_case("tau")
                    || name.eq_ignore_ascii_case("i");
                if !allowed {
                    return Err(format!(
                        "Unsupported variable '{name}' in complex shader (allowed: z, c, t, i, pi, e, tau)"
                    ));
                }
            }
            Expr::Unary { expr, .. } => stack.push(expr),
            Expr::Binary { left, right, .. } => {
                stack.push(right);
                stack.push(left);
            }
            Expr::Function { args, .. } => {
                for arg in args.iter().rev() {
                    stack.push(arg);
                }
            }
        }
    }
    Ok(())
}
