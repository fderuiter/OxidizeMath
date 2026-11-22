//! Iterative WGSL and GLSL shader expression emitters for [`Expr`].
//!
//! Lowers real scalar (`f32` / `float`) and complex (`vec2<f32>` / `vec2`) [`Expr`]
//! syntax trees using an explicit worklist bounded by [`MAX_EVAL_DEPTH`] without
//! call-stack recursion.

use super::eval::MAX_EVAL_DEPTH;
use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use super::shader_preludes::ShaderTarget;
use std::f64::consts::{E, PI, TAU};

/// Errors produced when lowering an [`Expr`] to WGSL or GLSL shader source.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ShaderEmitError {
    /// AST height exceeded [`MAX_EVAL_DEPTH`].
    #[error("Shader transpilation depth limit exceeded (max depth {0})")]
    RecursionLimitExceeded(usize),

    /// Built-in function node was given the wrong number of arguments.
    #[error("Function '{func:?}' expected {expected} arguments, found {found}")]
    WrongArgumentCount {
        /// Function identifier.
        func: Func,
        /// Required argument count.
        expected: usize,
        /// Provided argument count.
        found: usize,
    },

    /// Numeric literal is `NaN` or infinite.
    #[error("Cannot emit non-finite numeric literal in shader source")]
    NonFiniteLiteral,

    /// Variable identifier is invalid or unsafe for shader code generation.
    #[error("Invalid or reserved shader variable identifier '{0}'")]
    InvalidIdentifier(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DomainMode {
    Real,
    Complex,
}

enum EmitFrame<'a> {
    Eval { expr: &'a Expr, depth: usize },
    Raw(&'static str),
}

impl Expr {
    /// Lowers `self` iteratively to a real scalar (`f32`) WGSL expression string.
    ///
    /// Pair with [`super::wgsl_real_prelude`] when the expression uses division,
    /// exponentiation, natural logarithms, or base-10 logarithms.
    pub fn to_wgsl(&self) -> Result<String, ShaderEmitError> {
        emit_shader_expr(self, ShaderTarget::Wgsl, DomainMode::Real)
    }

    /// Lowers `self` iteratively to a real scalar (`float`) GLSL ES 3.00 expression string.
    ///
    /// Pair with [`super::glsl_real_prelude`] when the expression uses division,
    /// exponentiation, natural logarithms, or base-10 logarithms.
    pub fn to_glsl(&self) -> Result<String, ShaderEmitError> {
        emit_shader_expr(self, ShaderTarget::Glsl, DomainMode::Real)
    }

    /// Lowers `self` iteratively to a complex (`vec2<f32>`) WGSL expression string.
    ///
    /// Pair with [`super::wgsl_complex_prelude`] for `c_*` complex arithmetic and elementary helpers.
    pub fn to_wgsl_complex(&self) -> Result<String, ShaderEmitError> {
        emit_shader_expr(self, ShaderTarget::Wgsl, DomainMode::Complex)
    }

    /// Lowers `self` iteratively to a complex (`vec2`) GLSL ES 3.00 expression string.
    ///
    /// Pair with [`super::glsl_complex_prelude`] for `c_*` complex arithmetic and elementary helpers.
    pub fn to_glsl_complex(&self) -> Result<String, ShaderEmitError> {
        emit_shader_expr(self, ShaderTarget::Glsl, DomainMode::Complex)
    }
}

fn emit_shader_expr(
    root: &Expr,
    target: ShaderTarget,
    mode: DomainMode,
) -> Result<String, ShaderEmitError> {
    let mut out = String::with_capacity(64);
    let mut stack = Vec::with_capacity(16);
    stack.push(EmitFrame::Eval {
        expr: root,
        depth: 0,
    });

    while let Some(frame) = stack.pop() {
        match frame {
            EmitFrame::Raw(lit) => out.push_str(lit),
            EmitFrame::Eval { expr, depth } => {
                if depth > MAX_EVAL_DEPTH {
                    return Err(ShaderEmitError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
                }
                emit_node(expr, depth, target, mode, &mut stack, &mut out)?;
            }
        }
    }
    debug_assert!(!out.is_empty());
    Ok(out)
}

fn emit_node<'a>(
    expr: &'a Expr,
    depth: usize,
    target: ShaderTarget,
    mode: DomainMode,
    stack: &mut Vec<EmitFrame<'a>>,
    out: &mut String,
) -> Result<(), ShaderEmitError> {
    match expr {
        Expr::Number(val) => emit_number(*val, target, mode, out),
        Expr::Variable(name) => emit_variable(name, target, mode, out),
        Expr::Unary { op, expr: inner } => {
            push_unary(*op, inner, depth, stack, out);
            Ok(())
        }
        Expr::Binary { op, left, right } => {
            push_binary(*op, left, right, depth, mode, stack, out);
            Ok(())
        }
        Expr::Function { func, args } => push_function(*func, args, depth, mode, stack, out),
    }
}

/// Formats an `f64` value as an explicit `f32` shader float literal with a decimal point.
pub fn format_shader_f32(val: f64) -> Result<String, ShaderEmitError> {
    if !val.is_finite() {
        return Err(ShaderEmitError::NonFiniteLiteral);
    }
    let clamped = val.clamp(-f64::from(f32::MAX), f64::from(f32::MAX));
    let mut f = clamped as f32;
    if f == 0.0 {
        f = 0.0;
    }
    let mut s = format!("{f}");
    if let Some(exp_idx) = s.find(['e', 'E']) {
        if !s[..exp_idx].contains('.') {
            s.insert_str(exp_idx, ".0");
        }
    } else if !s.contains('.') {
        s.push_str(".0");
    }
    Ok(s)
}

fn write_vec2_literal(target: ShaderTarget, re_str: &str, im_str: &str, out: &mut String) {
    match target {
        ShaderTarget::Wgsl => out.push_str("vec2<f32>("),
        ShaderTarget::Glsl => out.push_str("vec2("),
    }
    out.push_str(re_str);
    out.push_str(", ");
    out.push_str(im_str);
    out.push(')');
}

fn emit_number(
    val: f64,
    target: ShaderTarget,
    mode: DomainMode,
    out: &mut String,
) -> Result<(), ShaderEmitError> {
    let lit = format_shader_f32(val)?;
    match mode {
        DomainMode::Real => out.push_str(&lit),
        DomainMode::Complex => write_vec2_literal(target, &lit, "0.0", out),
    }
    Ok(())
}

fn resolve_builtin_const(name: &str, mode: DomainMode) -> Option<(f64, f64)> {
    if name.eq_ignore_ascii_case("pi") {
        Some((PI, 0.0))
    } else if name.eq_ignore_ascii_case("e") {
        Some((E, 0.0))
    } else if name.eq_ignore_ascii_case("tau") {
        Some((TAU, 0.0))
    } else if mode == DomainMode::Complex && name.eq_ignore_ascii_case("i") {
        Some((0.0, 1.0))
    } else {
        None
    }
}

fn is_reserved_shader_word(ident: &str) -> bool {
    matches!(
        ident,
        "fn" | "let"
            | "var"
            | "const"
            | "struct"
            | "return"
            | "discard"
            | "uniform"
            | "in"
            | "out"
            | "inout"
            | "void"
            | "float"
            | "int"
            | "uint"
            | "bool"
            | "vec2"
            | "vec3"
            | "vec4"
            | "f32"
            | "i32"
            | "u32"
            | "main"
            | "fs_main"
            | "mainImage"
    ) || ident.starts_with("om_")
        || ident.starts_with("c_")
        || ident.starts_with("gl_")
}

fn sanitize_shader_ident(raw: &str, mode: DomainMode) -> Result<String, ShaderEmitError> {
    if raw.is_empty() {
        return Err(ShaderEmitError::InvalidIdentifier(String::new()));
    }
    if mode == DomainMode::Complex {
        if raw.eq_ignore_ascii_case("z") {
            return Ok("z".to_string());
        }
        if raw.eq_ignore_ascii_case("c") {
            return Ok("c".to_string());
        }
    }
    let mut ident = String::with_capacity(raw.len());
    for (idx, ch) in raw.chars().enumerate() {
        if idx == 0 {
            if ch.is_ascii_alphabetic() || ch == '_' {
                ident.push(ch);
            } else {
                return Err(ShaderEmitError::InvalidIdentifier(raw.to_string()));
            }
        } else if ch.is_ascii_alphanumeric() || ch == '_' {
            ident.push(ch);
        } else if ch == '\'' {
            ident.push_str("_prime");
        } else {
            return Err(ShaderEmitError::InvalidIdentifier(raw.to_string()));
        }
    }
    if is_reserved_shader_word(&ident) {
        return Err(ShaderEmitError::InvalidIdentifier(raw.to_string()));
    }
    Ok(ident)
}

fn emit_variable(
    name: &str,
    target: ShaderTarget,
    mode: DomainMode,
    out: &mut String,
) -> Result<(), ShaderEmitError> {
    if let Some((re, im)) = resolve_builtin_const(name, mode) {
        let re_s = format_shader_f32(re)?;
        match mode {
            DomainMode::Real => out.push_str(&re_s),
            DomainMode::Complex => {
                let im_s = format_shader_f32(im)?;
                write_vec2_literal(target, &re_s, &im_s, out);
            }
        }
        return Ok(());
    }
    let ident = sanitize_shader_ident(name, mode)?;
    out.push_str(&ident);
    Ok(())
}

fn push_unary<'a>(
    op: UnaryOp,
    inner: &'a Expr,
    depth: usize,
    stack: &mut Vec<EmitFrame<'a>>,
    out: &mut String,
) {
    match op {
        UnaryOp::Pos => stack.push(EmitFrame::Eval {
            expr: inner,
            depth: depth + 1,
        }),
        UnaryOp::Neg => {
            out.push_str("(-");
            stack.push(EmitFrame::Raw(")"));
            stack.push(EmitFrame::Eval {
                expr: inner,
                depth: depth + 1,
            });
        }
    }
}

fn push_binary<'a>(
    op: BinaryOp,
    left: &'a Expr,
    right: &'a Expr,
    depth: usize,
    mode: DomainMode,
    stack: &mut Vec<EmitFrame<'a>>,
    out: &mut String,
) {
    let (prefix, sep, suffix) = match (mode, op) {
        (_, BinaryOp::Add) => ("(", " + ", ")"),
        (_, BinaryOp::Sub) => ("(", " - ", ")"),
        (DomainMode::Real, BinaryOp::Mul) => ("(", " * ", ")"),
        (DomainMode::Real, BinaryOp::Div) => ("om_safe_div(", ", ", ")"),
        (DomainMode::Real, BinaryOp::Pow) => ("om_safe_pow(", ", ", ")"),
        (DomainMode::Complex, BinaryOp::Mul) => ("c_mul(", ", ", ")"),
        (DomainMode::Complex, BinaryOp::Div) => ("c_div(", ", ", ")"),
        (DomainMode::Complex, BinaryOp::Pow) => ("c_pow(", ", ", ")"),
    };
    out.push_str(prefix);
    stack.push(EmitFrame::Raw(suffix));
    stack.push(EmitFrame::Eval {
        expr: right,
        depth: depth + 1,
    });
    stack.push(EmitFrame::Raw(sep));
    stack.push(EmitFrame::Eval {
        expr: left,
        depth: depth + 1,
    });
}

fn scalar_func_wrap(func: Func) -> (&'static str, &'static str) {
    match func {
        Func::Sin => ("sin(", ")"),
        Func::Cos => ("cos(", ")"),
        Func::Tan => ("tan(", ")"),
        Func::Exp => ("exp(clamp(", ", -80.0, 80.0))"),
        Func::Ln => ("om_ln(", ")"),
        Func::Log10 => ("om_log10(", ")"),
        Func::Sqrt => ("sqrt(max(", ", 0.0))"),
        Func::Abs => ("abs(", ")"),
        Func::Sinh => ("sinh(clamp(", ", -80.0, 80.0))"),
        Func::Cosh => ("cosh(clamp(", ", -80.0, 80.0))"),
        Func::Tanh => ("tanh(", ")"),
        Func::Asin => ("asin(clamp(", ", -1.0, 1.0))"),
        Func::Acos => ("acos(clamp(", ", -1.0, 1.0))"),
        Func::Atan => ("atan(", ")"),
    }
}

fn complex_func_wrap(func: Func) -> (&'static str, &'static str) {
    match func {
        Func::Sin => ("c_sin(", ")"),
        Func::Cos => ("c_cos(", ")"),
        Func::Tan => ("c_tan(", ")"),
        Func::Exp => ("c_exp(", ")"),
        Func::Ln => ("c_ln(", ")"),
        Func::Log10 => ("c_log10(", ")"),
        Func::Sqrt => ("c_sqrt(", ")"),
        Func::Abs => ("c_abs(", ")"),
        Func::Sinh => ("c_sinh(", ")"),
        Func::Cosh => ("c_cosh(", ")"),
        Func::Tanh => ("c_tanh(", ")"),
        Func::Asin => ("c_asin(", ")"),
        Func::Acos => ("c_acos(", ")"),
        Func::Atan => ("c_atan(", ")"),
    }
}

fn push_function<'a>(
    func: Func,
    args: &'a [Expr],
    depth: usize,
    mode: DomainMode,
    stack: &mut Vec<EmitFrame<'a>>,
    out: &mut String,
) -> Result<(), ShaderEmitError> {
    if args.len() != 1 {
        return Err(ShaderEmitError::WrongArgumentCount {
            func,
            expected: 1,
            found: args.len(),
        });
    }
    let (prefix, suffix) = match mode {
        DomainMode::Real => scalar_func_wrap(func),
        DomainMode::Complex => complex_func_wrap(func),
    };
    out.push_str(prefix);
    stack.push(EmitFrame::Raw(suffix));
    stack.push(EmitFrame::Eval {
        expr: &args[0],
        depth: depth + 1,
    });
    Ok(())
}
