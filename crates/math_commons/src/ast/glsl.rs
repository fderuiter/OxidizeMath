//! Iterative AST-to-GLSL ES 3.00 transpiler for scalar (`float`) and complex (`vec2`) expressions.

use super::eval::MAX_EVAL_DEPTH;
use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use super::glsl_helpers::is_valid_glsl_ident;
pub use super::glsl_helpers::{GLSL_COMPLEX_HELPERS, GLSL_SCALAR_HELPERS};
use std::f64::consts::{E, PI, TAU};
use std::fmt::Write;

/// Errors that can occur when transpiling an [`Expr`] tree to GLSL source code.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GlslError {
    /// AST subtree height exceeded [`MAX_EVAL_DEPTH`].
    #[error("GLSL transpilation depth limit exceeded (max depth {0})")]
    DepthLimitExceeded(usize),

    /// Built-in function was invoked with an unsupported argument count.
    #[error("Function '{func:?}' expected {expected} arguments, found {found}")]
    WrongArgumentCount {
        /// Built-in function variant.
        func: Func,
        /// Required arity.
        expected: usize,
        /// Provided argument count.
        found: usize,
    },

    /// Numeric constant is `NaN`, infinite, or outside IEEE-754 `f32` range.
    #[error("Non-finite or out-of-range GLSL float literal: {0}")]
    NonFiniteLiteral(f64),

    /// Variable identifier is reserved, unsafe, or unbound in the target shader scope.
    #[error("Invalid or unbound GLSL variable identifier '{0}'")]
    InvalidIdentifier(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GlslMode {
    Scalar,
    Complex,
}

enum GlslFrame<'a> {
    Eval {
        expr: &'a Expr,
        parent_prec: u8,
        depth: usize,
    },
    Literal(&'static str),
}

impl Expr {
    /// Transpiles the expression to a scalar (`float`) GLSL ES 3.00 expression iteratively.
    pub fn to_glsl_scalar(&self) -> Result<String, GlslError> {
        emit_glsl_iterative(self, GlslMode::Scalar, None)
    }

    /// Transpiles the expression to a scalar (`float`) GLSL expression restricted to `allowed_vars`.
    pub fn to_glsl_scalar_with_vars(&self, allowed_vars: &[&str]) -> Result<String, GlslError> {
        emit_glsl_iterative(self, GlslMode::Scalar, Some(allowed_vars))
    }

    /// Transpiles the expression to a complex (`vec2`) GLSL ES 3.00 expression iteratively.
    pub fn to_glsl_complex(&self) -> Result<String, GlslError> {
        emit_glsl_iterative(self, GlslMode::Complex, None)
    }

    /// Transpiles the expression to a complex (`vec2`) GLSL expression restricted to `allowed_vars`.
    pub fn to_glsl_complex_with_vars(&self, allowed_vars: &[&str]) -> Result<String, GlslError> {
        emit_glsl_iterative(self, GlslMode::Complex, Some(allowed_vars))
    }
}

struct GlslEmitter<'a> {
    mode: GlslMode,
    allowed_vars: Option<&'a [&'a str]>,
    stack: Vec<GlslFrame<'a>>,
    out: String,
}

fn emit_glsl_iterative<'a>(
    root: &'a Expr,
    mode: GlslMode,
    allowed_vars: Option<&'a [&'a str]>,
) -> Result<String, GlslError> {
    let mut emitter = GlslEmitter {
        mode,
        allowed_vars,
        stack: vec![GlslFrame::Eval {
            expr: root,
            parent_prec: 0,
            depth: 0,
        }],
        out: String::new(),
    };
    while let Some(frame) = emitter.stack.pop() {
        match frame {
            GlslFrame::Literal(lit) => emitter.out.push_str(lit),
            GlslFrame::Eval {
                expr,
                parent_prec,
                depth,
            } => {
                if depth > MAX_EVAL_DEPTH {
                    return Err(GlslError::DepthLimitExceeded(MAX_EVAL_DEPTH));
                }
                emitter.step_frame(expr, parent_prec, depth)?;
            }
        }
    }
    Ok(emitter.out)
}

impl<'a> GlslEmitter<'a> {
    fn step_frame(
        &mut self,
        expr: &'a Expr,
        parent_prec: u8,
        depth: usize,
    ) -> Result<(), GlslError> {
        match expr {
            Expr::Number(n) => write_glsl_literal(&mut self.out, *n, parent_prec, self.mode),
            Expr::Variable(name) => {
                write_glsl_variable(&mut self.out, name, self.mode, self.allowed_vars)
            }
            Expr::Binary { op, left, right } => {
                self.push_binary(*op, left, right, parent_prec, depth + 1);
                Ok(())
            }
            Expr::Unary { op, expr } => {
                self.push_unary(*op, expr, parent_prec, depth + 1);
                Ok(())
            }
            Expr::Function { func, args } => self.push_func(*func, args, depth + 1),
        }
    }

    fn push_binary(
        &mut self,
        op: BinaryOp,
        left: &'a Expr,
        right: &'a Expr,
        parent_prec: u8,
        child_depth: usize,
    ) {
        match (self.mode, op) {
            (_, BinaryOp::Add) => {
                let spec = InfixSpec {
                    sep: " + ",
                    prec: 1,
                    left_prec: 1,
                    right_prec: 1,
                };
                self.push_infix(left, right, spec, parent_prec, child_depth);
            }
            (_, BinaryOp::Sub) => {
                let spec = InfixSpec {
                    sep: " - ",
                    prec: 1,
                    left_prec: 1,
                    right_prec: 2,
                };
                self.push_infix(left, right, spec, parent_prec, child_depth);
            }
            (GlslMode::Scalar, BinaryOp::Mul) => {
                let spec = InfixSpec {
                    sep: " * ",
                    prec: 3,
                    left_prec: 3,
                    right_prec: 3,
                };
                self.push_infix(left, right, spec, parent_prec, child_depth);
            }
            (GlslMode::Complex, BinaryOp::Mul) => {
                self.push_call2("cmul(", left, right, child_depth)
            }
            (GlslMode::Scalar, BinaryOp::Div) => {
                self.push_call2("glsl_div(", left, right, child_depth);
            }
            (GlslMode::Complex, BinaryOp::Div) => {
                self.push_call2("cdiv(", left, right, child_depth)
            }
            (GlslMode::Scalar, BinaryOp::Pow) => {
                self.push_call2("glsl_pow(", left, right, child_depth);
            }
            (GlslMode::Complex, BinaryOp::Pow) => {
                self.push_call2("cpow(", left, right, child_depth)
            }
        }
    }

    fn push_infix(
        &mut self,
        left: &'a Expr,
        right: &'a Expr,
        spec: InfixSpec,
        parent_prec: u8,
        child_depth: usize,
    ) {
        let wrap = spec.prec < parent_prec;
        if wrap {
            self.out.push('(');
            self.stack.push(GlslFrame::Literal(")"));
        }
        self.stack.push(GlslFrame::Eval {
            expr: right,
            parent_prec: spec.right_prec,
            depth: child_depth,
        });
        self.stack.push(GlslFrame::Literal(spec.sep));
        self.stack.push(GlslFrame::Eval {
            expr: left,
            parent_prec: spec.left_prec,
            depth: child_depth,
        });
    }

    fn push_call2(
        &mut self,
        prefix: &'static str,
        left: &'a Expr,
        right: &'a Expr,
        child_depth: usize,
    ) {
        self.out.push_str(prefix);
        self.stack.push(GlslFrame::Literal(")"));
        self.stack.push(GlslFrame::Eval {
            expr: right,
            parent_prec: 0,
            depth: child_depth,
        });
        self.stack.push(GlslFrame::Literal(", "));
        self.stack.push(GlslFrame::Eval {
            expr: left,
            parent_prec: 0,
            depth: child_depth,
        });
    }

    fn push_unary(&mut self, op: UnaryOp, expr: &'a Expr, parent_prec: u8, child_depth: usize) {
        match op {
            UnaryOp::Pos => {
                self.stack.push(GlslFrame::Eval {
                    expr,
                    parent_prec,
                    depth: child_depth,
                });
            }
            UnaryOp::Neg => {
                let wrap = parent_prec > 0;
                if wrap {
                    self.out.push('(');
                    self.stack.push(GlslFrame::Literal(")"));
                }
                self.out.push('-');
                self.stack.push(GlslFrame::Eval {
                    expr,
                    parent_prec: 4,
                    depth: child_depth,
                });
            }
        }
    }

    fn push_func(
        &mut self,
        func: Func,
        args: &'a [Expr],
        child_depth: usize,
    ) -> Result<(), GlslError> {
        let expected = func.arity();
        if args.len() != expected {
            return Err(GlslError::WrongArgumentCount {
                func,
                expected,
                found: args.len(),
            });
        }
        self.out.push_str(glsl_func_prefix(func, self.mode));
        self.stack.push(GlslFrame::Literal(")"));
        for (idx, arg) in args.iter().enumerate().rev() {
            self.stack.push(GlslFrame::Eval {
                expr: arg,
                parent_prec: 0,
                depth: child_depth,
            });
            if idx > 0 {
                self.stack.push(GlslFrame::Literal(", "));
            }
        }
        Ok(())
    }
}

/// Formats a finite `f64` as a guaranteed GLSL ES 3.00 `float` literal.
pub fn format_glsl_float(n: f64) -> Result<String, GlslError> {
    if !n.is_finite() || n.abs() > f64::from(f32::MAX) {
        return Err(GlslError::NonFiniteLiteral(n));
    }
    let clean = if n == 0.0 { 0.0 } else { n };
    let mut buf = String::with_capacity(24);
    write!(buf, "{clean:?}").expect("writing to String cannot fail");
    if !buf.contains('.') && !buf.contains('e') && !buf.contains('E') {
        buf.push_str(".0");
    }
    Ok(buf)
}

fn write_glsl_literal(
    out: &mut String,
    n: f64,
    parent_prec: u8,
    mode: GlslMode,
) -> Result<(), GlslError> {
    let lit = format_glsl_float(n)?;
    match mode {
        GlslMode::Scalar => {
            let wrap = lit.starts_with('-') && parent_prec > 0;
            if wrap {
                out.push('(');
            }
            out.push_str(&lit);
            if wrap {
                out.push(')');
            }
        }
        GlslMode::Complex => {
            write!(out, "vec2({lit}, 0.0)").expect("writing to String cannot fail");
        }
    }
    Ok(())
}

fn write_glsl_variable(
    out: &mut String,
    name: &str,
    mode: GlslMode,
    allowed_vars: Option<&[&str]>,
) -> Result<(), GlslError> {
    if name.eq_ignore_ascii_case("pi") {
        return write_glsl_literal(out, PI, 0, mode);
    }
    if name.eq_ignore_ascii_case("e") {
        return write_glsl_literal(out, E, 0, mode);
    }
    if name.eq_ignore_ascii_case("tau") {
        return write_glsl_literal(out, TAU, 0, mode);
    }
    if mode == GlslMode::Complex && name.eq_ignore_ascii_case("i") {
        out.push_str("vec2(0.0, 1.0)");
        return Ok(());
    }
    let resolved = resolve_glsl_ident(name, allowed_vars)?;
    out.push_str(resolved);
    Ok(())
}

fn resolve_glsl_ident<'a>(
    name: &'a str,
    allowed_vars: Option<&[&'a str]>,
) -> Result<&'a str, GlslError> {
    if let Some(allowed) = allowed_vars {
        let matched = allowed
            .iter()
            .copied()
            .find(|cand| *cand == name)
            .or_else(|| {
                allowed
                    .iter()
                    .copied()
                    .find(|cand| cand.eq_ignore_ascii_case(name))
            })
            .ok_or_else(|| GlslError::InvalidIdentifier(name.to_string()))?;
        if !is_valid_glsl_ident(matched) {
            return Err(GlslError::InvalidIdentifier(matched.to_string()));
        }
        Ok(matched)
    } else if is_valid_glsl_ident(name) {
        Ok(name)
    } else {
        Err(GlslError::InvalidIdentifier(name.to_string()))
    }
}

struct InfixSpec {
    sep: &'static str,
    prec: u8,
    left_prec: u8,
    right_prec: u8,
}

const fn glsl_func_prefix(func: Func, mode: GlslMode) -> &'static str {
    match (mode, func) {
        (GlslMode::Scalar, Func::Sin) => "sin(",
        (GlslMode::Scalar, Func::Cos) => "cos(",
        (GlslMode::Scalar, Func::Tan) => "tan(",
        (GlslMode::Scalar, Func::Exp) => "glsl_exp(",
        (GlslMode::Scalar, Func::Ln) => "glsl_ln(",
        (GlslMode::Scalar, Func::Log10) => "glsl_log10(",
        (GlslMode::Scalar, Func::Sqrt) => "glsl_sqrt(",
        (GlslMode::Scalar, Func::Abs) => "abs(",
        (GlslMode::Scalar, Func::Sinh) => "glsl_sinh(",
        (GlslMode::Scalar, Func::Cosh) => "glsl_cosh(",
        (GlslMode::Scalar, Func::Tanh) => "tanh(",
        (GlslMode::Scalar, Func::Asin) => "glsl_asin(",
        (GlslMode::Scalar, Func::Acos) => "glsl_acos(",
        (GlslMode::Scalar, Func::Atan) => "atan(",
        (GlslMode::Scalar, Func::Min) => "min(",
        (GlslMode::Scalar, Func::Max) => "max(",
        (GlslMode::Scalar, Func::Clamp) => "glsl_clamp(",
        (GlslMode::Scalar, Func::Step) => "glsl_step(",
        (GlslMode::Scalar, Func::Rect) => "glsl_rect(",
        (GlslMode::Scalar, Func::Atan2) => "glsl_atan2(",
        (GlslMode::Complex, Func::Sin) => "csin(",
        (GlslMode::Complex, Func::Cos) => "ccos(",
        (GlslMode::Complex, Func::Tan) => "ctan(",
        (GlslMode::Complex, Func::Exp) => "cexp(",
        (GlslMode::Complex, Func::Ln) => "cln(",
        (GlslMode::Complex, Func::Log10) => "clog10(",
        (GlslMode::Complex, Func::Sqrt) => "csqrt(",
        (GlslMode::Complex, Func::Abs) => "cabs(",
        (GlslMode::Complex, Func::Sinh) => "csinh(",
        (GlslMode::Complex, Func::Cosh) => "ccosh(",
        (GlslMode::Complex, Func::Tanh) => "ctanh(",
        (GlslMode::Complex, Func::Asin) => "casin(",
        (GlslMode::Complex, Func::Acos) => "cacos(",
        (GlslMode::Complex, Func::Atan) => "catan(",
        (GlslMode::Complex, Func::Min) => "cmin(",
        (GlslMode::Complex, Func::Max) => "cmax(",
        (GlslMode::Complex, Func::Clamp) => "cclamp(",
        (GlslMode::Complex, Func::Step) => "cstep(",
        (GlslMode::Complex, Func::Rect) => "crect(",
        (GlslMode::Complex, Func::Atan2) => "catan2(",
    }
}
