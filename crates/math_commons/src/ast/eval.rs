//! Dynamic evaluation engine for AST expressions supporting both real and complex numbers.
// [cite:math_commons]

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use math_scalars::{Complex64, ComplexField, RealField};
use std::collections::HashMap;
use std::f64::consts::{E, PI, TAU};

/// Maximum allowed recursion and operand stack depth during expression evaluation.
pub const MAX_EVAL_DEPTH: usize = 64;

/// Errors that can occur during AST compilation or evaluation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EvalError {
    /// Variable was not defined in the scope.
    #[error("Unknown variable '{0}'")]
    UnknownVariable(String),

    /// Function was called with incorrect number of arguments.
    #[error("Function '{}' expected {expected} argument(s), found {found}", func.as_str())]
    WrongArgumentCount {
        /// Function.
        func: Func,
        /// Expected count.
        expected: usize,
        /// Actual count provided.
        found: usize,
    },

    /// Evaluation recursion or stack limit exceeded.
    #[error("Evaluation recursion limit exceeded (max depth {0})")]
    RecursionLimitExceeded(usize),

    /// Compiled instruction or AST node traversal budget exceeded.
    #[error("Compiled expression exceeds maximum limit of {0} instructions")]
    InstructionLimitExceeded(usize),

    /// Formatted string output budget exceeded.
    #[error("Formatted expression output exceeds maximum limit of {0} bytes")]
    OutputLimitExceeded(usize),

    /// Domain error (e.g. division by zero in strict mode).
    #[error("Domain evaluation error: {0}")]
    DomainError(String),
}

impl math_diagnostics::Diagnostic for EvalError {
    fn severity(&self) -> math_diagnostics::Severity {
        math_diagnostics::Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn math_diagnostics::DiagnosticVisitor) {
        visitor.record_str("error_type", "EvalError");
        visitor.record_display("description", self);
        match self {
            Self::UnknownVariable(name) => {
                visitor.record_str("variable", name);
            }
            Self::WrongArgumentCount {
                func,
                expected,
                found,
            } => {
                visitor.record_display("function", &format_args!("{func:?}"));
                visitor.record_usize("expected_args", *expected);
                visitor.record_usize("found_args", *found);
            }
            Self::RecursionLimitExceeded(max_depth) => {
                visitor.record_usize("max_depth", *max_depth);
            }
            Self::InstructionLimitExceeded(max_instructions) => {
                visitor.record_usize("max_instructions", *max_instructions);
            }
            Self::OutputLimitExceeded(max_bytes) => {
                visitor.record_usize("max_output_bytes", *max_bytes);
            }
            Self::DomainError(reason) => {
                visitor.record_str("reason", reason);
            }
        }
    }
}

impl Expr {
    /// Evaluates the expression with the given `f64` variable map.
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Result<f64, EvalError> {
        self.eval_checked(vars, 0)
    }

    /// Evaluates the expression over any [`ComplexField`] (`f32`, `f64`, `Complex32`, `Complex64`, `Dual<T>`).
    pub fn eval_generic<T: ComplexField>(&self, vars: &HashMap<String, T>) -> Result<T, EvalError> {
        self.eval_checked(vars, 0)
    }

    /// Evaluates the expression for an ODE solver given `t` (time), `y` (state), and parameter map.
    ///
    /// Optimized for high-throughput loops without per-step heap allocations.
    /// Returns `T::NAN` on divergence or error.
    #[must_use]
    pub fn eval_2d<T: RealField>(&self, t: T, y: T, params: &HashMap<String, T>) -> T {
        self.eval_fast(0, &|name| {
            if name.eq_ignore_ascii_case("t") {
                Some(t)
            } else if name.eq_ignore_ascii_case("y") {
                Some(y)
            } else {
                lookup_var(params, name)
            }
        })
    }

    /// Evaluates a 2D planar ODE expression given `t` (time), `x`, `y`, and parameter map.
    ///
    /// Optimized for high-throughput vector-field and phase-portrait loops without per-step heap allocations.
    /// Returns `T::NAN` on divergence or error.
    #[must_use]
    pub fn eval_planar<T: RealField>(&self, t: T, x: T, y: T, params: &HashMap<String, T>) -> T {
        self.eval_fast(0, &|name| {
            if name.eq_ignore_ascii_case("t") {
                Some(t)
            } else if name.eq_ignore_ascii_case("x") {
                Some(x)
            } else if name.eq_ignore_ascii_case("y") {
                Some(y)
            } else {
                lookup_var(params, name)
            }
        })
    }

    /// Evaluates the expression over complex numbers `Complex64`.
    pub fn eval_complex(&self, vars: &HashMap<String, Complex64>) -> Result<Complex64, EvalError> {
        self.eval_checked(vars, 0)
    }

    /// Evaluates the expression for conformal complex mapping given coordinate `z`.
    #[must_use]
    pub fn eval_complex_z<T: ComplexField>(&self, z: T) -> T {
        self.eval_fast(0, &|name| name.eq_ignore_ascii_case("z").then_some(z))
    }

    /// Evaluates the expression for complex recurrences given state `z` and parameter `c`.
    #[must_use]
    pub fn eval_complex_zc<T: ComplexField>(&self, z: T, c: T) -> T {
        self.eval_fast(0, &|name| {
            if name.eq_ignore_ascii_case("z") {
                Some(z)
            } else if name.eq_ignore_ascii_case("c") {
                Some(c)
            } else {
                None
            }
        })
    }

    fn eval_checked<T: ComplexField>(
        &self,
        vars: &HashMap<String, T>,
        depth: usize,
    ) -> Result<T, EvalError> {
        if depth > MAX_EVAL_DEPTH {
            return Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
        }

        match self {
            Self::Number(n) => Ok(T::from_real_f64(*n)),
            Self::Variable(name) => lookup_const(name)
                .or_else(|| lookup_var(vars, name))
                .ok_or_else(|| EvalError::UnknownVariable(name.clone())),
            Self::Binary { op, left, right } => {
                let l = left.eval_checked(vars, depth + 1)?;
                let r = right.eval_checked(vars, depth + 1)?;
                Ok(eval_binop(*op, l, r))
            }
            Self::Unary { op, expr } => Ok(eval_unary(*op, expr.eval_checked(vars, depth + 1)?)),
            Self::Function { func, args } => {
                let expected = func.arity();
                if args.len() != expected {
                    return Err(EvalError::WrongArgumentCount {
                        func: *func,
                        expected,
                        found: args.len(),
                    });
                }
                let mut buf = [T::nan(); 3];
                for (slot, arg) in buf[..expected].iter_mut().zip(args.iter()) {
                    *slot = arg.eval_checked(vars, depth + 1)?;
                }
                Ok(eval_func(*func, &buf[..expected]))
            }
        }
    }

    fn eval_fast<T: ComplexField, F: Fn(&str) -> Option<T>>(
        &self,
        depth: usize,
        resolve_var: &F,
    ) -> T {
        if depth > MAX_EVAL_DEPTH {
            return T::nan();
        }

        match self {
            Self::Number(n) => T::from_real_f64(*n),
            Self::Variable(name) => lookup_const(name)
                .or_else(|| resolve_var(name))
                .unwrap_or_else(T::nan),
            Self::Binary { op, left, right } => {
                let l = self_eval_fast(left, depth + 1, resolve_var);
                let r = self_eval_fast(right, depth + 1, resolve_var);
                eval_binop(*op, l, r)
            }
            Self::Unary { op, expr } => {
                eval_unary(*op, self_eval_fast(expr, depth + 1, resolve_var))
            }
            Self::Function { func, args } => {
                let expected = func.arity();
                if args.len() != expected {
                    return T::nan();
                }
                let mut buf = [T::nan(); 3];
                for (slot, arg) in buf[..expected].iter_mut().zip(args.iter()) {
                    *slot = self_eval_fast(arg, depth + 1, resolve_var);
                }
                eval_func(*func, &buf[..expected])
            }
        }
    }
}

#[inline]
fn self_eval_fast<T: ComplexField, F: Fn(&str) -> Option<T>>(
    expr: &Expr,
    depth: usize,
    resolve_var: &F,
) -> T {
    expr.eval_fast(depth, resolve_var)
}

#[inline]
pub(super) fn eval_unary<T: ComplexField>(op: UnaryOp, val: T) -> T {
    match op {
        UnaryOp::Pos => val,
        UnaryOp::Neg => -val,
    }
}

#[inline]
pub(super) fn eval_binop<T: ComplexField>(op: BinaryOp, l: T, r: T) -> T {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => {
            if !T::imag_unit().is_any_nan() {
                if r.norm_sqr() < T::Real::from_f64(1e-12) {
                    T::nan()
                } else {
                    l / r
                }
            } else if r.real().to_f64() == 0.0 {
                T::nan()
            } else {
                l / r
            }
        }
        BinaryOp::Pow => {
            if r.imag() == T::Real::ZERO {
                l.powf(r.real())
            } else {
                l.powc(r)
            }
        }
    }
}

pub(super) fn eval_func<T: ComplexField>(func: Func, args: &[T]) -> T {
    if args.len() != func.arity() {
        return T::nan();
    }
    match func {
        Func::Sin => args[0].sin(),
        Func::Cos => args[0].cos(),
        Func::Tan => args[0].tan(),
        Func::Exp => args[0].exp(),
        Func::Ln => args[0].ln(),
        Func::Log10 => args[0].log10(),
        Func::Sqrt => args[0].csqrt(),
        Func::Abs => T::from_real(args[0].abs()),
        Func::Sinh => args[0].sinh(),
        Func::Cosh => args[0].cosh(),
        Func::Tanh => args[0].tanh(),
        Func::Asin => args[0].asin(),
        Func::Acos => args[0].acos(),
        Func::Atan => args[0].atan(),
        Func::Step | Func::Rect | Func::Min | Func::Max | Func::Clamp | Func::Atan2 => {
            eval_real_projected_func(func, args)
        }
    }
}

fn eval_real_projected_func<T: ComplexField>(func: Func, args: &[T]) -> T {
    if args.iter().any(|v| v.is_any_nan()) {
        return T::nan();
    }
    let r0 = args[0].real();
    let val = match func {
        Func::Step => {
            if r0 < T::Real::ZERO {
                T::Real::ZERO
            } else {
                T::Real::ONE
            }
        }
        Func::Rect => {
            if ComplexField::abs(r0) <= T::Real::HALF {
                T::Real::ONE
            } else {
                T::Real::ZERO
            }
        }
        Func::Min => r0.min(args[1].real()),
        Func::Max => r0.max(args[1].real()),
        Func::Atan2 => r0.atan2(args[1].real()),
        Func::Clamp => {
            let lo = args[1].real();
            let hi = args[2].real();
            if lo > hi {
                T::Real::NAN
            } else {
                r0.max(lo).min(hi)
            }
        }
        _ => T::Real::NAN,
    };
    if val.is_nan() {
        T::nan()
    } else {
        T::from_real(val)
    }
}

fn lookup_var<V: Copy>(map: &HashMap<String, V>, name: &str) -> Option<V> {
    if let Some(&val) = map.get(name) {
        return Some(val);
    }
    map.iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, &v)| v)
}

pub(super) fn lookup_universal_real_const(name: &str) -> Option<f64> {
    if name.eq_ignore_ascii_case("pi") {
        Some(PI)
    } else if name.eq_ignore_ascii_case("e") {
        Some(E)
    } else if name.eq_ignore_ascii_case("tau") {
        Some(TAU)
    } else {
        None
    }
}

fn lookup_const<T: ComplexField>(name: &str) -> Option<T> {
    if name.eq_ignore_ascii_case("pi") {
        Some(T::from_real(T::Real::PI))
    } else if name.eq_ignore_ascii_case("e") {
        Some(T::from_real(T::Real::E))
    } else if name.eq_ignore_ascii_case("tau") {
        Some(T::from_real(T::Real::TAU))
    } else if !T::imag_unit().is_any_nan() && name.eq_ignore_ascii_case("i") {
        Some(T::imag_unit())
    } else {
        None
    }
}
