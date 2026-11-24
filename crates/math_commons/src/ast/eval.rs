//! Dynamic evaluation engine for AST expressions supporting both real and complex numbers.
// [cite:math_commons]

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use engine_symbolic::{Dual, DualComplex64};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::{E, PI, TAU};

/// Maximum allowed recursion and operand stack depth during expression evaluation.
pub const MAX_EVAL_DEPTH: usize = 64;

/// Errors that can occur during AST compilation or evaluation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
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

impl diagnostics::Diagnostic for EvalError {
    fn severity(&self) -> diagnostics::Severity {
        diagnostics::Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "EvalError".to_string());
        map.insert("description".to_string(), self.to_string());
        match self {
            Self::UnknownVariable(name) => {
                map.insert("variable".to_string(), name.clone());
            }
            Self::WrongArgumentCount {
                func,
                expected,
                found,
            } => {
                map.insert("function".to_string(), format!("{func:?}"));
                map.insert("expected_args".to_string(), expected.to_string());
                map.insert("found_args".to_string(), found.to_string());
            }
            Self::RecursionLimitExceeded(max_depth) => {
                map.insert("max_depth".to_string(), max_depth.to_string());
            }
            Self::InstructionLimitExceeded(max_instructions) => {
                map.insert("max_instructions".to_string(), max_instructions.to_string());
            }
            Self::OutputLimitExceeded(max_bytes) => {
                map.insert("max_output_bytes".to_string(), max_bytes.to_string());
            }
            Self::DomainError(reason) => {
                map.insert("reason".to_string(), reason.clone());
            }
        }
        map
    }
}

impl Expr {
    /// Evaluates the expression with the given variable map.
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Result<f64, EvalError> {
        self.eval_checked(vars, 0)
    }

    /// Evaluates the expression for an ODE solver given `t` (time), `y` (state), and parameter map.
    ///
    /// Optimized for high-throughput loops without per-step heap allocations.
    /// Returns `f64::NAN` on divergence or error.
    #[must_use]
    pub fn eval_2d(&self, t: f64, y: f64, params: &HashMap<String, f64>) -> f64 {
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
    /// Returns `f64::NAN` on divergence or error.
    #[must_use]
    pub fn eval_planar(&self, t: f64, x: f64, y: f64, params: &HashMap<String, f64>) -> f64 {
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

    /// Evaluates the expression for conformal complex mapping given complex coordinate `z`.
    #[must_use]
    pub fn eval_complex_z(&self, z: Complex64) -> Complex64 {
        self.eval_fast(0, &|name| name.eq_ignore_ascii_case("z").then_some(z))
    }

    /// Evaluates the expression over forward-mode real [`Dual<f64>`] variables.
    pub fn eval_dual(&self, vars: &HashMap<String, Dual<f64>>) -> Result<Dual<f64>, EvalError> {
        self.eval_checked(vars, 0)
    }

    /// Evaluates the expression over forward-mode complex [`DualComplex64`] variables.
    pub fn eval_complex_dual(
        &self,
        vars: &HashMap<String, DualComplex64>,
    ) -> Result<DualComplex64, EvalError> {
        self.eval_checked(vars, 0)
    }

    fn eval_checked<T: EvalScalar>(
        &self,
        vars: &HashMap<String, T>,
        depth: usize,
    ) -> Result<T, EvalError> {
        if depth > MAX_EVAL_DEPTH {
            return Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
        }

        match self {
            Self::Number(n) => Ok(T::from_real(*n)),
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
                let mut buf = [T::NAN; 3];
                for (slot, arg) in buf[..expected].iter_mut().zip(args.iter()) {
                    *slot = arg.eval_checked(vars, depth + 1)?;
                }
                Ok(eval_func(*func, &buf[..expected]))
            }
        }
    }

    fn eval_fast<T: EvalScalar, F: Fn(&str) -> Option<T>>(
        &self,
        depth: usize,
        resolve_var: &F,
    ) -> T {
        if depth > MAX_EVAL_DEPTH {
            return T::NAN;
        }

        match self {
            Self::Number(n) => T::from_real(*n),
            Self::Variable(name) => lookup_const(name)
                .or_else(|| resolve_var(name))
                .unwrap_or(T::NAN),
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
                    return T::NAN;
                }
                let mut buf = [T::NAN; 3];
                for (slot, arg) in buf[..expected].iter_mut().zip(args.iter()) {
                    *slot = self_eval_fast(arg, depth + 1, resolve_var);
                }
                eval_func(*func, &buf[..expected])
            }
        }
    }
}

#[inline]
fn self_eval_fast<T: EvalScalar, F: Fn(&str) -> Option<T>>(
    expr: &Expr,
    depth: usize,
    resolve_var: &F,
) -> T {
    expr.eval_fast(depth, resolve_var)
}

/// Scalar algebra contract unifying primal (`f64`, `Complex64`) and forward-mode (`Dual`, `Jet2`) AST evaluation.
pub trait EvalScalar:
    Copy
    + PartialEq
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    /// Canonical quiet NaN representation.
    const NAN: Self;

    /// Lifts a real `f64` constant into `Self`.
    fn from_real(val: f64) -> Self;
    /// Returns the imaginary unit $i$ (or `Self::NAN` on purely real domains).
    fn imag_unit() -> Self;
    /// Returns `true` if `self` represents an invalid / NaN scalar.
    fn is_invalid_scalar(self) -> bool;
    /// Checked division returning `Self::NAN` on zero/singular divisors.
    fn div_checked(self, rhs: Self) -> Self;
    /// Exponentiation `self ^ rhs`.
    fn pow_scalar(self, rhs: Self) -> Self;
    /// Resolves domain-specific named constants (such as `i` on complex scalars).
    fn extra_const(name: &str) -> Option<Self> {
        let _ = name;
        None
    }

    /// Sine $\sin(x)$.
    fn sin_s(self) -> Self;
    /// Cosine $\cos(x)$.
    fn cos_s(self) -> Self;
    /// Tangent $\tan(x)$.
    fn tan_s(self) -> Self;
    /// Exponential $e^x$.
    fn exp_s(self) -> Self;
    /// Natural logarithm $\ln(x)$.
    fn ln_s(self) -> Self;
    /// Base-10 logarithm $\log_{10}(x)$.
    fn log10_s(self) -> Self;
    /// Square root $\sqrt{x}$.
    fn sqrt_s(self) -> Self;
    /// Absolute value $|x|$ with Clarke subgradient.
    fn abs_s(self) -> Self;
    /// Hyperbolic sine $\sinh(x)$.
    fn sinh_s(self) -> Self;
    /// Hyperbolic cosine $\cosh(x)$.
    fn cosh_s(self) -> Self;
    /// Hyperbolic tangent $\tanh(x)$.
    fn tanh_s(self) -> Self;
    /// Inverse sine $\arcsin(x)$.
    fn asin_s(self) -> Self;
    /// Inverse cosine $\arccos(x)$.
    fn acos_s(self) -> Self;
    /// Inverse tangent $\arctan(x)$.
    fn atan_s(self) -> Self;
    /// Heaviside step $\operatorname{step}(x)$ with Clarke subgradient.
    fn step_s(self) -> Self;
    /// Boxcar pulse $\operatorname{rect}(x)$ with Clarke subgradient.
    fn rect_s(self) -> Self;
    /// Minimum $\min(a, b)$ with Clarke subgradient.
    fn min_s(self, other: Self) -> Self;
    /// Maximum $\max(a, b)$ with Clarke subgradient.
    fn max_s(self, other: Self) -> Self;
    /// Clamping $\operatorname{clamp}(x, \text{lo}, \text{hi})$ with Clarke subgradient.
    fn clamp_s(self, lo: Self, hi: Self) -> Self;
    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x)$.
    fn atan2_s(self, other: Self) -> Self;
}

#[inline]
pub(super) fn eval_unary<T: EvalScalar>(op: UnaryOp, val: T) -> T {
    match op {
        UnaryOp::Pos => val,
        UnaryOp::Neg => -val,
    }
}

#[inline]
pub(super) fn eval_binop<T: EvalScalar>(op: BinaryOp, l: T, r: T) -> T {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l.div_checked(r),
        BinaryOp::Pow => l.pow_scalar(r),
    }
}

pub(super) fn eval_func<T: EvalScalar>(func: Func, args: &[T]) -> T {
    if args.len() != func.arity() {
        return T::NAN;
    }
    match func {
        Func::Sin => args[0].sin_s(),
        Func::Cos => args[0].cos_s(),
        Func::Tan => args[0].tan_s(),
        Func::Exp => args[0].exp_s(),
        Func::Ln => args[0].ln_s(),
        Func::Log10 => args[0].log10_s(),
        Func::Sqrt => args[0].sqrt_s(),
        Func::Abs => args[0].abs_s(),
        Func::Sinh => args[0].sinh_s(),
        Func::Cosh => args[0].cosh_s(),
        Func::Tanh => args[0].tanh_s(),
        Func::Asin => args[0].asin_s(),
        Func::Acos => args[0].acos_s(),
        Func::Atan => args[0].atan_s(),
        Func::Step => args[0].step_s(),
        Func::Rect => args[0].rect_s(),
        Func::Min => args[0].min_s(args[1]),
        Func::Max => args[0].max_s(args[1]),
        Func::Clamp => args[0].clamp_s(args[1], args[2]),
        Func::Atan2 => args[0].atan2_s(args[1]),
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

fn lookup_const<T: EvalScalar>(name: &str) -> Option<T> {
    lookup_universal_real_const(name)
        .map(T::from_real)
        .or_else(|| T::extra_const(name))
}
