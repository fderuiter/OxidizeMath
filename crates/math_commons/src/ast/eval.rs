//! Dynamic evaluation engine for AST expressions supporting both real and complex numbers.
// [cite:math_commons]

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use num_complex::{Complex64, ComplexFloat};
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

    /// Evaluates a 3D spatial expression given `x`, `y`, `z` coordinates and parameter map.
    ///
    /// Uses zero-allocation ASCII matching for `x`, `y`, and `z` so volumetric samplers
    /// can evaluate millions of voxels without heap churn. Returns `f64::NAN` on error.
    #[must_use]
    pub fn eval_3d(&self, x: f64, y: f64, z: f64, params: &HashMap<String, f64>) -> f64 {
        self.eval_fast(0, &|name| {
            if name.eq_ignore_ascii_case("x") {
                Some(x)
            } else if name.eq_ignore_ascii_case("y") {
                Some(y)
            } else if name.eq_ignore_ascii_case("z") {
                Some(z)
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

pub(super) trait EvalScalar: ComplexFloat<Real = f64> {
    const NAN: Self;

    fn from_real(val: f64) -> Self;
    fn imag_unit() -> Self;
    fn real_part(self) -> f64;
    fn is_invalid_scalar(self) -> bool;
    fn div_checked(self, rhs: Self) -> Self;
    fn pow_scalar(self, rhs: Self) -> Self;
    fn extra_const(name: &str) -> Option<Self> {
        let _ = name;
        None
    }
}

impl EvalScalar for f64 {
    const NAN: Self = f64::NAN;

    fn from_real(val: f64) -> Self {
        val
    }

    fn imag_unit() -> Self {
        f64::NAN
    }

    fn real_part(self) -> f64 {
        self
    }

    fn is_invalid_scalar(self) -> bool {
        self.is_nan()
    }

    fn div_checked(self, rhs: Self) -> Self {
        if rhs == 0.0 { f64::NAN } else { self / rhs }
    }

    fn pow_scalar(self, rhs: Self) -> Self {
        self.powf(rhs)
    }
}

impl EvalScalar for Complex64 {
    const NAN: Self = Self::new(f64::NAN, f64::NAN);

    fn from_real(val: f64) -> Self {
        Self::new(val, 0.0)
    }

    fn imag_unit() -> Self {
        Self::i()
    }

    fn real_part(self) -> f64 {
        self.re
    }

    fn is_invalid_scalar(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    fn div_checked(self, rhs: Self) -> Self {
        if rhs.norm_sqr() < 1e-12 {
            Self::NAN
        } else {
            self / rhs
        }
    }

    fn pow_scalar(self, rhs: Self) -> Self {
        if rhs.im == 0.0 {
            self.powf(rhs.re)
        } else {
            self.powc(rhs)
        }
    }

    fn extra_const(name: &str) -> Option<Self> {
        name.eq_ignore_ascii_case("i").then(Self::i)
    }
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
        Func::Sin => args[0].sin(),
        Func::Cos => args[0].cos(),
        Func::Tan => args[0].tan(),
        Func::Exp => args[0].exp(),
        Func::Ln => args[0].ln(),
        Func::Log10 => args[0].log10(),
        Func::Sqrt => args[0].sqrt(),
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

fn eval_real_projected_func<T: EvalScalar>(func: Func, args: &[T]) -> T {
    if args.iter().any(|v| v.is_invalid_scalar()) {
        return T::NAN;
    }
    let val = match func {
        Func::Step => {
            if args[0].real_part() < 0.0 {
                0.0
            } else {
                1.0
            }
        }
        Func::Rect => {
            if args[0].real_part().abs() <= 0.5 {
                1.0
            } else {
                0.0
            }
        }
        Func::Min => args[0].real_part().min(args[1].real_part()),
        Func::Max => args[0].real_part().max(args[1].real_part()),
        Func::Atan2 => args[0].real_part().atan2(args[1].real_part()),
        Func::Clamp => {
            let x = args[0].real_part();
            let lo = args[1].real_part();
            let hi = args[2].real_part();
            if lo > hi { f64::NAN } else { x.max(lo).min(hi) }
        }
        _ => f64::NAN,
    };
    if val.is_nan() {
        T::NAN
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

fn lookup_const<T: EvalScalar>(name: &str) -> Option<T> {
    lookup_universal_real_const(name)
        .map(T::from_real)
        .or_else(|| T::extra_const(name))
}
