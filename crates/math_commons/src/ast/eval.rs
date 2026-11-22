//! Dynamic evaluation engine for AST expressions supporting both real and complex numbers.

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use num_complex::{Complex64, ComplexFloat};
use std::collections::HashMap;
use std::f64::consts::{E, PI, TAU};

/// Maximum allowed recursion depth during expression evaluation.
pub const MAX_EVAL_DEPTH: usize = 64;

/// Errors that can occur during AST evaluation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum EvalError {
    /// Variable was not defined in the scope.
    #[error("Unknown variable '{0}'")]
    UnknownVariable(String),

    /// Function was called with incorrect number of arguments.
    #[error("Function '{func:?}' expected {expected} arguments, found {found}")]
    WrongArgumentCount {
        /// Function.
        func: Func,
        /// Expected count.
        expected: usize,
        /// Actual count provided.
        found: usize,
    },

    /// Evaluation recursion limit exceeded.
    #[error("Evaluation recursion limit exceeded (max depth {0})")]
    RecursionLimitExceeded(usize),

    /// Domain error (e.g. division by zero in strict mode).
    #[error("Domain evaluation error: {0}")]
    DomainError(String),
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

    /// Evaluates the expression for 3D spatial fields given coordinates `(x, y, z)`.
    ///
    /// Optimized for high-throughput marching-cubes and implicit CAD loops without
    /// per-sample heap allocations. Returns `f64::NAN` on divergence or error.
    #[must_use]
    pub fn eval_3d(&self, x: f64, y: f64, z: f64) -> f64 {
        self.eval_fast(0, &|name| {
            if name.eq_ignore_ascii_case("x") {
                Some(x)
            } else if name.eq_ignore_ascii_case("y") {
                Some(y)
            } else if name.eq_ignore_ascii_case("z") {
                Some(z)
            } else {
                None
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
            Self::Function { func, args } => eval_function_checked(*func, args, vars, depth),
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
                let l = self_or_child_fast(left, depth + 1, resolve_var);
                let r = self_or_child_fast(right, depth + 1, resolve_var);
                eval_binop(*op, l, r)
            }
            Self::Unary { op, expr } => eval_unary(*op, expr.eval_fast(depth + 1, resolve_var)),
            Self::Function { func, args } => eval_function_fast(*func, args, depth, resolve_var),
        }
    }
}

#[inline]
fn self_or_child_fast<T: EvalScalar, F: Fn(&str) -> Option<T>>(
    expr: &Expr,
    depth: usize,
    resolve_var: &F,
) -> T {
    expr.eval_fast(depth, resolve_var)
}

fn eval_function_checked<T: EvalScalar>(
    func: Func,
    args: &[Expr],
    vars: &HashMap<String, T>,
    depth: usize,
) -> Result<T, EvalError> {
    if matches!(func, Func::Min | Func::Max) {
        if args.len() < 2 {
            return Err(EvalError::WrongArgumentCount {
                func,
                expected: 2,
                found: args.len(),
            });
        }
        let mut acc = args[0].eval_checked(vars, depth + 1)?;
        for arg in &args[1..] {
            let rhs = arg.eval_checked(vars, depth + 1)?;
            acc = if func == Func::Min {
                acc.min_checked(rhs)?
            } else {
                acc.max_checked(rhs)?
            };
        }
        return Ok(acc);
    }

    if args.len() != 1 {
        return Err(EvalError::WrongArgumentCount {
            func,
            expected: 1,
            found: args.len(),
        });
    }
    let arg = args[0].eval_checked(vars, depth + 1)?;
    Ok(eval_func(func, arg))
}

fn eval_function_fast<T: EvalScalar, F: Fn(&str) -> Option<T>>(
    func: Func,
    args: &[Expr],
    depth: usize,
    resolve_var: &F,
) -> T {
    if matches!(func, Func::Min | Func::Max) {
        if args.len() < 2 {
            return T::NAN;
        }
        let mut acc = args[0].eval_fast(depth + 1, resolve_var);
        for arg in &args[1..] {
            let rhs = arg.eval_fast(depth + 1, resolve_var);
            acc = if func == Func::Min {
                acc.min_fast(rhs)
            } else {
                acc.max_fast(rhs)
            };
        }
        return acc;
    }

    if args.len() != 1 {
        return T::NAN;
    }
    let arg = args[0].eval_fast(depth + 1, resolve_var);
    eval_func(func, arg)
}

trait EvalScalar: ComplexFloat<Real = f64> {
    const NAN: Self;

    fn from_real(val: f64) -> Self;
    fn div_checked(self, rhs: Self) -> Self;
    fn pow_scalar(self, rhs: Self) -> Self;
    fn min_checked(self, rhs: Self) -> Result<Self, EvalError>;
    fn max_checked(self, rhs: Self) -> Result<Self, EvalError>;
    fn min_fast(self, rhs: Self) -> Self;
    fn max_fast(self, rhs: Self) -> Self;
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

    fn div_checked(self, rhs: Self) -> Self {
        if rhs == 0.0 { f64::NAN } else { self / rhs }
    }

    fn pow_scalar(self, rhs: Self) -> Self {
        self.powf(rhs)
    }

    fn min_checked(self, rhs: Self) -> Result<Self, EvalError> {
        Ok(self.min_fast(rhs))
    }

    fn max_checked(self, rhs: Self) -> Result<Self, EvalError> {
        Ok(self.max_fast(rhs))
    }

    fn min_fast(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            f64::NAN
        } else {
            self.min(rhs)
        }
    }

    fn max_fast(self, rhs: Self) -> Self {
        if self.is_nan() || rhs.is_nan() {
            f64::NAN
        } else {
            self.max(rhs)
        }
    }
}

impl EvalScalar for Complex64 {
    const NAN: Self = Self::new(f64::NAN, f64::NAN);

    fn from_real(val: f64) -> Self {
        Self::new(val, 0.0)
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

    fn min_checked(self, _rhs: Self) -> Result<Self, EvalError> {
        Err(EvalError::DomainError(
            "min() is only defined for real numbers".to_string(),
        ))
    }

    fn max_checked(self, _rhs: Self) -> Result<Self, EvalError> {
        Err(EvalError::DomainError(
            "max() is only defined for real numbers".to_string(),
        ))
    }

    fn min_fast(self, _rhs: Self) -> Self {
        Self::NAN
    }

    fn max_fast(self, _rhs: Self) -> Self {
        Self::NAN
    }

    fn extra_const(name: &str) -> Option<Self> {
        name.eq_ignore_ascii_case("i").then(Self::i)
    }
}

fn eval_unary<T: EvalScalar>(op: UnaryOp, val: T) -> T {
    match op {
        UnaryOp::Pos => val,
        UnaryOp::Neg => -val,
    }
}

fn eval_binop<T: EvalScalar>(op: BinaryOp, l: T, r: T) -> T {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => l.div_checked(r),
        BinaryOp::Pow => l.pow_scalar(r),
    }
}

fn eval_func<T: EvalScalar>(func: Func, arg: T) -> T {
    match func {
        Func::Sin => arg.sin(),
        Func::Cos => arg.cos(),
        Func::Tan => arg.tan(),
        Func::Exp => arg.exp(),
        Func::Ln => arg.ln(),
        Func::Log10 => arg.log10(),
        Func::Sqrt => arg.sqrt(),
        Func::Abs => T::from_real(arg.abs()),
        Func::Sinh => arg.sinh(),
        Func::Cosh => arg.cosh(),
        Func::Tanh => arg.tanh(),
        Func::Asin => arg.asin(),
        Func::Acos => arg.acos(),
        Func::Atan => arg.atan(),
        Func::Min | Func::Max => T::NAN,
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

fn lookup_const<T: EvalScalar>(name: &str) -> Option<T> {
    if name.eq_ignore_ascii_case("pi") {
        Some(T::from_real(PI))
    } else if name.eq_ignore_ascii_case("e") {
        Some(T::from_real(E))
    } else if name.eq_ignore_ascii_case("tau") {
        Some(T::from_real(TAU))
    } else {
        T::extra_const(name)
    }
}
