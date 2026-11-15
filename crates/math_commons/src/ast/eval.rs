//! Dynamic evaluation engine for AST expressions supporting both real and complex numbers.

use super::expr::{BinaryOp, Expr, Func, UnaryOp};
use num_complex::Complex64;
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
        self.eval_internal(vars, 0)
    }

    fn eval_internal(&self, vars: &HashMap<String, f64>, depth: usize) -> Result<f64, EvalError> {
        if depth > MAX_EVAL_DEPTH {
            return Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
        }

        match self {
            Self::Number(n) => Ok(*n),
            Self::Variable(name) => {
                if let Some(val) = lookup_const_f64(name) {
                    return Ok(val);
                }
                if let Some(&val) = vars.get(name) {
                    return Ok(val);
                }
                // Try case-insensitive lookup
                let lower = name.to_ascii_lowercase();
                for (k, &v) in vars {
                    if k.to_ascii_lowercase() == lower {
                        return Ok(v);
                    }
                }
                Err(EvalError::UnknownVariable(name.clone()))
            }
            Self::Binary { op, left, right } => {
                let l = left.eval_internal(vars, depth + 1)?;
                let r = right.eval_internal(vars, depth + 1)?;
                Ok(eval_binop_f64(*op, l, r))
            }
            Self::Unary { op, expr } => {
                let val = expr.eval_internal(vars, depth + 1)?;
                match op {
                    UnaryOp::Pos => Ok(val),
                    UnaryOp::Neg => Ok(-val),
                }
            }
            Self::Function { func, args } => {
                if args.len() != 1 {
                    return Err(EvalError::WrongArgumentCount {
                        func: *func,
                        expected: 1,
                        found: args.len(),
                    });
                }
                let arg = args[0].eval_internal(vars, depth + 1)?;
                Ok(eval_func_f64(*func, arg))
            }
        }
    }

    /// Evaluates the expression for an ODE solver given `t` (time), `y` (state), and parameter map.
    ///
    /// Optimized for high-throughput loops without per-step heap allocations.
    /// Returns `f64::NAN` on divergence or error.
    #[must_use]
    pub fn eval_2d(&self, t: f64, y: f64, params: &HashMap<String, f64>) -> f64 {
        self.eval_2d_internal(t, y, params, 0)
    }

    fn eval_2d_internal(&self, t: f64, y: f64, params: &HashMap<String, f64>, depth: usize) -> f64 {
        if depth > MAX_EVAL_DEPTH {
            return f64::NAN;
        }

        match self {
            Self::Number(n) => *n,
            Self::Variable(name) => {
                if name.eq_ignore_ascii_case("t") {
                    t
                } else if name.eq_ignore_ascii_case("y") {
                    y
                } else if let Some(val) = lookup_const_f64(name) {
                    val
                } else if let Some(&val) = params.get(name) {
                    val
                } else {
                    let lower = name.to_ascii_lowercase();
                    params
                        .iter()
                        .find(|(k, _)| k.to_ascii_lowercase() == lower)
                        .map(|(_, &v)| v)
                        .unwrap_or(f64::NAN)
                }
            }
            Self::Binary { op, left, right } => {
                let l = left.eval_2d_internal(t, y, params, depth + 1);
                let r = right.eval_2d_internal(t, y, params, depth + 1);
                eval_binop_f64(*op, l, r)
            }
            Self::Unary { op, expr } => {
                let val = expr.eval_2d_internal(t, y, params, depth + 1);
                match op {
                    UnaryOp::Pos => val,
                    UnaryOp::Neg => -val,
                }
            }
            Self::Function { func, args } => {
                if args.is_empty() {
                    return f64::NAN;
                }
                let arg = args[0].eval_2d_internal(t, y, params, depth + 1);
                eval_func_f64(*func, arg)
            }
        }
    }

    /// Evaluates the expression over complex numbers `Complex64`.
    pub fn eval_complex(&self, vars: &HashMap<String, Complex64>) -> Result<Complex64, EvalError> {
        self.eval_complex_internal(vars, 0)
    }

    fn eval_complex_internal(
        &self,
        vars: &HashMap<String, Complex64>,
        depth: usize,
    ) -> Result<Complex64, EvalError> {
        if depth > MAX_EVAL_DEPTH {
            return Err(EvalError::RecursionLimitExceeded(MAX_EVAL_DEPTH));
        }

        match self {
            Self::Number(n) => Ok(Complex64::new(*n, 0.0)),
            Self::Variable(name) => {
                if let Some(val) = lookup_const_complex(name) {
                    return Ok(val);
                }
                if let Some(&val) = vars.get(name) {
                    return Ok(val);
                }
                let lower = name.to_ascii_lowercase();
                for (k, &v) in vars {
                    if k.to_ascii_lowercase() == lower {
                        return Ok(v);
                    }
                }
                Err(EvalError::UnknownVariable(name.clone()))
            }
            Self::Binary { op, left, right } => {
                let l = left.eval_complex_internal(vars, depth + 1)?;
                let r = right.eval_complex_internal(vars, depth + 1)?;
                Ok(eval_binop_complex(*op, l, r))
            }
            Self::Unary { op, expr } => {
                let val = expr.eval_complex_internal(vars, depth + 1)?;
                match op {
                    UnaryOp::Pos => Ok(val),
                    UnaryOp::Neg => Ok(-val),
                }
            }
            Self::Function { func, args } => {
                if args.len() != 1 {
                    return Err(EvalError::WrongArgumentCount {
                        func: *func,
                        expected: 1,
                        found: args.len(),
                    });
                }
                let arg = args[0].eval_complex_internal(vars, depth + 1)?;
                Ok(eval_func_complex(*func, arg))
            }
        }
    }

    /// Evaluates the expression for conformal complex mapping given complex coordinate `z`.
    #[must_use]
    pub fn eval_complex_z(&self, z: Complex64) -> Complex64 {
        self.eval_complex_z_internal(z, 0)
    }

    fn eval_complex_z_internal(&self, z: Complex64, depth: usize) -> Complex64 {
        if depth > MAX_EVAL_DEPTH {
            return Complex64::new(f64::NAN, f64::NAN);
        }

        match self {
            Self::Number(n) => Complex64::new(*n, 0.0),
            Self::Variable(name) => {
                if name.eq_ignore_ascii_case("z") {
                    z
                } else if let Some(c) = lookup_const_complex(name) {
                    c
                } else {
                    Complex64::new(f64::NAN, f64::NAN)
                }
            }
            Self::Binary { op, left, right } => {
                let l = left.eval_complex_z_internal(z, depth + 1);
                let r = right.eval_complex_z_internal(z, depth + 1);
                eval_binop_complex(*op, l, r)
            }
            Self::Unary { op, expr } => {
                let val = expr.eval_complex_z_internal(z, depth + 1);
                match op {
                    UnaryOp::Pos => val,
                    UnaryOp::Neg => -val,
                }
            }
            Self::Function { func, args } => {
                if args.is_empty() {
                    return Complex64::new(f64::NAN, f64::NAN);
                }
                let arg = args[0].eval_complex_z_internal(z, depth + 1);
                eval_func_complex(*func, arg)
            }
        }
    }
}

fn eval_binop_f64(op: BinaryOp, l: f64, r: f64) -> f64 {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => {
            if r == 0.0 {
                f64::NAN
            } else {
                l / r
            }
        }
        BinaryOp::Pow => l.powf(r),
    }
}

fn eval_func_f64(func: Func, arg: f64) -> f64 {
    match func {
        Func::Sin => arg.sin(),
        Func::Cos => arg.cos(),
        Func::Tan => arg.tan(),
        Func::Exp => arg.exp(),
        Func::Ln => arg.ln(),
        Func::Log10 => arg.log10(),
        Func::Sqrt => arg.sqrt(),
        Func::Abs => arg.abs(),
        Func::Sinh => arg.sinh(),
        Func::Cosh => arg.cosh(),
        Func::Tanh => arg.tanh(),
        Func::Asin => arg.asin(),
        Func::Acos => arg.acos(),
        Func::Atan => arg.atan(),
    }
}

fn eval_binop_complex(op: BinaryOp, l: Complex64, r: Complex64) -> Complex64 {
    match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => {
            if r.norm_sqr() < 1e-12 {
                Complex64::new(f64::NAN, f64::NAN)
            } else {
                l / r
            }
        }
        BinaryOp::Pow => {
            if r.im == 0.0 {
                l.powf(r.re)
            } else {
                l.powc(r)
            }
        }
    }
}

fn eval_func_complex(func: Func, arg: Complex64) -> Complex64 {
    match func {
        Func::Sin => arg.sin(),
        Func::Cos => arg.cos(),
        Func::Tan => arg.tan(),
        Func::Exp => arg.exp(),
        Func::Ln => arg.ln(),
        Func::Log10 => arg.ln() / std::f64::consts::LN_10,
        Func::Sqrt => arg.sqrt(),
        Func::Abs => Complex64::new(arg.norm(), 0.0),
        Func::Sinh => arg.sinh(),
        Func::Cosh => arg.cosh(),
        Func::Tanh => arg.tanh(),
        Func::Asin => arg.asin(),
        Func::Acos => arg.acos(),
        Func::Atan => arg.atan(),
    }
}

fn lookup_const_f64(name: &str) -> Option<f64> {
    match name.to_ascii_lowercase().as_str() {
        "pi" => Some(PI),
        "e" => Some(E),
        "tau" => Some(TAU),
        _ => None,
    }
}

fn lookup_const_complex(name: &str) -> Option<Complex64> {
    match name.to_ascii_lowercase().as_str() {
        "pi" => Some(Complex64::new(PI, 0.0)),
        "e" => Some(Complex64::new(E, 0.0)),
        "tau" => Some(Complex64::new(TAU, 0.0)),
        "i" => Some(Complex64::i()),
        _ => None,
    }
}
