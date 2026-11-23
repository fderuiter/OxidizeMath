//! Zero-allocation `#![no_std]` error definitions for ODE solvers, quadrature, and Hermite root-bracketing.
// [cite:engine_diffeq]

use core::fmt;

/// Typed `Copy` error variants returned by `engine_diffeq` numerical routines.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DiffeqError {
    /// Integration or root-bracketing bounds `[a, b]` are inverted (`a > b`) or non-finite.
    InvalidBounds {
        /// Lower bound `a`.
        lower: f64,
        /// Upper bound `b`.
        upper: f64,
    },
    /// Requested numerical tolerance `tol` is non-positive or non-finite.
    InvalidTolerance {
        /// Supplied tolerance scalar.
        tol: f64,
    },
    /// Time step `dt` or interval width `h` is zero or non-finite.
    InvalidStepSize {
        /// Supplied step size scalar.
        dt: f64,
    },
    /// Endpoint evaluations do not bracket a root (`f(a) * f(b) > 0` or non-finite).
    NotBracketed {
        /// Endpoint evaluation at the lower bound.
        fa: f64,
        /// Endpoint evaluation at the upper bound.
        fb: f64,
    },
    /// Iterative solver exhausted `max_iter` steps without meeting tolerance.
    MaxIterationsExceeded {
        /// Maximum iterations attempted.
        max_iter: usize,
        /// Final residual magnitude or interval width.
        residual: f64,
    },
    /// Linear stage or Newton system encountered a singular or near-singular matrix pivot.
    SingularJacobian {
        /// Zero-based pivot column where breakdown occurred.
        pivot_index: usize,
    },
    /// Caller-supplied `SolverWorkspace` slice capacity is smaller than required.
    InsufficientWorkspace {
        /// Minimum `f64` elements required.
        required: usize,
        /// Actual `f64` elements available.
        available: usize,
    },
    /// Objective, derivative, or integrand returned `NaN` or `±Inf`.
    NonFiniteEvaluation,
}

impl fmt::Display for DiffeqError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidBounds { lower, upper } => {
                write!(f, "Invalid interval bounds [{lower}, {upper}]")
            }
            Self::InvalidTolerance { tol } => {
                write!(
                    f,
                    "Tolerance must be finite and strictly positive (got {tol})"
                )
            }
            Self::InvalidStepSize { dt } => {
                write!(f, "Step size must be finite and non-zero (got {dt})")
            }
            Self::NotBracketed { fa, fb } => {
                write!(f, "Root not bracketed at endpoints (fa={fa}, fb={fb})")
            }
            Self::MaxIterationsExceeded { max_iter, residual } => write!(
                f,
                "Solver exceeded {max_iter} iterations (residual={residual})"
            ),
            Self::SingularJacobian { pivot_index } => {
                write!(f, "Singular Jacobian matrix at pivot column {pivot_index}")
            }
            Self::InsufficientWorkspace {
                required,
                available,
            } => write!(
                f,
                "Insufficient workspace scratch capacity: need {required} f64s, got {available}"
            ),
            Self::NonFiniteEvaluation => {
                write!(f, "Function or derivative evaluation returned NaN or Inf")
            }
        }
    }
}

impl core::error::Error for DiffeqError {}
