//! Compact arena-indexed symbolic operators and DAG representation (`alloc` feature).
// [cite:engine_symbolic]

use alloc::string::String;
use alloc::vec::Vec;

/// Built-in univariate symbolic function tag inside [`SymOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymFunc1 {
    /// Sine $\sin(x)$.
    Sin,
    /// Cosine $\cos(x)$.
    Cos,
    /// Tangent $\tan(x)$.
    Tan,
    /// Exponential $e^x$.
    Exp,
    /// Natural logarithm $\ln(x)$.
    Ln,
    /// Base-10 logarithm $\log_{10}(x)$.
    Log10,
    /// Square root $\sqrt{x}$.
    Sqrt,
    /// Absolute value $|x|$.
    Abs,
    /// Hyperbolic sine $\sinh(x)$.
    Sinh,
    /// Hyperbolic cosine $\cosh(x)$.
    Cosh,
    /// Hyperbolic tangent $\tanh(x)$.
    Tanh,
    /// Inverse sine $\arcsin(x)$.
    Asin,
    /// Inverse cosine $\arccos(x)$.
    Acos,
    /// Inverse tangent $\arctan(x)$.
    Atan,
    /// Unit Heaviside step $\operatorname{step}(x)$.
    Step,
    /// Unit boxcar pulse $\operatorname{rect}(x)$.
    Rect,
}

/// Built-in binary symbolic function tag inside [`SymOp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymFunc2 {
    /// Minimum $\min(a, b)$.
    Min,
    /// Maximum $\max(a, b)$.
    Max,
    /// Four-quadrant inverse tangent $\operatorname{atan2}(y, x)$.
    Atan2,
}

/// Canonical e-node / symbolic DAG operator referencing child e-classes or DAG slots by `u32` ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymOp {
    /// Canonical IEEE-754 `f64` bit pattern (`-0.0` normalized to `+0.0`).
    Const(u64),
    /// Interned variable or symbolic constant index.
    Symbol(u16),
    /// Unary negation `-a`.
    Neg(u32),
    /// Binary addition `a + b`.
    Add([u32; 2]),
    /// Binary subtraction `a - b`.
    Sub([u32; 2]),
    /// Binary multiplication `a * b`.
    Mul([u32; 2]),
    /// Binary division `a / b`.
    Div([u32; 2]),
    /// Binary exponentiation `a ^ b`.
    Pow([u32; 2]),
    /// Univariate function application.
    Func1(SymFunc1, u32),
    /// Binary function application.
    Func2(SymFunc2, [u32; 2]),
    /// Ternary clamping `clamp(x, lo, hi)`.
    Clamp([u32; 3]),
}

/// Canonicalizes `f64` bit patterns so `-0.0` and `+0.0` share a single hashcons key.
#[inline]
#[must_use]
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn canonical_f64_bits(val: f64) -> u64 {
    if val == 0.0 {
        0.0_f64.to_bits()
    } else if val.is_nan() {
        f64::NAN.to_bits()
    } else {
        val.to_bits()
    }
}

impl SymOp {
    /// Constructs a canonical numeric literal node.
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn num(val: f64) -> Self {
        Self::Const(canonical_f64_bits(val))
    }

    /// Extracts the `f64` literal value if `self` is [`SymOp::Const`].
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::Const(bits) => Some(f64::from_bits(bits)),
            _ => None,
        }
    }

    /// Writes child IDs into `out` and returns the number of children (`0..=3`).
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn children_into(&self, out: &mut [u32; 3]) -> usize {
        match *self {
            Self::Const(_) | Self::Symbol(_) => 0,
            Self::Neg(a) | Self::Func1(_, a) => {
                out[0] = a;
                1
            }
            Self::Add([a, b])
            | Self::Sub([a, b])
            | Self::Mul([a, b])
            | Self::Div([a, b])
            | Self::Pow([a, b])
            | Self::Func2(_, [a, b]) => {
                out[0] = a;
                out[1] = b;
                2
            }
            Self::Clamp([a, b, c]) => {
                out[0] = a;
                out[1] = b;
                out[2] = c;
                3
            }
        }
    }

    /// Maps each child ID via `f` and normalizes commutative argument order (`Add`, `Mul`, `Min`, `Max`).
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn map_children<F: Fn(u32) -> u32>(self, f: F) -> Self {
        match self {
            Self::Const(_) | Self::Symbol(_) => self,
            Self::Neg(a) => Self::Neg(f(a)),
            Self::Add([a, b]) => {
                let (u, v) = (f(a), f(b));
                Self::Add(if u <= v { [u, v] } else { [v, u] })
            }
            Self::Sub([a, b]) => Self::Sub([f(a), f(b)]),
            Self::Mul([a, b]) => {
                let (u, v) = (f(a), f(b));
                Self::Mul(if u <= v { [u, v] } else { [v, u] })
            }
            Self::Div([a, b]) => Self::Div([f(a), f(b)]),
            Self::Pow([a, b]) => Self::Pow([f(a), f(b)]),
            Self::Func1(op, a) => Self::Func1(op, f(a)),
            Self::Func2(op, [a, b]) => {
                let (u, v) = (f(a), f(b));
                let pair = match op {
                    SymFunc2::Min | SymFunc2::Max if v < u => [v, u],
                    _ => [u, v],
                };
                Self::Func2(op, pair)
            }
            Self::Clamp([a, b, c]) => Self::Clamp([f(a), f(b), f(c)]),
        }
    }

    /// Local AST extraction cost of a single operator node (`1` for leaves, `2..=4` for operators).
    #[inline]
    #[must_use]
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub const fn local_cost(&self) -> u64 {
        match *self {
            Self::Const(_) | Self::Symbol(_) => 1,
            Self::Neg(_) => 2,
            Self::Add(_) | Self::Sub(_) | Self::Mul(_) => 2,
            Self::Div(_) | Self::Pow(_) => 3,
            Self::Func1(_, _) | Self::Func2(_, _) => 3,
            Self::Clamp(_) => 4,
        }
    }
}

/// Extracted flat post-order symbolic expression DAG produced by [`crate::egraph::EGraph::extract`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymExpr {
    /// Post-order arena of symbolic nodes (`0..=root`).
    pub nodes: Vec<SymOp>,
    /// Interned variable/symbol names indexed by `SymOp::Symbol(id)`.
    pub symbols: Vec<String>,
    /// Index of the root expression node inside `nodes`.
    pub root: u32,
}
