//! Abstract Syntax Tree (AST) definitions for mathematical expressions.

/// Supported binary mathematical operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BinaryOp {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
    /// Exponentiation (`^`).
    Pow,
}

impl BinaryOp {
    /// Returns the symbol representation of the operator.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Pow => "^",
        }
    }
}

/// Supported unary mathematical operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnaryOp {
    /// Unary positive (`+`).
    Pos,
    /// Unary negative (`-`).
    Neg,
}

impl UnaryOp {
    /// Returns the symbol representation of the operator.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pos => "+",
            Self::Neg => "-",
        }
    }
}

/// Supported built-in mathematical functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Func {
    /// Sine function.
    Sin,
    /// Cosine function.
    Cos,
    /// Tangent function.
    Tan,
    /// Natural exponential function ($e^x$).
    Exp,
    /// Natural logarithm ($\ln(x)$).
    Ln,
    /// Base-10 logarithm.
    Log10,
    /// Square root ($\sqrt{x}$).
    Sqrt,
    /// Absolute value ($|x|$).
    Abs,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Hyperbolic tangent.
    Tanh,
    /// Inverse sine ($\arcsin(x)$).
    Asin,
    /// Inverse cosine ($\arccos(x)$).
    Acos,
    /// Inverse tangent ($\arctan(x)$).
    Atan,
}

impl Func {
    /// Returns the name of the function.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Exp => "exp",
            Self::Ln => "ln",
            Self::Log10 => "log10",
            Self::Sqrt => "sqrt",
            Self::Abs => "abs",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
        }
    }

    /// Looks up a built-in function by name (case-insensitive).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "sin" => Some(Self::Sin),
            "cos" => Some(Self::Cos),
            "tan" => Some(Self::Tan),
            "exp" => Some(Self::Exp),
            "ln" | "log" => Some(Self::Ln),
            "log10" => Some(Self::Log10),
            "sqrt" => Some(Self::Sqrt),
            "abs" => Some(Self::Abs),
            "sinh" => Some(Self::Sinh),
            "cosh" => Some(Self::Cosh),
            "tanh" => Some(Self::Tanh),
            "asin" | "arcsin" => Some(Self::Asin),
            "acos" | "arccos" => Some(Self::Acos),
            "atan" | "arctan" => Some(Self::Atan),
            _ => None,
        }
    }
}

/// Abstract Syntax Tree node representing a mathematical expression.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expr {
    /// Numeric constant.
    Number(f64),
    /// Named variable or constant identifier.
    Variable(String),
    /// Binary operation.
    Binary {
        /// Operator.
        op: BinaryOp,
        /// Left-hand operand.
        left: Box<Expr>,
        /// Right-hand operand.
        right: Box<Expr>,
    },
    /// Unary operation.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Sub-expression operand.
        expr: Box<Expr>,
    },
    /// Function call with arguments.
    Function {
        /// Built-in function.
        func: Func,
        /// Function argument expressions.
        args: Vec<Expr>,
    },
}

impl Expr {
    /// Creates a binary operation node.
    #[must_use]
    pub fn binary(op: BinaryOp, left: Expr, right: Expr) -> Self {
        Self::Binary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Creates a unary operation node.
    #[must_use]
    pub fn unary(op: UnaryOp, expr: Expr) -> Self {
        Self::Unary {
            op,
            expr: Box::new(expr),
        }
    }

    /// Creates a function call node.
    #[must_use]
    pub fn function(func: Func, args: Vec<Expr>) -> Self {
        Self::Function { func, args }
    }

    const fn has_children(&self) -> bool {
        matches!(
            self,
            Self::Binary { .. } | Self::Unary { .. } | Self::Function { .. }
        )
    }

    fn take_children(&mut self, worklist: &mut Vec<Expr>) {
        match self {
            Self::Number(_) | Self::Variable(_) => {}
            Self::Binary { left, right, .. } => {
                let lhs = std::mem::replace(&mut **left, Self::Number(0.0));
                let rhs = std::mem::replace(&mut **right, Self::Number(0.0));
                if lhs.has_children() {
                    worklist.push(lhs);
                }
                if rhs.has_children() {
                    worklist.push(rhs);
                }
            }
            Self::Unary { expr, .. } => {
                let inner = std::mem::replace(&mut **expr, Self::Number(0.0));
                if inner.has_children() {
                    worklist.push(inner);
                }
            }
            Self::Function { args, .. } => {
                for arg in std::mem::take(args) {
                    if arg.has_children() {
                        worklist.push(arg);
                    }
                }
            }
        }
    }
}

impl Drop for Expr {
    fn drop(&mut self) {
        if !self.has_children() {
            return;
        }
        let mut worklist = Vec::new();
        self.take_children(&mut worklist);
        while let Some(mut node) = worklist.pop() {
            node.take_children(&mut worklist);
        }
    }
}
