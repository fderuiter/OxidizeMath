//! LaTeX representation generator for Abstract Syntax Tree expressions.

use super::expr::{BinaryOp, Expr, Func, UnaryOp};

impl Expr {
    /// Converts the AST expression to a formatted LaTeX string.
    #[must_use]
    pub fn to_latex(&self) -> String {
        self.to_latex_prec(0)
    }

    fn to_latex_prec(&self, parent_prec: u8) -> String {
        match self {
            Self::Number(n) => format_number(*n),
            Self::Variable(name) => format_variable(name),
            Self::Binary { op, left, right } => format_binary(*op, left, right, parent_prec),
            Self::Unary { op, expr } => format_unary(*op, expr),
            Self::Function { func, args } => format_function(*func, args),
        }
    }
}

fn format_number(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e12 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn format_variable(name: &str) -> String {
    if name.eq_ignore_ascii_case("pi") {
        r"\pi".to_string()
    } else if name.eq_ignore_ascii_case("tau") {
        r"\tau".to_string()
    } else if name.contains('_') {
        name.replace('_', r"\_")
    } else {
        name.to_string()
    }
}

fn format_binary(op: BinaryOp, left: &Expr, right: &Expr, parent_prec: u8) -> String {
    let (prec, s) = match op {
        BinaryOp::Add => (
            1,
            format!("{} + {}", left.to_latex_prec(1), right.to_latex_prec(1)),
        ),
        BinaryOp::Sub => (
            1,
            format!("{} - {}", left.to_latex_prec(1), right.to_latex_prec(2)),
        ),
        BinaryOp::Mul => (
            3,
            format!(
                r"{} \cdot {}",
                left.to_latex_prec(3),
                right.to_latex_prec(3)
            ),
        ),
        BinaryOp::Div => (
            4,
            format!(r"\frac{{{}}}{{{}}}", left.to_latex(), right.to_latex()),
        ),
        BinaryOp::Pow => (
            5,
            format!(r"{{{}}}^{{{}}}", left.to_latex_prec(5), right.to_latex()),
        ),
    };

    if prec < parent_prec && op != BinaryOp::Div {
        format!(r"\left({s}\right)")
    } else {
        s
    }
}

fn format_unary(op: UnaryOp, expr: &Expr) -> String {
    let inner = expr.to_latex_prec(6);
    match op {
        UnaryOp::Pos => format!("+{inner}"),
        UnaryOp::Neg => format!("-{inner}"),
    }
}

fn format_function(func: Func, args: &[Expr]) -> String {
    match func {
        Func::Sqrt if !args.is_empty() => format!(r"\sqrt{{{}}}", args[0].to_latex()),
        Func::Abs if !args.is_empty() => format!(r"\left|{}\right|", args[0].to_latex()),
        _ => {
            let name = latex_func_name(func);
            let arg_strs: Vec<String> = args.iter().map(Expr::to_latex).collect();
            format!(r"{name}\left({}\right)", arg_strs.join(", "))
        }
    }
}

fn latex_func_name(func: Func) -> &'static str {
    match func {
        Func::Sin => r"\sin",
        Func::Cos => r"\cos",
        Func::Tan => r"\tan",
        Func::Exp => r"\exp",
        Func::Ln => r"\ln",
        Func::Log10 => r"\log",
        Func::Sinh => r"\sinh",
        Func::Cosh => r"\cosh",
        Func::Tanh => r"\tanh",
        Func::Asin => r"\arcsin",
        Func::Acos => r"\arccos",
        Func::Atan => r"\arctan",
        _ => func.as_str(),
    }
}
