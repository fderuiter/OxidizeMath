//! Parameter-calibrated LaTeX, Rust 2024, and Python (`numpy`) code export for [`super::ComplexMappingTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ComplexMappingTool, MappingFunction};
use crate::framework::{CodeArtifact, InteractiveTool};
use math_commons::ast::{
    format_latex_scalar, format_python_float, format_rust_f64, parse_equation, BinaryOp, Expr,
    Func, UnaryOp, MAX_EVAL_DEPTH,
};
use std::collections::HashMap;

const RUST_COMPLEX_PRELUDE: &str = r#"#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self { Self { re, im } }
    pub const fn real(re: f64) -> Self { Self { re, im: 0.0 } }
    pub fn norm(self) -> f64 { self.re.hypot(self.im) }
    pub fn arg(self) -> f64 { self.im.atan2(self.re) }
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self::new(r * self.im.cos(), r * self.im.sin())
    }
    pub fn ln(self) -> Self { Self::new(self.norm().ln(), self.arg()) }
    pub fn log10(self) -> Self {
        let ln10 = std::f64::consts::LN_10;
        let l = self.ln();
        Self::new(l.re / ln10, l.im / ln10)
    }
    pub fn powf(self, exp: Self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::real(0.0);
        }
        (self.ln() * exp).exp()
    }
    pub fn sqrt(self) -> Self { self.powf(Self::real(0.5)) }
    pub fn cbrt(self) -> Self { self.powf(Self::real(1.0 / 3.0)) }
    pub fn sin(self) -> Self {
        Self::new(self.re.sin() * self.im.cosh(), self.re.cos() * self.im.sinh())
    }
    pub fn cos(self) -> Self {
        Self::new(self.re.cos() * self.im.cosh(), -self.re.sin() * self.im.sinh())
    }
    pub fn tan(self) -> Self { self.sin() / self.cos() }
    pub fn sinh(self) -> Self {
        Self::new(self.re.sinh() * self.im.cos(), self.re.cosh() * self.im.sin())
    }
    pub fn cosh(self) -> Self {
        Self::new(self.re.cosh() * self.im.cos(), self.re.sinh() * self.im.sin())
    }
    pub fn tanh(self) -> Self { self.sinh() / self.cosh() }
    pub fn asin(self) -> Self {
        let i = Self::new(0.0, 1.0);
        let one = Self::real(1.0);
        -i * (i * self + (one - self * self).sqrt()).ln()
    }
    pub fn acos(self) -> Self { Self::real(std::f64::consts::FRAC_PI_2) - self.asin() }
    pub fn atan(self) -> Self {
        let i = Self::new(0.0, 1.0);
        let one = Self::real(1.0);
        Self::new(0.0, 0.5) * ((one - i * self) / (one + i * self)).ln()
    }
    pub fn abs(self) -> Self { Self::real(self.norm()) }
    pub fn signum(self) -> Self {
        let n = self.norm();
        if n == 0.0 { Self::real(0.0) } else { Self::new(self.re / n, self.im / n) }
    }
    pub fn floor(self) -> Self { Self::new(self.re.floor(), self.im.floor()) }
    pub fn ceil(self) -> Self { Self::new(self.re.ceil(), self.im.ceil()) }
    pub fn round(self) -> Self { Self::new(self.re.round(), self.im.round()) }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self::new(self.re + rhs.re, self.im + rhs.im) }
}
impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self::new(self.re - rhs.re, self.im - rhs.im) }
}
impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.re * rhs.re - self.im * rhs.im, self.re * rhs.im + self.im * rhs.re)
    }
}
impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new((self.re * rhs.re + self.im * rhs.im) / denom, (self.im * rhs.re - self.re * rhs.im) / denom)
    }
}
impl std::ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self { Self::new(-self.re, -self.im) }
}"#;

impl ComplexMappingTool {
    pub(super) fn build_code_artifact(&self) -> Option<CodeArtifact> {
        if !self.has_finite_parameters() {
            return None;
        }
        let parsed_preset;
        let expr: &Expr = match self.function {
            MappingFunction::ZSquared => {
                parsed_preset = parse_equation("z^2").ok()?.1;
                &parsed_preset
            }
            MappingFunction::Exponential => {
                parsed_preset = parse_equation("exp(z)").ok()?.1;
                &parsed_preset
            }
            MappingFunction::Sine => {
                parsed_preset = parse_equation("sin(z)").ok()?.1;
                &parsed_preset
            }
            MappingFunction::Inverse => {
                parsed_preset = parse_equation("1 / z").ok()?.1;
                &parsed_preset
            }
            MappingFunction::Custom => self.math_input.expr()?,
        };

        let (symbolic_latex, calibrated_latex) = self.build_complex_latex(expr)?;
        let rust_source = self.build_complex_rust(expr)?;
        let python_source = self.build_complex_python(expr)?;

        Some(CodeArtifact {
            symbolic_latex,
            calibrated_latex,
            rust_source,
            python_source,
        })
    }

    fn build_complex_latex(&self, expr: &Expr) -> Option<(String, String)> {
        let rhs_latex = expr.to_latex();
        let z_re_tex = format_latex_scalar(self.inspection_point[0]).ok()?;
        let z_im_tex = format_latex_scalar(self.inspection_point[1]).ok()?;
        let gmin_tex = format_latex_scalar(self.grid_min).ok()?;
        let gmax_tex = format_latex_scalar(self.grid_max).ok()?;

        let symbolic = format!("\\begin{{aligned}}\nw(z) &= {rhs_latex}\n\\end{{aligned}}");
        let calibrated = format!(
            "\\begin{{aligned}}\nw(z) &= {rhs_latex} \\\\\nz_0 &= {z_re_tex} + {z_im_tex}i, \\quad \\text{{grid}} \\in [{gmin_tex}, {gmax_tex}]^2, \\quad N = {}\n\\end{{aligned}}",
            self.grid_density
        );
        Some((symbolic, calibrated))
    }

    fn build_complex_rust(&self, expr: &Expr) -> Option<String> {
        let rust_body = format_complex_rust_iterative(expr)?;
        let z_re_rs = format_rust_f64(self.inspection_point[0]).ok()?;
        let z_im_rs = format_rust_f64(self.inspection_point[1]).ok()?;
        Some(format!(
            "// Standalone Rust 2024 conformal map evaluator generated by OxidizeMath (zero external crates)\n\
             {RUST_COMPLEX_PRELUDE}\n\n\
             pub fn map_z(z: Complex) -> Complex {{\n\
             \x20   {rust_body}\n\
             }}\n\n\
             fn main() {{\n\
             \x20   let z0 = Complex::new({z_re_rs}, {z_im_rs});\n\
             \x20   let w0 = map_z(z0);\n\
             \x20   println!(\"w({{:.4}} + {{:.4}}i) = {{:.9}} + {{:.9}}i\", z0.re, z0.im, w0.re, w0.im);\n\
             }}\n"
        ))
    }

    fn build_complex_python(&self, expr: &Expr) -> Option<String> {
        let py_expr = expr.to_python_expr(&HashMap::new()).ok()?;
        let z_re_py = format_python_float(self.inspection_point[0]).ok()?;
        let z_im_py = format_python_float(self.inspection_point[1]).ok()?;
        let gmin_py = format_python_float(self.grid_min).ok()?;
        let gmax_py = format_python_float(self.grid_max).ok()?;
        Some(format!(
            "\"\"\"Standalone Python conformal mapping evaluator generated by OxidizeMath.\"\"\"\n\
             import numpy as np\n\n\
             def map_z(z: complex | np.ndarray):\n\
             \x20   z = np.asarray(z, dtype=np.complex128)\n\
             \x20   return {py_expr}\n\n\
             if __name__ == \"__main__\":\n\
             \x20   z0 = complex({z_re_py}, {z_im_py})\n\
             \x20   w0 = complex(map_z(z0))\n\
             \x20   axis = np.linspace({gmin_py}, {gmax_py}, {})\n\
             \x20   print(f\"w({{z0}}) = {{w0.real:.9f}} + {{w0.imag:.9f}}j (grid lines={{len(axis)}})\")\n",
            self.grid_density
        ))
    }
}

enum CFrame<'a> {
    Eval(&'a Expr, u8, usize),
    Lit(&'static str),
}

fn format_complex_rust_iterative(root: &Expr) -> Option<String> {
    let mut out = String::with_capacity(128);
    let mut stack = vec![CFrame::Eval(root, 0, 1)];
    while let Some(frame) = stack.pop() {
        match frame {
            CFrame::Lit(s) => out.push_str(s),
            CFrame::Eval(expr, parent_prec, depth) => {
                if depth > MAX_EVAL_DEPTH {
                    return None;
                }
                step_complex_frame(expr, parent_prec, depth, &mut stack, &mut out)?;
            }
        }
    }
    Some(out)
}

fn step_complex_frame<'a>(
    expr: &'a Expr,
    parent_prec: u8,
    depth: usize,
    stack: &mut Vec<CFrame<'a>>,
    out: &mut String,
) -> Option<()> {
    match expr {
        Expr::Number(n) => out.push_str(&format!("Complex::real({})", format_rust_f64(*n).ok()?)),
        Expr::Variable(v) => out.push_str(complex_symbol_literal(v)?),
        Expr::Unary { op, expr: inner } => {
            if *op == UnaryOp::Neg {
                if parent_prec >= 2 {
                    out.push_str("(-");
                    stack.push(CFrame::Lit(")"));
                } else {
                    out.push('-');
                }
                stack.push(CFrame::Eval(inner, 2, depth + 1));
            } else {
                stack.push(CFrame::Eval(inner, parent_prec, depth + 1));
            }
        }
        Expr::Binary { op, left, right } => {
            push_complex_binary(*op, (left, right), parent_prec, depth + 1, stack, out);
        }
        Expr::Function { func, args } => {
            let arg = args.first()?;
            stack.push(CFrame::Lit(complex_func_suffix(*func)));
            stack.push(CFrame::Eval(arg, 2, depth + 1));
        }
    }
    Some(())
}

fn complex_symbol_literal(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "z" => Some("z"),
        "i" => Some("Complex::new(0.0, 1.0)"),
        "pi" => Some("Complex::real(std::f64::consts::PI)"),
        "e" => Some("Complex::real(std::f64::consts::E)"),
        "tau" => Some("Complex::real(std::f64::consts::TAU)"),
        _ => None,
    }
}

fn push_complex_binary<'a>(
    op: BinaryOp,
    (left, right): (&'a Expr, &'a Expr),
    parent_prec: u8,
    next_depth: usize,
    stack: &mut Vec<CFrame<'a>>,
    out: &mut String,
) {
    if op == BinaryOp::Pow {
        stack.push(CFrame::Lit(")"));
        stack.push(CFrame::Eval(right, 0, next_depth));
        stack.push(CFrame::Lit(".powf("));
        stack.push(CFrame::Eval(left, 2, next_depth));
    } else {
        let sep = match op {
            BinaryOp::Add => " + ",
            BinaryOp::Sub => " - ",
            BinaryOp::Mul => " * ",
            BinaryOp::Div => " / ",
            BinaryOp::Pow => unreachable!(),
        };
        if parent_prec >= 1 {
            out.push('(');
            stack.push(CFrame::Lit(")"));
        }
        stack.push(CFrame::Eval(right, 2, next_depth));
        stack.push(CFrame::Lit(sep));
        stack.push(CFrame::Eval(left, 1, next_depth));
    }
}

const fn complex_func_suffix(func: Func) -> &'static str {
    match func {
        Func::Sin => ".sin()",
        Func::Cos => ".cos()",
        Func::Tan => ".tan()",
        Func::Asin => ".asin()",
        Func::Acos => ".acos()",
        Func::Atan => ".atan()",
        Func::Sinh => ".sinh()",
        Func::Cosh => ".cosh()",
        Func::Tanh => ".tanh()",
        Func::Exp => ".exp()",
        Func::Ln => ".ln()",
        Func::Log10 => ".log10()",
        Func::Sqrt => ".sqrt()",
        Func::Abs => ".abs()",
    }
}
