#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use math_commons::ast::{parse, CompiledExpr};
use math_explorer::physics::chaos::fractals::{renormalize_escape_step, SMOOTH_ESCAPE_RADIUS_SQR};
use num_complex::Complex;

pub const DEFAULT_FRACTAL_EXPR: &str = "z^3 + c";

/// Immutable, `Arc`-shareable compiled complex escape formula $z_{n+1} = f(z_n, c)$
/// evaluated on progressive background worker passes without heap allocation.
#[derive(Clone, Debug)]
pub struct CompiledFractalJob {
    compiled: CompiledExpr,
    params: [f64; MAX_SYNTHESIZED_PARAMS],
    param_len: usize,
}

impl CompiledFractalJob {
    /// Snapshots the active complex [`CompiledExpr`] and parameter slice from `bar`.
    #[must_use]
    pub fn from_bar(bar: &MathExpressionInputBar) -> Self {
        let compiled = bar.active_compiled().cloned().unwrap_or_else(|| {
            let fallback = parse("z^2 + c").expect("fallback parses");
            CompiledExpr::compile_complex(&fallback, &["z", "c"]).expect("fallback compiles")
        });
        let slice = bar.param_slice();
        let param_len = slice.len().min(MAX_SYNTHESIZED_PARAMS);
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        params[..param_len].copy_from_slice(&slice[..param_len]);
        Self {
            compiled,
            params,
            param_len,
        }
    }

    /// Evaluates escape-time iterations up to `max_iter` with escape radius $|z|^2 > 4$ on stack buffers.
    #[inline]
    #[must_use]
    pub fn escape_time(&self, z0: Complex<f64>, c: Complex<f64>, max_iter: u32) -> u32 {
        let mut vars = [Complex::new(0.0, 0.0); 2 + MAX_SYNTHESIZED_PARAMS];
        vars[0] = z0;
        vars[1] = c;
        for i in 0..self.param_len {
            vars[2 + i] = Complex::new(self.params[i], 0.0);
        }
        let total_vars = 2 + self.param_len;
        let mut iter = 0;
        while iter < max_iter {
            let z = vars[0];
            if !z.re.is_finite() || !z.im.is_finite() || z.norm_sqr() > 4.0 {
                break;
            }
            vars[0] = self.compiled.eval_complex_slice(&vars[..total_vars]);
            iter += 1;
        }
        iter
    }

    /// Evaluates continuous Douady-Hubbard renormalized escape time up to `max_iter` without heap allocation.
    #[inline]
    #[must_use]
    pub fn smooth_escape_time(&self, z0: Complex<f64>, c: Complex<f64>, max_iter: u32) -> f64 {
        let mut vars = [Complex::new(0.0, 0.0); 2 + MAX_SYNTHESIZED_PARAMS];
        vars[0] = z0;
        vars[1] = c;
        for i in 0..self.param_len {
            vars[2 + i] = Complex::new(self.params[i], 0.0);
        }
        let total_vars = 2 + self.param_len;
        let mut prev_r2 = z0.norm_sqr();
        for iter in 0..max_iter {
            let z = vars[0];
            let r2 = z.norm_sqr();
            if !z.re.is_finite() || !z.im.is_finite() || !r2.is_finite() {
                return f64::from(iter);
            }
            if r2 > SMOOTH_ESCAPE_RADIUS_SQR {
                return renormalize_escape_step(iter, prev_r2, r2)
                    .min(f64::from(max_iter) - 1.0e-6);
            }
            prev_r2 = r2;
            vars[0] = self.compiled.eval_complex_slice(&vars[..total_vars]);
        }
        f64::from(max_iter)
    }
}

/// Builds the default complex [`MathExpressionInputBar`] for custom escape-time fractals.
#[must_use]
pub fn default_fractal_input_bar() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
        "z_{n+1}",
        DEFAULT_FRACTAL_EXPR,
        vec![
            ("Multibrot Cubic", "z^3 + c"),
            ("Multibrot Quartic", "z^4 + c"),
            ("Scaled Quadratic", "a * z^2 + c"),
            ("Complex Sine", "sin(z) + c"),
            ("Complex Cosine", "c * cos(z)"),
        ],
    )
    .with_axes(&["z", "c"])
    .with_complex_mode(true)
    .with_param_range(-3.0..=3.0, 1.0)
}
