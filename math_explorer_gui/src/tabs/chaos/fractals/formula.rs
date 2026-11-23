#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use math_commons::ast::{parse, CompiledExpr};
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

    /// Evaluates continuous escape potential on stack buffers for Marching Squares contours and 16-bit textures.
    #[inline]
    #[must_use]
    pub fn smooth_escape(&self, z0: Complex<f64>, c: Complex<f64>, max_iter: u32) -> f64 {
        let mut vars = [Complex::new(0.0, 0.0); 2 + MAX_SYNTHESIZED_PARAMS];
        vars[0] = z0;
        vars[1] = c;
        for i in 0..self.param_len {
            vars[2 + i] = Complex::new(self.params[i], 0.0);
        }
        let total_vars = 2 + self.param_len;
        let max_pot = f64::from(max_iter);
        let mut iter = 0_u32;
        while iter < max_iter {
            let z = vars[0];
            let norm_sq = z.norm_sqr();
            if !z.re.is_finite() || !z.im.is_finite() || !norm_sq.is_finite() {
                return f64::from(iter).clamp(0.0, max_pot);
            }
            if norm_sq > 256.0 {
                let log_zn = norm_sq.ln() * 0.5;
                let frac = (log_zn.ln() / std::f64::consts::LN_2).clamp(0.0, 8.0);
                return (f64::from(iter) + 1.0 - frac).clamp(0.0, max_pot);
            }
            vars[0] = self.compiled.eval_complex_slice(&vars[..total_vars]);
            iter += 1;
        }
        max_pot
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
