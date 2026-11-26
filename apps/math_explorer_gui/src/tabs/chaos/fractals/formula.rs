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
    /// Builds the canonical quadratic polynomial $z_{n+1} = z_n^2 + c$ without parameters.
    #[must_use]
    pub fn quadratic() -> Self {
        let ast = parse("z^2 + c").expect("quadratic parses");
        let compiled =
            CompiledExpr::compile_complex(&ast, &["z", "c"]).expect("quadratic compiles");
        Self {
            compiled,
            params: [0.0; MAX_SYNTHESIZED_PARAMS],
            param_len: 0,
        }
    }

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

    /// Traces the critical orbit starting at $z_0 = 0$ for parameter `c` into `out` without heap allocation.
    #[inline]
    pub fn trace_critical_orbit(&self, c: Complex<f64>, out: &mut [Complex<f64>]) -> usize {
        if out.is_empty() || !c.re.is_finite() || !c.im.is_finite() {
            return 0;
        }
        let mut vars = [Complex::new(0.0, 0.0); 2 + MAX_SYNTHESIZED_PARAMS];
        vars[1] = c;
        for i in 0..self.param_len {
            vars[2 + i] = Complex::new(self.params[i], 0.0);
        }
        let total_vars = 2 + self.param_len;
        let mut written = 0;
        while written < out.len() {
            let z = vars[0];
            if !z.re.is_finite() || !z.im.is_finite() {
                break;
            }
            out[written] = z;
            written += 1;
            if z.norm_sqr() > 16.0 {
                break;
            }
            vars[0] = self.compiled.eval_complex_slice(&vars[..total_vars]);
        }
        written
    }

    #[inline]
    fn init_shadow_vars(
        &self,
        c: Complex<f64>,
        param0_override: Option<f64>,
    ) -> (
        [Complex<f64>; 2 + MAX_SYNTHESIZED_PARAMS],
        [Complex<f64>; 2 + MAX_SYNTHESIZED_PARAMS],
        usize,
    ) {
        let mut vars = [Complex::new(0.0, 0.0); 2 + MAX_SYNTHESIZED_PARAMS];
        let mut w_vars = [Complex::new(0.0, 0.0); 2 + MAX_SYNTHESIZED_PARAMS];
        vars[1] = c;
        w_vars[0] = Complex::new(1e-5, 0.0);
        w_vars[1] = c;
        for i in 0..self.param_len {
            let val = if i == 0 {
                param0_override.unwrap_or(self.params[i])
            } else {
                self.params[i]
            };
            vars[2 + i] = Complex::new(val, 0.0);
            w_vars[2 + i] = Complex::new(val, 0.0);
        }
        (vars, w_vars, 2 + self.param_len)
    }

    /// Classifies complex parameter `c` into `(regime_u8, primary_metric, secondary_metric)` on stack buffers.
    #[must_use]
    pub fn classify_c_plane_point(
        &self,
        c: Complex<f64>,
        param0_override: Option<f64>,
        max_iter: u32,
    ) -> (u8, f64, f64) {
        if !c.re.is_finite() || !c.im.is_finite() || max_iter == 0 {
            return (2, 0.0, 0.0);
        }
        let (mut vars, mut w_vars, total_vars) = self.init_shadow_vars(c, param0_override);
        let mut shadow_escaped = false;
        let mut tortoise = Complex::new(0.0, 0.0);
        let mut power = 1u32;
        let mut period = 1u32;
        let mut detected_period: Option<u32> = None;

        for iter in 0..max_iter {
            let z = vars[0];
            if !z.re.is_finite() || !z.im.is_finite() || z.norm_sqr() > 4.0 {
                let ratio = f64::from(iter) / f64::from(max_iter);
                let regime = if iter >= max_iter / 3 { 1 } else { 2 };
                return (regime, ratio, f64::from(iter));
            }
            let z_next = self.compiled.eval_complex_slice(&vars[..total_vars]);
            if iter >= max_iter / 2 && (z_next - tortoise).norm_sqr() < 1e-8 {
                detected_period = Some(detected_period.map_or(period, |p| p.min(period)));
            }
            if period == power {
                tortoise = z_next;
                power = power.saturating_mul(2);
                period = 0;
            }
            period += 1;
            vars[0] = z_next;

            if !shadow_escaped {
                let w_next = self.compiled.eval_complex_slice(&w_vars[..total_vars]);
                if !w_next.re.is_finite() || !w_next.im.is_finite() || w_next.norm_sqr() > 4.0 {
                    shadow_escaped = true;
                } else {
                    w_vars[0] = w_next;
                }
            }
        }

        let sep = (w_vars[0] - vars[0]).norm();
        if shadow_escaped || !sep.is_finite() || sep > 5e-3 {
            (1, 1.0, f64::from(max_iter))
        } else {
            let rate = (sep / 1e-5).max(1e-12).ln() / f64::from(max_iter);
            (0, rate, f64::from(detected_period.unwrap_or(1)))
        }
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
