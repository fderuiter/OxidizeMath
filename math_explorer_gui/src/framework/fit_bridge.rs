//! Bridges `egui_plot` custom regression callbacks to `math_commons::ast::CompiledExpr`
//! while keeping `egui_plot/Cargo.toml` completely independent of `math_commons`.

use std::sync::Arc;

use egui_plot::{CustomFitEvaluator, MAX_FIT_PARAMS};
use math_commons::ast::{is_universal_constant, parse_equation, CompiledExpr};

#[derive(Debug)]
struct CompiledCustomFitEvaluator {
    compiled: CompiledExpr,
    param_names: Vec<String>,
}

impl CustomFitEvaluator for CompiledCustomFitEvaluator {
    fn param_names(&self) -> &[String] {
        &self.param_names
    }

    fn eval(&self, x: f64, params: &[f64]) -> Option<f64> {
        let y = self.compiled.eval_1d(x, params);
        y.is_finite().then_some(y)
    }
}

/// Compiles a raw user formula (e.g. `"a * exp(-b * x) + c"`) into a zero-allocation
/// [`CustomFitEvaluator`] with up to [`MAX_FIT_PARAMS`] (6) free parameters.
pub fn compile_custom_fit_expr(raw_formula: &str) -> Result<Arc<dyn CustomFitEvaluator>, String> {
    let (_lhs, expr) = parse_equation(raw_formula).map_err(|e| e.user_message())?;
    let free = expr.free_variables();
    let axis = if free.iter().any(|v| v == "x") || !free.iter().any(|v| v == "t") {
        "x"
    } else {
        "t"
    };

    let mut params = Vec::new();
    for var in free {
        if var != axis && !is_universal_constant(&var) && !params.contains(&var) {
            params.push(var);
        }
    }
    if params.len() > MAX_FIT_PARAMS {
        return Err(format!(
            "Custom fit supports at most {MAX_FIT_PARAMS} free parameters (found {})",
            params.len()
        ));
    }

    let param_refs: Vec<&str> = params.iter().map(String::as_str).collect();
    let compiled = CompiledExpr::compile_with_vars_and_params(&expr, &[axis], &param_refs)
        .map_err(|e| e.to_string())?;

    Ok(Arc::new(CompiledCustomFitEvaluator {
        compiled,
        param_names: params,
    }))
}

/// Registers the `CompiledExpr`-backed custom regression compiler with `egui_plot`.
pub fn ensure_custom_fit_bridge() {
    egui_plot::register_custom_fit_callback(compile_custom_fit_expr);
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;

    #[test]
    fn test_compiled_custom_fit_bridge_six_params_and_zero_alloc() {
        ensure_custom_fit_bridge();
        assert!(egui_plot::has_custom_fit_callback());

        // 6-parameter custom regression through math_commons::ast::CompiledExpr
        let formula = "a + b * x + c * exp(-d * x) + f * cos(w * x)";
        let pts: Vec<[f64; 2]> = (0..=50)
            .map(|i| {
                let x = i as f64 * 0.2;
                let y = 0.8 + 0.25 * x + 2.1 * (-0.5 * x).exp() + 0.7 * (1.4 * x).cos();
                [x, y]
            })
            .collect();

        let seeds = [0.5, 0.2, 1.8, 0.4, 0.5, 1.3];
        let fit = egui_plot::fit_custom_with_registered_callback(&pts, formula, &seeds, "Wave6")
            .expect("6-param custom regression should converge");
        assert_eq!(fit.coefficients.len(), 6);
        assert!(fit.r_squared >= 0.9999, "R² = {}", fit.r_squared);
        assert!(fit.rmse < 1e-4, "RMSE = {}", fit.rmse);

        // Evaluating the fitted curve at 100 points performs zero heap allocations
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (acc, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let mut sum = 0.0;
            for i in 0..100 {
                sum += fit.eval(i as f64 * 0.05);
            }
            sum
        });
        assert!(acc.is_finite());

        // Reject > 6 free parameters cleanly
        assert!(compile_custom_fit_expr("p1 + p2 + p3 + p4 + p5 + p6 + p7 * x").is_err());
    }
}
