//! Zero-allocation `#[wasm_bindgen]` AST bytecode evaluator and WGSL shader compiler.
// [cite:math_commons]

use crate::error::WasmSdkError;
use math_commons::ast::{
    CompiledExpr, ComputeKernelConfig, Expr, emit_wgsl_3d_compute_kernel, parse,
};
use wasm_bindgen::prelude::*;

/// Compiled symbolic expression handle supporting zero-allocation 1D/3D batch evaluation and WGSL codegen.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmAstKernel {
    expr: Expr,
    compiled: CompiledExpr,
    param_values: Vec<f64>,
}

#[wasm_bindgen]
impl WasmAstKernel {
    /// Parses and compiles a 1D formula `f(x; params)` into flat postfix stack bytecode.
    #[wasm_bindgen(js_name = tryCompile1d)]
    pub fn try_compile_1d(source: &str) -> Result<Self, WasmSdkError> {
        let expr = parse(source)?;
        let compiled = CompiledExpr::compile_with_vars(&expr, &["x"])?;
        let param_values = vec![1.0; compiled.param_names().len()];
        Ok(Self {
            expr,
            compiled,
            param_values,
        })
    }

    /// Parses and compiles a 3D spatial formula `f(x, y, z; params)` into flat postfix stack bytecode.
    #[wasm_bindgen(js_name = tryCompile3d)]
    pub fn try_compile_3d(source: &str) -> Result<Self, WasmSdkError> {
        let expr = parse(source)?;
        let compiled = CompiledExpr::compile_with_vars(&expr, &["x", "y", "z"])?;
        let param_values = vec![1.0; compiled.param_names().len()];
        Ok(Self {
            expr,
            compiled,
            param_values,
        })
    }

    /// Number of discovered free scalar parameters after the coordinate axes.
    #[wasm_bindgen(js_name = paramCount)]
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.param_values.len()
    }

    /// Updates a named free parameter in place without recompiling bytecode.
    #[wasm_bindgen(js_name = setParam)]
    pub fn set_param(&mut self, name: &str, value: f64) -> Result<(), WasmSdkError> {
        let names = self.compiled.param_names();
        let Some(pos) = names.iter().position(|n| n == name) else {
            return Err(WasmSdkError::invalid_argument(
                "E_AST_UNKNOWN_PARAM",
                format!("Unknown free parameter '{name}'"),
                names.len(),
                0,
            ));
        };
        self.param_values[pos] = value;
        Ok(())
    }

    /// Evaluates a 1D formula over `xs` into `ys_out` (`Float64Array`) with zero heap allocations.
    #[wasm_bindgen(js_name = eval1dBatch)]
    pub fn eval_1d_batch(&self, xs: &[f64], ys_out: &mut [f64]) -> Result<(), WasmSdkError> {
        if xs.len() != ys_out.len() {
            return Err(WasmSdkError::invalid_argument(
                "E_AST_BATCH_LEN",
                "xs and ys_out slices must have identical lengths",
                xs.len(),
                ys_out.len(),
            ));
        }
        for (&x, y) in xs.iter().zip(ys_out.iter_mut()) {
            *y = self.compiled.eval_1d(x, &self.param_values);
        }
        Ok(())
    }

    /// Evaluates a 3D formula at `(x, y, z)` using currently bound parameter values.
    #[wasm_bindgen(js_name = eval3dPoint)]
    #[must_use]
    pub fn eval_3d_point(&self, x: f64, y: f64, z: f64) -> f64 {
        self.compiled.eval_3d(x, y, z, &self.param_values)
    }

    /// Evaluates a 3D formula across an isotropic `dim^3` grid over `[-half_extent, +half_extent]^3` into `out`.
    #[wasm_bindgen(js_name = eval3dGrid)]
    pub fn eval_3d_grid(
        &self,
        dim: usize,
        half_extent: f64,
        out: &mut [f64],
    ) -> Result<(), WasmSdkError> {
        let total = dim
            .checked_mul(dim)
            .and_then(|d2| d2.checked_mul(dim))
            .unwrap_or(usize::MAX);
        if dim < 2 || out.len() != total {
            return Err(WasmSdkError::invalid_argument(
                "E_AST_GRID_LEN",
                "Output buffer length must equal dim^3 with dim >= 2",
                total,
                out.len(),
            ));
        }
        let step = (2.0 * half_extent) / ((dim - 1) as f64);
        for z_i in 0..dim {
            let wz = -half_extent + (z_i as f64) * step;
            for y_i in 0..dim {
                let wy = -half_extent + (y_i as f64) * step;
                for x_i in 0..dim {
                    let wx = -half_extent + (x_i as f64) * step;
                    let idx = (z_i * dim + y_i) * dim + x_i;
                    out[idx] = self.compiled.eval_3d(wx, wy, wz, &self.param_values);
                }
            }
        }
        Ok(())
    }

    /// Lowers this expression into a scalar (`f32`) WGSL expression string.
    #[wasm_bindgen(js_name = toWgslExpr)]
    pub fn to_wgsl_expr(&self) -> Result<String, WasmSdkError> {
        Ok(self.expr.to_wgsl()?)
    }

    /// Emits a standalone WebGPU `@compute @workgroup_size(4, 4, 4)` WGSL shader for this 3D expression.
    #[wasm_bindgen(js_name = toComputeWgsl)]
    pub fn to_compute_wgsl(&self, dim: u32, half_extent: f64) -> Result<String, WasmSdkError> {
        let mut cfg = ComputeKernelConfig::new_3d(dim, half_extent);
        for (name, &val) in self.compiled.param_names().iter().zip(&self.param_values) {
            cfg.params.push((name.clone(), val));
        }
        Ok(emit_wgsl_3d_compute_kernel(&self.expr, &cfg)?)
    }
}
