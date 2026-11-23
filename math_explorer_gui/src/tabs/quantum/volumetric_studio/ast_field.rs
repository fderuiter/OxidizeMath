#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation 3D `CompiledExpr::eval_3d` kernel and `MathExpressionInputBar` controller
//! for [`super::VolumetricFieldStudioTool`].
// [cite:isosurface_extraction]
// [cite:quantum_mechanics]

use super::state::{clamp_finite, VolumetricStudioConfig, STUDIO_VIEWPORT_EXTENT};
use crate::framework::math_input::{MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use eframe::egui;
use math_commons::ast::{parse_equation, CompiledExpr};
use math_explorer::applied::isosurface::VoxelGrid;
use math_explorer::math_kernel::types::flatten_3d_index;
use std::collections::BTreeMap;

/// Default 3D Schwarz P minimal-surface implicit formula.
pub const DEFAULT_CUSTOM_3D_EXPR: &str = "cos(k * x) + cos(k * y) + cos(k * z)";
/// Triply-periodic Schoen Gyroid minimal-surface formula.
pub const GYROID_3D_EXPR: &str =
    "sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)";
/// Hydrogenic-like `4f_xyz` octant nodal orbital formula.
pub const ORBITAL_4F_XYZ_EXPR: &str = "x * y * z * exp(-a * sqrt(x^2 + y^2 + z^2))";
/// Singular inverse-square pole formula used to verify finite clamping before `FiniteStateGuard`.
pub const SINGULAR_POLE_3D_EXPR: &str = "1 / (x^2 + y^2 + z^2)";
/// Maximum allowed byte length for persisted `custom_expr` strings.
pub const MAX_CUSTOM_3D_EXPR_BYTES: usize = 512;
/// Downsampled cubic voxel resolution (`20^3 = 8,000` voxels) used during active slider drags.
pub const PREVIEW_DRAG_GRID: usize = 20;
/// Finite magnitude ceiling applied to evaluated 3D AST samples prior to `FiniteStateGuard`.
const VOXEL_SAMPLE_CLAMP: f64 = 500.0;

/// Curated 3D implicit AST templates surfaced in [`MathExpressionInputBar`].
#[must_use]
pub fn custom_3d_formula_templates() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Schwarz P Minimal Surface", DEFAULT_CUSTOM_3D_EXPR),
        ("Schoen Gyroid Labyrinth", GYROID_3D_EXPR),
        ("4f_xyz Octant Orbital", ORBITAL_4F_XYZ_EXPR),
        ("Clamped Singular Pole 1/r²", SINGULAR_POLE_3D_EXPR),
    ]
}

/// Validates that `raw_expr` fits byte bounds, parses, and compiles over `["x", "y", "z"]`
/// with at most [`MAX_SYNTHESIZED_PARAMS`] free parameters.
pub fn validate_custom_3d_formula(raw_expr: &str) -> Result<CompiledExpr, String> {
    if raw_expr.len() > MAX_CUSTOM_3D_EXPR_BYTES {
        return Err(format!(
            "custom_expr length {} exceeds {MAX_CUSTOM_3D_EXPR_BYTES} bytes",
            raw_expr.len()
        ));
    }
    let trimmed = raw_expr.trim();
    if trimmed.is_empty() {
        return Err("custom_expr cannot be empty".into());
    }
    let (_, expr) =
        parse_equation(trimmed).map_err(|e| format!("Invalid 3D custom_expr syntax: {e:?}"))?;
    let compiled = CompiledExpr::compile_with_vars(&expr, &["x", "y", "z"])
        .map_err(|e| format!("Invalid 3D custom_expr bytecode: {e}"))?;
    if compiled.param_names().len() > MAX_SYNTHESIZED_PARAMS {
        return Err(format!(
            "3D custom_expr defines {} free parameters (max {MAX_SYNTHESIZED_PARAMS})",
            compiled.param_names().len()
        ));
    }
    Ok(compiled)
}

/// Validates `custom_expr` and `custom_params` on [`VolumetricStudioConfig`].
pub fn validate_custom_ast_config(cfg: &VolumetricStudioConfig) -> Result<(), String> {
    validate_custom_3d_formula(&cfg.custom_expr)?;
    if cfg.custom_params.len() > MAX_SYNTHESIZED_PARAMS {
        return Err(format!(
            "custom_params count {} exceeds {MAX_SYNTHESIZED_PARAMS}",
            cfg.custom_params.len()
        ));
    }
    for (name, &val) in &cfg.custom_params {
        if name.is_empty() || name.len() > 32 {
            return Err(format!("Invalid custom_params key '{name}'"));
        }
        if !val.is_finite() || !(-1.0e4..=1.0e4).contains(&val) {
            return Err(format!("custom_params['{name}']={val} out of finite range"));
        }
    }
    Ok(())
}

/// Clamps `custom_expr` and `custom_params` on [`VolumetricStudioConfig`] in place.
pub fn clamp_custom_ast_config_in_place(cfg: &mut VolumetricStudioConfig) {
    if cfg.custom_expr.len() > MAX_CUSTOM_3D_EXPR_BYTES
        || validate_custom_3d_formula(&cfg.custom_expr).is_err()
    {
        cfg.custom_expr = DEFAULT_CUSTOM_3D_EXPR.to_string();
    }
    let mut cleaned = BTreeMap::new();
    for (k, &v) in cfg.custom_params.iter().take(MAX_SYNTHESIZED_PARAMS) {
        if !k.is_empty() && k.len() <= 32 {
            cleaned.insert(k.clone(), clamp_finite(v, -1.0e4, 1.0e4, 1.0));
        }
    }
    cfg.custom_params = cleaned;
}

/// Sanitizes a single 3D AST evaluation sample to a finite `f32` in `[-500.0, 500.0]`.
#[inline]
#[must_use]
pub fn sanitize_voxel_sample(value: f64) -> f32 {
    if value.is_finite() {
        value.clamp(-VOXEL_SAMPLE_CLAMP, VOXEL_SAMPLE_CLAMP) as f32
    } else {
        0.0
    }
}

/// Pre-compiled stack-parameterized 3D AST kernel executing zero-allocation `VoxelGrid` sweeps.
#[derive(Clone, Debug)]
pub struct CompiledAstFieldKernel {
    compiled: CompiledExpr,
    params: [f64; MAX_SYNTHESIZED_PARAMS],
    param_len: usize,
}

impl Default for CompiledAstFieldKernel {
    fn default() -> Self {
        let compiled = validate_custom_3d_formula(DEFAULT_CUSTOM_3D_EXPR)
            .expect("default 3D Schwarz P expression compiles");
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        params[0] = 2.5;
        let param_len = compiled.param_names().len().min(MAX_SYNTHESIZED_PARAMS);
        Self {
            compiled,
            params,
            param_len,
        }
    }
}

impl CompiledAstFieldKernel {
    /// Replaces the compiled bytecode and synchronizes inline parameter values from `param_map`.
    pub fn sync_from_config_expr(
        &mut self,
        raw_expr: &str,
        param_map: &BTreeMap<String, f64>,
    ) -> bool {
        let Ok(compiled) = validate_custom_3d_formula(raw_expr) else {
            return false;
        };
        let names = compiled.param_names();
        let len = names.len().min(MAX_SYNTHESIZED_PARAMS);
        let mut next_params = [0.0; MAX_SYNTHESIZED_PARAMS];
        for (i, name) in names.iter().take(len).enumerate() {
            let fallback = default_param_value(name);
            let raw_val = param_map.get(name).copied().unwrap_or(fallback);
            next_params[i] = if raw_val.is_finite() {
                raw_val.clamp(-1.0e4, 1.0e4)
            } else {
                fallback
            };
        }
        self.compiled = compiled;
        self.params = next_params;
        self.param_len = len;
        true
    }

    /// Copies already-compiled bytecode and inline parameter slice without map lookups.
    pub fn sync_from_bar(&mut self, compiled: &CompiledExpr, slice: &[f64]) {
        if self.compiled != *compiled {
            self.compiled = compiled.clone();
        }
        self.set_params_slice(slice);
    }

    /// Updates the fixed stack parameter array in place with zero heap allocations.
    #[inline]
    pub fn set_params_slice(&mut self, slice: &[f64]) {
        let len = slice.len().min(MAX_SYNTHESIZED_PARAMS);
        for (dst, &src) in self.params.iter_mut().zip(slice.iter()).take(len) {
            *dst = if src.is_finite() {
                src.clamp(-1.0e4, 1.0e4)
            } else {
                0.0
            };
        }
        self.param_len = len;
    }

    /// Populates `grid` across `[-2.5, +2.5]^3` via [`CompiledExpr::eval_3d`] with zero heap allocations.
    pub fn populate_voxel_grid_in_place(&self, grid: &mut VoxelGrid) {
        let dim = grid.width().max(2);
        let step = (2.0 * STUDIO_VIEWPORT_EXTENT) / ((dim - 1) as f64);
        let origin = -STUDIO_VIEWPORT_EXTENT;
        let param_slice = &self.params[..self.param_len];
        let out = grid.data_mut();
        for z in 0..dim {
            let wz = origin + (z as f64) * step;
            for y in 0..dim {
                let wy = origin + (y as f64) * step;
                for x in 0..dim {
                    let wx = origin + (x as f64) * step;
                    let raw = self.compiled.eval_3d(wx, wy, wz, param_slice);
                    out[flatten_3d_index(x, y, z, dim, dim)] = sanitize_voxel_sample(raw);
                }
            }
        }
    }
}

/// UI controller pairing [`MathExpressionInputBar`] (`["x", "y", "z"]`) with `ParameterSliderBank`.
#[derive(Clone, Debug)]
pub struct CustomAstFieldController {
    /// Interactive formula bar with synthesized parameter sliders (`<= 6`).
    pub bar: MathExpressionInputBar,
}

impl Default for CustomAstFieldController {
    fn default() -> Self {
        let bar = MathExpressionInputBar::new(
            "f(x, y, z)",
            DEFAULT_CUSTOM_3D_EXPR,
            custom_3d_formula_templates(),
        )
        .with_axes(&["x", "y", "z"])
        .with_param_range(-4.0..=4.0, 1.0)
        .with_param_spec("k", "Spatial Frequency k", 0.4..=4.0, 2.5)
        .with_param_spec("a", "Radial Decay a", 0.2..=3.5, 1.5)
        .with_param_spec("b", "Coupling b", -4.0..=4.0, 1.0)
        .with_param_spec("c", "Offset c", -2.0..=2.0, 0.0)
        .with_param_spec("r0", "Shell Radius r0", 0.2..=2.4, 1.2)
        .with_param_spec("w", "Scale w", 0.2..=4.0, 1.0);
        Self { bar }
    }
}

impl CustomAstFieldController {
    /// Synchronizes the formula bar from persisted `(custom_expr, custom_params)`.
    pub fn load_from_config(&mut self, expr: &str, params: &BTreeMap<String, f64>) {
        if self.bar.raw_text != expr {
            self.bar.set_expression(expr);
        }
        if !params.is_empty() {
            self.bar.set_params_map(params);
        }
    }

    /// Renders the 3D expression bar and inline `ParameterSliderBank`, writing updates to `(expr, params)`.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        expr_out: &mut String,
        params_out: &mut BTreeMap<String, f64>,
    ) -> bool {
        let changed = self.bar.ui(ui);
        if changed && self.bar.is_valid() && self.bar.raw_text.len() <= MAX_CUSTOM_3D_EXPR_BYTES {
            *expr_out = self.bar.raw_text.clone();
            *params_out = self.bar.params_map();
        }
        changed
    }
}

fn default_param_value(name: &str) -> f64 {
    match name {
        "k" => 2.5,
        "a" => 1.5,
        "c" => 0.0,
        "r0" => 1.2,
        _ => 1.0,
    }
}
