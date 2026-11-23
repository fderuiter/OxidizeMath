#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Zero-allocation `CompiledExpr::eval_3d` implicit CAD evaluator, parameter slider bank,
//! and ordered `[min, max]` X/Y/Z domain bound controls for [`super::VolumetricFieldStudioTool`].
// [cite:isosurface_extraction]

use super::state::{
    clamp_finite, ImplicitClosureMode, VolumetricStudioConfig, DEFAULT_EXPORT_SIZE_MM,
    DEFAULT_IMPLICIT_BOUNDS, DEFAULT_IMPLICIT_EXPR, DEFAULT_IMPLICIT_WALL, IMPLICIT_DOMAIN_MAX,
    IMPLICIT_DOMAIN_MIN, IMPLICIT_DOMAIN_MIN_SPAN, MAX_EXPORT_SIZE_MM, MAX_IMPLICIT_WALL,
    MIN_EXPORT_SIZE_MM, MIN_IMPLICIT_WALL,
};
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range, MathExpressionInputBar};
use eframe::egui;
use math_commons::ast::{parse, parse_equation, BinaryOp, CompiledExpr, Expr};
use math_explorer::applied::isosurface::VoxelGrid;

const MAX_IMPLICIT_SCALAR: f64 = 1.0e4;

const IMPLICIT_FORMULA_PRESETS: &[(&str, &str)] = &[
    (
        "Gyroid TPMS Lattice",
        "sin(k * x) * cos(k * y) + sin(k * y) * cos(k * z) + sin(k * z) * cos(k * x)",
    ),
    (
        "Schwarz-P Minimal Shell",
        "cos(k * x) + cos(k * y) + cos(k * z)",
    ),
    (
        "Schwarz-D Diamond",
        "sin(k * x) * sin(k * y) * sin(k * z) + sin(k * x) * cos(k * y) * cos(k * z) + cos(k * x) * sin(k * y) * cos(k * z) + cos(k * x) * cos(k * y) * sin(k * z)",
    ),
    (
        "CSG Hollow Cube-Sphere",
        "max(max(abs(x) - a, max(abs(y) - a, abs(z) - a)), r - sqrt(x^2 + y^2 + z^2))",
    ),
    (
        "CSG Cylinder Cross",
        "min(x^2 + y^2 - r^2, min(y^2 + z^2 - r^2, x^2 + z^2 - r^2))",
    ),
    (
        "Clamped Gaussian Shell",
        "exp(-0.35 * (x^2 + y^2 + z^2)) - clamp(0.5 + a * z, 0.15, 0.85)",
    ),
];

/// Default scalar value assigned to newly discovered implicit parameters.
pub fn default_implicit_param_value(name: &str) -> f64 {
    match name {
        "k" => 1.0,
        "a" => 1.8,
        "b" => 1.0,
        "r" => 2.2,
        _ => 1.0,
    }
}

/// Parses and compiles a 3D implicit formula $f(x, y, z)$ or equation `LHS = RHS` into [`CompiledExpr`].
pub fn compile_implicit_formula(raw: &str) -> Result<CompiledExpr, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("Implicit formula must not be empty".into());
    }
    let expr = match parse_equation(trimmed) {
        Ok((_, ast)) => ast,
        Err(primary_err) => {
            if let Some((lhs, rhs)) = trimmed.split_once('=') {
                if !rhs.contains('=') {
                    if let (Ok(l_ast), Ok(r_ast)) = (parse(lhs.trim()), parse(rhs.trim())) {
                        return compile_ast_with_limit(&Expr::binary(BinaryOp::Sub, l_ast, r_ast));
                    }
                }
            }
            return Err(primary_err.user_message_for_input(trimmed));
        }
    };
    compile_ast_with_limit(&expr)
}

fn compile_ast_with_limit(expr: &Expr) -> Result<CompiledExpr, String> {
    let compiled =
        CompiledExpr::compile_with_vars(expr, &["x", "y", "z"]).map_err(|e| e.to_string())?;
    if compiled.param_names().len() > MAX_SYNTHESIZED_PARAMS {
        return Err(format!(
            "Implicit formula defines {} parameters (maximum {MAX_SYNTHESIZED_PARAMS})",
            compiled.param_names().len()
        ));
    }
    Ok(compiled)
}

/// Pre-allocated runtime evaluator executing `CompiledExpr::eval_3d` with zero heap allocations.
pub struct ImplicitRuntimeEvaluator {
    compiled: CompiledExpr,
    cached_expr: String,
    param_vals: [f64; MAX_SYNTHESIZED_PARAMS],
    param_len: usize,
}

impl ImplicitRuntimeEvaluator {
    /// Compiles the initial formula from `cfg` and populates inline parameter slots.
    pub fn new(cfg: &VolumetricStudioConfig) -> Self {
        let compiled = compile_implicit_formula(&cfg.implicit_expr).unwrap_or_else(|_| {
            compile_implicit_formula(DEFAULT_IMPLICIT_EXPR).expect("default gyroid compiles")
        });
        let mut eval = Self {
            compiled,
            cached_expr: cfg.implicit_expr.clone(),
            param_vals: [1.0; MAX_SYNTHESIZED_PARAMS],
            param_len: 0,
        };
        eval.refresh_param_slots_zero_alloc(cfg);
        eval
    }

    /// Synchronizes bytecode (only when `implicit_expr` text changed) and parameter slots with zero heap allocations on unchanged text.
    pub fn sync_zero_alloc(&mut self, cfg: &VolumetricStudioConfig) {
        if self.cached_expr != cfg.implicit_expr {
            if let Ok(compiled) = compile_implicit_formula(&cfg.implicit_expr) {
                self.compiled = compiled;
                self.cached_expr.clear();
                self.cached_expr.push_str(&cfg.implicit_expr);
            }
        }
        self.refresh_param_slots_zero_alloc(cfg);
    }

    fn refresh_param_slots_zero_alloc(&mut self, cfg: &VolumetricStudioConfig) {
        let names = self.compiled.param_names();
        let count = names.len().min(MAX_SYNTHESIZED_PARAMS);
        self.param_len = count;
        for (idx, name) in names.iter().take(count).enumerate() {
            self.param_vals[idx] = cfg
                .implicit_params
                .get(name.as_str())
                .copied()
                .unwrap_or_else(|| default_implicit_param_value(name));
        }
    }

    /// Evaluates $f(x, y, z)$ across `grid` in-place with zero heap allocations (`>= 0` inside solid).
    pub fn populate_grid_in_place(&self, cfg: &VolumetricStudioConfig, grid: &mut VoxelGrid) {
        let (nx, ny, nz) = (grid.width(), grid.height(), grid.depth());
        if nx < 2 || ny < 2 || nz < 2 {
            return;
        }
        let dx = (cfg.x_bounds[1] - cfg.x_bounds[0]) / ((nx - 1) as f64);
        let dy = (cfg.y_bounds[1] - cfg.y_bounds[0]) / ((ny - 1) as f64);
        let dz = (cfg.z_bounds[1] - cfg.z_bounds[0]) / ((nz - 1) as f64);
        let params = &self.param_vals[..self.param_len];
        let level = cfg.implicit_level;
        let half_w = 0.5 * cfg.wall_thickness;
        let closure = cfg.implicit_closure;
        let buf = grid.data_mut();

        for iz in 0..nz {
            let z = cfg.z_bounds[0] + (iz as f64) * dz;
            for iy in 0..ny {
                let y = cfg.y_bounds[0] + (iy as f64) * dy;
                let row = (iz * ny + iy) * nx;
                for ix in 0..nx {
                    let x = cfg.x_bounds[0] + (ix as f64) * dx;
                    let f_val = self.compiled.eval_3d(x, y, z, params);
                    let signed_inside = if f_val.is_finite() {
                        match closure {
                            ImplicitClosureMode::HalfSpaceSolid => level - f_val,
                            ImplicitClosureMode::ThickenedWall => half_w - (f_val - level).abs(),
                        }
                        .clamp(-MAX_IMPLICIT_SCALAR, MAX_IMPLICIT_SCALAR)
                            as f32
                    } else {
                        -1.0_f32
                    };
                    buf[row + ix] = signed_inside;
                }
            }
        }
    }
}

/// Side-panel UI controller wrapping [`MathExpressionInputBar`] and X/Y/Z domain bound editors.
pub struct ImplicitFormulaUi {
    bar: MathExpressionInputBar,
}

impl ImplicitFormulaUi {
    /// Creates an [`ImplicitFormulaUi`] initialized from `cfg`.
    pub fn new(cfg: &VolumetricStudioConfig) -> Self {
        let mut ui_state = Self {
            bar: MathExpressionInputBar::new(
                "f(x,y,z)",
                &cfg.implicit_expr,
                IMPLICIT_FORMULA_PRESETS.to_vec(),
            )
            .with_axes(&["x", "y", "z"])
            .with_param_range(-5.0..=5.0, 1.0)
            .with_param_spec("k", "k (spatial freq)", 0.2..=4.0, 1.0)
            .with_param_spec("a", "a (box half-width)", 0.2..=4.0, 1.8)
            .with_param_spec("b", "b (modulation)", -3.0..=3.0, 1.0)
            .with_param_spec("r", "r (radius)", 0.2..=4.0, 2.2),
        };
        ui_state.sync_from_config(cfg);
        ui_state
    }

    /// Synchronizes the expression bar and parameter slider values from `cfg`.
    pub fn sync_from_config(&mut self, cfg: &VolumetricStudioConfig) {
        let normalized = normalize_for_input_bar(&cfg.implicit_expr);
        if self.bar.raw_text != normalized {
            self.bar.set_expression(&normalized);
        }
        for name in self.bar.active_param_names().to_vec() {
            let val = cfg
                .implicit_params
                .get(&name)
                .copied()
                .unwrap_or_else(|| default_implicit_param_value(&name));
            self.bar.set_param(&name, val);
        }
    }

    /// Renders implicit formula, parameter sliders, closure mode, level `c`, and X/Y/Z bounds.
    pub fn draw_controls(&mut self, ui: &mut egui::Ui, cfg: &mut VolumetricStudioConfig) -> bool {
        let mut changed = self.sync_bar_and_params_ui(ui, cfg);
        changed |= ui
            .add(
                egui::Slider::new(&mut cfg.implicit_level, -5.0..=5.0)
                    .clamping(egui::SliderClamping::Never)
                    .text("Level c"),
            )
            .changed();
        let prev_closure = cfg.implicit_closure;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut cfg.implicit_closure,
                ImplicitClosureMode::HalfSpaceSolid,
                ImplicitClosureMode::HalfSpaceSolid.label(),
            );
            ui.selectable_value(
                &mut cfg.implicit_closure,
                ImplicitClosureMode::ThickenedWall,
                ImplicitClosureMode::ThickenedWall.label(),
            );
        });
        changed |= cfg.implicit_closure != prev_closure;
        if cfg.implicit_closure == ImplicitClosureMode::ThickenedWall {
            changed |= ui
                .add(
                    egui::Slider::new(
                        &mut cfg.wall_thickness,
                        MIN_IMPLICIT_WALL..=MAX_IMPLICIT_WALL,
                    )
                    .text("Wall Width w"),
                )
                .changed();
        }
        changed |= draw_xyz_bounds_editor(ui, cfg);
        changed
    }

    fn sync_bar_and_params_ui(
        &mut self,
        ui: &mut egui::Ui,
        cfg: &mut VolumetricStudioConfig,
    ) -> bool {
        let bar_changed = self.bar.ui(ui);
        if !self.bar.is_valid() && !ui.ctx().wants_keyboard_input() {
            let normalized = normalize_for_input_bar(&self.bar.raw_text);
            if normalized != self.bar.raw_text {
                self.bar.set_expression(&normalized);
            }
        }
        if bar_changed && self.bar.is_valid() {
            cfg.implicit_expr = self.bar.raw_text.clone();
            cfg.implicit_params = self.bar.params_map();
            return true;
        }
        if self.bar.is_valid() {
            let latest_map = self.bar.params_map();
            if cfg.implicit_params != latest_map || cfg.implicit_expr != self.bar.raw_text {
                cfg.implicit_expr = self.bar.raw_text.clone();
                cfg.implicit_params = latest_map;
                return true;
            }
        }
        false
    }
}

fn draw_xyz_bounds_editor(ui: &mut egui::Ui, cfg: &mut VolumetricStudioConfig) -> bool {
    let mut changed = false;
    ui.label("Domain Bounds [min, max]:");
    for (label, bounds) in [
        ("X:", &mut cfg.x_bounds),
        ("Y:", &mut cfg.y_bounds),
        ("Z:", &mut cfg.z_bounds),
    ] {
        let prev = *bounds;
        let mut min_ch = false;
        let mut max_ch = false;
        ui.horizontal(|ui| {
            ui.label(label);
            min_ch = ui
                .add(bounded_drag_value(
                    &mut bounds[0],
                    IMPLICIT_DOMAIN_MIN..=IMPLICIT_DOMAIN_MAX,
                    0.1,
                ))
                .changed();
            ui.label("to");
            max_ch = ui
                .add(bounded_drag_value(
                    &mut bounds[1],
                    IMPLICIT_DOMAIN_MIN..=IMPLICIT_DOMAIN_MAX,
                    0.1,
                ))
                .changed();
        });
        let (mut lo, mut hi) = (bounds[0], bounds[1]);
        enforce_ordered_drag_range(
            &mut lo,
            &mut hi,
            IMPLICIT_DOMAIN_MIN,
            IMPLICIT_DOMAIN_MAX,
            IMPLICIT_DOMAIN_MIN_SPAN,
            min_ch,
            max_ch,
        );
        *bounds = [lo, hi];
        changed |= *bounds != prev;
    }
    if ui.small_button("Reset Domain [-π, π]³").clicked() {
        cfg.x_bounds = DEFAULT_IMPLICIT_BOUNDS;
        cfg.y_bounds = DEFAULT_IMPLICIT_BOUNDS;
        cfg.z_bounds = DEFAULT_IMPLICIT_BOUNDS;
        changed = true;
    }
    changed
}

fn normalize_for_input_bar(raw: &str) -> String {
    let trimmed = raw.trim();
    if parse_equation(trimmed).is_ok() {
        return trimmed.to_string();
    }
    if let Some((lhs, rhs)) = trimmed.split_once('=') {
        if !rhs.contains('=') && parse(lhs.trim()).is_ok() && parse(rhs.trim()).is_ok() {
            return format!("({}) - ({})", lhs.trim(), rhs.trim());
        }
    }
    trimmed.to_string()
}

pub(super) fn validate_implicit_config(cfg: &VolumetricStudioConfig) -> Result<(), String> {
    for (axis, [lo, hi]) in [
        ("x_bounds", cfg.x_bounds),
        ("y_bounds", cfg.y_bounds),
        ("z_bounds", cfg.z_bounds),
    ] {
        if !lo.is_finite()
            || !hi.is_finite()
            || !(IMPLICIT_DOMAIN_MIN..=IMPLICIT_DOMAIN_MAX).contains(&lo)
            || !(IMPLICIT_DOMAIN_MIN..=IMPLICIT_DOMAIN_MAX).contains(&hi)
            || lo + IMPLICIT_DOMAIN_MIN_SPAN > hi
        {
            return Err(format!("Implicit {axis} out of valid ordered bounds"));
        }
    }
    if cfg.implicit_expr.trim().is_empty() || cfg.implicit_expr.len() > 512 {
        return Err("implicit_expr must be 1..=512 characters".into());
    }
    compile_implicit_formula(&cfg.implicit_expr)?;
    if cfg.implicit_params.len() > MAX_SYNTHESIZED_PARAMS {
        return Err(format!(
            "implicit_params exceeds maximum of {MAX_SYNTHESIZED_PARAMS}"
        ));
    }
    for (k, v) in &cfg.implicit_params {
        if k.is_empty() || k.len() > 32 || !v.is_finite() || !(-50.0..=50.0).contains(v) {
            return Err(format!("Invalid implicit parameter '{k}'={v}"));
        }
    }
    Ok(())
}

pub(super) fn clamp_implicit_config_in_place(cfg: &mut VolumetricStudioConfig) {
    cfg.implicit_level = clamp_finite(cfg.implicit_level, -20.0, 20.0, 0.0);
    cfg.wall_thickness = clamp_finite(
        cfg.wall_thickness,
        MIN_IMPLICIT_WALL,
        MAX_IMPLICIT_WALL,
        DEFAULT_IMPLICIT_WALL,
    );
    cfg.export_size_mm = clamp_finite(
        cfg.export_size_mm,
        MIN_EXPORT_SIZE_MM,
        MAX_EXPORT_SIZE_MM,
        DEFAULT_EXPORT_SIZE_MM,
    );
    cfg.x_bounds = clamp_ordered_bounds(cfg.x_bounds);
    cfg.y_bounds = clamp_ordered_bounds(cfg.y_bounds);
    cfg.z_bounds = clamp_ordered_bounds(cfg.z_bounds);
    if compile_implicit_formula(&cfg.implicit_expr).is_err() {
        cfg.implicit_expr = DEFAULT_IMPLICIT_EXPR.to_string();
    }
    for val in cfg.implicit_params.values_mut() {
        *val = clamp_finite(*val, -50.0, 50.0, 1.0);
    }
}

fn clamp_ordered_bounds([b0, b1]: [f64; 2]) -> [f64; 2] {
    let lo = clamp_finite(
        b0,
        IMPLICIT_DOMAIN_MIN,
        IMPLICIT_DOMAIN_MAX - IMPLICIT_DOMAIN_MIN_SPAN,
        DEFAULT_IMPLICIT_BOUNDS[0],
    );
    let hi = clamp_finite(
        b1,
        lo + IMPLICIT_DOMAIN_MIN_SPAN,
        IMPLICIT_DOMAIN_MAX,
        DEFAULT_IMPLICIT_BOUNDS[1].max(lo + IMPLICIT_DOMAIN_MIN_SPAN),
    );
    [lo, hi]
}
