#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Shared `CompiledExpr`-backed custom heightfield `z = f(x, y)` and parametric triple
//! `(x(u, v), y(u, v), z(u, v))` formula and domain bound controls for `SurfaceViewer`
//! and `CurvatureHeatmap`.

use crate::framework::math_input::{ParameterSliderBank, MAX_SYNTHESIZED_PARAMS};
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range, MathExpressionInputBar};
use eframe::egui;
use math_commons::ast::{is_universal_constant, parse, CompiledExpr};
use math_explorer::pure_math::differential_geometry::surface::{
    CompiledHeightfieldSurface, CompiledParametricSurface, ParametricSurfaceKind,
};
use std::collections::BTreeMap;
use std::f64::consts::PI;

pub const DEFAULT_HEIGHTFIELD_EXPR: &str = "a * (x^2 - y^2)";
pub const DEFAULT_PARAM_X_EXPR: &str = "(2 + a * cos(v)) * cos(u)";
pub const DEFAULT_PARAM_Y_EXPR: &str = "(2 + a * cos(v)) * sin(u)";
pub const DEFAULT_PARAM_Z_EXPR: &str = "a * sin(v) + 0.3 * sin(2 * u)";

pub const CUSTOM_DOMAIN_MIN_BOUND: f64 = -50.0;
pub const CUSTOM_DOMAIN_MAX_BOUND: f64 = 50.0;
pub const CUSTOM_DOMAIN_MIN_SPAN: f64 = 0.05;
pub const DEFAULT_HF_BOUNDS: [f64; 2] = [-2.0, 2.0];
pub const DEFAULT_PARAM_BOUNDS: [f64; 2] = [0.0, 2.0 * PI];
pub struct CustomSurfaceSandbox {
    pub heightfield_bar: MathExpressionInputBar,
    pub param_x_bar: MathExpressionInputBar,
    pub param_y_bar: MathExpressionInputBar,
    pub param_z_bar: MathExpressionInputBar,
    pub param_sliders: ParameterSliderBank,
    pub parametric_error: Option<String>,
    pub hf_u_bounds: [f64; 2],
    pub hf_v_bounds: [f64; 2],
    pub param_u_bounds: [f64; 2],
    pub param_v_bounds: [f64; 2],
    cached_heightfield_zero: CompiledExpr,
    cached_parametric: (CompiledExpr, CompiledExpr, CompiledExpr),
}

impl Default for CustomSurfaceSandbox {
    fn default() -> Self {
        let heightfield_bar = MathExpressionInputBar::new(
            "z",
            DEFAULT_HEIGHTFIELD_EXPR,
            vec![
                ("Hyperbolic Saddle", "a * (x^2 - y^2)"),
                ("Radial Ripple", "a * cos(k * sqrt(x^2 + y^2 + 0.01))"),
                ("Monkey Saddle", "a * (x^3 - 3 * x * y^2)"),
                ("Eggcrate Wave", "a * sin(k * x) * cos(m * y)"),
            ],
        )
        .with_axes(&["x", "y"]);

        let param_x_bar = MathExpressionInputBar::new("x(u,v)", DEFAULT_PARAM_X_EXPR, vec![])
            .with_axes(&["u", "v"])
            .with_inline_sliders(false);
        let param_y_bar = MathExpressionInputBar::new("y(u,v)", DEFAULT_PARAM_Y_EXPR, vec![])
            .with_axes(&["u", "v"])
            .with_inline_sliders(false);
        let param_z_bar = MathExpressionInputBar::new("z(u,v)", DEFAULT_PARAM_Z_EXPR, vec![])
            .with_axes(&["u", "v"])
            .with_inline_sliders(false);

        let fallback = fallback_compiled_triple();
        let zero_hf =
            CompiledExpr::compile_with_vars(&parse("0").expect("zero parses"), &["x", "y"])
                .expect("zero compiles");
        let mut sandbox = Self {
            heightfield_bar,
            param_x_bar,
            param_y_bar,
            param_z_bar,
            param_sliders: ParameterSliderBank::new().with_range(-3.0..=3.0, 0.6),
            parametric_error: None,
            hf_u_bounds: DEFAULT_HF_BOUNDS,
            hf_v_bounds: DEFAULT_HF_BOUNDS,
            param_u_bounds: DEFAULT_PARAM_BOUNDS,
            param_v_bounds: DEFAULT_PARAM_BOUNDS,
            cached_heightfield_zero: zero_hf,
            cached_parametric: fallback,
        };
        sandbox.recompile_parametric_triple();
        sandbox
    }
}

impl CustomSurfaceSandbox {
    #[must_use]
    pub const fn uv_domain_for(
        &self,
        surface_type: ParametricSurfaceKind,
    ) -> ((f64, f64), (f64, f64)) {
        match surface_type {
            ParametricSurfaceKind::CustomHeightfield => (
                (self.hf_u_bounds[0], self.hf_u_bounds[1]),
                (self.hf_v_bounds[0], self.hf_v_bounds[1]),
            ),
            ParametricSurfaceKind::CustomParametric => (
                (self.param_u_bounds[0], self.param_u_bounds[1]),
                (self.param_v_bounds[0], self.param_v_bounds[1]),
            ),
            other => other.uv_domain(),
        }
    }

    pub fn set_uv_bounds(
        &mut self,
        surface_type: ParametricSurfaceKind,
        u_bounds: [f64; 2],
        v_bounds: [f64; 2],
    ) -> Result<(), String> {
        validate_domain_pair(u_bounds, "u")?;
        validate_domain_pair(v_bounds, "v")?;
        match surface_type {
            ParametricSurfaceKind::CustomHeightfield => {
                self.hf_u_bounds = u_bounds;
                self.hf_v_bounds = v_bounds;
            }
            ParametricSurfaceKind::CustomParametric => {
                self.param_u_bounds = u_bounds;
                self.param_v_bounds = v_bounds;
            }
            _ => {}
        }
        Ok(())
    }

    pub fn state_fingerprint(&self, surface_type: ParametricSurfaceKind) -> u64 {
        use std::hash::{Hash as _, Hasher as _};
        match surface_type {
            ParametricSurfaceKind::CustomHeightfield => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                self.heightfield_bar.state_fingerprint().hash(&mut h);
                self.hf_u_bounds[0].to_bits().hash(&mut h);
                self.hf_u_bounds[1].to_bits().hash(&mut h);
                self.hf_v_bounds[0].to_bits().hash(&mut h);
                self.hf_v_bounds[1].to_bits().hash(&mut h);
                h.finish().max(1)
            }
            ParametricSurfaceKind::CustomParametric => {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                self.param_x_bar.raw_text.hash(&mut h);
                self.param_y_bar.raw_text.hash(&mut h);
                self.param_z_bar.raw_text.hash(&mut h);
                self.parametric_error.is_some().hash(&mut h);
                self.param_sliders.state_fingerprint().hash(&mut h);
                self.param_u_bounds[0].to_bits().hash(&mut h);
                self.param_u_bounds[1].to_bits().hash(&mut h);
                self.param_v_bounds[0].to_bits().hash(&mut h);
                self.param_v_bounds[1].to_bits().hash(&mut h);
                h.finish().max(1)
            }
            _ => 0,
        }
    }

    pub fn ui_for_type(&mut self, ui: &mut egui::Ui, surface_type: ParametricSurfaceKind) {
        match surface_type {
            ParametricSurfaceKind::CustomHeightfield => {
                self.heightfield_bar.ui(ui);
                ui.add_space(6.0);
                draw_domain_bounds_editor(
                    ui,
                    "Domain Bounds",
                    ("x [u_min, u_max]:", &mut self.hf_u_bounds),
                    ("y [v_min, v_max]:", &mut self.hf_v_bounds),
                    DEFAULT_HF_BOUNDS,
                );
            }
            ParametricSurfaceKind::CustomParametric => {
                let mut edited = false;
                edited |= self.param_x_bar.ui(ui);
                edited |= self.param_y_bar.ui(ui);
                edited |= self.param_z_bar.ui(ui);
                if edited {
                    self.recompile_parametric_triple();
                }
                if let Some(ref err) = self.parametric_error {
                    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                    ui.colored_label(palette.error, format!("⚠️ {err}"));
                }
                self.param_sliders.ui(ui);
                ui.add_space(6.0);
                draw_domain_bounds_editor(
                    ui,
                    "Domain Bounds",
                    ("u [u_min, u_max]:", &mut self.param_u_bounds),
                    ("v [v_min, v_max]:", &mut self.param_v_bounds),
                    DEFAULT_PARAM_BOUNDS,
                );
            }
            _ => {}
        }
    }

    pub fn build_heightfield(&self) -> CompiledHeightfieldSurface {
        let compiled = self
            .heightfield_bar
            .active_compiled()
            .unwrap_or(&self.cached_heightfield_zero)
            .clone();
        CompiledHeightfieldSurface::new(compiled, self.heightfield_bar.param_slice())
    }

    pub fn build_parametric(&self) -> CompiledParametricSurface {
        let (cx, cy, cz) = self.cached_parametric.clone();
        CompiledParametricSurface::new(cx, cy, cz, self.param_sliders.param_slice())
    }

    pub fn recompile_parametric_triple(&mut self) {
        let (Some(ex), Some(ey), Some(ez)) = (
            self.param_x_bar.expr(),
            self.param_y_bar.expr(),
            self.param_z_bar.expr(),
        ) else {
            return;
        };

        let mut discovered = Vec::new();
        for expr in [ex, ey, ez] {
            for v in expr.free_variables() {
                if !v.eq_ignore_ascii_case("u")
                    && !v.eq_ignore_ascii_case("v")
                    && !is_universal_constant(&v)
                    && !discovered.iter().any(|existing| existing == &v)
                {
                    discovered.push(v);
                }
            }
        }
        if discovered.len() > MAX_SYNTHESIZED_PARAMS {
            self.parametric_error = Some(format!(
                "Parametric surface defines {} free parameters (maximum {MAX_SYNTHESIZED_PARAMS})",
                discovered.len()
            ));
            return;
        }
        let shared_refs: Vec<&str> = discovered.iter().map(String::as_str).collect();
        if let (Ok(cx), Ok(cy), Ok(cz)) = (
            CompiledExpr::compile_with_vars_and_params(ex, &["u", "v"], &shared_refs),
            CompiledExpr::compile_with_vars_and_params(ey, &["u", "v"], &shared_refs),
            CompiledExpr::compile_with_vars_and_params(ez, &["u", "v"], &shared_refs),
        ) {
            self.parametric_error = None;
            self.param_sliders.sync_names(&discovered);
            self.cached_parametric = (cx, cy, cz);
        }
    }

    pub fn export_fields(
        &self,
        surface_type: ParametricSurfaceKind,
    ) -> (Option<String>, Option<[String; 3]>, BTreeMap<String, f64>) {
        let custom_expr = (surface_type == ParametricSurfaceKind::CustomHeightfield
            || self.heightfield_bar.raw_text != DEFAULT_HEIGHTFIELD_EXPR)
            .then(|| self.heightfield_bar.raw_text.clone());

        let has_nondefault_triple = self.param_x_bar.raw_text != DEFAULT_PARAM_X_EXPR
            || self.param_y_bar.raw_text != DEFAULT_PARAM_Y_EXPR
            || self.param_z_bar.raw_text != DEFAULT_PARAM_Z_EXPR;
        let parametric_exprs = (surface_type == ParametricSurfaceKind::CustomParametric
            || has_nondefault_triple)
            .then(|| {
                [
                    self.param_x_bar.raw_text.clone(),
                    self.param_y_bar.raw_text.clone(),
                    self.param_z_bar.raw_text.clone(),
                ]
            });

        let custom_params = match surface_type {
            ParametricSurfaceKind::CustomHeightfield => self.heightfield_bar.params_map(),
            ParametricSurfaceKind::CustomParametric => self.param_sliders.export_active_map(),
            _ => BTreeMap::new(),
        };
        (custom_expr, parametric_exprs, custom_params)
    }

    #[must_use]
    pub fn export_domain_bounds(
        &self,
        surface_type: ParametricSurfaceKind,
    ) -> (Option<[f64; 2]>, Option<[f64; 2]>) {
        match surface_type {
            ParametricSurfaceKind::CustomHeightfield => (
                (!approx_pair(self.hf_u_bounds, DEFAULT_HF_BOUNDS)).then_some(self.hf_u_bounds),
                (!approx_pair(self.hf_v_bounds, DEFAULT_HF_BOUNDS)).then_some(self.hf_v_bounds),
            ),
            ParametricSurfaceKind::CustomParametric => (
                (!approx_pair(self.param_u_bounds, DEFAULT_PARAM_BOUNDS))
                    .then_some(self.param_u_bounds),
                (!approx_pair(self.param_v_bounds, DEFAULT_PARAM_BOUNDS))
                    .then_some(self.param_v_bounds),
            ),
            _ => (None, None),
        }
    }

    pub fn load_fields(
        &mut self,
        custom_expr: Option<&str>,
        parametric_exprs: Option<&[String; 3]>,
        custom_params: &BTreeMap<String, f64>,
    ) {
        self.heightfield_bar
            .set_expression(custom_expr.unwrap_or(DEFAULT_HEIGHTFIELD_EXPR));
        self.heightfield_bar.set_params_map(custom_params);

        let [px, py, pz] = parametric_exprs.map_or(
            [
                DEFAULT_PARAM_X_EXPR,
                DEFAULT_PARAM_Y_EXPR,
                DEFAULT_PARAM_Z_EXPR,
            ],
            |[x, y, z]| [x.as_str(), y.as_str(), z.as_str()],
        );
        self.param_x_bar.set_expression(px);
        self.param_y_bar.set_expression(py);
        self.param_z_bar.set_expression(pz);
        self.param_sliders.set_params_map(custom_params);
        self.recompile_parametric_triple();
    }

    pub fn load_domain_bounds(
        &mut self,
        surface_type: ParametricSurfaceKind,
        u_bounds: Option<[f64; 2]>,
        v_bounds: Option<[f64; 2]>,
    ) -> Result<(), String> {
        self.hf_u_bounds = DEFAULT_HF_BOUNDS;
        self.hf_v_bounds = DEFAULT_HF_BOUNDS;
        self.param_u_bounds = DEFAULT_PARAM_BOUNDS;
        self.param_v_bounds = DEFAULT_PARAM_BOUNDS;
        if u_bounds.is_none() && v_bounds.is_none() {
            return Ok(());
        }
        let default_pair = match surface_type {
            ParametricSurfaceKind::CustomHeightfield => DEFAULT_HF_BOUNDS,
            _ => DEFAULT_PARAM_BOUNDS,
        };
        let u = u_bounds.unwrap_or(default_pair);
        let v = v_bounds.unwrap_or(default_pair);
        self.set_uv_bounds(surface_type, u, v)
    }
}

fn approx_pair(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12
}

fn validate_domain_pair(bounds: [f64; 2], axis: &str) -> Result<(), String> {
    let [lo, hi] = bounds;
    if !lo.is_finite()
        || !hi.is_finite()
        || !(CUSTOM_DOMAIN_MIN_BOUND..=CUSTOM_DOMAIN_MAX_BOUND).contains(&lo)
        || !(CUSTOM_DOMAIN_MIN_BOUND..=CUSTOM_DOMAIN_MAX_BOUND).contains(&hi)
        || lo + CUSTOM_DOMAIN_MIN_SPAN > hi
    {
        return Err(format!(
            "Custom surface {axis}_bounds out of valid ordered range"
        ));
    }
    Ok(())
}

fn draw_domain_bounds_editor(
    ui: &mut egui::Ui,
    heading: &str,
    u_axis: (&str, &mut [f64; 2]),
    v_axis: (&str, &mut [f64; 2]),
    defaults: [f64; 2],
) {
    ui.label(heading);
    for (label, bounds) in [(u_axis.0, &mut *u_axis.1), (v_axis.0, &mut *v_axis.1)] {
        let mut min_changed = false;
        let mut max_changed = false;
        ui.horizontal(|ui| {
            ui.label(label);
            min_changed = ui
                .add(bounded_drag_value(
                    &mut bounds[0],
                    CUSTOM_DOMAIN_MIN_BOUND..=CUSTOM_DOMAIN_MAX_BOUND,
                    0.1,
                ))
                .changed();
            ui.label("to");
            max_changed = ui
                .add(bounded_drag_value(
                    &mut bounds[1],
                    CUSTOM_DOMAIN_MIN_BOUND..=CUSTOM_DOMAIN_MAX_BOUND,
                    0.1,
                ))
                .changed();
        });
        let (mut lo, mut hi) = (bounds[0], bounds[1]);
        enforce_ordered_drag_range(
            &mut lo,
            &mut hi,
            CUSTOM_DOMAIN_MIN_BOUND,
            CUSTOM_DOMAIN_MAX_BOUND,
            CUSTOM_DOMAIN_MIN_SPAN,
            min_changed,
            max_changed,
        );
        *bounds = [lo, hi];
    }
    if ui.small_button("Reset Domain").clicked() {
        *u_axis.1 = defaults;
        *v_axis.1 = defaults;
    }
}

fn fallback_compiled_triple() -> (CompiledExpr, CompiledExpr, CompiledExpr) {
    let zero = parse("0").expect("zero");
    let c0 = CompiledExpr::compile_with_vars(&zero, &["u", "v"]).expect("compile zero");
    (c0.clone(), c0.clone(), c0)
}
