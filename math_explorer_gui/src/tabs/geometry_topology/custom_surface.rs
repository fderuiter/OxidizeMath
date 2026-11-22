#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Shared `CompiledExpr`-backed custom heightfield `z = f(x, y)` and parametric triple
//! `(x(u, v), y(u, v), z(u, v))` formula controls for `SurfaceViewer` and `CurvatureHeatmap`.

use crate::framework::math_input::{ParameterSliderBank, MAX_SYNTHESIZED_PARAMS};
use crate::framework::MathExpressionInputBar;
use eframe::egui;
use math_commons::ast::{is_universal_constant, parse, CompiledExpr};
use math_explorer::pure_math::differential_geometry::surface::{
    CompiledHeightfieldSurface, CompiledParametricSurface,
};
use std::collections::BTreeMap;
use std::f64::consts::PI;

pub const DEFAULT_HEIGHTFIELD_EXPR: &str = "a * (x^2 - y^2)";
pub const DEFAULT_PARAM_X_EXPR: &str = "(2 + a * cos(v)) * cos(u)";
pub const DEFAULT_PARAM_Y_EXPR: &str = "(2 + a * cos(v)) * sin(u)";
pub const DEFAULT_PARAM_Z_EXPR: &str = "a * sin(v) + 0.3 * sin(2 * u)";

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub enum SurfaceType {
    Sphere,
    Torus,
    KleinBottle,
    CustomHeightfield,
    CustomParametric,
}

impl SurfaceType {
    pub const fn uv_domain(self) -> ((f64, f64), (f64, f64)) {
        match self {
            Self::CustomHeightfield => ((-2.0, 2.0), (-2.0, 2.0)),
            Self::Sphere | Self::Torus | Self::KleinBottle | Self::CustomParametric => {
                ((0.0, 2.0 * PI), (0.0, 2.0 * PI))
            }
        }
    }
}

pub struct CustomSurfaceSandbox {
    pub heightfield_bar: MathExpressionInputBar,
    pub param_x_bar: MathExpressionInputBar,
    pub param_y_bar: MathExpressionInputBar,
    pub param_z_bar: MathExpressionInputBar,
    pub param_sliders: ParameterSliderBank,
    pub parametric_error: Option<String>,
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
        let mut sandbox = Self {
            heightfield_bar,
            param_x_bar,
            param_y_bar,
            param_z_bar,
            param_sliders: ParameterSliderBank::new().with_range(-3.0..=3.0, 0.6),
            parametric_error: None,
            cached_parametric: fallback,
        };
        sandbox.recompile_parametric_triple();
        sandbox
    }
}

impl CustomSurfaceSandbox {
    pub fn ui_for_type(&mut self, ui: &mut egui::Ui, surface_type: SurfaceType) {
        match surface_type {
            SurfaceType::CustomHeightfield => {
                self.heightfield_bar.ui(ui);
            }
            SurfaceType::CustomParametric => {
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
            }
            _ => {}
        }
    }

    pub fn build_heightfield(&self) -> CompiledHeightfieldSurface {
        let compiled = self
            .heightfield_bar
            .active_compiled()
            .cloned()
            .unwrap_or_else(|| {
                CompiledExpr::compile_with_vars(&parse("0").expect("zero parses"), &["x", "y"])
                    .expect("zero compiles")
            });
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
        surface_type: SurfaceType,
    ) -> (Option<String>, Option<[String; 3]>, BTreeMap<String, f64>) {
        let custom_expr = (surface_type == SurfaceType::CustomHeightfield
            || self.heightfield_bar.raw_text != DEFAULT_HEIGHTFIELD_EXPR)
            .then(|| self.heightfield_bar.raw_text.clone());

        let has_nondefault_triple = self.param_x_bar.raw_text != DEFAULT_PARAM_X_EXPR
            || self.param_y_bar.raw_text != DEFAULT_PARAM_Y_EXPR
            || self.param_z_bar.raw_text != DEFAULT_PARAM_Z_EXPR;
        let parametric_exprs =
            (surface_type == SurfaceType::CustomParametric || has_nondefault_triple).then(|| {
                [
                    self.param_x_bar.raw_text.clone(),
                    self.param_y_bar.raw_text.clone(),
                    self.param_z_bar.raw_text.clone(),
                ]
            });

        let custom_params = match surface_type {
            SurfaceType::CustomHeightfield => self.heightfield_bar.params_map(),
            SurfaceType::CustomParametric => self.param_sliders.export_active_map(),
            _ => BTreeMap::new(),
        };
        (custom_expr, parametric_exprs, custom_params)
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

    pub fn hash_state(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        self.heightfield_bar.raw_text.hash(h);
        for &p in self.heightfield_bar.param_slice() {
            p.to_bits().hash(h);
        }
        self.param_x_bar.raw_text.hash(h);
        self.param_y_bar.raw_text.hash(h);
        self.param_z_bar.raw_text.hash(h);
        for &p in self.param_sliders.param_slice() {
            p.to_bits().hash(h);
        }
    }
}

fn fallback_compiled_triple() -> (CompiledExpr, CompiledExpr, CompiledExpr) {
    let zero = parse("0").expect("zero");
    let c0 = CompiledExpr::compile_with_vars(&zero, &["u", "v"]).expect("compile zero");
    (c0.clone(), c0.clone(), c0)
}
