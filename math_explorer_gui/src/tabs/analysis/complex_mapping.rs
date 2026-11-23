#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use num_complex::Complex64;
use std::collections::BTreeMap;

#[path = "complex_mapping/cache.rs"]
mod cache;
#[path = "complex_mapping/shader.rs"]
mod shader;
#[path = "complex_mapping/ui.rs"]
mod ui;

pub(crate) const COMPLEX_COORD_MIN: f64 = -50.0;
pub(crate) const COMPLEX_COORD_MAX: f64 = 50.0;
pub(super) const COMPLEX_MIN_SPAN: f64 = 0.1;
const EXP_SIN_GUARD_LIMIT: f64 = 50.0;
const NAN_COMPLEX: Complex64 = Complex64::new(f64::NAN, f64::NAN);
const DEFAULT_CUSTOM_EXPR: &str = "z^3 - 1/z";

#[derive(PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) enum MappingFunction {
    ZSquared,
    Exponential,
    Sine,
    Inverse,
    Custom,
}

fn sanitize_complex_pole(w: Complex64) -> Complex64 {
    if w.re.is_finite() && w.im.is_finite() {
        w
    } else {
        NAN_COMPLEX
    }
}

fn sanitize_complex_deriv(w: Complex64) -> Complex64 {
    if w.re.is_finite() && w.im.is_finite() {
        w
    } else {
        Complex64::new(0.0, 0.0)
    }
}

impl MappingFunction {
    pub(super) fn name(&self) -> &'static str {
        match self {
            MappingFunction::ZSquared => "w = z^2",
            MappingFunction::Exponential => "w = e^z",
            MappingFunction::Sine => "w = sin(z)",
            MappingFunction::Inverse => "w = 1/z",
            MappingFunction::Custom => "w = f(z) (Custom)",
        }
    }

    fn evaluate(&self, z: Complex64) -> Complex64 {
        if !z.re.is_finite() || !z.im.is_finite() {
            return NAN_COMPLEX;
        }
        let raw = match self {
            MappingFunction::ZSquared => z * z,
            MappingFunction::Exponential => {
                Complex64::new(z.re.clamp(-EXP_SIN_GUARD_LIMIT, EXP_SIN_GUARD_LIMIT), z.im).exp()
            }
            MappingFunction::Sine => {
                Complex64::new(z.re, z.im.clamp(-EXP_SIN_GUARD_LIMIT, EXP_SIN_GUARD_LIMIT)).sin()
            }
            MappingFunction::Inverse => {
                if z.norm_sqr() < 1e-12 {
                    NAN_COMPLEX
                } else {
                    1.0 / z
                }
            }
            MappingFunction::Custom => NAN_COMPLEX,
        };
        sanitize_complex_pole(raw)
    }

    fn derivative(&self, z: Complex64) -> Complex64 {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Complex64::new(0.0, 0.0);
        }
        let raw = match self {
            MappingFunction::ZSquared => 2.0 * z,
            MappingFunction::Exponential => {
                Complex64::new(z.re.clamp(-EXP_SIN_GUARD_LIMIT, EXP_SIN_GUARD_LIMIT), z.im).exp()
            }
            MappingFunction::Sine => {
                Complex64::new(z.re, z.im.clamp(-EXP_SIN_GUARD_LIMIT, EXP_SIN_GUARD_LIMIT)).cos()
            }
            MappingFunction::Inverse => {
                if z.norm_sqr() < 1e-12 {
                    NAN_COMPLEX
                } else {
                    -1.0 / (z * z)
                }
            }
            MappingFunction::Custom => Complex64::new(0.0, 0.0),
        };
        sanitize_complex_deriv(raw)
    }
}

pub struct ComplexMappingTool {
    pub(crate) function: MappingFunction,
    pub(crate) grid_min: f64,
    pub(crate) grid_max: f64,
    pub(crate) grid_density: usize,
    pub(crate) inspection_point: [f64; 2],
    pub(crate) math_input: MathExpressionInputBar,
    pub(crate) high_contrast: bool,
    pub(crate) cache: cache::ConformalGridCache,
}

impl Default for ComplexMappingTool {
    fn default() -> Self {
        Self {
            function: MappingFunction::ZSquared,
            grid_min: -2.0,
            grid_max: 2.0,
            grid_density: 10,
            inspection_point: [1.0, 1.0],
            math_input: MathExpressionInputBar::new(
                "w",
                DEFAULT_CUSTOM_EXPR,
                vec![
                    ("w = z + a^2 / z (Parameterized Joukowsky)", "z + a^2 / z"),
                    ("w = z^3 - 1/z", "z^3 - 1/z"),
                    ("w = z^2", "z^2"),
                    ("w = exp(z)", "exp(z)"),
                    ("w = sin(z)", "sin(z)"),
                    ("w = 1/z", "1/z"),
                ],
            )
            .with_axes(&["z"])
            .with_complex_mode(true)
            .with_allowed_vars(&["z", "a"]),
            high_contrast: false,
            cache: cache::ConformalGridCache::default(),
        }
    }
}

impl ComplexMappingTool {
    pub(crate) fn eval_z(&self, z: Complex64) -> Complex64 {
        if !z.re.is_finite() || !z.im.is_finite() {
            return NAN_COMPLEX;
        }
        match self.function {
            MappingFunction::Custom => self.math_input.compiled().map_or(NAN_COMPLEX, |c| {
                sanitize_complex_pole(c.eval_complex_z_params(z, self.math_input.param_slice()))
            }),
            _ => self.function.evaluate(z),
        }
    }

    pub(crate) fn deriv_z(&self, z: Complex64) -> Complex64 {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Complex64::new(0.0, 0.0);
        }
        match self.function {
            MappingFunction::Custom => {
                let h = Complex64::new(1e-7, 0.0);
                let f_plus = self.eval_z(z + h);
                let f_minus = self.eval_z(z - h);
                sanitize_complex_deriv((f_plus - f_minus) / (2.0 * h))
            }
            _ => self.function.derivative(z),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ComplexMappingState {
    function: MappingFunction,
    grid_min: f64,
    grid_max: f64,
    grid_density: usize,
    inspection_point: [f64; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
}

impl InteractiveTool for ComplexMappingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Complex Mapping"
    }

    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
    }

    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::plot().with_shader()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_export_primitives()
    }

    fn sample_animation_frames(
        &mut self,
        _num_frames: usize,
    ) -> Vec<egui_plot::export::DatasetFrame> {
        Vec::new()
    }
    fn export_glsl_shader(
        &self,
        dialect: crate::framework::ShaderDialect,
    ) -> Result<String, String> {
        self.build_glsl_shader(dialect)
    }

    fn has_finite_parameters(&self) -> bool {
        self.grid_min.is_finite()
            && self.grid_max.is_finite()
            && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_min)
            && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&self.grid_max)
            && self.grid_min < self.grid_max
            && (2..=crate::presets::MAX_INTERACTIVE_GRID_SIZE).contains(&self.grid_density)
            && self
                .inspection_point
                .iter()
                .all(|v| v.is_finite() && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(v))
    }

    fn save_state(&self) -> Option<String> {
        let is_custom = self.function == MappingFunction::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_CUSTOM_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&ComplexMappingState {
            function: self.function,
            grid_min: self.grid_min,
            grid_max: self.grid_max,
            grid_density: self.grid_density,
            inspection_point: self.inspection_point,
            custom_expr,
            custom_params,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: ComplexMappingState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid complex mapping state: {e}"))?;
        if !s.grid_min.is_finite()
            || !s.grid_max.is_finite()
            || !(COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&s.grid_min)
            || !(COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(&s.grid_max)
            || s.grid_min >= s.grid_max
            || !(2..=math_commons::registry::MAX_GRID_SIZE).contains(&s.grid_density)
            || !s
                .inspection_point
                .iter()
                .all(|v| v.is_finite() && (COMPLEX_COORD_MIN..=COMPLEX_COORD_MAX).contains(v))
        {
            return Err("Complex mapping bounds or density out of valid range".into());
        }
        self.function = s.function;
        self.grid_min = s.grid_min;
        self.grid_max = s.grid_max;
        self.grid_density = s
            .grid_density
            .min(crate::presets::MAX_INTERACTIVE_GRID_SIZE);
        self.inspection_point = s.inspection_point;
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_CUSTOM_EXPR));
        self.math_input.set_params_map(&s.custom_params);
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.high_contrast = crate::app::ThemeMode::from_visuals(parent_ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        self.draw_controls(parent_ui);
        let grid_ok = self.ensure_cached_contours();
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Conformal Mapping Visualization");
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.columns(2, |columns| {
                self.draw_z_plane(&mut columns[0], &palette, grid_ok);
                self.draw_w_plane(&mut columns[1], &palette, grid_ok);
            });
        });
    }
}

// [cite:partitions_implementation]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ComplexMappingTool",
        display_name: "Complex Mapping",
        domain: "analysis",
        tags: &[],
        build: || Box::new(ComplexMappingTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for ComplexMappingTool {
    fn theory_description(&self) -> String {
        let fn_desc = match self.function {
            MappingFunction::ZSquared => "quadratic power map $w = z^2$",
            MappingFunction::Exponential => "entire exponential map $w = e^z$",
            MappingFunction::Sine => "complex trigonometric map $w = \\sin(z)$",
            MappingFunction::Inverse => "inversion map $w = 1/z$",
            MappingFunction::Custom => "custom holomorphic map $w = f(z)$",
        };
        format!(
            "Conformal complex mapping {} preserving angles and orientation under Cauchy-Riemann equations with local scaling factor $|f'(z)|$.",
            fn_desc
        )
    }

    fn phonetic_description(&self) -> String {
        "Conformal complex mapping and Cauchy-Riemann equations theory".into()
    }

    fn theory_citation(&self) -> String {
        "Visual Complex Analysis (Needham, 1997); Complex Analysis (Ahlfors, 1979)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "ZSquared".into(),
            "w = z^2 doubling angles at the origin".into(),
        );
        map.insert(
            "Exponential".into(),
            "w = e^z mapping vertical lines to circles".into(),
        );
        map.insert(
            "Sine".into(),
            "w = sin(z) mapping strips to upper half-plane".into(),
        );
        map.insert(
            "Inverse".into(),
            "w = 1/z conformal inversion through unit circle".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Conformal Mapping",
    "domain": "analysis",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "function": "ZSquared",
    "grid_min": -2.0,
    "grid_max": 2.0,
    "grid_density": 15,
    "inspection_point": [1.0, 0.5]
  }
}"#;

const JOUKOWSKY_POLE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Joukowsky Pole Explorer",
    "domain": "analysis",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "function": "Custom",
    "grid_min": -2.0,
    "grid_max": 2.0,
    "grid_density": 15,
    "inspection_point": [1.2, 0.4],
    "custom_expr": "z + a^2 / z",
    "custom_params": {
      "a": 0.85
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "complex_mapping_starter",
        title: "Conformal Mapping",
        description: "w = f(z): Complex holomorphic functions mapping orthogonal grid coordinate contours.",
        category: "Complex Analysis",
        icon: "🌀",
        domain: "analysis",
        tool_name: Some("ComplexMappingTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "complex_mapping_joukowsky_pole",
        title: "Joukowsky Pole Explorer",
        description: "w = z + a^2/z: Parameterized Joukowsky airfoil map with pole-split contours and live slider a.",
        category: "Complex Analysis",
        icon: "✈️",
        domain: "analysis",
        tool_name: Some("ComplexMappingTool"),
        preset_generator: || JOUKOWSKY_POLE_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "complex_mapping/tests.rs"]
mod tests;
