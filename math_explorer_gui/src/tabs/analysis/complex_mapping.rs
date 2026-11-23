#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::shader_foundry::{
    bind_complex_shader_params, emit_domain_coloring_shader, ShaderPreviewState,
    ShaderUniformDefaults,
};
use crate::framework::{
    ExportCapabilities, InteractiveTool, MathExpressionInputBar, ShaderExportFormat,
};
use eframe::egui;
use math_commons::ast::{parse, Expr};
use math_commons::math_kernel::colormap::{domain_color_with_palette, ShaderPalette};
use math_commons::math_kernel::types::flatten_2d_index;
use num_complex::Complex64;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[path = "complex_mapping/state.rs"]
mod state;
use state::ComplexMappingState;

#[path = "complex_mapping/cache.rs"]
mod cache;
#[path = "complex_mapping/ui.rs"]
mod ui;

pub(crate) const COMPLEX_COORD_MIN: f64 = -50.0;
pub(crate) const COMPLEX_COORD_MAX: f64 = 50.0;
pub(super) const COMPLEX_MIN_SPAN: f64 = 0.1;
pub(super) const DOMAIN_COLOR_PREVIEW_SIDE: usize = 160;
const EXP_SIN_GUARD_LIMIT: f64 = 50.0;
const NAN_COMPLEX: Complex64 = Complex64::new(f64::NAN, f64::NAN);
pub(super) const MAX_CUSTOM_EXPR_BYTES: usize = 512;
pub(super) const DEFAULT_CUSTOM_MAPPING_EXPR: &str = "z^3 - 1/z";

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
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
    pub(crate) domain_coloring: bool,
    pub(crate) palette: ShaderPalette,
    pub(crate) high_contrast: bool,
    pub(crate) shader_preview: ShaderPreviewState,
    pub(crate) domain_texture: Option<egui::TextureHandle>,
    pub(crate) domain_texture_key: Option<u64>,
    pub(crate) cache: cache::ContourGridCache,
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
                DEFAULT_CUSTOM_MAPPING_EXPR,
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
            .with_complex_mode(true),
            domain_coloring: true,
            palette: ShaderPalette::PhaseWheel,
            high_contrast: false,
            shader_preview: ShaderPreviewState::default(),
            domain_texture: None,
            domain_texture_key: None,
            cache: cache::ContourGridCache::default(),
        }
    }
}

impl ComplexMappingTool {
    pub(crate) fn eval_z(&self, z: Complex64) -> Complex64 {
        if !z.re.is_finite() || !z.im.is_finite() {
            return NAN_COMPLEX;
        }
        match self.function {
            MappingFunction::Custom => self.math_input.active_compiled().map_or(NAN_COMPLEX, |c| {
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

    pub(super) fn shader_revision(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.function.hash(&mut hasher);
        self.grid_min.to_bits().hash(&mut hasher);
        self.grid_max.to_bits().hash(&mut hasher);
        self.domain_coloring.hash(&mut hasher);
        self.palette.hash(&mut hasher);
        self.high_contrast.hash(&mut hasher);
        self.math_input.raw_text.hash(&mut hasher);
        for &param in self.math_input.param_slice() {
            param.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    pub(super) fn resolve_active_shader_expr(&self) -> Result<Expr, String> {
        let builtin = match self.function {
            MappingFunction::ZSquared => Some("z^2"),
            MappingFunction::Exponential => Some("exp(z)"),
            MappingFunction::Sine => Some("sin(z)"),
            MappingFunction::Inverse => Some("1 / z"),
            MappingFunction::Custom => None,
        };
        match builtin {
            Some(src) => parse(src).map_err(|e| e.to_string()),
            None => {
                let expr = self.math_input.expr().ok_or_else(|| {
                    self.math_input
                        .error_message()
                        .unwrap_or_else(|| "Invalid custom complex mapping expression".to_string())
                })?;
                bind_complex_shader_params(expr, &self.math_input.params_map(), &["z"])
            }
        }
    }

    pub(super) fn shader_uniform_defaults(&self) -> ShaderUniformDefaults {
        let lo = self.grid_min.clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX);
        let hi = self
            .grid_max
            .clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX)
            .max(lo + COMPLEX_MIN_SPAN);
        ShaderUniformDefaults {
            resolution: [512, 512],
            viewport: [lo, lo, hi, hi],
            max_iter: 1,
            julia_c: [
                self.inspection_point[0].clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX),
                self.inspection_point[1].clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX),
            ],
            palette: self.palette,
            high_contrast: self.high_contrast,
            smooth_coloring: true,
        }
    }

    /// Renders a continuous phase-modulus domain coloring raster over `[grid_min, grid_max]^2`.
    #[must_use]
    pub fn render_domain_coloring_image(&self, side: usize) -> egui::ColorImage {
        let dim = side.clamp(8, 512);
        let lo = if self.grid_min.is_finite() {
            self.grid_min.clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX)
        } else {
            -2.0
        };
        let hi = if self.grid_max.is_finite() && self.grid_max > lo {
            self.grid_max.clamp(COMPLEX_COORD_MIN, COMPLEX_COORD_MAX)
        } else {
            lo + 4.0
        };
        let span = hi - lo;
        let denom = (dim.saturating_sub(1)).max(1) as f64;
        let mut pixels = vec![egui::Color32::BLACK; dim * dim];

        for y in 0..dim {
            let im = hi - (y as f64 / denom) * span;
            for x in 0..dim {
                let re = lo + (x as f64 / denom) * span;
                let w = self.eval_z(Complex64::new(re, im));
                let rgb = domain_color_with_palette(w.re, w.im, self.palette, self.high_contrast);
                pixels[flatten_2d_index(x, y, dim)] =
                    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
            }
        }
        egui::ColorImage::new([dim, dim], pixels)
    }
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

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot_with_shader()
    }

    fn export_shader_source(&self, format: ShaderExportFormat) -> Result<String, String> {
        let expr = self.resolve_active_shader_expr()?;
        emit_domain_coloring_shader(format, &expr, &self.shader_uniform_defaults())
    }

    fn state_revision(&self) -> u64 {
        self.compute_state_revision()
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
            && self
                .math_input
                .param_slice()
                .iter()
                .all(|v| v.is_finite() && (-1.0e4..=1.0e4).contains(v))
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.to_serializable_state()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: ComplexMappingState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid complex mapping state: {e}"))?;
        let palette = s.validate()?;
        self.function = s.function;
        self.grid_min = s.grid_min;
        self.grid_max = s.grid_max;
        self.grid_density = s
            .grid_density
            .min(crate::presets::MAX_INTERACTIVE_GRID_SIZE);
        self.inspection_point = s.inspection_point;
        self.domain_coloring = s.domain_coloring;
        self.palette = palette;
        self.math_input.set_expression(
            s.custom_expr
                .as_deref()
                .unwrap_or(DEFAULT_CUSTOM_MAPPING_EXPR),
        );
        self.math_input.set_params_map(&s.custom_params);
        self.domain_texture_key = None;
        self.shader_preview.invalidate();
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
        self.draw_central_canvas(parent_ui, grid_ok);
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
