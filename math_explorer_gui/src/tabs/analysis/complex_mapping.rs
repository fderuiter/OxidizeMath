#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::shader_foundry::{
    emit_domain_coloring_shader, ShaderPreviewState, ShaderUniformDefaults,
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

#[path = "complex_mapping/ui.rs"]
mod ui;

pub(crate) const COMPLEX_COORD_MIN: f64 = -50.0;
pub(crate) const COMPLEX_COORD_MAX: f64 = 50.0;
pub(super) const COMPLEX_MIN_SPAN: f64 = 0.1;
pub(super) const DOMAIN_COLOR_PREVIEW_SIDE: usize = 160;
const EXP_SIN_GUARD_LIMIT: f64 = 50.0;
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

fn sanitize_complex(w: Complex64, fallback: Complex64) -> Complex64 {
    if w.re.is_finite() && w.im.is_finite() {
        w
    } else {
        fallback
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
        let fallback = Complex64::new(1e6, 1e6);
        if !z.re.is_finite() || !z.im.is_finite() {
            return fallback;
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
                if z.norm() < 1e-6 {
                    fallback
                } else {
                    1.0 / z
                }
            }
            MappingFunction::Custom => fallback,
        };
        sanitize_complex(raw, fallback)
    }

    fn derivative(&self, z: Complex64) -> Complex64 {
        let fallback = Complex64::new(0.0, 0.0);
        if !z.re.is_finite() || !z.im.is_finite() {
            return fallback;
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
                if z.norm() < 1e-6 {
                    Complex64::new(1e6, 1e6)
                } else {
                    -1.0 / (z * z)
                }
            }
            MappingFunction::Custom => fallback,
        };
        sanitize_complex(raw, fallback)
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
                    ("w = z^3 - 1/z", "z^3 - 1/z"),
                    ("w = z^2", "z^2"),
                    ("w = exp(z)", "exp(z)"),
                    ("w = sin(z)", "sin(z)"),
                    ("w = 1/z", "1/z"),
                    ("w = z + 1/z (Joukowsky)", "z + 1/z"),
                ],
            ),
            domain_coloring: true,
            palette: ShaderPalette::PhaseWheel,
            high_contrast: false,
            shader_preview: ShaderPreviewState::default(),
            domain_texture: None,
            domain_texture_key: None,
        }
    }
}

impl ComplexMappingTool {
    pub(crate) fn eval_z(&self, z: Complex64) -> Complex64 {
        let fallback = Complex64::new(1e6, 1e6);
        if !z.re.is_finite() || !z.im.is_finite() {
            return fallback;
        }
        match self.function {
            MappingFunction::Custom => self.math_input.expr().map_or(fallback, |expr| {
                sanitize_complex(expr.eval_complex_z(z), fallback)
            }),
            _ => self.function.evaluate(z),
        }
    }

    fn eval_domain_z(&self, z: Complex64) -> Complex64 {
        if !z.re.is_finite() || !z.im.is_finite() {
            return Complex64::new(f64::NAN, f64::NAN);
        }
        match self.function {
            MappingFunction::ZSquared => z * z,
            MappingFunction::Exponential => {
                Complex64::new(z.re.clamp(-EXP_SIN_GUARD_LIMIT, EXP_SIN_GUARD_LIMIT), z.im).exp()
            }
            MappingFunction::Sine => {
                Complex64::new(z.re, z.im.clamp(-EXP_SIN_GUARD_LIMIT, EXP_SIN_GUARD_LIMIT)).sin()
            }
            MappingFunction::Inverse => {
                if z.norm_sqr() < 1.0e-24 {
                    Complex64::new(f64::NAN, f64::NAN)
                } else {
                    1.0 / z
                }
            }
            MappingFunction::Custom => self
                .math_input
                .expr()
                .map_or(Complex64::new(f64::NAN, f64::NAN), |e| e.eval_complex_z(z)),
        }
    }

    pub(crate) fn deriv_z(&self, z: Complex64) -> Complex64 {
        let fallback = Complex64::new(0.0, 0.0);
        if !z.re.is_finite() || !z.im.is_finite() {
            return fallback;
        }
        match self.function {
            MappingFunction::Custom => {
                let h = Complex64::new(1e-7, 0.0);
                let f_plus = self.eval_z(z + h);
                let f_minus = self.eval_z(z - h);
                sanitize_complex((f_plus - f_minus) / (2.0 * h), fallback)
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
            None => self.math_input.expr().cloned().ok_or_else(|| {
                self.math_input
                    .error_message()
                    .unwrap_or_else(|| "Invalid custom complex mapping expression".to_string())
            }),
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
                let w = self.eval_domain_z(Complex64::new(re, im));
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

    fn has_finite_parameters(&self) -> bool {
        self.to_serializable_state().validate().is_ok()
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
        self.grid_density = s.grid_density;
        self.inspection_point = s.inspection_point;
        self.domain_coloring = s.domain_coloring;
        self.palette = palette;
        self.math_input.set_expression(
            s.custom_expr
                .as_deref()
                .unwrap_or(DEFAULT_CUSTOM_MAPPING_EXPR),
        );
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
        self.draw_central_canvas(parent_ui);
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

#[cfg(test)]
#[path = "complex_mapping/tests.rs"]
mod tests;
