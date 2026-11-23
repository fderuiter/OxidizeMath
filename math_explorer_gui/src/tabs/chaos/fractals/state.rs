#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::shader_foundry::{
    bind_complex_shader_params, validate_complex_shader_vars, ShaderUniformDefaults,
};
use crate::framework::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::{parse, parse_equation, CompiledExpr, Expr};
use math_commons::math_kernel::colormap::ShaderPalette;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

const fn default_smooth_coloring() -> bool {
    true
}

const fn is_true(v: &bool) -> bool {
    *v
}

fn default_palette_slug() -> String {
    ShaderPalette::CyclicCosine.as_slug().to_string()
}

fn is_default_palette_slug(slug: &str) -> bool {
    ShaderPalette::from_slug(slug) == Some(ShaderPalette::CyclicCosine)
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FractalState {
    pub(super) center_re: f64,
    pub(super) center_im: f64,
    pub(super) zoom: f64,
    pub(super) max_iter: u32,
    pub(super) mode: FractalMode,
    pub(super) julia_re: f64,
    pub(super) julia_im: f64,
    #[serde(default = "default_smooth_coloring", skip_serializing_if = "is_true")]
    pub(super) smooth_coloring: bool,
    #[serde(
        default = "default_palette_slug",
        skip_serializing_if = "is_default_palette_slug"
    )]
    pub(super) palette: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) custom_params: BTreeMap<String, f64>,
}

impl FractalState {
    fn validate_bounds(&self) -> Result<(), String> {
        let coords_valid = [self.center_re, self.center_im, self.julia_re, self.julia_im]
            .iter()
            .all(|v| v.is_finite() && (COORD_MIN..=COORD_MAX).contains(v));
        let zoom_valid = self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom);
        let iter_valid =
            (10..=math_commons::registry::MAX_ITERATIONS as u32).contains(&self.max_iter);
        if !coords_valid || !zoom_valid || !iter_valid {
            return Err("Fractal view parameters out of bounds".into());
        }
        if self.custom_params.len() > MAX_SYNTHESIZED_PARAMS
            || !self.custom_params.iter().all(|(k, &v)| {
                !k.is_empty() && k.len() <= 64 && v.is_finite() && (-1.0e4..=1.0e4).contains(&v)
            })
        {
            return Err("Custom fractal parameters out of valid range".into());
        }
        Ok(())
    }

    pub(super) fn validate(&self) -> Result<ShaderPalette, String> {
        self.validate_bounds()?;
        let palette = ShaderPalette::from_slug(&self.palette)
            .ok_or_else(|| format!("Unknown fractal shader palette '{}'", self.palette))?;
        if let Some(raw) = &self.custom_expr {
            if raw.len() > MAX_CUSTOM_SHADER_EXPR_BYTES {
                return Err("Custom fractal expression exceeds 512-byte limit".into());
            }
            let (_, expr) = parse_equation(raw.trim())
                .map_err(|e| format!("Invalid custom fractal expression: {e}"))?;
            let compiled = CompiledExpr::compile_complex(&expr, &["z", "c"])
                .map_err(|e| format!("Invalid custom fractal expression: {e}"))?;
            if compiled.param_names().len() > MAX_SYNTHESIZED_PARAMS {
                return Err(format!(
                    "Custom fractal expression exceeds {MAX_SYNTHESIZED_PARAMS} parameters"
                ));
            }
            let mut effective = BTreeMap::new();
            for name in compiled.param_names() {
                effective.insert(
                    name.clone(),
                    self.custom_params.get(name).copied().unwrap_or(1.0),
                );
            }
            let bound = bind_complex_shader_params(&expr, &effective, &["z", "c"])?;
            validate_complex_shader_vars(&bound)?;
        }
        Ok(palette)
    }
}

impl FractalViewer {
    pub(super) fn to_serializable_state(&self) -> FractalState {
        let is_custom = self.mode.is_custom();
        let custom_expr = (is_custom || self.math_input.raw_text.trim() != DEFAULT_FRACTAL_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        FractalState {
            center_re: self.center.re,
            center_im: self.center.im,
            zoom: self.zoom,
            max_iter: self.max_iter,
            mode: self.mode,
            julia_re: self.julia_c.re,
            julia_im: self.julia_c.im,
            smooth_coloring: self.smooth_coloring,
            palette: self.palette.as_slug().to_string(),
            custom_expr,
            custom_params,
        }
    }

    pub(super) fn shader_revision(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.mode.hash(&mut hasher);
        self.center.re.to_bits().hash(&mut hasher);
        self.center.im.to_bits().hash(&mut hasher);
        self.zoom.to_bits().hash(&mut hasher);
        self.max_iter.hash(&mut hasher);
        self.julia_c.re.to_bits().hash(&mut hasher);
        self.julia_c.im.to_bits().hash(&mut hasher);
        self.smooth_coloring.hash(&mut hasher);
        self.palette.hash(&mut hasher);
        self.high_contrast.hash(&mut hasher);
        self.math_input.raw_text.hash(&mut hasher);
        for &param in self.math_input.param_slice() {
            param.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    pub(super) fn resolve_active_shader_expr(&self) -> Result<(Expr, bool), String> {
        match self.mode {
            FractalMode::Mandelbrot => parse("z^2 + c")
                .map(|e| (e, false))
                .map_err(|e| e.to_string()),
            FractalMode::Julia => parse("z^2 + c")
                .map(|e| (e, true))
                .map_err(|e| e.to_string()),
            FractalMode::CustomMandelbrot | FractalMode::CustomJulia => {
                let expr = self.math_input.expr().ok_or_else(|| {
                    self.math_input
                        .error_message()
                        .unwrap_or_else(|| "Invalid custom fractal expression".to_string())
                })?;
                let bound =
                    bind_complex_shader_params(expr, &self.math_input.params_map(), &["z", "c"])?;
                Ok((bound, self.mode.uses_julia_c()))
            }
        }
    }

    pub(super) fn shader_uniform_defaults(&self) -> ShaderUniformDefaults {
        let raw = if self.viewport_size.0 > 0 && self.viewport_size.1 > 0 {
            self.viewport_size
        } else {
            DEFAULT_EXPORT_VIEWPORT_SIZE
        };
        let (w, h) = clamp_fractal_dimensions(raw.0, raw.1, MAX_FRACTAL_TEXTURE_SIDE);
        let safe_zoom = if self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom) {
            self.zoom
        } else {
            1.0
        };
        let scale_y = 3.0 / safe_zoom;
        let scale_x = scale_y * (w.max(1) as f64 / h.max(1) as f64);
        let cx = self.center.re.clamp(COORD_MIN, COORD_MAX);
        let cy = self.center.im.clamp(COORD_MIN, COORD_MAX);
        ShaderUniformDefaults {
            resolution: [w.max(1) as u32, h.max(1) as u32],
            viewport: [
                cx - scale_x * 0.5,
                cy - scale_y * 0.5,
                cx + scale_x * 0.5,
                cy + scale_y * 0.5,
            ],
            max_iter: self
                .max_iter
                .clamp(10, math_commons::registry::MAX_ITERATIONS as u32),
            julia_c: [
                self.julia_c.re.clamp(COORD_MIN, COORD_MAX),
                self.julia_c.im.clamp(COORD_MIN, COORD_MAX),
            ],
            palette: self.palette,
            high_contrast: self.high_contrast,
            smooth_coloring: self.smooth_coloring,
        }
    }
}

impl TheoryDescribable for FractalViewer {
    fn theory_description(&self) -> String {
        let name = match self.mode {
            FractalMode::Mandelbrot => "Mandelbrot set".to_string(),
            FractalMode::Julia => format!(
                "Julia set with c=({:.2}, {:.2})",
                self.julia_c.re, self.julia_c.im
            ),
            FractalMode::CustomMandelbrot => {
                format!(
                    "Custom escape fractal z_{{n+1}} = {}",
                    self.math_input.raw_text
                )
            }
            FractalMode::CustomJulia => format!(
                "Custom Julia fractal z_{{n+1}} = {} with c=({:.2}, {:.2})",
                self.math_input.raw_text, self.julia_c.re, self.julia_c.im
            ),
        };
        format!(
            "Fractal viewer: {}, center: ({:.2}, {:.2}), zoom: {:.2}",
            name, self.center.re, self.center.im, self.zoom
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Mandelbrot, B. B. (1982). The Fractal Geometry of Nature. W. H. Freeman; Douady, A. & Hubbard, J. H. (1982). Itération des polynômes quadratiques complexes. C. R. Acad. Sci. Paris, 294, 123-126.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("default".to_string(), "Fractal viewer".to_string());
        map
    }
}
