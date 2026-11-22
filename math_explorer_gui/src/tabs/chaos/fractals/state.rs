#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::shader_foundry::validate_complex_shader_vars;
use math_commons::ast::parse_equation;
use math_commons::math_kernel::colormap::ShaderPalette;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

const fn default_smooth_coloring() -> bool {
    true
}

const fn is_true(v: &bool) -> bool {
    *v
}

const fn is_false(v: &bool) -> bool {
    !*v
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
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) custom_julia_seed: bool,
}

impl FractalState {
    pub(super) fn validate(&self) -> Result<ShaderPalette, String> {
        let coords_valid = [self.center_re, self.center_im, self.julia_re, self.julia_im]
            .iter()
            .all(|v| v.is_finite() && (COORD_MIN..=COORD_MAX).contains(v));
        let zoom_valid = self.zoom.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&self.zoom);
        let iter_valid =
            (10..=math_commons::registry::MAX_ITERATIONS as u32).contains(&self.max_iter);
        if !coords_valid || !zoom_valid || !iter_valid {
            return Err("Fractal view parameters out of bounds".into());
        }
        let palette = ShaderPalette::from_slug(&self.palette)
            .ok_or_else(|| format!("Unknown fractal shader palette '{}'", self.palette))?;
        if let Some(raw) = &self.custom_expr {
            if raw.len() > MAX_CUSTOM_SHADER_EXPR_BYTES {
                return Err("Custom fractal expression exceeds 512-byte limit".into());
            }
            let (_, expr) = parse_equation(raw.trim())
                .map_err(|e| format!("Invalid custom fractal expression: {e}"))?;
            validate_complex_shader_vars(&expr)?;
        }
        Ok(palette)
    }
}

impl FractalViewer {
    pub(super) fn to_serializable_state(&self) -> FractalState {
        let custom_expr = (self.mode == FractalMode::Custom
            || self.math_input.raw_text.trim() != DEFAULT_CUSTOM_FRACTAL_EXPR)
            .then(|| self.math_input.raw_text.clone());
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
            custom_julia_seed: self.custom_julia_seed,
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
            FractalMode::Custom => format!("Custom recurrence z' = {}", self.math_input.raw_text),
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
