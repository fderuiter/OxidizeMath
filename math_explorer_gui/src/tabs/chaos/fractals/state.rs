#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::shader_foundry::{
    bind_complex_shader_params, validate_complex_shader_vars, ShaderUniformDefaults,
};
use crate::framework::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::{parse, parse_equation, CompiledExpr, Expr};
use math_commons::math_kernel::colormap::ShaderPalette;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
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

type SerializedPlaneView = (Option<f64>, Option<f64>, Option<f64>);

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
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) c_pinned: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) linked_explorer: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) mandelbrot_center_re: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) mandelbrot_center_im: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) mandelbrot_zoom: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) julia_center_re: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) julia_center_im: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) julia_zoom: Option<f64>,
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
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub(super) atlas_visible: bool,
    #[serde(
        default = "super::atlas::default_fractal_atlas_axes",
        skip_serializing_if = "super::atlas::is_default_fractal_atlas_axes"
    )]
    pub(super) atlas_axes: [String; 2],
}

impl FractalState {
    pub(super) fn validate_bounds(&self) -> Result<(), String> {
        let coords_valid = [
            Some(self.center_re),
            Some(self.center_im),
            Some(self.julia_re),
            Some(self.julia_im),
            self.mandelbrot_center_re,
            self.mandelbrot_center_im,
            self.julia_center_re,
            self.julia_center_im,
        ]
        .into_iter()
        .flatten()
        .all(|v| v.is_finite() && (COORD_MIN..=COORD_MAX).contains(&v));
        let zoom_valid = [Some(self.zoom), self.mandelbrot_zoom, self.julia_zoom]
            .into_iter()
            .flatten()
            .all(|z| z.is_finite() && (ZOOM_MIN..=ZOOM_MAX).contains(&z));
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
    /// Saves the active plane viewport and switches `FractalMode` while preserving `(c, z)` plane viewports.
    pub fn switch_mode(&mut self, next_mode: FractalMode) {
        if self.mode == next_mode {
            return;
        }
        self.sync_active_plane_view();
        self.mode = next_mode;
        let (next_center, next_zoom) = if next_mode.uses_julia_c() {
            self.z_plane_view
        } else {
            self.c_plane_view
        };
        self.center = next_center;
        self.zoom = next_zoom.clamp(ZOOM_MIN, ZOOM_MAX);
        self.mark_dirty();
    }

    /// Teleports `julia_c` to `c` and switches into progressive Julia rendering while preserving viewports.
    pub fn teleport_to_julia(&mut self, c: Complex<f64>) {
        if !c.re.is_finite() || !c.im.is_finite() {
            return;
        }
        self.julia_c = Complex::new(
            c.re.clamp(COORD_MIN, COORD_MAX),
            c.im.clamp(COORD_MIN, COORD_MAX),
        );
        let target = if self.mode.is_custom() {
            FractalMode::CustomJulia
        } else {
            FractalMode::Julia
        };
        self.switch_mode(target);
    }

    pub(crate) fn sync_active_plane_view(&mut self) {
        if self.mode.uses_julia_c() {
            self.z_plane_view = (self.center, self.zoom);
        } else {
            self.c_plane_view = (self.center, self.zoom);
        }
    }

    fn serialized_inactive_plane_views(&self) -> (SerializedPlaneView, SerializedPlaneView) {
        let (c_center, c_zoom) = self.c_plane_view;
        let (z_center, z_zoom) = self.z_plane_view;
        let m_non_default = (c_center.re - DEFAULT_MANDELBROT_CENTER.re).abs() > 1e-12
            || (c_center.im - DEFAULT_MANDELBROT_CENTER.im).abs() > 1e-12
            || (c_zoom - 1.0).abs() > 1e-12;
        let j_non_default = (z_center.re - DEFAULT_JULIA_CENTER.re).abs() > 1e-12
            || (z_center.im - DEFAULT_JULIA_CENTER.im).abs() > 1e-12
            || (z_zoom - 1.0).abs() > 1e-12;
        let m_view = if self.mode.uses_julia_c() && m_non_default {
            (Some(c_center.re), Some(c_center.im), Some(c_zoom))
        } else {
            (None, None, None)
        };
        let j_view = if !self.mode.uses_julia_c() && j_non_default {
            (Some(z_center.re), Some(z_center.im), Some(z_zoom))
        } else {
            (None, None, None)
        };
        (m_view, j_view)
    }

    pub(super) fn build_state(&self, include_custom_expr: bool) -> FractalState {
        let ((m_re, m_im, m_z), (j_re, j_im, j_z)) = self.serialized_inactive_plane_views();
        let is_custom = self.mode.is_custom();
        let custom_expr = (include_custom_expr
            && (is_custom || self.math_input.raw_text.trim() != DEFAULT_FRACTAL_EXPR))
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        let (xk, yk) = self.atlas.axis_keys();
        FractalState {
            center_re: self.center.re,
            center_im: self.center.im,
            zoom: self.zoom,
            max_iter: self.max_iter,
            mode: self.mode,
            julia_re: self.julia_c.re,
            julia_im: self.julia_c.im,
            c_pinned: self.c_pinned,
            linked_explorer: self.linked_explorer,
            mandelbrot_center_re: m_re,
            mandelbrot_center_im: m_im,
            mandelbrot_zoom: m_z,
            julia_center_re: j_re,
            julia_center_im: j_im,
            julia_zoom: j_z,
            smooth_coloring: self.smooth_coloring,
            palette: self.palette.as_slug().to_string(),
            custom_expr,
            custom_params,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        }
    }

    pub(super) fn apply_loaded_state(&mut self, state: &str) -> Result<(), String> {
        let s: FractalState =
            serde_json::from_str(state).map_err(|e| format!("Invalid fractal state: {e}"))?;
        let palette = s.validate()?;
        self.center = Complex::new(s.center_re, s.center_im);
        self.zoom = s.zoom;
        self.max_iter = s.max_iter;
        self.mode = s.mode;
        self.julia_c = Complex::new(s.julia_re, s.julia_im);
        self.c_pinned = s.c_pinned;
        self.linked_explorer = s.linked_explorer;
        self.smooth_coloring = s.smooth_coloring;
        self.palette = palette;
        self.c_plane_view = (
            Complex::new(
                s.mandelbrot_center_re
                    .unwrap_or(DEFAULT_MANDELBROT_CENTER.re),
                s.mandelbrot_center_im
                    .unwrap_or(DEFAULT_MANDELBROT_CENTER.im),
            ),
            s.mandelbrot_zoom.unwrap_or(1.0),
        );
        self.z_plane_view = (
            Complex::new(
                s.julia_center_re.unwrap_or(DEFAULT_JULIA_CENTER.re),
                s.julia_center_im.unwrap_or(DEFAULT_JULIA_CENTER.im),
            ),
            s.julia_zoom.unwrap_or(1.0),
        );
        self.sync_active_plane_view();
        self.math_input
            .set_expression(s.custom_expr.as_deref().unwrap_or(DEFAULT_FRACTAL_EXPR));
        self.math_input.set_params_map(&s.custom_params);
        self.atlas.set_visible(s.atlas_visible);
        let _ = self.atlas.set_axis_keys(&s.atlas_axes[0], &s.atlas_axes[1]);
        self.mark_dirty();
        Ok(())
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

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        self.build_theory_parameters()
    }
}
