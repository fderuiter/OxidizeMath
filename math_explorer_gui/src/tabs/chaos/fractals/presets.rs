#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    FractalMode, FractalViewer, COORD_MAX, COORD_MIN, DEFAULT_FRACTAL_EXPR, DEFAULT_JULIA_CENTER,
    DEFAULT_MANDELBROT_CENTER, ZOOM_MAX, ZOOM_MIN,
};
use std::collections::BTreeMap;

const MAX_CUSTOM_FRACTAL_EXPR_BYTES: usize = 256;

fn is_false(flag: &bool) -> bool {
    !*flag
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
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) c_pinned: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) custom_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) atlas_visible: bool,
    #[serde(
        default = "super::atlas::default_fractal_atlas_axes",
        skip_serializing_if = "super::atlas::is_default_fractal_atlas_axes"
    )]
    pub(super) atlas_axes: [String; 2],
}

impl FractalState {
    pub(super) fn validate(&self) -> Result<(), String> {
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
        let params_valid = self.custom_params.values().all(|v| v.is_finite());
        if !coords_valid || !zoom_valid || !iter_valid || !params_valid {
            return Err("Fractal view parameters out of bounds".into());
        }
        if let Some(ref expr_str) = self.custom_expr {
            if expr_str.len() > MAX_CUSTOM_FRACTAL_EXPR_BYTES {
                return Err("Custom fractal expression exceeds maximum length".into());
            }
            if self.mode.is_custom() && math_commons::ast::parse_equation(expr_str.trim()).is_err()
            {
                return Err("Invalid custom fractal formula syntax".into());
            }
        }
        Ok(())
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
        self.dirty = true;
    }

    pub(crate) fn sync_active_plane_view(&mut self) {
        if self.mode.uses_julia_c() {
            self.z_plane_view = (self.center, self.zoom);
        } else {
            self.c_plane_view = (self.center, self.zoom);
        }
    }

    pub(super) fn build_state(&self, include_custom_expr: bool) -> FractalState {
        let (c_center, c_zoom) = self.c_plane_view;
        let (z_center, z_zoom) = self.z_plane_view;
        let m_non_default = (c_center.re - DEFAULT_MANDELBROT_CENTER.re).abs() > 1e-12
            || (c_center.im - DEFAULT_MANDELBROT_CENTER.im).abs() > 1e-12
            || (c_zoom - 1.0).abs() > 1e-12;
        let j_non_default = (z_center.re - DEFAULT_JULIA_CENTER.re).abs() > 1e-12
            || (z_center.im - DEFAULT_JULIA_CENTER.im).abs() > 1e-12
            || (z_zoom - 1.0).abs() > 1e-12;

        let (m_re, m_im, m_z) = if self.mode.uses_julia_c() && m_non_default {
            (Some(c_center.re), Some(c_center.im), Some(c_zoom))
        } else {
            (None, None, None)
        };
        let (j_re, j_im, j_z) = if !self.mode.uses_julia_c() && j_non_default {
            (Some(z_center.re), Some(z_center.im), Some(z_zoom))
        } else {
            (None, None, None)
        };
        let is_custom = self.mode.is_custom();
        let custom_expr = (include_custom_expr
            && (is_custom || self.math_input.raw_text != DEFAULT_FRACTAL_EXPR))
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
            mandelbrot_center_re: m_re,
            mandelbrot_center_im: m_im,
            mandelbrot_zoom: m_z,
            julia_center_re: j_re,
            julia_center_im: j_im,
            julia_zoom: j_z,
            custom_expr,
            custom_params,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        }
    }
}

/// Legacy starter preset JSON envelope for the Fractal Viewer.
pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Explorer",
    "domain": "chaos",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "center_re": -0.75,
    "center_im": 0.0,
    "zoom": 1.0,
    "max_iter": 150,
    "mode": "Mandelbrot",
    "julia_re": -0.7,
    "julia_im": 0.27015
  }
}"#;

/// Starter preset JSON envelope just inside the Mandelbrot main cardioid cusp (`c = 0.24`).
pub const STARTER_CUSP_CONNECTED_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Cusp: Connected Julia (c = 0.24)",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "center_re": 0.25,
    "center_im": 0.0,
    "zoom": 4.0,
    "max_iter": 200,
    "mode": "Mandelbrot",
    "julia_re": 0.24,
    "julia_im": 0.0,
    "c_pinned": true
  }
}"#;

/// Starter preset JSON envelope just outside the Mandelbrot cusp (`c = 0.27`, Cantor dust `J_c`).
pub const STARTER_CUSP_DUST_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Mandelbrot Cusp: Cantor Dust (c = 0.27)",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "center_re": 0.25,
    "center_im": 0.0,
    "zoom": 4.0,
    "max_iter": 200,
    "mode": "Mandelbrot",
    "julia_re": 0.27,
    "julia_im": 0.0,
    "c_pinned": true
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_viewer_starter",
        title: "Mandelbrot Explorer",
        description: "z_{n+1} = z_n^2 + c: Complex quadratic polynomial iteration and boundary fractal escape times.",
        category: "Fractals",
        icon: "❄️",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_mandelbrot_cusp_connected",
        title: "Mandelbrot Cusp: Connected Julia (c = 0.24)",
        description: "Pinned c = 0.24 inside the main cardioid cusp with bounded critical orbit and connected J_c.",
        category: "Fractals",
        icon: "🔗",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_CUSP_CONNECTED_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "fractal_mandelbrot_cusp_dust",
        title: "Mandelbrot Cusp: Cantor Dust (c = 0.27)",
        description: "Pinned c = 0.27 past the c = 0.25 cusp bottleneck where escaping critical orbits shatter J_c into dust.",
        category: "Fractals",
        icon: "✨",
        domain: "chaos",
        tool_name: Some("FractalViewer"),
        preset_generator: || STARTER_CUSP_DUST_JSON,
    }
}
