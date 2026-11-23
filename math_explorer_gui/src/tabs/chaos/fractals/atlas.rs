#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::formula::CompiledFractalJob;
use super::runner::{COORD_MAX, COORD_MIN, ZOOM_MAX, ZOOM_MIN};
use super::FractalViewer;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use num_complex::Complex;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Supported `(x, y)` parameter axis pairs for [`FractalCartographer`].
pub const FRACTAL_ATLAS_AXIS_PAIRS: &[(&str, &str)] =
    &[("julia_re", "julia_im"), ("julia_re", "param_0")];

const ATLAS_ESCAPE_ITER: u32 = 64;

/// Discrete connectedness and orbital stability regimes in the complex parameter plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FractalRegime {
    /// Connected Julia set with an attracting fixed point or periodic cycle.
    ConnectedCycle = 0,
    /// Dendrite filament, Misiurewicz pre-periodic orbit, or slow-escaping boundary.
    DendriteBoundary = 1,
    /// Totally disconnected Cantor dust where the critical orbit escapes rapidly.
    CantorDust = 2,
}

impl FractalRegime {
    /// Converts a raw `u8` cell regime code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::ConnectedCycle,
            1 => Self::DendriteBoundary,
            _ => Self::CantorDust,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ConnectedCycle => "Connected Cycle",
            Self::DendriteBoundary => "Dendrite / Boundary",
            Self::CantorDust => "Cantor Dust",
        }
    }
}

/// Default serialized atlas axis keys for `FractalState`.
#[must_use]
pub fn default_fractal_atlas_axes() -> [String; 2] {
    ["julia_re".to_string(), "julia_im".to_string()]
}

/// Predicate skipping default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_fractal_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "julia_re" && axes[1] == "julia_im"
}

fn fractal_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (resolved, label, min, max, step) = match key {
        "julia_re" => ("julia_re", "Re(c)", COORD_MIN, COORD_MAX, 0.01),
        "param_0" => ("param_0", "Custom Param 0", -3.0, 3.0, 0.05),
        _ => ("julia_im", "Im(c)", COORD_MIN, COORD_MAX, 0.01),
    };
    AtlasAxisSpec::from_theory(
        theory_params,
        resolved,
        label,
        ParameterConstraint { min, max, step },
    )
}

/// Zero-allocation 2D connectedness and cycle cartographer for complex escape fractals.
#[derive(Clone, Debug)]
pub struct FractalCartographer {
    axes: AtlasAxisPair,
    job: CompiledFractalJob,
    base_julia: Complex<f64>,
    base_param0: f64,
}

impl FractalCartographer {
    /// Builds a [`FractalCartographer`] bounded by `theory_params` and backed by `job`.
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        job: CompiledFractalJob,
        base_julia: Complex<f64>,
        base_param0: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: fractal_axis_spec(theory_params, axis_keys.0),
                y: fractal_axis_spec(theory_params, axis_keys.1),
            },
            job,
            base_julia,
            base_param0,
        }
    }

    #[inline]
    fn resolve_point(&self, p1: f64, p2: f64) -> (Complex<f64>, Option<f64>) {
        let mut c = self.base_julia;
        let mut p0 = self.base_param0;
        let mut override_p0 = false;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "julia_re" => c.re = v,
                "julia_im" => c.im = v,
                "param_0" => {
                    p0 = v;
                    override_p0 = true;
                }
                _ => {}
            }
        }
        (c, override_p0.then_some(p0))
    }
}

impl RegimeCartographer for FractalCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (c, p0_override) = self.resolve_point(p1, p2);
        let (regime, primary, secondary) =
            self.job
                .classify_c_plane_point(c, p0_override, ATLAS_ESCAPE_ITER);
        AtlasCell {
            regime,
            primary,
            secondary,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match FractalRegime::from_u8(cell.regime) {
            FractalRegime::ConnectedCycle => (1.0 / (1.0 - cell.primary).max(1.0)).clamp(0.25, 1.0),
            FractalRegime::DendriteBoundary => cell.primary.clamp(0.35, 1.0),
            FractalRegime::CantorDust => cell.primary.clamp(0.15, 0.95),
        };
        atlas_regime_color(cell.regime, 3, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        FractalRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Stability / Escape Ratio"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = FractalRegime::from_u8(cell.regime);
        match reg {
            FractalRegime::ConnectedCycle => format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nCycle Period ≈ {:.0} (λ ≈ {:+.3})",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.secondary.max(1.0),
                cell.primary
            ),
            FractalRegime::DendriteBoundary | FractalRegime::CantorDust => format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nEscape Steps: {:.0} / {}",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.secondary,
                ATLAS_ESCAPE_ITER
            ),
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            FractalRegime::ConnectedCycle,
            FractalRegime::DendriteBoundary,
            FractalRegime::CantorDust,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 3, 0.7, high_contrast),
            )
        })
        .collect()
    }
}

impl FractalViewer {
    pub(crate) fn build_theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let mut map = HashMap::new();
        let coord = ParameterConstraint {
            min: COORD_MIN,
            max: COORD_MAX,
            step: 0.01,
        };
        map.insert("center_re".to_string(), coord.clone());
        map.insert("center_im".to_string(), coord.clone());
        map.insert("julia_re".to_string(), coord.clone());
        map.insert("julia_im".to_string(), coord);
        map.insert(
            "zoom".to_string(),
            ParameterConstraint {
                min: ZOOM_MIN,
                max: ZOOM_MAX,
                step: 0.1,
            },
        );
        map.insert(
            "param_0".to_string(),
            ParameterConstraint {
                min: -3.0,
                max: 3.0,
                step: 0.05,
            },
        );
        map
    }

    fn first_custom_param_value(&self) -> f64 {
        self.math_input
            .param_slice()
            .first()
            .copied()
            .unwrap_or(1.0)
    }

    fn set_first_custom_param_value(&mut self, value: f64) {
        let mut pmap = self.math_input.params_map();
        if let Some(first_val) = pmap.values_mut().next() {
            *first_val = value.clamp(-3.0, 3.0);
            self.math_input.set_params_map(&pmap);
        }
    }

    /// Applies an [`AtlasTeleportEvent`] to `julia_c` and/or custom parameter 0.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "julia_re" => self.julia_c.re = val.clamp(COORD_MIN, COORD_MAX),
                "julia_im" => self.julia_c.im = val.clamp(COORD_MIN, COORD_MAX),
                "param_0" => self.set_first_custom_param_value(val),
                _ => {}
            }
        }
        self.dirty = true;
    }

    pub(super) fn render_fractal_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let p0 = self.first_custom_param_value();
        let get_val = |k: &str| match k {
            "julia_re" => self.julia_c.re,
            "julia_im" => self.julia_c.im,
            "param_0" => p0,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.mode.is_custom().hash(&mut h);
        if self.mode.is_custom() {
            self.math_input.raw_text.hash(&mut h);
            for (idx, &pv) in self.math_input.param_slice().iter().enumerate() {
                if !(idx == 0 && (xk == "param_0" || yk == "param_0")) {
                    pv.to_bits().hash(&mut h);
                }
            }
        }
        if xk != "julia_re" && yk != "julia_re" {
            self.julia_c.re.to_bits().hash(&mut h);
        }
        if xk != "julia_im" && yk != "julia_im" {
            self.julia_c.im.to_bits().hash(&mut h);
        }

        let theory_params = self.theory_parameters();
        let job = if self.mode.is_custom() {
            CompiledFractalJob::from_bar(&self.math_input)
        } else {
            self.quadratic_job.clone()
        };
        let base_julia = self.julia_c;
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "fractals",
            get_val(xk),
            get_val(yk),
            h.finish(),
            |keys| {
                Arc::new(FractalCartographer::new(
                    &theory_params,
                    keys,
                    job,
                    base_julia,
                    p0,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}
