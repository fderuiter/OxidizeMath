#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Progressive asynchronous 2D parameter regime atlas engine.
//!
//! Generalizes progressive stride rasterization (`8x -> 4x -> 2x -> 1x`) onto
//! [`crate::async_sim::BackgroundWorker`] for arbitrary 2D parameter slices bounded by
//! [`scientific_metadata::theory::ParameterConstraint`].

mod controller;
mod runner;
mod ui;

pub use controller::{AtlasTeleportEvent, ParameterAtlasController};
pub use runner::{
    clamp_atlas_dimensions, AtlasPassMeta, AtlasRenderRequest, AtlasRunner, ATLAS_SLICE_BUDGET,
    ATLAS_STRIDES, DEFAULT_ATLAS_SIDE, MAX_ATLAS_SIDE,
};

use eframe::egui;
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Metadata and domain bounds for one axis of a 2D parameter atlas.
#[derive(Clone, Debug)]
pub struct AtlasAxisSpec {
    /// Canonical parameter key matching [`scientific_metadata::theory::TheoryDescribable::theory_parameters`].
    pub key: &'static str,
    /// Human-readable axis label rendered on the atlas plot and tooltips.
    pub label: &'static str,
    /// Validated numeric domain bounds and step size.
    pub constraint: ParameterConstraint,
}

impl AtlasAxisSpec {
    /// Resolves an [`AtlasAxisSpec`] from `theory_params`, falling back to `fallback` only if absent.
    #[must_use]
    pub fn from_theory(
        theory_params: &HashMap<String, ParameterConstraint>,
        key: &'static str,
        label: &'static str,
        fallback: ParameterConstraint,
    ) -> Self {
        let constraint = theory_params
            .get(key)
            .cloned()
            .filter(|c| c.min.is_finite() && c.max.is_finite() && c.max > c.min)
            .unwrap_or(fallback);
        Self {
            key,
            label,
            constraint,
        }
    }

    /// Positive width of the parameter interval.
    #[inline]
    #[must_use]
    pub fn span(&self) -> f64 {
        (self.constraint.max - self.constraint.min).max(1e-9)
    }

    /// Midpoint of the parameter interval.
    #[inline]
    #[must_use]
    pub fn midpoint(&self) -> f64 {
        0.5 * (self.constraint.min + self.constraint.max)
    }

    /// Clamps `value` to the finite closed domain `[min, max]`.
    #[inline]
    #[must_use]
    pub fn clamp(&self, value: f64) -> f64 {
        if value.is_finite() {
            value.clamp(self.constraint.min, self.constraint.max)
        } else {
            self.midpoint()
        }
    }
}

/// Pair of orthogonal `(x, y)` parameter axes spanning a 2D regime atlas.
#[derive(Clone, Debug)]
pub struct AtlasAxisPair {
    /// Horizontal parameter axis (`p1`).
    pub x: AtlasAxisSpec,
    /// Vertical parameter axis (`p2`).
    pub y: AtlasAxisSpec,
}

impl AtlasAxisPair {
    /// Maps raster sample coordinates `(sx, sy)` into `(p1, p2)` parameter space.
    ///
    /// Row `0` corresponds to `y.constraint.max` (top of `egui::ColorImage`), matching `PlotImage`.
    #[inline]
    #[must_use]
    pub fn pixel_to_coords(&self, sx: usize, sy: usize, width: usize, height: usize) -> (f64, f64) {
        let w = width.max(1) as f64;
        let h = height.max(1) as f64;
        let tx = ((sx as f64) + 0.5) / w;
        let ty = ((sy as f64) + 0.5) / h;
        let p1 = self.x.constraint.min + tx * self.x.span();
        let p2 = self.y.constraint.max - ty * self.y.span();
        (p1, p2)
    }

    /// Maps `(p1, p2)` parameter coordinates to clamped `(sx, sy)` raster indices.
    #[inline]
    #[must_use]
    pub fn coords_to_pixel(&self, p1: f64, p2: f64, width: usize, height: usize) -> (usize, usize) {
        if width == 0 || height == 0 {
            return (0, 0);
        }
        let tx = ((self.x.clamp(p1) - self.x.constraint.min) / self.x.span()).clamp(0.0, 1.0);
        let ty = ((self.y.constraint.max - self.y.clamp(p2)) / self.y.span()).clamp(0.0, 1.0);
        let sx = ((tx * width as f64).floor() as usize).min(width - 1);
        let sy = ((ty * height as f64).floor() as usize).min(height - 1);
        (sx, sy)
    }
}

/// Zero-allocation per-coordinate evaluation result produced by a [`RegimeCartographer`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtlasCell {
    /// Discrete dynamical or numerical regime index.
    pub regime: u8,
    /// Primary scalar observable (e.g. dominant growth rate, final outbreak fraction, spike count).
    pub primary: f64,
    /// Secondary scalar observable (e.g. $R_0$, dominant $k^2$, or peak membrane voltage).
    pub secondary: f64,
}

/// Maps a normalized scalar `t` in `[0.0, 1.0]` to an accessible heatmap color.
///
/// Routes `high_contrast == true` through [`heatmap_color_high_contrast`] (`>= 3.0:1` on `#000000`).
#[inline]
#[must_use]
pub fn atlas_scalar_color(t: f64, high_contrast: bool) -> egui::Color32 {
    let norm = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let rgb = if high_contrast {
        heatmap_color_high_contrast(norm)
    } else {
        heatmap_color(norm)
    };
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

/// Maps a discrete `(regime_slot, regime_count)` and local `[0.0, 1.0]` intensity to `Color32`.
///
/// In `HighContrast` mode, partitions `[0.0, 1.0]` across [`heatmap_color_high_contrast`] so every
/// regime band maintains `>= 3.0:1` luminance contrast against `#000000`.
#[must_use]
pub fn atlas_regime_color(
    regime_slot: u8,
    regime_count: u8,
    intensity: f64,
    high_contrast: bool,
) -> egui::Color32 {
    let local = if intensity.is_finite() {
        intensity.clamp(0.0, 1.0)
    } else {
        0.5
    };
    if high_contrast {
        let count = f64::from(regime_count.max(1));
        let slot = f64::from(regime_slot.min(regime_count.saturating_sub(1)));
        let band_t = (slot + 0.15 + 0.70 * local) / count;
        return atlas_scalar_color(band_t, true);
    }
    let base: [u8; 3] = match regime_slot {
        0 => [36, 86, 138],
        1 => [34, 168, 132],
        2 => [235, 168, 42],
        _ => [206, 66, 87],
    };
    let scale = 0.72 + 0.28 * local;
    let r = (f64::from(base[0]) * scale).round().clamp(0.0, 255.0) as u8;
    let g = (f64::from(base[1]) * scale).round().clamp(0.0, 255.0) as u8;
    let b = (f64::from(base[2]) * scale).round().clamp(0.0, 255.0) as u8;
    egui::Color32::from_rgb(r, g, b)
}

/// Zero-allocation 2D parameter-space evaluator for [`AtlasRunner`] and [`ParameterAtlasController`].
pub trait RegimeCartographer: Send + Sync + 'static {
    /// Returns the active `(x, y)` parameter axis metadata and [`ParameterConstraint`] bounds.
    fn axes(&self) -> AtlasAxisPair;

    /// Evaluates the regime classification and scalar observables at `(p1, p2)`.
    ///
    /// Implementations MUST perform zero heap allocations so evaluation is safe inside
    /// `VerifiedEngine::run_verified`.
    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell;

    /// Maps an [`AtlasCell`] to an opaque pixel color (`HighContrast` >= 3.0:1 against `#000000`).
    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32;

    /// Returns the human-readable name for `regime`.
    fn regime_label(&self, regime: u8) -> &'static str;

    /// Returns the label of the primary scalar observable.
    fn primary_metric_label(&self) -> &'static str;

    /// Formats a multi-line hover/keyboard tooltip for `(p1, p2)` and its evaluated `cell`.
    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String;

    /// Optional analytical bifurcation or threshold curve `[p1, p2]` rendered over the heatmap.
    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        None
    }

    /// Optional discrete `(regime_label, swatch_color)` legend entries shown above the plot.
    fn legend_entries(&self, _high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests;

#[cfg(all(
    test,
    target_arch = "wasm32",
    feature = "biology",
    feature = "epidemiology"
))]
mod wasm_tests;
