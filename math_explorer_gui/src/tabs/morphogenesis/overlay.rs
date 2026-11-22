#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::types::MorphogenesisOverlayMode;
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui::{Color32, ColorImage};
use math_commons::math_kernel::colormap::{
    dual_concentration_vorticity_color, heatmap_color, heatmap_color_high_contrast,
    vorticity_color, vorticity_color_high_contrast, vorticity_color_light,
};

fn min_max_finite(data: &[f64]) -> (f64, f64, f64) {
    let (min, max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &val| {
            if val.is_finite() {
                (lo.min(val), hi.max(val))
            } else {
                (lo, hi)
            }
        });
    let (min, max) = if min.is_finite() && max.is_finite() && max >= min {
        (min, max)
    } else {
        (0.0, 1.0)
    };
    let range = max - min;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 1.0 };
    (min, max, inv_range)
}

fn max_abs_finite(data: &[f64]) -> f64 {
    let peak = data.iter().fold(0.0_f64, |acc, &val| {
        if val.is_finite() {
            acc.max(val.abs())
        } else {
            acc
        }
    });
    if peak > 1e-6 {
        peak
    } else {
        1.0
    }
}

fn obstacle_color_for_theme(theme_mode: ThemeMode) -> Color32 {
    let palette = SemanticPalette::for_theme(theme_mode);
    match theme_mode {
        ThemeMode::HighContrast => palette.graph_edge,
        ThemeMode::Dark | ThemeMode::Light => palette.surface_fill,
    }
}

/// Renders concentration, signed vorticity `\omega_z`, or dual composite overlays with obstacle masks
/// across [`ThemeMode::Dark`], [`ThemeMode::Light`], and [`ThemeMode::HighContrast`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn render_morphogenesis_overlay(
    concentration: &[f64],
    vorticity: &[f64],
    obstacles: &[bool],
    width: usize,
    height: usize,
    overlay_mode: MorphogenesisOverlayMode,
    theme_mode: ThemeMode,
) -> ColorImage {
    let num_pixels = width * height;
    let mut pixels = Vec::with_capacity(num_pixels);
    let (c_min, _c_max, c_inv_range) = min_max_finite(concentration);
    let omega_scale = 1.0 / max_abs_finite(vorticity);
    let high_contrast = theme_mode == ThemeMode::HighContrast;
    let light_theme = theme_mode == ThemeMode::Light;
    let obs_color = obstacle_color_for_theme(theme_mode);

    for i in 0..num_pixels {
        if i < obstacles.len() && obstacles[i] {
            pixels.push(obs_color);
            continue;
        }
        let raw_c = concentration.get(i).copied().unwrap_or(c_min);
        let safe_c = if raw_c.is_finite() { raw_c } else { c_min };
        let c_norm = ((safe_c - c_min) * c_inv_range).clamp(0.0, 1.0);

        let raw_w = vorticity.get(i).copied().unwrap_or(0.0);
        let w_norm = if raw_w.is_finite() {
            (raw_w * omega_scale).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let rgb = match overlay_mode {
            MorphogenesisOverlayMode::Concentration => {
                if high_contrast {
                    heatmap_color_high_contrast(c_norm)
                } else {
                    heatmap_color(c_norm)
                }
            }
            MorphogenesisOverlayMode::Vorticity => {
                if high_contrast {
                    vorticity_color_high_contrast(w_norm)
                } else if light_theme {
                    vorticity_color_light(w_norm)
                } else {
                    vorticity_color(w_norm)
                }
            }
            MorphogenesisOverlayMode::DualOverlay => {
                dual_concentration_vorticity_color(c_norm, w_norm, light_theme, high_contrast)
            }
        };
        pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }

    ColorImage {
        size: [width, height],
        pixels,
        ..Default::default()
    }
}
