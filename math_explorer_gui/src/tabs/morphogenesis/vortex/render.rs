#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation rasterizer mapping coupled vorticity and morphogen fields onto [`Color32`] buffers.

use super::state::VortexDisplayMode;
use crate::app::palette::{DARK_PALETTE, HIGH_CONTRAST_PALETTE};
use eframe::egui::{Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::vortex_morphogenesis::CoupledVortexMorphogenesis;

#[inline]
fn lerp_channel(a: u8, b: u8, weight_256: u16) -> u8 {
    let inv = 256_u16.saturating_sub(weight_256);
    (((a as u16) * inv + (b as u16) * weight_256) >> 8) as u8
}

#[inline]
fn blend_rgb(base: [u8; 3], top: [u8; 3], alpha: f64) -> Color32 {
    let w = (alpha.clamp(0.0, 1.0) * 256.0).round() as u16;
    Color32::from_rgb(
        lerp_channel(base[0], top[0], w),
        lerp_channel(base[1], top[1], w),
        lerp_channel(base[2], top[2], w),
    )
}

fn min_max_finite(slice: &[f64], obstacles: &[bool]) -> (f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for (i, &val) in slice.iter().enumerate() {
        if !obstacles.get(i).copied().unwrap_or(false) && val.is_finite() {
            min_v = min_v.min(val);
            max_v = max_v.max(val);
        }
    }
    if min_v.is_finite() && max_v.is_finite() && max_v >= min_v {
        (min_v, max_v)
    } else {
        (0.0, 1.0)
    }
}

fn sample_colormap(norm: f64, high_contrast: bool) -> [u8; 3] {
    let t = if norm.is_finite() {
        norm.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if high_contrast {
        heatmap_color_high_contrast(t)
    } else {
        heatmap_color(t)
    }
}

fn coupled_cell_color(
    v_norm: f64,
    omega: f64,
    omega_scale: f64,
    high_contrast: bool,
) -> Color32 {
    let curl = if omega.is_finite() {
        (omega * omega_scale).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let bg = if high_contrast {
        if curl >= 0.0 {
            blend_rgb([10, 14, 24], [0, 190, 245], curl.abs() * 0.85)
        } else {
            blend_rgb([10, 14, 24], [190, 95, 250], curl.abs() * 0.85)
        }
    } else if curl >= 0.0 {
        blend_rgb([15, 24, 42], [28, 162, 218], curl.abs() * 0.78)
    } else {
        blend_rgb([15, 24, 42], [138, 78, 214], curl.abs() * 0.78)
    };

    let morph = sample_colormap(v_norm, high_contrast);
    let base_rgb = [bg.r(), bg.g(), bg.b()];
    let alpha = (v_norm * 1.15).clamp(0.0, 0.96);
    blend_rgb(base_rgb, morph, alpha)
}

struct FieldScales {
    u_min: f64,
    u_inv: f64,
    v_min: f64,
    v_inv: f64,
    omega_scale: f64,
}

fn mode_cell_color(
    mode: VortexDisplayMode,
    u_val: f64,
    v_val: f64,
    omega: f64,
    scales: &FieldScales,
    high_contrast: bool,
) -> Color32 {
    match mode {
        VortexDisplayMode::CoupledWake => {
            let v_norm = if v_val.is_finite() {
                ((v_val - scales.v_min) * scales.v_inv).clamp(0.0, 1.0)
            } else {
                0.0
            };
            coupled_cell_color(v_norm, omega, scales.omega_scale, high_contrast)
        }
        VortexDisplayMode::MorphogenV => {
            let rgb = sample_colormap((v_val - scales.v_min) * scales.v_inv, high_contrast);
            Color32::from_rgb(rgb[0], rgb[1], rgb[2])
        }
        VortexDisplayMode::MorphogenU => {
            let rgb = sample_colormap((u_val - scales.u_min) * scales.u_inv, high_contrast);
            Color32::from_rgb(rgb[0], rgb[1], rgb[2])
        }
        VortexDisplayMode::VorticityCurl => {
            let norm = 0.5 + 0.5 * (omega * scales.omega_scale).clamp(-1.0, 1.0);
            let rgb = sample_colormap(norm, high_contrast);
            Color32::from_rgb(rgb[0], rgb[1], rgb[2])
        }
    }
}

/// Renders the coupled fluid-morphogen state into `pixels` (`width * height`) with `y = 0` at bottom.
pub fn render_vortex_field_into(
    engine: &CoupledVortexMorphogenesis,
    mode: VortexDisplayMode,
    high_contrast: bool,
    pixels: &mut Vec<Color32>,
) {
    let w = engine.config.width;
    let h = engine.config.height;
    let total = w * h;
    if pixels.len() != total {
        pixels.resize(total, Color32::BLACK);
    }

    let u_slice = engine.morphogens.u();
    let v_slice = engine.morphogens.v();
    let omega_slice = engine.vorticity_row();
    let obs_slice = engine.obstacles_row();
    let emit_slice = engine.emitters_row();

    let (u_min, u_max) = min_max_finite(u_slice, obs_slice);
    let (v_min, v_max) = min_max_finite(v_slice, obs_slice);
    let omega_peak = omega_slice
        .iter()
        .fold(0.02_f64, |acc, &w_z| if w_z.is_finite() { acc.max(w_z.abs()) } else { acc });
    let scales = FieldScales {
        u_min,
        u_inv: if u_max - u_min > 1e-6 { 1.0 / (u_max - u_min) } else { 1.0 },
        v_min,
        v_inv: 1.0 / (v_max - v_min).max(0.18),
        omega_scale: 1.0 / omega_peak,
    };
    let palette = if high_contrast { HIGH_CONTRAST_PALETTE } else { DARK_PALETTE };

    for py in 0..h {
        let gy = h - 1 - py;
        for gx in 0..w {
            let src_idx = flatten_2d_index(gx, gy, w);
            let dst_idx = flatten_2d_index(gx, py, w);
            let checker = (gx + gy).is_multiple_of(2);
            if obs_slice[src_idx] {
                pixels[dst_idx] = if high_contrast {
                    palette.node_stroke
                } else if checker {
                    palette.graph_edge
                } else {
                    palette.surface_fill
                };
                continue;
            }
            let base = mode_cell_color(
                mode,
                u_slice[src_idx],
                v_slice[src_idx],
                omega_slice[src_idx],
                &scales,
                high_contrast,
            );
            pixels[dst_idx] = if emit_slice[src_idx] && checker {
                blend_rgb([base.r(), base.g(), base.b()], [52, 245, 197], 0.55)
            } else {
                base
            };
        }
    }
}

/// Builds a standalone [`ColorImage`] snapshot for PNG, SVG, and GIF figure exports.
pub fn build_export_image(
    engine: &CoupledVortexMorphogenesis,
    mode: VortexDisplayMode,
    high_contrast: bool,
) -> ColorImage {
    let mut pixels = Vec::with_capacity(engine.config.width * engine.config.height);
    render_vortex_field_into(engine, mode, high_contrast, &mut pixels);
    ColorImage {
        size: [engine.config.width, engine.config.height],
        pixels,
        ..Default::default()
    }
}
