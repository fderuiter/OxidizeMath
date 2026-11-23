#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::formula::CompiledFractalJob;
use super::{
    FractalColorStyle, FractalMode, FractalRenderRequest, MAX_FRACTAL_TEXTURE_SIDE, ZOOM_MAX,
    ZOOM_MIN,
};
use crate::async_sim::StateSnapshot;
use eframe::egui;
use math_commons::math_kernel::colormap::{evaluate_fractal_palette, ShaderPalette};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::physics::chaos::fractals::{
    escape_time_julia, escape_time_mandelbrot, smooth_escape_julia, smooth_escape_mandelbrot,
};
use num_complex::Complex;

/// Clamps `(width, height)` proportionally so neither side exceeds `min(max_texture_side, 2048)`.
#[must_use]
pub fn clamp_fractal_dimensions(
    width: usize,
    height: usize,
    max_texture_side: usize,
) -> (usize, usize) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let limit = if max_texture_side == 0 {
        MAX_FRACTAL_TEXTURE_SIDE
    } else {
        max_texture_side.min(MAX_FRACTAL_TEXTURE_SIDE)
    };
    let max_dim = width.max(height);
    if max_dim <= limit {
        return (width, height);
    }
    let scale = limit as f64 / max_dim as f64;
    let clamped_w = ((width as f64 * scale).round() as usize).clamp(1, limit);
    let clamped_h = ((height as f64 * scale).round() as usize).clamp(1, limit);
    (clamped_w, clamped_h)
}

pub(in crate::tabs::chaos::fractals) fn snapshot_to_clamped_color_image(
    snapshot: &StateSnapshot,
    max_texture_side: usize,
) -> Option<egui::ColorImage> {
    let (clamped_w, clamped_h) =
        clamp_fractal_dimensions(snapshot.width, snapshot.height, max_texture_side);
    if clamped_w == 0 || clamped_h == 0 || snapshot.width == 0 || snapshot.height == 0 {
        return None;
    }
    let px = snapshot.pixels.try_read().ok()?;
    if px.len() != snapshot.width.saturating_mul(snapshot.height) {
        return None;
    }
    if (clamped_w, clamped_h) == (snapshot.width, snapshot.height) {
        return Some(egui::ColorImage::new([clamped_w, clamped_h], px.clone()));
    }
    let mut out = vec![egui::Color32::BLACK; clamped_w * clamped_h];
    for y in 0..clamped_h {
        let src_y = (y * snapshot.height / clamped_h).min(snapshot.height - 1);
        for x in 0..clamped_w {
            let src_x = (x * snapshot.width / clamped_w).min(snapshot.width - 1);
            out[flatten_2d_index(x, y, clamped_w)] =
                px[flatten_2d_index(src_x, src_y, snapshot.width)];
        }
    }
    Some(egui::ColorImage::new([clamped_w, clamped_h], out))
}

/// Maps a continuous or discrete escape-time value to an opaque `Color32` under `palette`.
#[must_use]
pub fn map_escape_to_color32(
    iter_val: f64,
    max_iter: u32,
    palette: ShaderPalette,
    high_contrast: bool,
) -> egui::Color32 {
    if !iter_val.is_finite() || iter_val >= f64::from(max_iter) {
        egui::Color32::BLACK
    } else {
        let rgb = evaluate_fractal_palette(iter_val, max_iter, palette, high_contrast);
        egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
    }
}

/// Maps an integer escape-time iteration count to an opaque `Color32` under the default cosine palette.
#[cfg(test)]
#[must_use]
pub fn map_iter_to_color32(iter: u32, max_iter: u32, high_contrast: bool) -> egui::Color32 {
    map_escape_to_color32(
        f64::from(iter),
        max_iter,
        ShaderPalette::CyclicCosine,
        high_contrast,
    )
}

fn eval_custom_mode_escape(
    z0: Complex<f64>,
    c: Complex<f64>,
    max_iter: u32,
    smooth: bool,
    formula: Option<&CompiledFractalJob>,
    fallback: impl FnOnce() -> f64,
) -> f64 {
    match (formula, smooth) {
        (Some(job), true) => job.smooth_escape_time(z0, c, max_iter),
        (Some(job), false) => f64::from(job.escape_time(z0, c, max_iter)),
        (None, _) => fallback(),
    }
}

pub(super) fn eval_pixel_color(
    re: f64,
    im: f64,
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
    style: FractalColorStyle,
) -> egui::Color32 {
    let pt = Complex::new(re, im);
    let iter_val = match (req.mode, style.smooth_coloring) {
        (FractalMode::Mandelbrot, true) => smooth_escape_mandelbrot(pt, req.max_iter),
        (FractalMode::Mandelbrot, false) => f64::from(escape_time_mandelbrot(pt, req.max_iter)),
        (FractalMode::Julia, true) => smooth_escape_julia(pt, req.julia_c, req.max_iter),
        (FractalMode::Julia, false) => f64::from(escape_time_julia(pt, req.julia_c, req.max_iter)),
        (FractalMode::CustomMandelbrot, smooth) => eval_custom_mode_escape(
            Complex::new(0.0, 0.0),
            pt,
            req.max_iter,
            smooth,
            formula,
            || {
                if smooth {
                    smooth_escape_mandelbrot(pt, req.max_iter)
                } else {
                    f64::from(escape_time_mandelbrot(pt, req.max_iter))
                }
            },
        ),
        (FractalMode::CustomJulia, smooth) => {
            eval_custom_mode_escape(pt, req.julia_c, req.max_iter, smooth, formula, || {
                if smooth {
                    smooth_escape_julia(pt, req.julia_c, req.max_iter)
                } else {
                    f64::from(escape_time_julia(pt, req.julia_c, req.max_iter))
                }
            })
        }
    };
    map_escape_to_color32(iter_val, req.max_iter, style.palette, req.high_contrast)
}

/// Synchronously renders a full-resolution fractal image (used for small seed previews and tests).
#[cfg(test)]
#[must_use]
pub fn render_fractal_color_image(req: &FractalRenderRequest) -> egui::ColorImage {
    render_fractal_color_image_with_formula(req, None)
}

/// Synchronously renders a full-resolution fractal image with an optional [`CompiledFractalJob`].
#[cfg(test)]
#[must_use]
pub fn render_fractal_color_image_with_formula(
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
) -> egui::ColorImage {
    render_fractal_color_image_styled(req, formula, FractalColorStyle::default())
}

/// Synchronously renders a full-resolution fractal image with explicit formula and [`FractalColorStyle`].
#[must_use]
pub fn render_fractal_color_image_styled(
    req: &FractalRenderRequest,
    formula: Option<&CompiledFractalJob>,
    style: FractalColorStyle,
) -> egui::ColorImage {
    let (width, height) = clamp_fractal_dimensions(req.width, req.height, MAX_FRACTAL_TEXTURE_SIDE);
    if width == 0 || height == 0 {
        return egui::ColorImage::new([1, 1], vec![egui::Color32::BLACK]);
    }
    let zoom = if req.zoom.is_finite() && req.zoom >= ZOOM_MIN {
        req.zoom.min(ZOOM_MAX)
    } else {
        1.0
    };
    let scale_y = 3.0 / zoom;
    let scale_x = scale_y * (width as f64 / height as f64);
    let x_start = req.center.re - scale_x * 0.5;
    let y_start = req.center.im + scale_y * 0.5;
    let mut pixels = vec![egui::Color32::BLACK; width * height];

    for y in 0..height {
        let im = y_start - (y as f64 / height as f64) * scale_y;
        for x in 0..width {
            let re = x_start + (x as f64 / width as f64) * scale_x;
            pixels[flatten_2d_index(x, y, width)] = eval_pixel_color(re, im, req, formula, style);
        }
    }
    egui::ColorImage::new([width, height], pixels)
}
