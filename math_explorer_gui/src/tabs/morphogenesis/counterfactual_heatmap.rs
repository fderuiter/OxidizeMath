//! Theme-aware spatial heatmap renderers for counterfactual morphogenesis twin comparison.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::MorphogenesisHeatmapMode;
use crate::app::ThemeMode;
use eframe::egui::{Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * t.clamp(0.0, 1.0)).round() as u8
}

fn concentration_rgb(norm: f64, theme: ThemeMode) -> [u8; 3] {
    let t = if norm.is_finite() {
        norm.clamp(0.0, 1.0)
    } else {
        0.0
    };
    match theme {
        ThemeMode::Dark => heatmap_color(t),
        ThemeMode::HighContrast => heatmap_color_high_contrast(t),
        ThemeMode::Light => {
            if t < 0.5 {
                let s = t * 2.0;
                [
                    lerp_u8(244, 32, s),
                    lerp_u8(247, 138, s),
                    lerp_u8(252, 156, s),
                ]
            } else {
                let s = (t - 0.5) * 2.0;
                [
                    lerp_u8(32, 18, s),
                    lerp_u8(138, 34, s),
                    lerp_u8(156, 108, s),
                ]
            }
        }
    }
}

fn difference_rgb(norm: f64, theme: ThemeMode) -> [u8; 3] {
    let t = if norm.is_finite() {
        norm.clamp(0.0, 1.0)
    } else {
        0.0
    };
    match theme {
        ThemeMode::Dark => {
            if t < 0.5 {
                let s = t * 2.0;
                [lerp_u8(16, 198, s), lerp_u8(20, 48, s), lerp_u8(32, 108, s)]
            } else {
                let s = (t - 0.5) * 2.0;
                [
                    lerp_u8(198, 255, s),
                    lerp_u8(48, 214, s),
                    lerp_u8(108, 72, s),
                ]
            }
        }
        ThemeMode::Light => {
            if t < 0.5 {
                let s = t * 2.0;
                [
                    lerp_u8(246, 217, s),
                    lerp_u8(248, 95, s),
                    lerp_u8(252, 14, s),
                ]
            } else {
                let s = (t - 0.5) * 2.0;
                [lerp_u8(217, 127, s), lerp_u8(95, 0, s), lerp_u8(14, 38, s)]
            }
        }
        ThemeMode::HighContrast => {
            if t < 0.5 {
                let s = t * 2.0;
                [0, lerp_u8(0, 220, s), lerp_u8(0, 255, s)]
            } else {
                let s = (t - 0.5) * 2.0;
                [lerp_u8(0, 255, s), lerp_u8(220, 255, s), lerp_u8(255, 0, s)]
            }
        }
    }
}

fn finite_bounds(primary: &[f64], secondary: Option<&[f64]>) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in primary.iter().chain(secondary.unwrap_or(&[]).iter()) {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if min.is_finite() && max.is_finite() && max >= min {
        let range = max - min;
        (min, if range > 1e-6 { 1.0 / range } else { 1.0 })
    } else {
        (0.0, 1.0)
    }
}

fn render_split_pixels(
    u_b: &[f64],
    ua: &[f64],
    width: usize,
    n: usize,
    theme: ThemeMode,
) -> Vec<Color32> {
    let mut pixels = Vec::with_capacity(n);
    let (min, inv) = finite_bounds(u_b, Some(ua));
    let seam_x = width / 2;
    let seam_rgb = match theme {
        ThemeMode::Dark => [226, 232, 240],
        ThemeMode::Light => [15, 23, 42],
        ThemeMode::HighContrast => [255, 255, 0],
    };
    for idx in 0..n {
        let x = idx % width;
        let rgb = if x == seam_x {
            seam_rgb
        } else {
            let val = if x < seam_x { ua[idx] } else { u_b[idx] };
            let safe = if val.is_finite() { val } else { min };
            concentration_rgb((safe - min) * inv, theme)
        };
        pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }
    pixels
}

fn render_difference_pixels(u_b: &[f64], ua: &[f64], n: usize, theme: ThemeMode) -> Vec<Color32> {
    let mut pixels = Vec::with_capacity(n);
    let mut max_diff = 0.0_f64;
    for (&a, &b) in ua.iter().zip(u_b.iter()) {
        let d = (b - a).abs();
        if d.is_finite() {
            max_diff = max_diff.max(d);
        }
    }
    let inv = if max_diff > 1e-12 {
        1.0 / max_diff
    } else {
        0.0
    };
    for (&a, &b) in ua.iter().zip(u_b.iter()) {
        let d = (b - a).abs();
        let rgb = difference_rgb(if d.is_finite() { d * inv } else { 0.0 }, theme);
        pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }
    pixels
}

fn render_single_pixels(u_b: &[f64], n: usize, theme: ThemeMode) -> Vec<Color32> {
    let mut pixels = Vec::with_capacity(n);
    let (min, inv) = finite_bounds(u_b, None);
    for &val in u_b.iter().take(n) {
        let safe = if val.is_finite() { val } else { min };
        let rgb = concentration_rgb((safe - min) * inv, theme);
        pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }
    pixels.resize(n, Color32::BLACK);
    pixels
}

/// Renders `Active u_B`, `Split u_A | u_B`, or `Difference |u_B - u_A|` across `Dark`, `Light`, and `HighContrast`.
#[must_use]
pub fn render_morphogenesis_heatmap(
    u_b: &[f64],
    u_a: Option<&[f64]>,
    width: usize,
    height: usize,
    mode: MorphogenesisHeatmapMode,
    theme: ThemeMode,
) -> ColorImage {
    let n = width * height;
    let effective_ua = u_a.filter(|a| a.len() == n && u_b.len() == n);
    let pixels = match (mode, effective_ua) {
        (MorphogenesisHeatmapMode::SplitUaUb, Some(ua)) if width >= 2 => {
            render_split_pixels(u_b, ua, width, n, theme)
        }
        (MorphogenesisHeatmapMode::DifferenceAbs, Some(ua)) => {
            render_difference_pixels(u_b, ua, n, theme)
        }
        _ => render_single_pixels(u_b, n, theme),
    };
    ColorImage {
        size: [width, height],
        pixels,
        ..Default::default()
    }
}
