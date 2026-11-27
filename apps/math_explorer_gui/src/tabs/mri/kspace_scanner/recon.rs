//! `ColorImage` panel renderers for [`super::tool::MriKSpaceScannerTool`] backed by
//! [`math_explorer::physics::mri::reconstruction`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{KSPACE_DIM, KSPACE_LEN};
use eframe::egui::{Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
pub use math_explorer::physics::mri::reconstruction::{
    compute_forward_kspace, ForwardKSpaceCache, KSpaceReconWorkspace, ReconOutput,
};

/// Renders a `64x64` anatomical magnitude image (`Ground-Truth` or `Reconstructed`) in clinical grayscale.
#[must_use]
pub fn render_grayscale_panel(values: &[f64], high_contrast: bool) -> ColorImage {
    let peak = values
        .iter()
        .fold(
            0.0_f64,
            |acc, &v| if v.is_finite() { acc.max(v) } else { acc },
        )
        .max(1e-6);
    let floor = if high_contrast { 24.0 } else { 8.0 };
    let mut pixels = Vec::with_capacity(KSPACE_LEN);
    for &v in values {
        let norm = if v.is_finite() {
            (v / peak).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lum = (floor + norm * (255.0 - floor)).round() as u8;
        pixels.push(Color32::from_rgb(lum, lum, lum));
    }
    ColorImage {
        size: [KSPACE_DIM, KSPACE_DIM],
        pixels,
        ..Default::default()
    }
}

/// Renders the centered `64x64` log-magnitude k-space spectrum highlighting sampled vs erased bins.
#[must_use]
pub fn render_kspace_panel(log_power: &[f64], mask: &[bool], high_contrast: bool) -> ColorImage {
    let max_log = log_power
        .iter()
        .fold(
            0.0_f64,
            |acc, &v| if v.is_finite() { acc.max(v) } else { acc },
        )
        .max(1e-6);
    let mut pixels = Vec::with_capacity(KSPACE_LEN);
    for (&lp, &sampled) in log_power.iter().zip(mask.iter()) {
        let norm = if lp.is_finite() {
            (lp / max_log).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if sampled {
            let rgb = if high_contrast {
                heatmap_color_high_contrast(norm)
            } else {
                let r = (35.0 + 215.0 * norm.powf(1.4)) as u8;
                let g = (65.0 + 185.0 * norm) as u8;
                let b = (115.0 + 140.0 * norm.sqrt()) as u8;
                [r, g, b]
            };
            pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        } else {
            let ghost = (10.0 + 26.0 * norm) as u8;
            pixels.push(Color32::from_rgb(ghost, ghost / 2, ghost / 2));
        }
    }
    ColorImage {
        size: [KSPACE_DIM, KSPACE_DIM],
        pixels,
        ..Default::default()
    }
}

/// Renders the `64x64` absolute reconstruction error heatmap.
#[must_use]
pub fn render_error_panel(abs_error: &[f64], high_contrast: bool) -> ColorImage {
    let scale = abs_error
        .iter()
        .fold(
            0.0_f64,
            |acc, &v| if v.is_finite() { acc.max(v) } else { acc },
        )
        .max(0.05);
    let mut pixels = Vec::with_capacity(KSPACE_LEN);
    for &err in abs_error {
        let norm = if err.is_finite() {
            (err / scale).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rgb = if high_contrast {
            heatmap_color_high_contrast(norm)
        } else {
            heatmap_color(norm)
        };
        pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }
    ColorImage {
        size: [KSPACE_DIM, KSPACE_DIM],
        pixels,
        ..Default::default()
    }
}

/// Stitches all four `64x64` panels into a `2x2` composite `ColorImage` for PNG/SVG/GIF figure export.
#[must_use]
pub fn compose_quad_export_image(panels: [&ColorImage; 4], high_contrast: bool) -> ColorImage {
    let gap = 4_usize;
    let side = KSPACE_DIM * 2 + gap;
    let theme = if high_contrast {
        crate::app::ThemeMode::HighContrast
    } else {
        crate::app::ThemeMode::Dark
    };
    let border = crate::app::SemanticPalette::for_theme(theme).surface_fill;
    let mut pixels = vec![border; side * side];
    let offsets = [
        (0, 0),
        (0, KSPACE_DIM + gap),
        (KSPACE_DIM + gap, 0),
        (KSPACE_DIM + gap, KSPACE_DIM + gap),
    ];
    for (panel, (row_off, col_off)) in panels.into_iter().zip(offsets) {
        for r in 0..KSPACE_DIM {
            for c in 0..KSPACE_DIM {
                let dst = flatten_2d_index(col_off + c, row_off + r, side);
                pixels[dst] = panel.pixels[flatten_2d_index(c, r, KSPACE_DIM)];
            }
        }
    }
    ColorImage {
        size: [side, side],
        pixels,
        ..Default::default()
    }
}
