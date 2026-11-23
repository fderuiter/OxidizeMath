//! Perceptual raster difference rendering and dataset/figure export helpers for shadow branches.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::hud::BranchMetadata;
use eframe::egui;
use egui_plot::ExportPrimitive;
use math_commons::math_kernel::colormap::{ensure_dark_canvas_contrast, heatmap_color};

/// Display mode for raster simulations when a shadow-twin branch is active.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RasterBranchViewMode {
    /// Render the perturbed counterfactual primary field $u_{\text{branch}}(x, y, t)$.
    #[default]
    CounterfactualBranch,
    /// Render the unperturbed baseline shadow field $u_{\text{shadow}}(x, y, t)$.
    ShadowBaseline,
    /// Render the perceptual absolute difference heatmap $|u_{\text{branch}} - u_{\text{shadow}}|$.
    DifferenceHeatmap,
}

impl RasterBranchViewMode {
    /// Short accessible UI label for radio/segmented controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::CounterfactualBranch => "Counterfactual",
            Self::ShadowBaseline => "Shadow Baseline",
            Self::DifferenceHeatmap => "|Δu| Diff Map",
        }
    }
}

/// Renders a perceptual high-contrast difference map into `out_rgb` (`3 * n` bytes) with zero heap allocations.
///
/// Returns `(l2_distance, peak_abs_delta)` across the overlapping scalar cells.
/// Identical cells (`|Δ| == 0`) render as neutral dark slate `[18, 22, 28]` while non-zero
/// deltas use WCAG `>= 3.0:1` contrast-lifted perceptual colors via [`ensure_dark_canvas_contrast`].
pub fn render_perceptual_difference_rgb(
    primary: &[f64],
    shadow: &[f64],
    out_rgb: &mut [u8],
) -> (f64, f64) {
    let cell_count = primary.len().min(shadow.len()).min(out_rgb.len() / 3);
    if cell_count == 0 {
        return (0.0, 0.0);
    }

    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f64;
    for idx in 0..cell_count {
        let diff = (primary[idx] - shadow[idx]).abs();
        if diff.is_finite() {
            sum_sq += diff * diff;
            if diff > peak {
                peak = diff;
            }
        }
    }

    if peak <= 1.0e-15 {
        for idx in 0..cell_count {
            let base = idx * 3;
            out_rgb[base] = 18;
            out_rgb[base + 1] = 22;
            out_rgb[base + 2] = 28;
        }
        return (0.0, 0.0);
    }

    let inv_peak = 1.0 / peak;
    for idx in 0..cell_count {
        let diff = (primary[idx] - shadow[idx]).abs();
        let base = idx * 3;
        if !diff.is_finite() || diff <= 1.0e-15 {
            out_rgb[base] = 18;
            out_rgb[base + 1] = 22;
            out_rgb[base + 2] = 28;
        } else {
            // Gamma-compressed perceptual normalization highlights subtle wavefront phase shifts.
            let norm = (diff * inv_peak).clamp(0.0, 1.0).sqrt();
            let rgb = ensure_dark_canvas_contrast(heatmap_color(norm));
            out_rgb[base] = rgb[0];
            out_rgb[base + 1] = rgb[1];
            out_rgb[base + 2] = rgb[2];
        }
    }

    (sum_sq.sqrt(), peak)
}

/// Builds a 2D dataset row encoding `BranchMetadata` from $t_k$ for CSV and JSON exports.
#[must_use]
pub fn build_branch_metadata_dataset_2d(meta: &BranchMetadata) -> (String, Vec<[f64; 2]>) {
    (
        format!("Branch Metadata [{}]", meta.summary_tag()),
        vec![[meta.fork_time, meta.divergence.instantaneous_l2]],
    )
}

/// Builds a 3D dataset row encoding `BranchMetadata` from $t_k$ for 3D CSV and JSON exports.
#[must_use]
pub fn build_branch_metadata_dataset_3d(
    meta: &BranchMetadata,
    fork_anchor: [f64; 3],
) -> (String, Vec<[f64; 3]>) {
    (
        format!("Branch Metadata [{}]", meta.summary_tag()),
        vec![fork_anchor],
    )
}

/// Subdivides a 2D polyline into alternating dash segments for offscreen SVG and PNG exports.
#[must_use]
pub fn build_dashed_export_primitives(
    name: &str,
    points: &[[f64; 2]],
    color: egui::Color32,
    stroke_width: f32,
) -> Vec<ExportPrimitive> {
    if points.len() < 2 {
        return Vec::new();
    }
    let stride = (points.len() / 28).clamp(2, 24);
    let dash_len = (stride * 2 / 3).max(2);
    let mut out = Vec::new();
    let mut idx = 0;
    while idx + 1 < points.len() {
        let end = (idx + dash_len).min(points.len());
        if end > idx + 1 {
            out.push(ExportPrimitive::Polyline {
                name: name.to_string(),
                points: points[idx..end].to_vec().into(),
                color,
                stroke_width,
                closed: false,
                fill: None,
            });
        }
        idx += stride;
    }
    out
}

/// Builds a vertical fork marker primitive at $t_k$ spanning `[y_min, y_max]`.
#[must_use]
pub fn build_fork_vline_primitive(
    meta: &BranchMetadata,
    y_min: f64,
    y_max: f64,
    color: egui::Color32,
) -> ExportPrimitive {
    let lo = if y_min.is_finite() { y_min } else { 0.0 };
    let hi = if y_max.is_finite() && y_max > lo {
        y_max
    } else {
        lo + 1.0
    };
    ExportPrimitive::Polyline {
        name: format!("Branch Fork Marker [{}]", meta.summary_tag()),
        points: vec![[meta.fork_time, lo], [meta.fork_time, hi]].into(),
        color,
        stroke_width: 1.5,
        closed: false,
        fill: None,
    }
}

/// Embeds `<desc id="oxidizemath-branch-metadata">...</desc>` inside exported SVG markup.
#[must_use]
pub fn inject_svg_branch_metadata(svg: String, meta: &BranchMetadata) -> String {
    let Some(close_idx) = svg.find('>') else {
        return svg;
    };
    let escaped = meta
        .summary_tag()
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let mut out = String::with_capacity(svg.len() + escaped.len() + 64);
    out.push_str(&svg[..=close_idx]);
    out.push_str(&format!(
        "<desc id=\"oxidizemath-branch-metadata\">{escaped}</desc>"
    ));
    out.push_str(&svg[close_idx + 1..]);
    out
}

/// Embeds a standard PNG `tEXt` metadata chunk carrying `BranchMetadata` right before `IEND`.
#[must_use]
pub fn inject_png_branch_metadata(mut png_bytes: Vec<u8>, meta: &BranchMetadata) -> Vec<u8> {
    const IEND_LEN: usize = 12;
    if png_bytes.len() < 8 + IEND_LEN {
        return png_bytes;
    }
    let split_pos = png_bytes.len() - IEND_LEN;
    if &png_bytes[split_pos + 4..split_pos + 8] != b"IEND" {
        return png_bytes;
    }
    let keyword = b"OxidizeMath-Branch";
    let payload_str = meta.summary_tag();
    let payload = payload_str.as_bytes();
    let data_len = keyword.len() + 1 + payload.len();
    let Ok(len_u32) = u32::try_from(data_len) else {
        return png_bytes;
    };

    let iend = png_bytes[split_pos..].to_vec();
    png_bytes.truncate(split_pos);
    png_bytes.extend_from_slice(&len_u32.to_be_bytes());
    let crc_start = png_bytes.len();
    png_bytes.extend_from_slice(b"tEXt");
    png_bytes.extend_from_slice(keyword);
    png_bytes.push(0);
    png_bytes.extend_from_slice(payload);
    let crc = png_crc32(&png_bytes[crc_start..]);
    png_bytes.extend_from_slice(&crc.to_be_bytes());
    png_bytes.extend_from_slice(&iend);
    png_bytes
}

fn png_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
pub(crate) fn assert_framework_exports_all_branch_formats(
    tool: Box<dyn crate::framework::InteractiveTool>,
    expected_shadow_substr: &str,
) {
    use crate::framework::{ExportFormat, SimulationFramework};

    #[cfg(not(target_arch = "wasm32"))]
    let tool_name = tool.name().to_string();
    let ctx = egui::Context::default();
    let mut fw = SimulationFramework::new("__shared_branch_export_verify__");
    fw.export_options.width = 960;
    fw.export_options.height = 600;
    fw.active_tool = Some(tool);
    for fmt in [
        ExportFormat::Csv,
        ExportFormat::Json,
        ExportFormat::Svg,
        ExportFormat::Png,
    ] {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(960.0, 640.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(raw, |c| {
            fw.show_export_modal = true;
            fw.queue_export(fmt);
            fw.show(c, "shared_branch_export_verify");
        });
        match fmt {
            ExportFormat::Csv | ExportFormat::Json => {
                let body = fw.export_queue.last_exported_content.as_deref().unwrap();
                assert!(body.contains(expected_shadow_substr));
                assert!(body.contains("Branch Metadata"));
            }
            ExportFormat::Svg => {
                let svg = fw.export_queue.last_exported_content.as_deref().unwrap();
                assert!(svg.contains("oxidizemath-branch-metadata"));
            }
            ExportFormat::Png => {
                let png = fw.export_queue.last_exported_bytes.as_deref().unwrap();
                let latin1: String = png.iter().map(|&b| b as char).collect();
                assert!(latin1.contains("OxidizeMath-Branch"));
                #[cfg(not(target_arch = "wasm32"))]
                save_test_branch_png(&tool_name, png);
            }
            _ => {}
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn save_test_branch_png(tool_name: &str, png: &[u8]) {
    let filename = if tool_name.contains("SIR") {
        "/tmp/sir_timeline_shadow_branch.png"
    } else if tool_name.contains("Attractor") {
        "/tmp/attractor_shadow_twin_3d.png"
    } else if tool_name.contains("Wave") {
        "/tmp/wave_unified_diff_heatmap.png"
    } else if tool_name.contains("MRI") || tool_name.contains("Bloch") {
        "/tmp/mri_shadow_twin_branch.png"
    } else {
        return;
    };
    let _ = std::fs::write(filename, png);
}
