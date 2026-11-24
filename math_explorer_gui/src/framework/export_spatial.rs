//! Cold-path Marching Squares contour SVG and 16-bit texture export dispatchers.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::export::ExportOptions;
use egui_plot::export_primitives::{
    encode_heightmap_16bit_png, encode_normal_map_png, export_contour_layers_svg,
    sanitize_texture_size, SvgContourLayer, SvgContourPath, MAX_SVG_CONTOUR_LEVELS,
    MIN_SVG_CONTOUR_LEVELS,
};
use math_commons::math_kernel::colormap::{heatmap_color_high_contrast, mapped_gradient};
use math_commons::math_kernel::marching_squares::{MarchingSquares, ScalarField2D};

use super::export_figure::render_tool_svg_with_fallback;
use super::{ExportFormat, ExportStatusSeverity, SimulationFramework};

fn contour_stroke_color(idx: usize, total: usize, bg: egui::Color32) -> egui::Color32 {
    let frac = if total <= 1 {
        0.55
    } else {
        idx as f64 / (total - 1) as f64
    };
    let bg_sum = u16::from(bg.r()) + u16::from(bg.g()) + u16::from(bg.b());
    let [r, g, b] = if bg_sum < 384 {
        heatmap_color_high_contrast(frac)
    } else {
        mapped_gradient(frac, 0.0, 1.0)
    };
    egui::Color32::from_rgb(r, g, b)
}

pub(super) fn render_scalar_field_contours_svg(
    field: &ScalarField2D,
    options: &ExportOptions,
    contour_levels: usize,
) -> Result<String, String> {
    if field.width < 2 || field.height < 2 || field.values.len() != field.width * field.height {
        return Err("Invalid 2D scalar field dimensions for contour SVG export".to_string());
    }
    let count = contour_levels.clamp(MIN_SVG_CONTOUR_LEVELS, MAX_SVG_CONTOUR_LEVELS);
    let extracted = MarchingSquares::extract_levels(field, count);
    let total = extracted.len().max(1);
    let mut layers: Vec<SvgContourLayer> = extracted
        .into_iter()
        .enumerate()
        .filter(|(_, lvl)| !lvl.paths.is_empty())
        .map(|(idx, lvl)| SvgContourLayer {
            label: format!("iso-{idx}-{:.4}", lvl.threshold),
            color: contour_stroke_color(idx, total, options.background_color),
            stroke_width: 1.5,
            fill: None,
            paths: lvl
                .paths
                .into_iter()
                .map(|poly| SvgContourPath {
                    points: poly.points,
                    closed: poly.closed,
                })
                .collect(),
        })
        .collect();

    let (domain_w, domain_h) = field.domain_extents();

    if layers.is_empty() {
        layers.push(SvgContourLayer {
            label: "uniform-field-boundary".to_string(),
            color: contour_stroke_color(0, 1, options.background_color),
            stroke_width: 1.5,
            fill: None,
            paths: vec![SvgContourPath {
                points: vec![
                    [0.0, 0.0],
                    [domain_w, 0.0],
                    [domain_w, domain_h],
                    [0.0, domain_h],
                ],
                closed: true,
            }],
        });
    }

    export_contour_layers_svg(domain_w, domain_h, &layers, options)
}

impl SimulationFramework {
    fn ensure_scalar_field_exportable(&mut self) -> Option<String> {
        let tool = self.active_tool.as_ref()?;
        let caps = tool.export_capabilities();
        if !caps.scalar_field {
            self.set_export_status(
                ExportStatusSeverity::Warning,
                "16-bit texture export is unavailable: this tool does not expose a 2D scalar field.",
            );
            return None;
        }
        Some(tool.name().to_string())
    }

    pub(super) fn dispatch_svg_clipboard_export(&mut self, ctx: Option<&egui::Context>) {
        if self.ensure_figure_exportable().is_none() {
            return;
        }
        let fallback = self.export_datasets_with_fallback();
        let contour_levels = self.export_contour_levels;
        let svg_res = {
            let tool = self
                .active_tool
                .as_ref()
                .expect("verified active tool")
                .as_ref();
            render_tool_svg_with_fallback(tool, &fallback, &self.export_options, contour_levels)
        };
        match svg_res {
            Ok(svg) => self.dispatch_clipboard_text_request(ExportFormat::Svg, svg, ctx),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Clipboard SVG export failed: {err}"),
            ),
        }
    }

    pub(super) fn dispatch_heightmap16_export(&mut self) {
        let Some(tool_name) = self.ensure_scalar_field_exportable() else {
            return;
        };
        let size = self.export_texture_size;
        let dst = match sanitize_texture_size(size) {
            Ok(d) => d,
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, err);
                return;
            }
        };
        let filename = Self::build_export_filename(&tool_name, ExportFormat::Heightmap16Png);
        let field_opt = self
            .active_tool
            .as_ref()
            .and_then(|t| t.export_scalar_field_at(dst, dst));
        let Some(field) = field_opt else {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("No 2D scalar field available from {tool_name}."),
            );
            return;
        };
        match encode_heightmap_16bit_png(
            &field.values,
            field.width,
            field.height,
            size,
            field.periodic,
        ) {
            Ok(bytes) => self.complete_file_export(
                ExportFormat::Heightmap16Png,
                &filename,
                &bytes,
                None,
                "16-bit heightmap PNG",
                "Heightmap export failed",
            ),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Heightmap export failed: {err}"),
            ),
        }
    }

    pub(super) fn dispatch_normal_map_export(&mut self) {
        let Some(tool_name) = self.ensure_scalar_field_exportable() else {
            return;
        };
        let size = self.export_texture_size;
        let strength = self.export_normal_strength;
        let dst = match sanitize_texture_size(size) {
            Ok(d) => d,
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, err);
                return;
            }
        };
        let filename = Self::build_export_filename(&tool_name, ExportFormat::NormalMapPng);
        let field_opt = self
            .active_tool
            .as_ref()
            .and_then(|t| t.export_scalar_field_at(dst, dst));
        let Some(field) = field_opt else {
            self.set_export_status(
                ExportStatusSeverity::Error,
                format!("No 2D scalar field available from {tool_name}."),
            );
            return;
        };
        match encode_normal_map_png(
            &field.values,
            field.width,
            field.height,
            size,
            field.periodic,
            strength,
        ) {
            Ok(bytes) => self.complete_file_export(
                ExportFormat::NormalMapPng,
                &filename,
                &bytes,
                None,
                "Normal-map PNG",
                "Normal-map export failed",
            ),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Normal-map export failed: {err}"),
            ),
        }
    }
}
