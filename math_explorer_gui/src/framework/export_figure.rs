//! Cold-path offscreen figure rendering helpers for [`super::InteractiveTool`] exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::Arc;

use eframe::egui;
use egui_plot::export::{
    color32_image_to_png, compute_dataset_frames_bounds, DatasetFrame, ExportOptions,
};
use egui_plot::export_primitives::{
    compute_primitive_bounds_for_options, datasets_to_primitives, export_primitives_svg,
    export_raster_svg, has_drawable_primitives, has_finite_dataset_points, render_primitives_image,
    scale_color_image, ExportPrimitive,
};

use super::{ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework};

#[derive(Default)]
pub(super) struct CompactRasterGrid {
    width: usize,
    height: usize,
    pixels: Vec<[u8; 4]>,
}

impl CompactRasterGrid {
    pub(super) fn from_color_image(img: egui::ColorImage) -> Self {
        let width = img.width();
        let height = img.height();
        let pixels = img.pixels.iter().map(egui::Color32::to_array).collect();
        drop(img);
        Self {
            width,
            height,
            pixels,
        }
    }

    pub(super) fn into_scaled_color_image(
        self,
        target_w: usize,
        target_h: usize,
        options: &ExportOptions,
    ) -> egui::ColorImage {
        let color_pixels: Vec<egui::Color32> = self
            .pixels
            .into_iter()
            .map(|[r, g, b, a]| egui::Color32::from_rgba_premultiplied(r, g, b, a))
            .collect();
        let raw_img = egui::ColorImage::new([self.width, self.height], color_pixels);
        let scaled = scale_color_image(&raw_img, target_w, target_h, options);
        drop(raw_img);
        scaled
    }
}

pub(super) enum GifJobPayload {
    Raster {
        grids: Vec<CompactRasterGrid>,
        target_w: usize,
        target_h: usize,
    },
    Datasets {
        frames: Vec<DatasetFrame>,
        bounds: Option<[f64; 4]>,
    },
    ProgressivePrimitives {
        primitives: Vec<ExportPrimitive>,
        bounds: Option<[f64; 4]>,
        total_steps: usize,
    },
}

pub(super) fn sample_tool_gif_payload(
    tool: &mut dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
    frame_count: usize,
) -> Result<GifJobPayload, String> {
    let (target_w, target_h) = resolve_gif_canvas_dimensions(options)?;
    let count = frame_count.max(1);
    let saved_state = tool.save_state();
    let result = if let Some(first_raster) = valid_raster_image(tool) {
        Ok(sample_raster_grids(
            tool,
            first_raster,
            count,
            target_w,
            target_h,
        ))
    } else {
        sample_plot_or_primitive_payload(tool, fallback_datasets, options, count)
    };
    if let Some(state_json) = saved_state {
        tool.load_state(&state_json);
    }
    result
}

fn sample_raster_grids(
    tool: &mut dyn InteractiveTool,
    first_raster: egui::ColorImage,
    count: usize,
    target_w: usize,
    target_h: usize,
) -> GifJobPayload {
    let mut grids = Vec::with_capacity(count);
    grids.push(CompactRasterGrid::from_color_image(first_raster));
    for _ in 1..count {
        tool.step_simulation();
        if let Some(img) = valid_raster_image(tool) {
            grids.push(CompactRasterGrid::from_color_image(img));
        }
    }
    GifJobPayload::Raster {
        grids,
        target_w,
        target_h,
    }
}

fn sample_plot_or_primitive_payload(
    tool: &mut dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
    count: usize,
) -> Result<GifJobPayload, String> {
    let initial_prims = resolve_tool_primitives_with_fallback(tool, fallback_datasets, None);
    if has_non_polyline_primitives(&initial_prims) && has_drawable_primitives(&initial_prims) {
        let bounds = compute_primitive_bounds_for_options(&initial_prims, options);
        return Ok(GifJobPayload::ProgressivePrimitives {
            primitives: initial_prims,
            bounds,
            total_steps: count,
        });
    }
    let sampled = tool.sample_animation_frames(count);
    if sampled.iter().any(|frame| has_finite_dataset_points(frame)) {
        let bounds = compute_dataset_frames_bounds(&sampled, options);
        return Ok(GifJobPayload::Datasets {
            frames: sampled,
            bounds,
        });
    }
    if has_drawable_primitives(&initial_prims) {
        let bounds = compute_primitive_bounds_for_options(&initial_prims, options);
        return Ok(GifJobPayload::ProgressivePrimitives {
            primitives: initial_prims,
            bounds,
            total_steps: count,
        });
    }
    Err(format!(
        "{} produced no finite plot data or raster image for GIF export",
        tool.name()
    ))
}

pub(super) fn resolve_tool_primitives_with_fallback(
    tool: &dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    preloaded_plot_prims: Option<Vec<ExportPrimitive>>,
) -> Vec<ExportPrimitive> {
    let mut primitives = Vec::new();
    if let Some(raster) = valid_raster_image(tool) {
        let w = raster.width() as f64;
        let h = raster.height() as f64;
        primitives.push(ExportPrimitive::RasterImage {
            name: tool.name().to_string(),
            image: Arc::new(raster),
            bounds: Some([0.0, w, 0.0, h]),
        });
    }
    let plot_prims = preloaded_plot_prims.unwrap_or_else(|| tool.export_plot_primitives());
    if has_drawable_primitives(&plot_prims) {
        primitives.extend(plot_prims);
    } else if primitives.is_empty() {
        let datasets = tool.export_datasets();
        let effective = if datasets.is_empty() {
            fallback_datasets
        } else {
            datasets.as_slice()
        };
        primitives.extend(datasets_to_primitives(effective));
    }
    primitives
}

pub(super) fn valid_raster_image(tool: &dyn InteractiveTool) -> Option<egui::ColorImage> {
    tool.export_raster_image()
        .filter(|img| img.width() > 0 && img.height() > 0 && !img.pixels.is_empty())
}

pub(super) fn resolve_gif_canvas_dimensions(
    options: &ExportOptions,
) -> Result<(usize, usize), String> {
    let target_w = (options.width as f32 * options.scale).round().max(1.0) as u32;
    let mut target_h = (options.height as f32 * options.scale).round().max(1.0) as u32;
    if let Some(ratio) = options.aspect_ratio.ratio().filter(|&r| r > 0.0) {
        target_h = (target_w as f32 / ratio).round().max(1.0) as u32;
    }
    if target_w > egui_plot::MAX_EXPORT_DIMENSION || target_h > egui_plot::MAX_EXPORT_DIMENSION {
        return Err(format!(
            "Requested dimensions ({target_w}x{target_h}) exceed maximum limit"
        ));
    }
    Ok((target_w as usize, target_h as usize))
}

fn render_tool_color_image_with_fallback(
    tool: &dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<egui::ColorImage, String> {
    let mut preloaded_prims = None;
    if let Some(raster) = valid_raster_image(tool) {
        let plot_prims = tool.export_plot_primitives();
        if !has_drawable_primitives(&plot_prims) {
            let (target_w, target_h) = resolve_gif_canvas_dimensions(options)?;
            return Ok(scale_color_image(&raster, target_w, target_h, options));
        }
        preloaded_prims = Some(plot_prims);
    }
    let primitives =
        resolve_tool_primitives_with_fallback(tool, fallback_datasets, preloaded_prims);
    if !has_drawable_primitives(&primitives) {
        return Err(format!(
            "{} produced no finite plot primitives or raster image to export",
            tool.name()
        ));
    }
    render_primitives_image(&primitives, options)
}

fn render_tool_png_with_fallback(
    tool: &dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let image = render_tool_color_image_with_fallback(tool, fallback_datasets, options)?;
    color32_image_to_png(&image)
}

fn render_tool_svg_with_fallback(
    tool: &dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
) -> Result<String, String> {
    let mut preloaded_prims = None;
    if let Some(raster) = valid_raster_image(tool) {
        let plot_prims = tool.export_plot_primitives();
        if !has_drawable_primitives(&plot_prims) {
            return export_raster_svg(&raster, options);
        }
        preloaded_prims = Some(plot_prims);
    }
    let primitives =
        resolve_tool_primitives_with_fallback(tool, fallback_datasets, preloaded_prims);
    if !has_drawable_primitives(&primitives) {
        return Err(format!(
            "{} produced no finite plot primitives or raster image to export",
            tool.name()
        ));
    }
    export_primitives_svg(&primitives, options)
}

pub(super) fn has_non_polyline_primitives(primitives: &[ExportPrimitive]) -> bool {
    primitives.iter().any(|p| {
        matches!(
            p,
            ExportPrimitive::Bar { .. }
                | ExportPrimitive::Arrow { .. }
                | ExportPrimitive::BoxElem { .. }
                | ExportPrimitive::RasterImage { .. }
        )
    })
}

pub(super) fn render_progressive_primitive_slice(
    primitives: &[ExportPrimitive],
    bounds: Option<[f64; 4]>,
    step: usize,
    total_steps: usize,
    options: &ExportOptions,
) -> Result<egui::ColorImage, String> {
    let total_prims = primitives.len().max(1);
    let take_n = ((total_prims * step) / total_steps.max(1))
        .max(1)
        .min(primitives.len());
    let slice = &primitives[..take_n];
    if !has_drawable_primitives(slice) {
        return Ok(egui::ColorImage::new([0, 0], Vec::new()));
    }
    egui_plot::export_primitives::render_primitives_image_with_bounds(slice, options, bounds)
}

#[cfg(test)]
pub(super) fn encode_progressive_primitive_gif(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
    count: usize,
    fps: u32,
) -> Result<Vec<u8>, String> {
    let total_steps = count.max(1);
    let bounds = compute_primitive_bounds_for_options(primitives, options);
    let mut encoder = egui_plot::export::StreamingGifEncoder::new(fps)?;
    for step in 1..=total_steps {
        let img =
            render_progressive_primitive_slice(primitives, bounds, step, total_steps, options)?;
        encoder.encode_color_image(img)?;
    }
    encoder.finish()
}

fn copy_tool_image_to_clipboard_scoped(
    tool: &dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
    ctx: Option<&egui::Context>,
    scope_id: egui::Id,
) -> Result<&'static str, String> {
    let color_img = render_tool_color_image_with_fallback(tool, fallback_datasets, options)?;
    #[cfg(test)]
    {
        let _ = (&color_img, ctx, scope_id);
    }
    #[cfg(not(test))]
    {
        if let Some(egui_ctx) = ctx.cloned() {
            let status_id = egui_plot::clipboard_status_id(scope_id);
            egui_plot::export::copy_image_to_clipboard_with_callback(&color_img, move |res| {
                egui_ctx.data_mut(|d| {
                    let msg = match res {
                        Ok(()) => "Copied plot image to clipboard!".to_string(),
                        Err(e) => format!("Clipboard copy failed: {e}"),
                    };
                    d.insert_temp(status_id, msg);
                });
                egui_ctx.request_repaint();
            })?;
        } else {
            egui_plot::export::copy_image_to_clipboard(&color_img)?;
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok("Copying plot image to clipboard...")
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok("Copied plot image to clipboard!")
    }
}

impl SimulationFramework {
    pub(super) fn ensure_figure_exportable(&mut self) -> Option<String> {
        let tool = self.active_tool.as_ref()?;
        let caps = tool.export_capabilities();
        if !caps.figure {
            self.set_export_status(ExportStatusSeverity::Warning, caps.figure_disabled_reason);
            return None;
        }
        Some(tool.name().to_string())
    }

    pub(super) fn dispatch_png_export(&mut self) {
        let Some(tool_name) = self.ensure_figure_exportable() else {
            return;
        };
        let filename = Self::build_export_filename(&tool_name, ExportFormat::Png);
        let fallback = self.export_datasets_with_fallback();
        let tool = self
            .active_tool
            .as_ref()
            .expect("verified active tool")
            .as_ref();
        match render_tool_png_with_fallback(tool, &fallback, &self.export_options) {
            Ok(bytes) => self.complete_file_export(
                ExportFormat::Png,
                &filename,
                &bytes,
                None,
                "PNG image",
                "Export failed",
            ),
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format!("Export failed: {err}"))
            }
        }
    }

    pub(super) fn dispatch_png_clipboard_export(&mut self, ctx: Option<&egui::Context>) {
        if self.ensure_figure_exportable().is_none() {
            return;
        };
        let fallback = self.export_datasets_with_fallback();
        let scope_id = self
            .sync_state
            .last_modal_scope_id
            .unwrap_or_else(|| self.export_modal_scope_id(&self.domain));
        let tool = self
            .active_tool
            .as_ref()
            .expect("verified active tool")
            .as_ref();
        match copy_tool_image_to_clipboard_scoped(
            tool,
            &fallback,
            &self.export_options,
            ctx,
            scope_id,
        ) {
            Ok(msg) => self.set_export_status(ExportStatusSeverity::Success, msg),
            Err(err) => self.set_export_status(
                ExportStatusSeverity::Error,
                format!("Clipboard copy failed: {err}"),
            ),
        }
    }

    pub(super) fn dispatch_svg_export(&mut self) {
        let Some(tool_name) = self.ensure_figure_exportable() else {
            return;
        };
        let filename = Self::build_export_filename(&tool_name, ExportFormat::Svg);
        let fallback = self.export_datasets_with_fallback();
        let tool = self
            .active_tool
            .as_ref()
            .expect("verified active tool")
            .as_ref();
        match render_tool_svg_with_fallback(tool, &fallback, &self.export_options) {
            Ok(svg) => {
                let bytes = svg.as_bytes().to_vec();
                self.complete_file_export(
                    ExportFormat::Svg,
                    &filename,
                    &bytes,
                    Some(svg),
                    "SVG image",
                    "Export failed",
                );
            }
            Err(err) => {
                self.set_export_status(ExportStatusSeverity::Error, format!("Export failed: {err}"))
            }
        }
    }
}
