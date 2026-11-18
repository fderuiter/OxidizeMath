//! Cold-path offscreen figure rendering helpers for [`super::InteractiveTool`] exports.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::Arc;

use eframe::egui;
use egui_plot::export::{
    color32_image_to_png, color32_images_to_gif, export_custom_gif, ExportOptions,
};
use egui_plot::export_primitives::{
    compute_primitive_bounds, datasets_to_primitives, export_primitives_svg, export_raster_gif,
    export_raster_svg, has_drawable_primitives, has_finite_dataset_points, render_primitives_image,
    render_primitives_image_with_bounds, scale_color_image, ExportPrimitive,
};

use super::InteractiveTool;

/// Collects drawable [`ExportPrimitive`] items for `tool`, combining raster hooks, captured plot primitives, and fallback datasets.
pub fn resolve_tool_primitives(tool: &dyn InteractiveTool) -> Vec<ExportPrimitive> {
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
    let plot_prims = tool.export_plot_primitives();
    if has_drawable_primitives(&plot_prims) {
        primitives.extend(plot_prims);
    } else if primitives.is_empty() {
        let datasets = tool.export_datasets();
        primitives.extend(datasets_to_primitives(&datasets));
    }
    primitives
}

fn valid_raster_image(tool: &dyn InteractiveTool) -> Option<egui::ColorImage> {
    tool.export_raster_image()
        .filter(|img| img.width() > 0 && img.height() > 0 && !img.pixels.is_empty())
}

/// Renders the current frame of `tool` into an offscreen [`egui::ColorImage`], rejecting empty payloads.
pub fn render_tool_color_image(
    tool: &dyn InteractiveTool,
    options: &ExportOptions,
) -> Result<egui::ColorImage, String> {
    let plot_prims = tool.export_plot_primitives();
    if let Some(raster) = valid_raster_image(tool) {
        if !has_drawable_primitives(&plot_prims) {
            let target_w = (options.width as f32 * options.scale).round().max(1.0) as u32;
            let mut target_h = (options.height as f32 * options.scale).round().max(1.0) as u32;
            if let Some(ratio) = options.aspect_ratio.ratio().filter(|&r| r > 0.0) {
                target_h = (target_w as f32 / ratio).round().max(1.0) as u32;
            }
            if target_w > egui_plot::MAX_EXPORT_DIMENSION
                || target_h > egui_plot::MAX_EXPORT_DIMENSION
            {
                return Err(format!(
                    "Requested dimensions ({target_w}x{target_h}) exceed maximum limit"
                ));
            }
            return Ok(scale_color_image(
                &raster,
                target_w as usize,
                target_h as usize,
                options,
            ));
        }
    }
    let primitives = resolve_tool_primitives(tool);
    if !has_drawable_primitives(&primitives) {
        return Err(format!(
            "{} produced no finite plot primitives or raster image to export",
            tool.name()
        ));
    }
    render_primitives_image(&primitives, options)
}

/// Renders the current frame of `tool` into high-resolution PNG bytes.
pub fn render_tool_png(
    tool: &dyn InteractiveTool,
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let image = render_tool_color_image(tool, options)?;
    color32_image_to_png(&image)
}

/// Renders the current frame of `tool` into a vector (or embedded-raster) SVG string.
pub fn render_tool_svg(
    tool: &dyn InteractiveTool,
    options: &ExportOptions,
) -> Result<String, String> {
    let plot_prims = tool.export_plot_primitives();
    if let Some(raster) = valid_raster_image(tool) {
        if !has_drawable_primitives(&plot_prims) {
            return export_raster_svg(&raster, options);
        }
    }
    let primitives = resolve_tool_primitives(tool);
    if !has_drawable_primitives(&primitives) {
        return Err(format!(
            "{} produced no finite plot primitives or raster image to export",
            tool.name()
        ));
    }
    export_primitives_svg(&primitives, options)
}

/// Samples and encodes a multi-frame GIF animation for `tool`, rejecting empty payloads and restoring saved tool state.
pub fn render_tool_gif(
    tool: &mut dyn InteractiveTool,
    options: &ExportOptions,
    frame_count: usize,
    fps: u32,
) -> Result<Vec<u8>, String> {
    let count = frame_count.max(1);
    let saved_state = tool.save_state();

    let result = if let Some(first_raster) = valid_raster_image(tool) {
        sample_and_encode_raster_gif(tool, first_raster, options, count, fps)
    } else {
        sample_and_encode_plot_gif(tool, options, count, fps)
    };

    if let Some(state_json) = saved_state {
        tool.load_state(&state_json);
    }
    result
}

fn sample_and_encode_raster_gif(
    tool: &mut dyn InteractiveTool,
    first_raster: egui::ColorImage,
    options: &ExportOptions,
    count: usize,
    fps: u32,
) -> Result<Vec<u8>, String> {
    let mut frames = Vec::with_capacity(count);
    frames.push(first_raster);
    for _ in 1..count {
        tool.step_simulation();
        if let Some(img) = valid_raster_image(tool) {
            frames.push(img);
        }
    }
    export_raster_gif(&frames, options, fps)
}

fn sample_and_encode_plot_gif(
    tool: &mut dyn InteractiveTool,
    options: &ExportOptions,
    count: usize,
    fps: u32,
) -> Result<Vec<u8>, String> {
    let initial_prims = resolve_tool_primitives(tool);
    if has_non_polyline_primitives(&initial_prims) {
        return encode_progressive_primitive_gif(&initial_prims, options, count, fps);
    }
    let sampled = tool.sample_animation_frames(count);
    if sampled.iter().any(|frame| has_finite_dataset_points(frame)) {
        return export_custom_gif(&sampled, options, fps);
    }
    if has_drawable_primitives(&initial_prims) {
        return encode_progressive_primitive_gif(&initial_prims, options, count, fps);
    }
    Err(format!(
        "{} produced no finite plot data or raster image for GIF export",
        tool.name()
    ))
}

fn has_non_polyline_primitives(primitives: &[ExportPrimitive]) -> bool {
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

fn encode_progressive_primitive_gif(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
    count: usize,
    fps: u32,
) -> Result<Vec<u8>, String> {
    let bounds = compute_primitive_bounds(primitives);
    let total = primitives.len().max(1);
    let mut images = Vec::with_capacity(count);
    for step in 1..=count {
        let take_n = ((total * step) / count).max(1).min(primitives.len());
        let slice = &primitives[..take_n];
        if has_drawable_primitives(slice) {
            images.push(render_primitives_image_with_bounds(slice, options, bounds)?);
        }
    }
    color32_images_to_gif(&images, fps)
}
