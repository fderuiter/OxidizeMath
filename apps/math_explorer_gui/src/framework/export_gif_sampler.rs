//! GIF frame sampling helpers for raster, dataset, progressive-primitive, and stepped-primitive tools.

use eframe::egui;
use egui_plot::export::{compute_dataset_frames_bounds, DatasetFrame, ExportOptions};
use egui_plot::export_primitives::{
    compute_primitive_bounds_for_options, has_drawable_primitives, has_finite_dataset_points,
    scale_color_image, ExportPrimitive,
};

use super::{
    has_non_polyline_primitives, resolve_gif_canvas_dimensions,
    resolve_tool_primitives_with_fallback, valid_raster_image,
};
use crate::framework::InteractiveTool;

#[derive(Default)]
pub(in crate::framework) struct CompactRasterGrid {
    width: usize,
    height: usize,
    pixels: Vec<[u8; 4]>,
}

impl CompactRasterGrid {
    pub(in crate::framework) fn from_color_image(img: egui::ColorImage) -> Self {
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

    pub(in crate::framework) fn into_scaled_color_image(
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

pub(in crate::framework) enum GifJobPayload {
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
    PrimitiveFrames {
        frames: Vec<Vec<ExportPrimitive>>,
        bounds: Option<[f64; 4]>,
    },
}

pub(in crate::framework) fn sample_tool_gif_payload(
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

fn sample_stepped_primitive_frames(
    tool: &mut dyn InteractiveTool,
    initial_prims: Vec<ExportPrimitive>,
    options: &ExportOptions,
    count: usize,
) -> Option<GifJobPayload> {
    if count <= 1 {
        return None;
    }
    let rev0 = tool.state_revision();
    let mut frames = Vec::with_capacity(count);
    frames.push(initial_prims);
    let mut mutated = false;
    for _ in 1..count {
        tool.step_simulation();
        mutated |= tool.state_revision() != rev0;
        let next_prims = tool.export_plot_primitives();
        if has_drawable_primitives(&next_prims) {
            frames.push(next_prims);
        }
    }
    if !mutated || frames.len() < 2 {
        return None;
    }
    let flat: Vec<ExportPrimitive> = frames.iter().flatten().cloned().collect();
    let bounds = compute_primitive_bounds_for_options(&flat, options);
    Some(GifJobPayload::PrimitiveFrames { frames, bounds })
}

fn sample_plot_or_primitive_payload(
    tool: &mut dyn InteractiveTool,
    fallback_datasets: &[(String, Vec<[f64; 2]>)],
    options: &ExportOptions,
    count: usize,
) -> Result<GifJobPayload, String> {
    let initial_prims = resolve_tool_primitives_with_fallback(tool, fallback_datasets, None);
    if has_non_polyline_primitives(&initial_prims) && has_drawable_primitives(&initial_prims) {
        if let Some(stepped) =
            sample_stepped_primitive_frames(tool, initial_prims.clone(), options, count)
        {
            return Ok(stepped);
        }
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

pub(in crate::framework) fn render_progressive_primitive_slice(
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
pub(in crate::framework) fn encode_progressive_primitive_gif(
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

pub(in crate::framework) fn sample_tool_animation_frames<T: InteractiveTool + ?Sized>(
    tool: &mut T,
    num_frames: usize,
) -> Vec<egui_plot::export::DatasetFrame> {
    let count = num_frames.max(1);
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        tool.step_simulation();
        let ds = tool.export_datasets();
        if !ds.is_empty() {
            frames.push(ds);
        }
    }
    if frames.is_empty() {
        let ds = tool.export_datasets();
        if !ds.is_empty() {
            frames = egui_plot::export::sample_progressive_frames(&ds, count);
        }
    }
    frames
}
