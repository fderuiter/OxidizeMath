//! Multi-panel subplot and custom primitive GIF stepping helpers for [`super::export_figure`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::export::{color32_images_to_gif, ExportOptions};
use egui_plot::export_primitives::{
    compute_primitive_bounds_for_options, export_subplot_panels_svg, has_drawable_primitives,
    render_primitives_image_with_bounds, render_subplot_panels_image,
    render_subplot_panels_image_with_bounds, ExportPrimitive, ExportSubplotPanel,
};

use super::InteractiveTool;

pub(super) fn try_render_subplot_color_image(
    tool: &dyn InteractiveTool,
    options: &ExportOptions,
) -> Option<Result<eframe::egui::ColorImage, String>> {
    let panels = tool.export_subplot_panels();
    if panels.len() >= 2
        && panels
            .iter()
            .any(|p| has_drawable_primitives(&p.primitives))
    {
        Some(render_subplot_panels_image(&panels, options))
    } else {
        None
    }
}

pub(super) fn try_render_subplot_svg(
    tool: &dyn InteractiveTool,
    options: &ExportOptions,
) -> Option<Result<String, String>> {
    let panels = tool.export_subplot_panels();
    if panels.len() >= 2
        && panels
            .iter()
            .any(|p| has_drawable_primitives(&p.primitives))
    {
        Some(export_subplot_panels_svg(&panels, options))
    } else {
        None
    }
}

pub(super) fn try_sample_subplot_or_custom_primitive_gif(
    tool: &mut dyn InteractiveTool,
    options: &ExportOptions,
    count: usize,
    fps: u32,
) -> Option<Result<Vec<u8>, String>> {
    let initial_panels = tool.export_subplot_panels();
    if initial_panels.len() >= 2 {
        return Some(encode_stepped_subplot_gif(
            tool,
            &initial_panels,
            options,
            count,
            fps,
        ));
    }
    if has_custom_primitive_adapter(tool) {
        return Some(encode_stepped_custom_primitive_gif(
            tool, options, count, fps,
        ));
    }
    None
}

fn encode_stepped_subplot_gif(
    tool: &mut dyn InteractiveTool,
    initial_panels: &[ExportSubplotPanel],
    options: &ExportOptions,
    count: usize,
    fps: u32,
) -> Result<Vec<u8>, String> {
    let panel_count = initial_panels.len();
    let mut panel_frames: Vec<Vec<ExportSubplotPanel>> = Vec::with_capacity(count);
    for _ in 0..count {
        tool.step_simulation();
        let mut stepped = tool.export_subplot_panels();
        preserve_screen_rects(&mut stepped, initial_panels);
        if stepped
            .iter()
            .any(|p| has_drawable_primitives(&p.primitives))
        {
            panel_frames.push(stepped);
        }
    }
    if panel_frames.is_empty()
        && initial_panels
            .iter()
            .any(|p| has_drawable_primitives(&p.primitives))
    {
        panel_frames.push(initial_panels.to_vec());
    }
    if panel_frames.is_empty() {
        return Err(format!(
            "{} produced no drawable subplot primitives for GIF export",
            tool.name()
        ));
    }
    let locked = compute_locked_subplot_bounds(&panel_frames, panel_count, options);
    let mut images = Vec::with_capacity(panel_frames.len());
    for frame in &panel_frames {
        images.push(render_subplot_panels_image_with_bounds(
            frame,
            options,
            Some(&locked),
        )?);
    }
    color32_images_to_gif(&images, fps.max(1))
}

fn preserve_screen_rects(stepped: &mut [ExportSubplotPanel], reference: &[ExportSubplotPanel]) {
    if stepped.len() != reference.len() {
        return;
    }
    for (dst, src) in stepped.iter_mut().zip(reference.iter()) {
        if src.screen_rect.width() > 1.0 && src.screen_rect.height() > 1.0 {
            dst.screen_rect = src.screen_rect;
        }
    }
}

fn compute_locked_subplot_bounds(
    frames: &[Vec<ExportSubplotPanel>],
    panel_count: usize,
    options: &ExportOptions,
) -> Vec<Option<[f64; 4]>> {
    let mut locked = vec![None; panel_count];
    for frame in frames {
        for (idx, panel) in frame.iter().enumerate().take(panel_count) {
            let panel_opts = ExportOptions {
                log_x: options.log_x || panel.log_x,
                log_y: options.log_y || panel.log_y,
                ..options.clone()
            };
            if let Some(b) = compute_primitive_bounds_for_options(&panel.primitives, &panel_opts) {
                locked[idx] = Some(merge_bounds(locked[idx], b));
            }
        }
    }
    locked
}

fn merge_bounds(existing: Option<[f64; 4]>, next: [f64; 4]) -> [f64; 4] {
    match existing {
        Some([x0, x1, y0, y1]) => [
            x0.min(next[0]),
            x1.max(next[1]),
            y0.min(next[2]),
            y1.max(next[3]),
        ],
        None => next,
    }
}

fn capture_isolated_primitives(tool: &dyn InteractiveTool) -> Vec<ExportPrimitive> {
    let prev_prims = egui_plot::get_last_plot_primitives();
    let prev_subplots = egui_plot::get_last_plot_subplots();
    egui_plot::clear_last_plot_primitives();
    let prims = tool.export_plot_primitives();
    egui_plot::clear_last_plot_primitives();
    if !prev_prims.is_empty() {
        egui_plot::append_last_plot_primitives(&prev_prims);
    }
    for subplot in prev_subplots {
        egui_plot::append_last_plot_subplot(subplot);
    }
    prims
}

fn has_custom_primitive_adapter(tool: &dyn InteractiveTool) -> bool {
    let isolated = capture_isolated_primitives(tool);
    has_drawable_primitives(&isolated)
}

fn encode_stepped_custom_primitive_gif(
    tool: &mut dyn InteractiveTool,
    options: &ExportOptions,
    count: usize,
    fps: u32,
) -> Result<Vec<u8>, String> {
    let mut prim_frames: Vec<Vec<ExportPrimitive>> = Vec::with_capacity(count);
    let initial = capture_isolated_primitives(tool);
    if has_drawable_primitives(&initial) {
        prim_frames.push(initial);
    }
    let remaining = if prim_frames.is_empty() {
        count
    } else {
        count.saturating_sub(1)
    };
    for _ in 0..remaining {
        tool.step_simulation();
        let prims = capture_isolated_primitives(tool);
        if has_drawable_primitives(&prims) {
            prim_frames.push(prims);
        }
    }
    if prim_frames.is_empty() {
        return Err(format!(
            "{} produced no drawable primitives for GIF export",
            tool.name()
        ));
    }
    let mut global_bounds: Option<[f64; 4]> = None;
    for frame in &prim_frames {
        if let Some(b) = compute_primitive_bounds_for_options(frame, options) {
            global_bounds = Some(merge_bounds(global_bounds, b));
        }
    }
    let mut images = Vec::with_capacity(prim_frames.len());
    for frame in &prim_frames {
        images.push(render_primitives_image_with_bounds(
            frame,
            options,
            global_bounds,
        )?);
    }
    color32_images_to_gif(&images, fps.max(1))
}
