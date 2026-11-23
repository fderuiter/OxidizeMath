//! Offscreen raster (`egui::ColorImage`) renderer for [`super::ExportPrimitive`] items.

use crate::export::{ExportOptions, color32_image_to_png};

use super::pixel_ops::{blend_over, draw_dot, draw_segment, fill_triangle};
use super::{
    ExportPrimitive, PrimitiveCanvasLayout, compute_primitive_bounds_for_options,
    has_drawable_primitives, resolve_canvas_dimensions, resolve_color, sanitize_bar_corners,
};

/// Renders `primitives` into an offscreen RGBA PNG byte buffer using automatic bounds.
pub fn export_primitives_png(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
) -> Result<Vec<u8>, String> {
    let image = render_primitives_image(primitives, options)?;
    color32_image_to_png(&image)
}

/// Decodes an 8-bit RGBA PNG byte buffer into `(width, height, pixels)`.
pub fn decode_png_rgba(png_bytes: &[u8]) -> Result<(usize, usize, Vec<egui::Color32>), String> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes))
        .read_info()
        .map_err(|e| format!("Failed to parse PNG header: {e}"))?;
    let buf_len = reader
        .output_buffer_size()
        .ok_or_else(|| "PNG buffer size overflow".to_string())?;
    let mut buf = vec![0_u8; buf_len];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode PNG frame: {e}"))?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "Expected 8-bit RGBA PNG, got {:?} {:?}",
            info.color_type, info.bit_depth
        ));
    }
    let width = info.width as usize;
    let height = info.height as usize;
    let pixels = buf[..info.buffer_size()]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|px| egui::Color32::from_rgba_unmultiplied(px[0], px[1], px[2], px[3]))
        .collect();
    Ok((width, height, pixels))
}

/// Renders `primitives` into an offscreen `egui::ColorImage` using automatic bounds.
pub fn render_primitives_image(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
) -> Result<egui::ColorImage, String> {
    render_primitives_image_with_bounds(primitives, options, None)
}

/// Renders `primitives` into an offscreen `egui::ColorImage` with optional explicit `[min_x, max_x, min_y, max_y]` bounds.
pub fn render_primitives_image_with_bounds(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
    custom_bounds: Option<[f64; 4]>,
) -> Result<egui::ColorImage, String> {
    if !has_drawable_primitives(primitives) {
        return Err("No drawable plot primitives available for image export".to_string());
    }
    let (width, height) = resolve_canvas_dimensions(options)?;
    let bounds = custom_bounds
        .or_else(|| compute_primitive_bounds_for_options(primitives, options))
        .ok_or_else(|| "Unable to compute finite plot bounds for image export".to_string())?;
    let tf = PrimitiveCanvasLayout::new(width, height, bounds, options);
    let bg = background_pixel(options);
    let mut pixels = vec![bg; width * height];

    for (idx, primitive) in primitives.iter().enumerate() {
        draw_primitive(&mut pixels, &tf, primitive, idx);
    }

    Ok(egui::ColorImage::new([width, height], pixels))
}

/// Scales `source` onto a target `width x height` canvas preserving aspect ratio over `options` background.
pub fn scale_color_image(
    source: &egui::ColorImage,
    width: usize,
    height: usize,
    options: &ExportOptions,
) -> egui::ColorImage {
    let bg = background_pixel(options);
    let mut pixels = vec![bg; width * height];
    let src_w = source.width().max(1);
    let src_h = source.height().max(1);
    let padding = (16.0 * options.scale as f64).clamp(0.0, (width.min(height) as f64) * 0.1);
    let avail_w = (width as f64 - 2.0 * padding).max(1.0);
    let avail_h = (height as f64 - 2.0 * padding).max(1.0);
    let fit_scale = (avail_w / src_w as f64).min(avail_h / src_h as f64);
    let dst_w = (src_w as f64 * fit_scale).round().max(1.0);
    let dst_h = (src_h as f64 * fit_scale).round().max(1.0);
    let x0 = ((width as f64 - dst_w) * 0.5).floor();
    let y0 = ((height as f64 - dst_h) * 0.5).floor();
    blit_image(
        &mut pixels,
        width,
        height,
        source,
        (x0, y0, x0 + dst_w, y0 + dst_h),
    );
    egui::ColorImage::new([width, height], pixels)
}

pub(crate) fn background_pixel(options: &ExportOptions) -> egui::Color32 {
    let bg_a = (options.background_color.a() as f32 * options.background_opacity.clamp(0.0, 1.0))
        .round() as u8;
    egui::Color32::from_rgba_unmultiplied(
        options.background_color.r(),
        options.background_color.g(),
        options.background_color.b(),
        bg_a,
    )
}

fn draw_primitive(
    pixels: &mut [egui::Color32],
    tf: &PrimitiveCanvasLayout,
    primitive: &ExportPrimitive,
    idx: usize,
) {
    match primitive {
        ExportPrimitive::RasterImage { image, bounds, .. } => {
            draw_raster_primitive(pixels, tf, image, *bounds);
        }
        ExportPrimitive::Bar {
            min,
            max,
            fill,
            stroke_color,
            stroke_width,
            ..
        } => {
            let Some((c0, c1)) = sanitize_bar_corners(*min, *max, tf.log_x, tf.log_y) else {
                return;
            };
            let stroke = resolve_color(*stroke_color, idx);
            let fill_c = if *fill == egui::Color32::TRANSPARENT {
                stroke.linear_multiply(0.4)
            } else {
                *fill
            };
            let (p0, p1) = (tf.to_pixel(c0), tf.to_pixel(c1));
            let sw = ((*stroke_width as f64) * tf.scale).max(1.0);
            draw_rect(pixels, tf.width, tf.height, p0, p1, fill_c, stroke, sw);
        }
        ExportPrimitive::BoxElem { .. } => draw_box_elem(pixels, tf, primitive, idx),
        ExportPrimitive::Arrow {
            origin,
            tip,
            tip_length,
            color,
            ..
        } => {
            let stroke = resolve_color(*color, idx);
            draw_arrow(pixels, tf, *origin, *tip, *tip_length, stroke);
        }
        ExportPrimitive::Polyline {
            points,
            color,
            stroke_width,
            closed,
            fill,
            ..
        } => {
            let stroke = resolve_color(*color, idx);
            let radius = ((*stroke_width as f64) * tf.scale * 0.5).max(tf.base_stroke * 0.5);
            draw_polyline(pixels, tf, points, stroke, radius, *closed, *fill);
        }
        ExportPrimitive::Points {
            points,
            color,
            radius,
            ..
        } => {
            let fill = resolve_color(*color, idx);
            let r = ((*radius as f64) * tf.scale).max(1.5);
            for &pt in points {
                if tf.is_valid_point(pt) {
                    draw_dot(pixels, tf.width, tf.height, tf.to_pixel(pt), r, fill);
                }
            }
        }
    }
}

fn draw_raster_primitive(
    pixels: &mut [egui::Color32],
    tf: &PrimitiveCanvasLayout,
    image: &egui::ColorImage,
    bounds: Option<[f64; 4]>,
) {
    let rect = if let Some([x0, x1, y0, y1]) =
        bounds.filter(|b| !tf.log_x && !tf.log_y || (b[0] > 0.0 && b[2] > 0.0))
    {
        let (px0, py0) = tf.to_pixel([x0, y0]);
        let (px1, py1) = tf.to_pixel([x1, y1]);
        (px0.min(px1), py0.min(py1), px0.max(px1), py0.max(py1))
    } else {
        (
            tf.padding,
            tf.padding,
            tf.padding + tf.draw_w,
            tf.padding + tf.draw_h,
        )
    };
    blit_image(pixels, tf.width, tf.height, image, rect);
}

fn blit_image(
    pixels: &mut [egui::Color32],
    w: usize,
    h: usize,
    src: &egui::ColorImage,
    rect: (f64, f64, f64, f64),
) {
    let (rx0, ry0, rx1, ry1) = rect;
    let dst_w = (rx1 - rx0).max(1.0);
    let dst_h = (ry1 - ry0).max(1.0);
    let x_start = (rx0.floor() as isize).clamp(0, w as isize - 1) as usize;
    let x_end = (rx1.ceil() as isize).clamp(0, w as isize) as usize;
    let y_start = (ry0.floor() as isize).clamp(0, h as isize - 1) as usize;
    let y_end = (ry1.ceil() as isize).clamp(0, h as isize) as usize;
    let (sw, sh) = (src.width(), src.height());
    if sw == 0 || sh == 0 {
        return;
    }
    for y in y_start..y_end {
        let v = (((y as f64 + 0.5) - ry0) / dst_h).clamp(0.0, 0.999_999);
        let sy = (v * sh as f64) as usize;
        for x in x_start..x_end {
            let u = (((x as f64 + 0.5) - rx0) / dst_w).clamp(0.0, 0.999_999);
            let sx = (u * sw as f64) as usize;
            let color = src.pixels[sy * sw + sx];
            pixels[y * w + x] = blend_over(pixels[y * w + x], color, 1.0);
        }
    }
}

fn draw_box_elem(
    pixels: &mut [egui::Color32],
    tf: &PrimitiveCanvasLayout,
    primitive: &ExportPrimitive,
    idx: usize,
) {
    let ExportPrimitive::BoxElem {
        vertical,
        argument,
        lower_whisker,
        quartile1,
        median,
        quartile3,
        upper_whisker,
        box_width,
        whisker_width,
        fill,
        stroke_color,
        stroke_width,
        ..
    } = primitive
    else {
        return;
    };
    let pt_for = |arg: f64, val: f64| if *vertical { [arg, val] } else { [val, arg] };
    let (hb, hw) = (box_width * 0.5, whisker_width * 0.5);
    if !tf.is_valid_point(pt_for(argument - hb.max(hw), *lower_whisker))
        || !tf.is_valid_point(pt_for(argument + hb.max(hw), *upper_whisker))
    {
        return;
    }
    let stroke = resolve_color(*stroke_color, idx);
    let fill_c = if *fill == egui::Color32::TRANSPARENT {
        stroke.linear_multiply(0.35)
    } else {
        *fill
    };
    let sw = ((*stroke_width as f64) * tf.scale).max(1.0);
    let r = (sw * 0.5).max(0.75);
    let map = |arg: f64, val: f64| tf.to_pixel(pt_for(arg, val));
    let (b0, b1) = (
        map(argument - hb, *quartile1),
        map(argument + hb, *quartile3),
    );
    draw_rect(pixels, tf.width, tf.height, b0, b1, fill_c, stroke, sw);
    for (p0, p1) in [
        (map(argument - hb, *median), map(argument + hb, *median)),
        (map(*argument, *lower_whisker), map(*argument, *quartile1)),
        (map(*argument, *quartile3), map(*argument, *upper_whisker)),
        (
            map(argument - hw, *lower_whisker),
            map(argument + hw, *lower_whisker),
        ),
        (
            map(argument - hw, *upper_whisker),
            map(argument + hw, *upper_whisker),
        ),
    ] {
        draw_segment(pixels, tf.width, tf.height, p0, p1, r, stroke);
    }
}

fn draw_arrow(
    pixels: &mut [egui::Color32],
    tf: &PrimitiveCanvasLayout,
    origin: [f64; 2],
    tip: [f64; 2],
    tip_length: Option<f32>,
    color: egui::Color32,
) {
    if !tf.is_valid_point(origin) || !tf.is_valid_point(tip) {
        return;
    }
    let p0 = tf.to_pixel(origin);
    let p1 = tf.to_pixel(tip);
    let dx = p1.0 - p0.0;
    let dy = p1.1 - p0.1;
    let len = (dx * dx + dy * dy).sqrt();
    let r = (tf.base_stroke * 0.5).max(0.75);
    if len < 1e-4 {
        draw_dot(pixels, tf.width, tf.height, p0, r * 1.5, color);
        return;
    }
    draw_segment(pixels, tf.width, tf.height, p0, p1, r, color);
    let head_len = tip_length
        .map(|l| (l as f64) * tf.scale)
        .unwrap_or_else(|| (len * 0.25).clamp(4.0, 18.0 * tf.scale));
    let (ux, uy) = (dx / len, dy / len);
    let angle = std::f64::consts::TAU / 10.0;
    let (ca, sa) = (angle.cos(), angle.sin());
    let w1 = (
        p1.0 - head_len * (ux * ca + uy * sa),
        p1.1 - head_len * (-ux * sa + uy * ca),
    );
    let w2 = (
        p1.0 - head_len * (ux * ca - uy * sa),
        p1.1 - head_len * (ux * sa + uy * ca),
    );
    fill_triangle(pixels, tf.width, tf.height, p1, w1, w2, color);
    draw_segment(pixels, tf.width, tf.height, p1, w1, r, color);
    draw_segment(pixels, tf.width, tf.height, p1, w2, r, color);
}

fn draw_polyline(
    pixels: &mut [egui::Color32],
    tf: &PrimitiveCanvasLayout,
    points: &[[f64; 2]],
    stroke: egui::Color32,
    radius: f64,
    closed: bool,
    fill: Option<egui::Color32>,
) {
    let screen_pts: Vec<(f64, f64)> = points
        .iter()
        .filter(|&&p| tf.is_valid_point(p))
        .map(|&p| tf.to_pixel(p))
        .collect();
    if closed && screen_pts.len() >= 3 {
        let fill_c = fill.unwrap_or_else(|| stroke.linear_multiply(0.2));
        for i in 1..(screen_pts.len() - 1) {
            fill_triangle(
                pixels,
                tf.width,
                tf.height,
                screen_pts[0],
                screen_pts[i],
                screen_pts[i + 1],
                fill_c,
            );
        }
    }
    let mut prev: Option<(f64, f64)> = None;
    for &p in points {
        if !tf.is_valid_point(p) {
            prev = None;
            continue;
        }
        let curr = tf.to_pixel(p);
        if let Some(p0) = prev {
            draw_segment(pixels, tf.width, tf.height, p0, curr, radius, stroke);
        } else {
            draw_dot(pixels, tf.width, tf.height, curr, radius, stroke);
        }
        prev = Some(curr);
    }
    if closed && screen_pts.len() >= 2 {
        let last = screen_pts[screen_pts.len() - 1];
        draw_segment(
            pixels,
            tf.width,
            tf.height,
            last,
            screen_pts[0],
            radius,
            stroke,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_rect(
    pixels: &mut [egui::Color32],
    w: usize,
    h: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    fill: egui::Color32,
    stroke: egui::Color32,
    stroke_width: f64,
) {
    let left = p0.0.min(p1.0);
    let right = p0.0.max(p1.0).max(left + 1.0);
    let top = p0.1.min(p1.1);
    let bottom = p0.1.max(p1.1).max(top + 1.0);
    let x0 = (left.floor() as isize).clamp(0, w as isize - 1) as usize;
    let x1 = (right.ceil() as isize).clamp(0, w as isize) as usize;
    let y0 = (top.floor() as isize).clamp(0, h as isize - 1) as usize;
    let y1 = (bottom.ceil() as isize).clamp(0, h as isize) as usize;
    for y in y0..y1 {
        for x in x0..x1 {
            pixels[y * w + x] = blend_over(pixels[y * w + x], fill, 1.0);
        }
    }
    let r = (stroke_width * 0.5).max(0.6);
    draw_segment(pixels, w, h, (left, top), (right, top), r, stroke);
    draw_segment(pixels, w, h, (right, top), (right, bottom), r, stroke);
    draw_segment(pixels, w, h, (right, bottom), (left, bottom), r, stroke);
    draw_segment(pixels, w, h, (left, bottom), (left, top), r, stroke);
}
