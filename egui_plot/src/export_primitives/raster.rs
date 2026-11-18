//! Offscreen raster (`egui::ColorImage`) renderer for [`super::ExportPrimitive`] items.

use crate::export::ExportOptions;

use super::pixel_ops::{blend_over, draw_dot, draw_segment, fill_triangle};
use super::{
    ExportPrimitive, compute_primitive_bounds, has_drawable_primitives, padded_bounds,
    resolve_canvas_dimensions, resolve_color,
};

struct CanvasTransform {
    width: usize,
    height: usize,
    padding: f64,
    draw_w: f64,
    draw_h: f64,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    scale: f64,
    base_stroke: f64,
}

impl CanvasTransform {
    fn new(width: usize, height: usize, bounds: [f64; 4], options: &ExportOptions) -> Self {
        let [min_x, max_x, min_y, max_y] = padded_bounds(bounds);
        let padding = (20.0 * options.scale as f64).clamp(10.0, 100.0);
        let draw_w = (width as f64 - 2.0 * padding).max(1.0);
        let draw_h = (height as f64 - 2.0 * padding).max(1.0);
        Self {
            width,
            height,
            padding,
            draw_w,
            draw_h,
            min_x,
            max_x,
            min_y,
            max_y,
            scale: (options.scale as f64).max(0.5),
            base_stroke: (options.line_thickness * options.scale).max(1.0) as f64,
        }
    }

    fn to_pixel(&self, p: [f64; 2]) -> (f64, f64) {
        let x_norm = (p[0] - self.min_x) / (self.max_x - self.min_x);
        let y_norm = (p[1] - self.min_y) / (self.max_y - self.min_y);
        let px = self.padding + x_norm * self.draw_w;
        let py = (self.height as f64 - self.padding) - y_norm * self.draw_h;
        (px, py)
    }
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
        .or_else(|| compute_primitive_bounds(primitives))
        .ok_or_else(|| "Unable to compute finite plot bounds for image export".to_string())?;
    let tf = CanvasTransform::new(width, height, bounds, options);
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

fn background_pixel(options: &ExportOptions) -> egui::Color32 {
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
    tf: &CanvasTransform,
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
            let stroke = resolve_color(*stroke_color, idx);
            let fill_c = if *fill == egui::Color32::TRANSPARENT {
                stroke.linear_multiply(0.4)
            } else {
                *fill
            };
            let (p0, p1) = (tf.to_pixel(*min), tf.to_pixel(*max));
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
                if pt[0].is_finite() && pt[1].is_finite() {
                    draw_dot(pixels, tf.width, tf.height, tf.to_pixel(pt), r, fill);
                }
            }
        }
    }
}

fn draw_raster_primitive(
    pixels: &mut [egui::Color32],
    tf: &CanvasTransform,
    image: &egui::ColorImage,
    bounds: Option<[f64; 4]>,
) {
    let rect = if let Some([x0, x1, y0, y1]) = bounds {
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
    tf: &CanvasTransform,
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
    let stroke = resolve_color(*stroke_color, idx);
    let fill_c = if *fill == egui::Color32::TRANSPARENT {
        stroke.linear_multiply(0.35)
    } else {
        *fill
    };
    let sw = ((*stroke_width as f64) * tf.scale).max(1.0);
    let r = (sw * 0.5).max(0.75);
    let map = |arg: f64, val: f64| -> (f64, f64) {
        if *vertical {
            tf.to_pixel([arg, val])
        } else {
            tf.to_pixel([val, arg])
        }
    };
    let hb = box_width * 0.5;
    let hw = whisker_width * 0.5;
    let (b0, b1) = (
        map(argument - hb, *quartile1),
        map(argument + hb, *quartile3),
    );
    draw_rect(pixels, tf.width, tf.height, b0, b1, fill_c, stroke, sw);
    draw_segment(
        pixels,
        tf.width,
        tf.height,
        map(argument - hb, *median),
        map(argument + hb, *median),
        r,
        stroke,
    );
    draw_segment(
        pixels,
        tf.width,
        tf.height,
        map(*argument, *lower_whisker),
        map(*argument, *quartile1),
        r,
        stroke,
    );
    draw_segment(
        pixels,
        tf.width,
        tf.height,
        map(*argument, *quartile3),
        map(*argument, *upper_whisker),
        r,
        stroke,
    );
    draw_segment(
        pixels,
        tf.width,
        tf.height,
        map(argument - hw, *lower_whisker),
        map(argument + hw, *lower_whisker),
        r,
        stroke,
    );
    draw_segment(
        pixels,
        tf.width,
        tf.height,
        map(argument - hw, *upper_whisker),
        map(argument + hw, *upper_whisker),
        r,
        stroke,
    );
}

fn draw_arrow(
    pixels: &mut [egui::Color32],
    tf: &CanvasTransform,
    origin: [f64; 2],
    tip: [f64; 2],
    tip_length: Option<f32>,
    color: egui::Color32,
) {
    if !origin[0].is_finite() || !origin[1].is_finite() || !tip[0].is_finite() || !tip[1].is_finite()
    {
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
    tf: &CanvasTransform,
    points: &[[f64; 2]],
    stroke: egui::Color32,
    radius: f64,
    closed: bool,
    fill: Option<egui::Color32>,
) {
    let screen_pts: Vec<(f64, f64)> = points
        .iter()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
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
        if !p[0].is_finite() || !p[1].is_finite() {
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
        draw_segment(pixels, tf.width, tf.height, last, screen_pts[0], radius, stroke);
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
