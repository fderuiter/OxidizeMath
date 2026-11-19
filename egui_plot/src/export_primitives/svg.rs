//! Vector SVG serializer for [`super::ExportPrimitive`] items and embedded raster images.

use std::sync::Arc;

use crate::export::{ExportOptions, color32_image_to_png};

use super::{
    ExportPrimitive, PrimitiveCanvasLayout, compute_primitive_bounds_for_options,
    has_drawable_primitives, resolve_canvas_dimensions, resolve_color, sanitize_bar_corners,
};

/// Renders `primitives` into a vector SVG XML document.
pub fn export_primitives_svg(
    primitives: &[ExportPrimitive],
    options: &ExportOptions,
) -> Result<String, String> {
    if !has_drawable_primitives(primitives) {
        return Err("No drawable plot primitives available for SVG export".to_string());
    }
    let (width, height) = resolve_canvas_dimensions(options)?;
    let bounds = compute_primitive_bounds_for_options(primitives, options)
        .ok_or_else(|| "Unable to compute finite plot bounds for SVG export".to_string())?;
    let layout = PrimitiveCanvasLayout::new(width, height, bounds, options);

    let bg = options.background_color;
    let bg_opacity = (bg.a() as f32 / 255.0) * options.background_opacity.clamp(0.0, 1.0);
    let mut svg = String::new();
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str(&format!(
        "<svg width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    ));
    svg.push_str(&format!(
        "  <rect width=\"{width}\" height=\"{height}\" fill=\"rgb({},{},{})\" fill-opacity=\"{bg_opacity:.3}\" />\n",
        bg.r(),
        bg.g(),
        bg.b()
    ));

    for (idx, primitive) in primitives.iter().enumerate() {
        write_primitive_svg(&mut svg, &layout, primitive, idx)?;
    }

    svg.push_str("</svg>\n");
    Ok(svg)
}

/// Embeds a raster `ColorImage` inside a standalone SVG document at `options` resolution.
pub fn export_raster_svg(
    raster: &egui::ColorImage,
    options: &ExportOptions,
) -> Result<String, String> {
    if raster.width() == 0 || raster.height() == 0 || raster.pixels.is_empty() {
        return Err("Raster image is empty".to_string());
    }
    let min_c = if options.log_x || options.log_y {
        1.0
    } else {
        0.0
    };
    let max_x = (raster.width() as f64).max(min_c + 1.0);
    let max_y = (raster.height() as f64).max(min_c + 1.0);
    let primitive = ExportPrimitive::RasterImage {
        name: "Raster Field".to_string(),
        image: Arc::new(raster.clone()),
        bounds: Some([min_c, max_x, min_c, max_y]),
    };
    export_primitives_svg(&[primitive], options)
}

fn write_primitive_svg(
    svg: &mut String,
    layout: &PrimitiveCanvasLayout,
    primitive: &ExportPrimitive,
    idx: usize,
) -> Result<(), String> {
    match primitive {
        ExportPrimitive::RasterImage {
            name,
            image,
            bounds,
        } => write_raster_svg(svg, layout, name, image, *bounds, idx),
        ExportPrimitive::Bar {
            name,
            min,
            max,
            fill,
            stroke_color,
            stroke_width,
        } => {
            if let Some((c0, c1)) = sanitize_bar_corners(*min, *max, layout.log_x, layout.log_y) {
                let stroke = resolve_color(*stroke_color, idx);
                let fill_c = if *fill == egui::Color32::TRANSPARENT {
                    stroke.linear_multiply(0.4)
                } else {
                    *fill
                };
                let sw = ((*stroke_width as f64) * layout.scale).max(1.0);
                write_rect_svg(
                    svg,
                    name,
                    idx,
                    layout.to_pixel(c0),
                    layout.to_pixel(c1),
                    fill_c,
                    stroke,
                    sw,
                );
            }
            Ok(())
        }
        ExportPrimitive::BoxElem { .. } => {
            write_box_svg(svg, layout, primitive, idx);
            Ok(())
        }
        ExportPrimitive::Arrow {
            name,
            origin,
            tip,
            tip_length,
            color,
        } => {
            let stroke = resolve_color(*color, idx);
            write_arrow_svg(svg, layout, name, idx, *origin, *tip, *tip_length, stroke);
            Ok(())
        }
        ExportPrimitive::Polyline {
            name,
            points,
            color,
            stroke_width,
            closed,
            fill,
        } => {
            let stroke = resolve_color(*color, idx);
            let sw = ((*stroke_width as f64) * layout.scale).max(layout.base_stroke);
            write_polyline_svg(svg, layout, name, idx, points, stroke, sw, *closed, *fill);
            Ok(())
        }
        ExportPrimitive::Points {
            name,
            points,
            color,
            radius,
        } => {
            let fill = resolve_color(*color, idx);
            let r = ((*radius as f64) * layout.scale).max(1.5);
            write_points_svg(svg, layout, name, idx, points, fill, r);
            Ok(())
        }
    }
}

fn write_raster_svg(
    svg: &mut String,
    layout: &PrimitiveCanvasLayout,
    name: &str,
    image: &egui::ColorImage,
    bounds: Option<[f64; 4]>,
    idx: usize,
) -> Result<(), String> {
    let png_bytes = color32_image_to_png(image)?;
    let b64 = encode_base64(&png_bytes);
    let (x, y, w, h) = if let Some([x0, x1, y0, y1]) =
        bounds.filter(|b| !layout.log_x && !layout.log_y || (b[0] > 0.0 && b[2] > 0.0))
    {
        let (px0, py0) = layout.to_pixel([x0, y0]);
        let (px1, py1) = layout.to_pixel([x1, y1]);
        let left = px0.min(px1);
        let top = py0.min(py1);
        (
            left,
            top,
            (px0.max(px1) - left).max(1.0),
            (py0.max(py1) - top).max(1.0),
        )
    } else {
        (layout.padding, layout.padding, layout.draw_w, layout.draw_h)
    };
    let escaped = escape_xml(name);
    svg.push_str(&format!(
        "  <g id=\"primitive-{idx}\" data-name=\"{escaped}\">\n    <image x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" preserveAspectRatio=\"none\" image-rendering=\"pixelated\" href=\"data:image/png;base64,{b64}\" />\n  </g>\n"
    ));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_rect_svg(
    svg: &mut String,
    name: &str,
    idx: usize,
    p0: (f64, f64),
    p1: (f64, f64),
    fill: egui::Color32,
    stroke: egui::Color32,
    stroke_width: f64,
) {
    let x = p0.0.min(p1.0);
    let y = p0.1.min(p1.1);
    let w = (p0.0.max(p1.0) - x).max(1.0);
    let h = (p0.1.max(p1.1) - y).max(1.0);
    let escaped = escape_xml(name);
    let fill_hex = hex_rgb(fill);
    let fill_op = fill.a() as f32 / 255.0;
    let stroke_hex = hex_rgb(stroke);
    svg.push_str(&format!(
        "  <g id=\"primitive-{idx}\" data-name=\"{escaped}\">\n    <rect x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" fill=\"{fill_hex}\" fill-opacity=\"{fill_op:.3}\" stroke=\"{stroke_hex}\" stroke-width=\"{stroke_width:.2}\" />\n  </g>\n"
    ));
}

fn write_box_svg(
    svg: &mut String,
    layout: &PrimitiveCanvasLayout,
    primitive: &ExportPrimitive,
    idx: usize,
) {
    let ExportPrimitive::BoxElem {
        name,
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
    } = primitive
    else {
        return;
    };
    let pt_for = |arg: f64, val: f64| if *vertical { [arg, val] } else { [val, arg] };
    let (hb, hw) = (box_width * 0.5, whisker_width * 0.5);
    if !layout.is_valid_point(pt_for(argument - hb.max(hw), *lower_whisker))
        || !layout.is_valid_point(pt_for(argument + hb.max(hw), *upper_whisker))
    {
        return;
    }
    let stroke = resolve_color(*stroke_color, idx);
    let fill_c = if *fill == egui::Color32::TRANSPARENT {
        stroke.linear_multiply(0.35)
    } else {
        *fill
    };
    let sw = ((*stroke_width as f64) * layout.scale).max(1.0);
    let map = |arg: f64, val: f64| layout.to_pixel(pt_for(arg, val));
    let (b0, b1) = (
        map(argument - hb, *quartile1),
        map(argument + hb, *quartile3),
    );
    let (rx, ry) = (b0.0.min(b1.0), b0.1.min(b1.1));
    let (rw, rh) = (
        (b0.0.max(b1.0) - rx).max(1.0),
        (b0.1.max(b1.1) - ry).max(1.0),
    );
    let sh = hex_rgb(stroke);
    let fh = hex_rgb(fill_c);
    let fo = fill_c.a() as f32 / 255.0;
    svg.push_str(&format!(
        "  <g id=\"primitive-{idx}\" data-name=\"{}\">\n    <rect x=\"{rx:.2}\" y=\"{ry:.2}\" width=\"{rw:.2}\" height=\"{rh:.2}\" fill=\"{fh}\" fill-opacity=\"{fo:.3}\" stroke=\"{sh}\" stroke-width=\"{sw:.2}\" />\n",
        escape_xml(name)
    ));
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
        svg.push_str(&format!(
            "    <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{sh}\" stroke-width=\"{sw:.2}\" />\n",
            p0.0, p0.1, p1.0, p1.1
        ));
    }
    svg.push_str("  </g>\n");
}

#[allow(clippy::too_many_arguments)]
fn write_arrow_svg(
    svg: &mut String,
    layout: &PrimitiveCanvasLayout,
    name: &str,
    idx: usize,
    origin: [f64; 2],
    tip: [f64; 2],
    tip_length: Option<f32>,
    color: egui::Color32,
) {
    if !layout.is_valid_point(origin) || !layout.is_valid_point(tip) {
        return;
    }
    let p0 = layout.to_pixel(origin);
    let p1 = layout.to_pixel(tip);
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let len = (dx * dx + dy * dy).sqrt();
    let hex = hex_rgb(color);
    let sw = layout.base_stroke;
    let escaped = escape_xml(name);
    svg.push_str(&format!(
        "  <g id=\"primitive-{idx}\" data-name=\"{escaped}\">\n"
    ));
    if len < 1e-4 {
        svg.push_str(&format!(
            "    <circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{:.2}\" fill=\"{hex}\" />\n  </g>\n",
            p0.0, p0.1, sw
        ));
        return;
    }
    let head_len = tip_length
        .map(|l| (l as f64) * layout.scale)
        .unwrap_or_else(|| (len * 0.25).clamp(4.0, 18.0 * layout.scale));
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
    svg.push_str(&format!(
        "    <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{hex}\" stroke-width=\"{sw:.2}\" stroke-linecap=\"round\" />\n",
        p0.0, p0.1, p1.0, p1.1
    ));
    svg.push_str(&format!(
        "    <polygon points=\"{:.2},{:.2} {:.2},{:.2} {:.2},{:.2}\" fill=\"{hex}\" stroke=\"{hex}\" stroke-width=\"{:.2}\" />\n  </g>\n",
        p1.0, p1.1, w1.0, w1.1, w2.0, w2.1, sw * 0.75
    ));
}

#[allow(clippy::too_many_arguments)]
fn write_polyline_svg(
    svg: &mut String,
    layout: &PrimitiveCanvasLayout,
    name: &str,
    idx: usize,
    points: &[[f64; 2]],
    stroke: egui::Color32,
    stroke_width: f64,
    closed: bool,
    fill: Option<egui::Color32>,
) {
    let hex = hex_rgb(stroke);
    let escaped = escape_xml(name);
    svg.push_str(&format!(
        "  <g id=\"primitive-{idx}\" data-name=\"{escaped}\">\n"
    ));
    let finite: Vec<(f64, f64)> = points
        .iter()
        .filter(|&&p| layout.is_valid_point(p))
        .map(|&p| layout.to_pixel(p))
        .collect();
    if finite.len() == 1 {
        let (px, py) = finite[0];
        let r = (stroke_width * 0.5).max(1.5);
        svg.push_str(&format!(
            "    <circle cx=\"{px:.2}\" cy=\"{py:.2}\" r=\"{r:.2}\" fill=\"{hex}\" />\n"
        ));
    } else if finite.len() >= 2 {
        let pts_attr: String = finite
            .iter()
            .map(|(x, y)| format!("{x:.2},{y:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        if closed {
            let fc = fill.unwrap_or_else(|| stroke.linear_multiply(0.2));
            let fhex = hex_rgb(fc);
            let fop = fc.a() as f32 / 255.0;
            svg.push_str(&format!(
                "    <polygon points=\"{pts_attr}\" fill=\"{fhex}\" fill-opacity=\"{fop:.3}\" stroke=\"{hex}\" stroke-width=\"{stroke_width:.2}\" stroke-linejoin=\"round\" />\n"
            ));
        } else {
            svg.push_str(&format!(
                "    <polyline points=\"{pts_attr}\" fill=\"none\" stroke=\"{hex}\" stroke-width=\"{stroke_width:.2}\" stroke-linecap=\"round\" stroke-linejoin=\"round\" />\n"
            ));
        }
    }
    svg.push_str("  </g>\n");
}

fn write_points_svg(
    svg: &mut String,
    layout: &PrimitiveCanvasLayout,
    name: &str,
    idx: usize,
    points: &[[f64; 2]],
    fill: egui::Color32,
    radius: f64,
) {
    let hex = hex_rgb(fill);
    let escaped = escape_xml(name);
    svg.push_str(&format!(
        "  <g id=\"primitive-{idx}\" data-name=\"{escaped}\">\n"
    ));
    for &pt in points {
        if layout.is_valid_point(pt) {
            let (px, py) = layout.to_pixel(pt);
            svg.push_str(&format!(
                "    <circle cx=\"{px:.2}\" cy=\"{py:.2}\" r=\"{radius:.2}\" fill=\"{hex}\" />\n"
            ));
        }
    }
    svg.push_str("  </g>\n");
}

fn hex_rgb(color: egui::Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b())
}

fn escape_xml(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Encodes arbitrary bytes using standard RFC 4648 Base64 for SVG data URIs.
pub fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}
