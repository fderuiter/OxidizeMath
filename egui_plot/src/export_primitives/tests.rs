use std::sync::Arc;

use crate::{Arrows, Bar, BarChart, BoxElem, BoxPlot, BoxSpread, ExportOptions, Line, PlotItem};

use super::{
    ExportPrimitive, export_primitives_svg, export_raster_gif, export_raster_svg,
    render_primitives_image, scale_color_image,
};

fn count_non_bg_pixels(img: &egui::ColorImage, bg: egui::Color32) -> usize {
    img.pixels.iter().filter(|&&p| p != bg).count()
}

#[test]
fn test_barchart_extract_points_and_offscreen_render() {
    let chart = BarChart::new(
        "Partitions",
        vec![
            Bar::new(0.0, 1.0),
            Bar::new(1.0, 2.0),
            Bar::new(2.0, 5.0),
            Bar::new(3.0, 10.0),
        ],
    )
    .color(egui::Color32::from_rgb(60, 180, 240));

    let pts = chart.extract_points();
    assert_eq!(pts, vec![[0.0, 1.0], [1.0, 2.0], [2.0, 5.0], [3.0, 10.0]]);

    let prims = chart.extract_export_primitives();
    assert_eq!(prims.len(), 4);
    assert!(matches!(prims[0], ExportPrimitive::Bar { .. }));

    let opts = ExportOptions {
        width: 320,
        height: 200,
        ..ExportOptions::default()
    };
    let img = render_primitives_image(&prims, &opts).expect("bar raster render");
    assert!(count_non_bg_pixels(&img, opts.background_color) > 200);

    let png = crate::color32_image_to_png(&img).expect("bar png");
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));

    let svg = export_primitives_svg(&prims, &opts).expect("bar svg");
    assert!(svg.contains("<rect x="));
    assert!(svg.contains("data-name=\"Partitions\""));
}

#[test]
fn test_arrows_extract_points_bounds_and_offscreen_render() {
    let mut arrows = Arrows::new(
        "Velocity Vectors",
        vec![[0.0, 0.0], [1.0, 0.0]],
        vec![[0.8, 0.4], [1.8, 0.6]],
    )
    .color(egui::Color32::from_rgb(100, 220, 140))
    .tip_length(10.0);
    arrows.initialize(0.0..=2.0);

    let pts = arrows.extract_points();
    assert_eq!(pts, vec![[0.0, 0.0], [0.8, 0.4], [1.0, 0.0], [1.8, 0.6]]);

    let bounds = arrows.bounds();
    assert!(bounds.max()[0] >= 1.8 && bounds.max()[1] >= 0.6);

    let prims = arrows.extract_export_primitives();
    assert_eq!(prims.len(), 2);
    assert!(matches!(prims[0], ExportPrimitive::Arrow { .. }));

    let opts = ExportOptions {
        width: 320,
        height: 200,
        ..ExportOptions::default()
    };
    let img = render_primitives_image(&prims, &opts).expect("arrow raster render");
    assert!(count_non_bg_pixels(&img, opts.background_color) > 50);

    let svg = export_primitives_svg(&prims, &opts).expect("arrow svg");
    assert!(svg.contains("<line x1="));
    assert!(svg.contains("<polygon points="));
}

#[test]
fn test_boxplot_extract_points_and_offscreen_render() {
    let boxplot = BoxPlot::new(
        "Spread",
        vec![
            BoxElem::new(1.0, BoxSpread::new(0.5, 1.5, 2.2, 3.1, 4.5)),
            BoxElem::new(2.0, BoxSpread::new(1.0, 2.0, 2.8, 3.9, 5.0)),
        ],
    )
    .color(egui::Color32::from_rgb(240, 140, 60));

    let pts = boxplot.extract_points();
    assert_eq!(pts.len(), 10);
    assert_eq!(pts[0], [1.0, 0.5]);
    assert_eq!(pts[2], [1.0, 2.2]);
    assert_eq!(pts[4], [1.0, 4.5]);

    let prims = boxplot.extract_export_primitives();
    assert_eq!(prims.len(), 2);
    assert!(matches!(prims[0], ExportPrimitive::BoxElem { .. }));

    let opts = ExportOptions {
        width: 320,
        height: 200,
        ..ExportOptions::default()
    };
    let img = render_primitives_image(&prims, &opts).expect("boxplot raster");
    assert!(count_non_bg_pixels(&img, opts.background_color) > 150);

    let svg = export_primitives_svg(&prims, &opts).expect("boxplot svg");
    assert!(svg.contains("<rect x="));
    assert!(svg.contains("<line x1="));
}

#[test]
fn test_mixed_line_and_bar_and_raster_exports() {
    let mut line =
        Line::new("Trajectory", vec![[0.0, 0.2], [1.0, 0.6], [2.0, 0.8]]).color(egui::Color32::RED);
    line.initialize(0.0..=2.0);
    let bars = BarChart::new("Shares", vec![Bar::new(0.0, 0.6), Bar::new(1.0, 0.4)])
        .color(egui::Color32::BLUE);

    let mut prims = line.extract_export_primitives();
    prims.extend(bars.extract_export_primitives());
    assert_eq!(prims.len(), 3);

    let opts = ExportOptions {
        width: 320,
        height: 200,
        ..ExportOptions::default()
    };
    let svg = export_primitives_svg(&prims, &opts).expect("mixed svg");
    assert!(svg.contains("<polyline points="));
    assert!(svg.contains("<rect x="));

    let raster = egui::ColorImage::new(
        [4, 4],
        (0..16)
            .map(|i| egui::Color32::from_rgb((i * 15) as u8, 80, 200))
            .collect(),
    );
    let scaled_raster =
        scale_color_image(&raster, opts.width as usize, opts.height as usize, &opts);
    let raster_png = crate::color32_image_to_png(&scaled_raster).expect("raster png");
    assert!(raster_png.starts_with(b"\x89PNG\r\n\x1a\n"));
    let raster_svg = export_raster_svg(&raster, &opts).expect("raster svg");
    assert!(raster_svg.contains("data:image/png;base64,"));
    let gif = export_raster_gif(&[raster.clone(), raster], &opts, 10).expect("raster gif");
    assert!(gif.starts_with(b"GIF8"));
}

#[test]
fn test_empty_primitives_and_raster_rejected() {
    let opts = ExportOptions::default();
    assert!(render_primitives_image(&[], &opts).is_err());
    assert!(export_primitives_svg(&[], &opts).is_err());
    let empty_img = egui::ColorImage::new([0, 0], vec![]);
    assert!(export_raster_gif(std::slice::from_ref(&empty_img), &opts, 10).is_err());
    assert!(export_raster_svg(&empty_img, &opts).is_err());
    let empty_prim = ExportPrimitive::RasterImage {
        name: "empty".to_string(),
        image: Arc::new(empty_img),
        bounds: None,
    };
    assert!(render_primitives_image(&[empty_prim], &opts).is_err());
}

#[test]
fn test_primitives_log_axis_projection_png_and_svg() {
    let prim = ExportPrimitive::Points {
        name: "Decades".to_string(),
        points: vec![[-5.0, 10.0], [1.0, 10.0], [10.0, 100.0], [100.0, 1000.0]],
        color: egui::Color32::from_rgb(80, 220, 120),
        radius: 4.0,
    };
    let opts = ExportOptions {
        width: 400,
        height: 400,
        log_x: true,
        log_y: true,
        ..ExportOptions::default()
    };
    let img = render_primitives_image(std::slice::from_ref(&prim), &opts).expect("log png");
    assert!(count_non_bg_pixels(&img, opts.background_color) > 20);

    let svg = export_primitives_svg(std::slice::from_ref(&prim), &opts).expect("log svg");
    let cx_vals: Vec<f64> = svg
        .lines()
        .filter_map(|line| {
            let start = line.find("cx=\"")? + 4;
            let end = line[start..].find('"')? + start;
            line[start..end].parse::<f64>().ok()
        })
        .collect();
    assert_eq!(
        cx_vals.len(),
        3,
        "Non-positive point must be filtered on log axis"
    );
    let step_01 = cx_vals[1] - cx_vals[0];
    let step_12 = cx_vals[2] - cx_vals[1];
    assert!(
        (step_01 - step_12).abs() < 0.25,
        "Decade steps must be equal in log10 space: {cx_vals:?}"
    );
}

#[test]
fn test_canvas_layout_from_projected_bounds_and_log_clamp() {
    use super::{PrimitiveCanvasLayout, project_coord};

    let log_opts = ExportOptions {
        width: 500,
        height: 400,
        aspect_ratio: crate::AspectRatioOption::Free,
        log_x: true,
        log_y: true,
        ..ExportOptions::default()
    };

    // Direct 1e-6 clamping when raw_bounds are non-positive on log axes
    let clamped = PrimitiveCanvasLayout::new(500, 400, [-5.0, 0.0, 0.0, -2.0], &log_opts);
    let expected =
        PrimitiveCanvasLayout::from_projected_bounds(500, 400, [-6.0, -5.0, -6.0, -5.0], &log_opts);
    assert_eq!(
        clamped.to_pixel([1e-6, 1e-6]),
        expected.to_pixel([1e-6, 1e-6])
    );
    assert_eq!(
        clamped.to_pixel([1e-5, 1e-5]),
        expected.to_pixel([1e-5, 1e-5])
    );

    // Positive bounds parity between new and from_projected_bounds
    let raw = [2.0, 200.0, 5.0, 5000.0];
    let projected = [
        project_coord(raw[0], true),
        project_coord(raw[1], true),
        project_coord(raw[2], true),
        project_coord(raw[3], true),
    ];
    let via_new = PrimitiveCanvasLayout::new(500, 400, raw, &log_opts);
    let via_proj = PrimitiveCanvasLayout::from_projected_bounds(500, 400, projected, &log_opts);
    for pt in [[2.0, 5.0], [20.0, 500.0], [200.0, 5000.0]] {
        assert_eq!(via_new.to_pixel(pt), via_proj.to_pixel(pt));
    }
}

#[test]
fn test_shared_pixel_ops_and_svg_header_helpers() {
    use super::pixel_ops::{blend_over, draw_dot, draw_segment};
    use super::raster::background_pixel;
    use super::svg::{escape_xml, svg_document_header};

    assert_eq!(
        escape_xml("A & B <C> \"D\" 'E'"),
        "A &amp; B &lt;C&gt; &quot;D&quot; &apos;E&apos;"
    );

    let opts = ExportOptions {
        background_color: egui::Color32::from_rgb(10, 20, 30),
        background_opacity: 0.5,
        ..ExportOptions::default()
    };
    let bg = background_pixel(&opts);
    assert_eq!(bg, egui::Color32::from_rgba_unmultiplied(10, 20, 30, 128));

    let header = svg_document_header(320, 180, &opts);
    assert!(header.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
    assert!(header.contains("width=\"320\" height=\"180\""));
    assert!(header.contains("fill=\"rgb(10,20,30)\" fill-opacity=\"0.500\""));

    let mut pixels = vec![egui::Color32::TRANSPARENT; 64];
    let src = egui::Color32::from_rgb(200, 100, 50);
    draw_dot(&mut pixels, 8, 8, (4.0, 4.0), 1.5, src);
    draw_segment(&mut pixels, 8, 8, (1.0, 1.0), (6.0, 6.0), 1.0, src);
    assert!(pixels.iter().any(|&c| c.a() > 0));
    assert_eq!(blend_over(src, egui::Color32::TRANSPARENT, 1.0), src);
}

#[test]
fn test_svg_contours_and_rect_grid_emit_pure_vectors_without_base64_images() {
    use super::svg_contours::{
        SvgContourLayer, SvgContourPath, SvgRectCell, export_contour_layers_svg,
        export_rect_grid_svg,
    };

    let layers = vec![SvgContourLayer {
        label: "Level 0.50".to_string(),
        color: egui::Color32::from_rgb(64, 180, 255),
        stroke_width: 1.5,
        fill: None,
        paths: vec![SvgContourPath {
            points: vec![[2.0, 1.0], [6.0, 1.0], [6.0, 5.0], [2.0, 5.0]],
            closed: true,
        }],
    }];
    let svg = export_contour_layers_svg(8.0, 8.0, &layers, &ExportOptions::default())
        .expect("contour SVG");
    assert!(svg.contains("<path d=\"M "));
    assert!(svg.contains(" Z\""));
    assert!(!svg.contains("data:image/png"));
    assert!(!svg.contains("<image"));

    let cells = vec![
        SvgRectCell {
            col: 0,
            row: 0,
            fill: egui::Color32::from_rgb(240, 180, 40),
            stroke: None,
        },
        SvgRectCell {
            col: 1,
            row: 1,
            fill: egui::Color32::from_rgb(30, 90, 210),
            stroke: None,
        },
    ];
    let rect_svg =
        export_rect_grid_svg(2, 2, &cells, &ExportOptions::default()).expect("rect grid SVG");
    assert!(rect_svg.contains("<rect "));
    assert!(!rect_svg.contains("data:image/png"));
}

#[test]
fn test_16bit_heightmap_and_periodic_normal_map_encoding() {
    use super::texture_maps::{
        decode_heightmap_16bit_png, decode_normal_map_16bit_png, encode_heightmap_16bit_png,
        encode_normal_map_png, resample_scalar_field_borderless,
    };
    use std::collections::HashSet;
    use std::f64::consts::TAU;

    let (src_w, src_h) = (64_usize, 64_usize);
    let mut field = vec![0.0_f64; src_w * src_h];
    for y in 0..src_h {
        for x in 0..src_w {
            let u = (x as f64 / src_w as f64) * TAU;
            let v = (y as f64 / src_h as f64) * TAU;
            field[y * src_w + x] = 0.5 + 0.3 * u.cos() + 0.2 * v.sin();
        }
    }

    // Verify toroidal wrap error < 1e-9: shifting source by half-period (32, 32) must match
    // shifting the borderless 512x512 resampled output by (256, 256) across the seam!
    let mut shifted_src = vec![0.0_f64; src_w * src_h];
    for y in 0..src_h {
        for x in 0..src_w {
            shifted_src[((y + 32) % src_h) * src_w + ((x + 32) % src_w)] = field[y * src_w + x];
        }
    }
    let resampled =
        resample_scalar_field_borderless(&field, src_w, src_h, 512, 512, true).expect("resample");
    let resampled_shifted =
        resample_scalar_field_borderless(&shifted_src, src_w, src_h, 512, 512, true)
            .expect("resample shifted");
    for y in 0..512 {
        for x in [0_usize, 1, 255, 256, 510, 511] {
            let a = resampled[y * 512 + x];
            let b = resampled_shifted[((y + 256) % 512) * 512 + ((x + 256) % 512)];
            assert!(
                (a - b).abs() < 1e-12,
                "Toroidal wrap seam mismatch at ({x}, {y}): {a} vs {b}"
            );
        }
    }

    for target_size in [512_u32, 1024_u32] {
        let hm_bytes = encode_heightmap_16bit_png(&field, src_w, src_h, target_size, true)
            .expect("16-bit heightmap");
        let (w, h, samples) = decode_heightmap_16bit_png(&hm_bytes).expect("decode 16-bit");
        assert_eq!((w, h), (target_size, target_size));
        let unique: HashSet<u16> = samples.into_iter().collect();
        assert!(
            unique.len() > 256,
            "Expected > 256 unique 16-bit grayscale levels, got {}",
            unique.len()
        );

        let nm_bytes = encode_normal_map_png(&field, src_w, src_h, target_size, true, 2.5)
            .expect("16-bit normal map");
        let (nw, nh, normals) = decode_normal_map_16bit_png(&nm_bytes).expect("decode normal map");
        assert_eq!((nw, nh), (target_size, target_size));
        let stride = target_size as usize;
        for row in [0_usize, stride / 4, stride / 2, (3 * stride) / 4] {
            let left = normals[row * stride];
            let right = normals[row * stride + (stride - 1)];
            for ch in 0..3 {
                let diff = (f64::from(left[ch]) - f64::from(right[ch])).abs() / 65535.0;
                assert!(
                    diff < 0.02,
                    "Periodic normal seam jump {diff} too large at row {row} channel {ch}"
                );
            }
        }
    }
}
