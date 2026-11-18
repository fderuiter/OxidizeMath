use std::sync::Arc;

use crate::{Arrows, Bar, BarChart, BoxElem, BoxPlot, BoxSpread, ExportOptions, Line, PlotItem};

use super::{
    ExportPrimitive, export_primitives_png, export_primitives_svg, export_raster_gif,
    export_raster_png, export_raster_svg, render_primitives_image,
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

    let png = export_primitives_png(&prims, &opts).expect("bar png");
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
    let raster_png = export_raster_png(&raster, &opts).expect("raster png");
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
    assert!(export_raster_png(&empty_img, &opts).is_err());
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
