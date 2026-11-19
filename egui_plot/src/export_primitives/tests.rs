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

#[test]
fn test_plot_image_color_image_extracts_raster_primitive() {
    let img = Arc::new(egui::ColorImage::new(
        [8, 8],
        vec![egui::Color32::from_rgb(30, 210, 160); 64],
    ));
    let plot_img = crate::PlotImage::new(
        "turing_field",
        egui::TextureId::default(),
        crate::PlotPoint::new(0.5, 0.5),
        [1.0_f32, 1.0_f32],
    )
    .color_image(Arc::clone(&img));

    let prims = plot_img.extract_export_primitives();
    assert_eq!(prims.len(), 1);
    match &prims[0] {
        ExportPrimitive::RasterImage {
            name,
            image,
            bounds,
        } => {
            assert_eq!(name, "turing_field");
            assert_eq!(image.size, [8, 8]);
            assert_eq!(*bounds, Some([0.0, 1.0, 0.0, 1.0]));
        }
        other => panic!("Expected RasterImage, got {other:?}"),
    }

    let opts = ExportOptions {
        width: 200,
        height: 200,
        ..ExportOptions::default()
    };
    let rendered = render_primitives_image(&prims, &opts).expect("plot_image render");
    assert!(count_non_bg_pixels(&rendered, opts.background_color) > 1000);
}

#[test]
fn test_subplot_panels_preserve_disjoint_viewports_and_isolate_pole_bounds() {
    let z_panel = super::ExportSubplotPanel {
        id: "z_plane".to_string(),
        screen_rect: egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(390.0, 400.0)),
        log_x: false,
        log_y: false,
        datasets: vec![("grid".to_string(), vec![[-2.0, -2.0], [2.0, 2.0]])],
        primitives: vec![ExportPrimitive::Polyline {
            name: "z_plane: grid".to_string(),
            points: vec![[-2.0, -2.0], [2.0, 2.0]],
            color: egui::Color32::from_rgb(80, 180, 250),
            stroke_width: 3.0,
            closed: false,
            fill: None,
        }],
    };
    let w_panel = super::ExportSubplotPanel {
        id: "w_plane".to_string(),
        screen_rect: egui::Rect::from_min_max(egui::pos2(410.0, 0.0), egui::pos2(800.0, 400.0)),
        log_x: false,
        log_y: false,
        datasets: vec![("contour".to_string(), vec![[-1.0, -1.0], [1e6, 1e6]])],
        primitives: vec![ExportPrimitive::Polyline {
            name: "w_plane: contour".to_string(),
            points: vec![[-1.0, -1.0], [1e6, 1e6]],
            color: egui::Color32::from_rgb(240, 110, 80),
            stroke_width: 3.0,
            closed: false,
            fill: None,
        }],
    };

    let opts = ExportOptions {
        width: 800,
        height: 400,
        ..ExportOptions::default()
    };
    let vps = super::compute_subplot_viewports(&[z_panel.clone(), w_panel.clone()], 800, 400, 1.0);
    assert_eq!(vps.len(), 2);
    assert!(vps[0].x + vps[0].width <= vps[1].x);

    let img = super::render_subplot_panels_image(&[z_panel.clone(), w_panel.clone()], &opts)
        .expect("subplot image");
    let z_color = egui::Color32::from_rgb(80, 180, 250);
    let mut min_zx = usize::MAX;
    let mut max_zx = 0_usize;
    for y in 0..img.height() {
        for x in 0..img.width() {
            if img.pixels[y * img.width() + x] == z_color {
                min_zx = min_zx.min(x);
                max_zx = max_zx.max(x);
            }
        }
    }
    assert!(
        max_zx.saturating_sub(min_zx) > 250,
        "z_plane line must span its full subplot width despite 1e6 pole in w_plane ({min_zx}..={max_zx})"
    );
    assert!(
        max_zx < vps[1].x,
        "z_plane pixels must stay inside left viewport"
    );

    let svg = super::export_subplot_panels_svg(&[z_panel, w_panel], &opts).expect("subplot svg");
    assert!(svg.contains("data-subplot-id=\"z_plane\""));
    assert!(svg.contains("data-subplot-id=\"w_plane\""));
    assert!(svg.contains("clip-path=\"url(#subplot-clip-0)\""));
    assert!(svg.contains("clip-path=\"url(#subplot-clip-1)\""));
}
