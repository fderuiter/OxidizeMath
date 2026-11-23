use crate::{
    AmbientPlotLinkConfig, Line, Plot, PlotBounds, PlotImage, PlotPoint,
    set_ambient_plot_link_config,
};
use egui::{
    ColorImage, Context, Event, Pos2, RawInput, Rect, Shape, TextureOptions, Vec2, pos2, vec2,
};

fn run_dual_linked_plots(
    ctx: &Context,
    pane0_peak: f64,
    pane1_peak: f64,
    zoom_pane0: bool,
    reset_pane0: bool,
    events: Vec<Event>,
) -> (PlotBounds, PlotBounds, egui::FullOutput) {
    let mut b0 = PlotBounds::NOTHING;
    let mut b1 = PlotBounds::NOTHING;
    let screen = Rect::from_min_size(Pos2::ZERO, vec2(1000.0, 500.0));
    let out = ctx.run(
        RawInput {
            screen_rect: Some(screen),
            events,
            ..Default::default()
        },
        |c| {
            egui::CentralPanel::default().show(c, |ui| {
                ui.columns(2, |cols| {
                    cols[0].push_id("pane_left", |p0_ui| {
                        let plot_id = p0_ui.make_persistent_id(egui::Id::new("comparison_2d_plot"));
                        if reset_pane0 {
                            p0_ui
                                .ctx()
                                .data_mut(|d| d.insert_temp(plot_id.with("reset_view"), true));
                        }
                        let resp = Plot::new("comparison_2d_plot")
                            .width(440.0)
                            .height(360.0)
                            .show(p0_ui, |plot_ui| {
                                plot_ui.line(Line::new(
                                    "curve",
                                    vec![[0.0, 0.0], [5.0, pane0_peak], [10.0, 2.0]],
                                ));
                                if zoom_pane0 {
                                    plot_ui.zoom_bounds(Vec2::splat(2.0), PlotPoint::new(5.0, 5.0));
                                }
                            });
                        b0 = *resp.transform.bounds();
                    });
                    cols[1].push_id("pane_right", |p1_ui| {
                        let resp = Plot::new("comparison_2d_plot")
                            .width(440.0)
                            .height(360.0)
                            .show(p1_ui, |plot_ui| {
                                plot_ui.line(Line::new(
                                    "curve",
                                    vec![[0.0, 0.0], [5.0, pane1_peak], [10.0, 4.0]],
                                ));
                            });
                        b1 = *resp.transform.bounds();
                    });
                });
            });
        },
    );
    (b0, b1, out)
}

#[test]
fn test_ambient_link_axes_merges_union_auto_bounds_and_syncs_zoom_and_reset() {
    let ctx = Context::default();
    set_ambient_plot_link_config(
        &ctx,
        AmbientPlotLinkConfig {
            link_axes: true,
            link_cursors: false,
        },
    );

    let _ = run_dual_linked_plots(&ctx, 10.0, 100.0, false, false, Vec::new());
    let (b0, b1, _) = run_dual_linked_plots(&ctx, 10.0, 100.0, false, false, Vec::new());
    assert!(
        b0.max()[1] >= 100.0 && (b0.max()[1] - b1.max()[1]).abs() < 1e-6,
        "Both panes must share union Y-bounds >= 100: left={b0:?}, right={b1:?}"
    );

    // Changing parameter peak in Pane 1 updates shared Y-bounds without manual zooming.
    let _ = run_dual_linked_plots(&ctx, 10.0, 260.0, false, false, Vec::new());
    let (b0_up, b1_up, _) = run_dual_linked_plots(&ctx, 10.0, 260.0, false, false, Vec::new());
    assert!(
        b0_up.max()[1] >= 260.0 && (b0_up.max()[1] - b1_up.max()[1]).abs() < 1e-6,
        "Parameter update in Pane 1 must update shared Y-bounds: left={b0_up:?}, right={b1_up:?}"
    );

    // Manual zoom in Pane 0 synchronizes to Pane 1.
    let (z0, z1, _) = run_dual_linked_plots(&ctx, 10.0, 260.0, true, false, Vec::new());
    assert!(
        z0.max()[1] < 200.0 && (z0.max()[1] - z1.max()[1]).abs() < 1e-6,
        "Manual zoom must synchronize across panes: left={z0:?}, right={z1:?}"
    );

    // Reset in Pane 0 restores union auto-bounds across both panes.
    let _ = run_dual_linked_plots(&ctx, 10.0, 260.0, false, true, Vec::new());
    let (r0, r1, _) = run_dual_linked_plots(&ctx, 10.0, 260.0, false, false, Vec::new());
    assert!(
        r0.max()[1] >= 260.0 && (r0.max()[1] - r1.max()[1]).abs() < 1e-6,
        "Reset must restore union auto-bounds across both panes: left={r0:?}, right={r1:?}"
    );
}

#[test]
fn test_ambient_link_excludes_locked_3d_and_axis_hidden_raster_plots() {
    let ctx = Context::default();
    let tex = ctx.load_texture(
        "dummy_raster",
        ColorImage::example(),
        TextureOptions::NEAREST,
    );
    set_ambient_plot_link_config(
        &ctx,
        AmbientPlotLinkConfig {
            link_axes: true,
            link_cursors: true,
        },
    );

    let mut bounds_2d = PlotBounds::NOTHING;
    let screen = Rect::from_min_size(Pos2::ZERO, vec2(1200.0, 500.0));
    for _ in 0..2 {
        let _ = ctx.run(
            RawInput {
                screen_rect: Some(screen),
                ..Default::default()
            },
            |c| {
                egui::CentralPanel::default().show(c, |ui| {
                    ui.push_id("pane_2d", |u0| {
                        let resp = Plot::new("shared_id").show(u0, |pui| {
                            pui.line(Line::new("2d", vec![[0.0, 0.0], [10.0, 20.0]]));
                        });
                        bounds_2d = *resp.transform.bounds();
                    });
                    ui.push_id("pane_3d_locked", |u1| {
                        Plot::new("shared_id")
                            .allow_drag(false)
                            .allow_zoom(false)
                            .allow_scroll(false)
                            .allow_boxed_zoom(false)
                            .show_navigation_controls(false)
                            .show(u1, |pui| {
                                pui.line(Line::new(
                                    "3d_proj",
                                    vec![[-5000.0, -5000.0], [5000.0, 5000.0]],
                                ));
                            });
                    });
                    ui.push_id("pane_raster_hidden_axes", |u2| {
                        Plot::new("shared_id")
                            .show_axes([false, false])
                            .allow_drag(false)
                            .allow_zoom(false)
                            .allow_scroll(false)
                            .show(u2, |pui| {
                                pui.image(PlotImage::new(
                                    "raster",
                                    &tex,
                                    PlotPoint::new(9000.0, 9000.0),
                                    vec2(1000.0, 1000.0),
                                ));
                            });
                    });
                });
            },
        );
    }

    assert!(
        bounds_2d.max()[1] < 100.0 && bounds_2d.min()[1] > -50.0,
        "Locked 3D and axis-hidden raster canvases must be excluded from 2D axis linking: {bounds_2d:?}"
    );
}

fn count_full_span_crosshair_lines_in_right_half(
    shapes: &[egui::epaint::ClippedShape],
) -> (usize, usize) {
    let mut verticals = 0;
    let mut horizontals = 0;
    for cs in shapes {
        if cs.clip_rect.min.x < 480.0 {
            continue;
        }
        if let Shape::LineSegment { points: [a, b], .. } = cs.shape {
            let dx = (a.x - b.x).abs();
            let dy = (a.y - b.y).abs();
            if dx < 0.5 && dy > 250.0 {
                verticals += 1;
            } else if dy < 0.5 && dx > 300.0 {
                horizontals += 1;
            }
        }
    }
    (verticals, horizontals)
}

#[test]
fn test_ambient_link_cursors_renders_crosshair_in_second_pane_on_hover() {
    let ctx = Context::default();
    set_ambient_plot_link_config(
        &ctx,
        AmbientPlotLinkConfig {
            link_axes: true,
            link_cursors: false,
        },
    );
    let _ = run_dual_linked_plots(&ctx, 20.0, 20.0, false, false, Vec::new());
    let (_, _, unlinked_out) = run_dual_linked_plots(
        &ctx,
        20.0,
        20.0,
        false,
        false,
        vec![Event::PointerMoved(pos2(220.0, 200.0))],
    );
    let base_counts = count_full_span_crosshair_lines_in_right_half(&unlinked_out.shapes);

    set_ambient_plot_link_config(
        &ctx,
        AmbientPlotLinkConfig {
            link_axes: true,
            link_cursors: true,
        },
    );
    let (_, _, linked_out) = run_dual_linked_plots(
        &ctx,
        20.0,
        20.0,
        false,
        false,
        vec![Event::PointerMoved(pos2(220.0, 200.0))],
    );
    let linked_counts = count_full_span_crosshair_lines_in_right_half(&linked_out.shapes);

    assert!(
        linked_counts.0 > base_counts.0 && linked_counts.1 > base_counts.1,
        "Hovering Pane 1 with Link Plot Cursors active must render matching X/Y crosshair lines in Pane 2: before={base_counts:?}, after={linked_counts:?}"
    );
}
