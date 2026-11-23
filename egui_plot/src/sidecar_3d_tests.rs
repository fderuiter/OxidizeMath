#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::{
    Line, Plot, Plot3DSeries, Plot3DSeriesSidecar, PlotPoints, Points,
    a11y::PointTooltipState,
    data_table::{DataTableState, TableSeriesRef3D, format_tsv_3d, show_data_table_modal_3d},
    export::{format_csv_3d, format_json_3d},
    get_last_plot_datasets_3d,
};
use egui::{Color32, Context, Event, Id, Key, Modifiers, Pos2, RawInput, Rect, vec2};

fn frame_input(events: Vec<Event>) -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0))),
        events,
        ..Default::default()
    }
}

fn render_3d_plot_frame(
    ctx: &Context,
    input: RawInput,
    projected: [[f64; 2]; 2],
    world: [[f64; 3]; 2],
) -> (Id, Pos2) {
    let mut plot_id = Id::NULL;
    let mut first_screen_pos = Pos2::ZERO;
    let _ = ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp = Plot::new("test_3d_sidecar_plot")
                .show_axes(false)
                .show_grid(false)
                .default_x_bounds(-5.0, 5.0)
                .default_y_bounds(-5.0, 5.0)
                .show(ui, |plot_ui| {
                    plot_id = plot_ui.plot_id();
                    let item_id = plot_id.with("pt_series");
                    let mut sidecar = Plot3DSeriesSidecar::new();
                    sidecar.register_item_vertices(item_id, world.to_vec());
                    sidecar.push_series(Plot3DSeries::new(
                        "Atoms",
                        projected.to_vec(),
                        world.to_vec(),
                        Color32::LIGHT_BLUE,
                    ));
                    plot_ui.points(
                        Points::new("Atoms", projected.to_vec())
                            .id(item_id)
                            .radius(6.0_f32),
                    );
                    let line_id = plot_id.with("seg_item");
                    sidecar.register_item_vertices(line_id, world.to_vec());
                    plot_ui
                        .line(Line::new("Bond", PlotPoints::new(projected.to_vec())).id(line_id));
                    plot_ui.set_3d_sidecar(sidecar);
                });
            first_screen_pos = resp
                .transform
                .position_from_point(&crate::PlotPoint::new(projected[0][0], projected[0][1]));
            plot_id = resp.response.id;
        });
    });
    (plot_id, first_screen_pos)
}

#[test]
fn test_3d_sidecar_hover_tooltip_reports_world_xyz_and_suppresses_rulers() {
    let ctx = Context::default();
    let world = [[1.25, -2.5, 25.75], [-0.5, 3.0, 12.0]];
    let proj_pose_a = [[0.8, -0.6], [-1.2, 1.4]];
    let (_, screen_a) = render_3d_plot_frame(&ctx, frame_input(Vec::new()), proj_pose_a, world);
    let (plot_id, _) = render_3d_plot_frame(
        &ctx,
        frame_input(vec![Event::PointerMoved(screen_a)]),
        proj_pose_a,
        world,
    );

    let state_a = ctx
        .data(|d| d.get_temp::<PointTooltipState>(plot_id.with("a11y_point_tooltip_state")))
        .expect("3D point tooltip state should be recorded");
    assert!(state_a.suppress_rulers);
    assert_eq!(state_a.world_point, Some(world[0]));
    let text_a = state_a
        .announced_text
        .clone()
        .expect("3D tooltip announcement");
    assert!(text_a.contains("x = 1.2500"), "Got: {text_a}");
    assert!(text_a.contains("y = -2.5000"), "Got: {text_a}");
    assert!(text_a.contains("z = 25.7500"), "Got: {text_a}");

    let proj_pose_b = [[-2.1, 1.9], [1.7, -1.1]];
    let (_, screen_b) = render_3d_plot_frame(&ctx, frame_input(Vec::new()), proj_pose_b, world);
    let _ = render_3d_plot_frame(
        &ctx,
        frame_input(vec![Event::PointerMoved(screen_b)]),
        proj_pose_b,
        world,
    );
    let state_b = ctx
        .data(|d| d.get_temp::<PointTooltipState>(plot_id.with("a11y_point_tooltip_state")))
        .expect("3D tooltip state after camera orbit/pan");
    assert_eq!(state_a.body_lines, state_b.body_lines);
    assert_eq!(get_last_plot_datasets_3d()[0].1, world.to_vec());

    ctx.data_mut(|d| d.remove::<String>(Id::new("aria_live_message")));
    let far_corner = Pos2::new(30.0, 30.0);
    let _ = render_3d_plot_frame(
        &ctx,
        frame_input(vec![Event::PointerMoved(far_corner)]),
        proj_pose_b,
        world,
    );
    let empty_live = ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_message")));
    assert!(
        empty_live.is_none(),
        "Empty-space 2D ruler live readout must be suppressed"
    );
}

#[test]
fn test_3d_sidecar_keyboard_focus_navigation_announces_world_xyz() {
    let ctx = Context::default();
    let world = [[1.5, -0.25, 9.75], [2.5, 0.75, -4.5]];
    let projected = [[0.2, 0.3], [1.1, -0.8]];
    let (plot_id, _) = render_3d_plot_frame(&ctx, frame_input(Vec::new()), projected, world);
    ctx.memory_mut(|m| m.request_focus(plot_id));

    let arrow_right = vec![
        Event::Key {
            key: Key::ArrowRight,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        },
        Event::Key {
            key: Key::ArrowRight,
            physical_key: None,
            pressed: false,
            repeat: false,
            modifiers: Modifiers::NONE,
        },
    ];
    let _ = render_3d_plot_frame(&ctx, frame_input(arrow_right), projected, world);
    let msg = ctx
        .data(|d| d.get_temp::<String>(Id::new("aria_live_message")))
        .expect("Focused ArrowRight should announce 3D world point");
    assert!(msg.contains("Dataset Atoms"));
    assert!(
        msg.contains("X: 2.5000, Y: 0.7500, Z: -4.5000"),
        "Got: {msg}"
    );
}

#[test]
fn test_3d_data_table_and_exports_format_world_xyz() {
    let series = [
        Plot3DSeries::new(
            "Trajectory",
            vec![[0.1, 0.2], [0.3, 0.4]],
            vec![[1.0, 1.0, 25.0], [1.1, 1.2, 24.8]],
            Color32::RED,
        ),
        Plot3DSeries::new(
            "Initial State",
            vec![[0.0, 0.0]],
            vec![[0.0, 0.0, 0.0]],
            Color32::GREEN,
        ),
    ];
    let datasets: Vec<(String, Vec<[f64; 3]>)> = series
        .iter()
        .map(|s| (s.name.clone(), s.world.clone()))
        .collect();
    let csv = format_csv_3d(&datasets);
    assert!(csv.starts_with("Dataset,X,Y,Z\n"), "Got: {csv}");
    assert!(csv.contains("Trajectory,1,1,25\n"));

    let json = format_json_3d(&datasets);
    assert!(json.contains("\"z\": 25"));

    let refs = [TableSeriesRef3D {
        name: "Trajectory",
        points: &series[0].world,
        color: Color32::RED,
    }];
    let tsv = format_tsv_3d(&refs);
    assert!(tsv.starts_with("X\tY\tZ\n1\t1\t25\n"), "Got: {tsv}");

    let multi_refs = [
        refs[0],
        TableSeriesRef3D {
            name: "Initial State",
            points: &series[1].world,
            color: Color32::GREEN,
        },
    ];
    let multi_tsv = format_tsv_3d(&multi_refs);
    assert!(
        multi_tsv.starts_with("Dataset\tX\tY\tZ\nTrajectory\t1\t1\t25\n"),
        "Got: {multi_tsv}"
    );

    let ctx = Context::default();
    let mut show_table = true;
    let mut state = DataTableState::default();
    let plot_id = Id::new("table_3d_modal_test");
    let _ = ctx.run(frame_input(Vec::new()), |ctx| {
        show_data_table_modal_3d(ctx, plot_id, "Lorenz", &mut show_table, &mut state, &series);
    });
    let down = vec![Event::Key {
        key: Key::ArrowDown,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    }];
    let _ = ctx.run(frame_input(down), |ctx| {
        show_data_table_modal_3d(ctx, plot_id, "Lorenz", &mut show_table, &mut state, &series);
    });
    let row_msg = ctx
        .data(|d| d.get_temp::<String>(Id::new("aria_live_message")))
        .expect("ArrowDown in 3D Table View should announce X, Y, Z");
    assert!(row_msg.contains("Z = 24.8"), "Got: {row_msg}");
}
