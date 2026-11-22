use std::borrow::Cow;

use egui::{
    Color32, Context, Event, FullOutput, Id, Key, Modifiers, OutputCommand, PointerButton, Pos2,
    RawInput, Rect, TextureId, pos2, vec2,
};

use crate::{
    AccessState, Line, Plot, PlotIdSource, PlotImage, PlotMemory, PlotPoint,
    data_table::{
        DataTableState, TableSeriesRef, data_table_clipboard_status_id, format_table_value,
        format_tsv, series_share_x, show_data_table_modal,
    },
    humanize_plot_label,
    items::SavedCallout,
};

fn screen_input() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn key_input(key: Key) -> RawInput {
    let mut input = screen_input();
    input.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    input
}

fn click_at(
    ctx: &Context,
    pos: Pos2,
    button: PointerButton,
    mut ui_fn: impl FnMut(&Context),
) -> FullOutput {
    let mut down = screen_input();
    down.events = vec![
        Event::PointerMoved(pos),
        Event::PointerButton {
            pos,
            button,
            pressed: true,
            modifiers: Modifiers::NONE,
        },
    ];
    let _ = ctx.run(down, &mut ui_fn);
    let mut up = screen_input();
    up.events = vec![Event::PointerButton {
        pos,
        button,
        pressed: false,
        modifiers: Modifiers::NONE,
    }];
    ctx.run(up, ui_fn)
}

fn run_modal(
    ctx: &Context,
    raw: RawInput,
    id: Id,
    open: &mut bool,
    st: &mut DataTableState,
    ds: &[(String, Vec<[f64; 2]>, Color32)],
) -> FullOutput {
    ctx.run(raw, |c| {
        show_data_table_modal(c, id, "Data Table - SIR Plot", open, st, ds)
    })
}

fn collect_texts(output: &FullOutput) -> Vec<(String, Pos2)> {
    let mut out = Vec::new();
    let mut stack: Vec<&egui::epaint::Shape> = output.shapes.iter().map(|c| &c.shape).collect();
    while let Some(shape) = stack.pop() {
        match shape {
            egui::epaint::Shape::Text(t) => {
                out.push((
                    t.galley.job.text.clone(),
                    t.pos + t.galley.rect.center().to_vec2(),
                ));
            }
            egui::epaint::Shape::Vec(v) => stack.extend(v.iter().rev()),
            _ => {}
        }
    }
    out
}

fn find_text_center(output: &FullOutput, needle: &str) -> Option<Pos2> {
    collect_texts(output)
        .into_iter()
        .find_map(|(txt, pos)| txt.contains(needle).then_some(pos))
}

#[test]
fn test_plot_id_source_humanization_and_hash_parity() {
    let sir = ("sir_plot", "SIR Plot");
    assert_eq!(
        sir.0.into_plot_id_and_title(),
        (Id::new(sir.0), Some(sir.1.to_owned()))
    );
    let owned = String::from("band_plot");
    let (id_r, t_r) = (&owned).into_plot_id_and_title();
    let (id_o, t_o) = owned.into_plot_id_and_title();
    assert_eq!(
        (id_r, id_o, t_r.as_deref(), t_o),
        (
            Id::new("band_plot"),
            Id::new("band_plot"),
            Some("Band Plot"),
            t_r.clone()
        )
    );
    let cow: Cow<'_, str> = Cow::Borrowed("ode-solver_rk4");
    assert_eq!(
        cow.into_plot_id_and_title(),
        (Id::new("ode-solver_rk4"), Some("ODE Solver RK4".to_owned()))
    );
    let raw_id = Id::new("framework_test_plot_id");
    assert_eq!(raw_id.into_plot_id_and_title(), (Id::new(raw_id), None));
    assert!(humanize_plot_label("   ___   ").is_none());
}

fn seed_callout_memory(ctx: &Context, plot_id: Id) {
    let _ = ctx.run(screen_input(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new(plot_id).id(plot_id).show(ui, |p| {
                p.line(Line::new("Curve", vec![[0.0, 1.0], [1.0, 2.0]]))
            });
        });
    });
    let mut mem = PlotMemory::load(ctx, plot_id).expect("PlotMemory initialized");
    mem.callouts.push(SavedCallout {
        point: [0.5, 1.5],
        offset: [16.0, -16.0],
        label: "Peak".to_owned(),
    });
    mem.store(ctx, plot_id);
}

fn assert_modal_titles(
    ctx: &Context,
    id: Id,
    src: Option<&str>,
    custom: Option<&str>,
    suffix: Option<&str>,
) {
    seed_callout_memory(ctx, id);
    let bases = [
        "Data Table",
        "Export High-Res Plot",
        "Plot Settings 🔧",
        "Add Callout",
        "Edit Callout",
    ];
    for (which, base) in bases.into_iter().enumerate() {
        ctx.data_mut(|d| {
            let mut st = AccessState::default();
            match which {
                0 => st.show_table = true,
                1 => st.export_modal.open = true,
                2 => st.show_settings_modal = true,
                3 => st.show_add_callout_modal = true,
                _ => {
                    st.show_edit_callout_modal = true;
                    st.editing_callout_idx = Some(0);
                }
            }
            d.insert_temp(id, st);
        });
        let render = |ctx: &Context| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let mut plot =
                    src.map_or_else(|| Plot::new(id).id(id), |lbl| Plot::new(lbl).id(id));
                if let Some(t) = custom {
                    plot = plot.title(t);
                }
                plot.show(ui, |p| {
                    p.line(Line::new("Curve", vec![[0.0, 1.0], [1.0, 2.0]]))
                });
            });
        };
        let _ = ctx.run(screen_input(), render);
        let out = ctx.run(screen_input(), render);
        let texts: Vec<String> = collect_texts(&out).into_iter().map(|(t, _)| t).collect();
        let expected = suffix.map_or_else(|| base.to_owned(), |s| format!("{base} - {s}"));
        assert!(
            texts.iter().any(|t| t == &expected),
            "Missing '{expected}' in {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t.contains("Id(")),
            "Leaked Id debug in {texts:?}"
        );
    }
}

#[test]
fn test_all_five_modals_humanized_custom_and_id_fallback_titles() {
    let ctx = Context::default();
    assert_modal_titles(
        &ctx,
        Id::new("h_sir"),
        Some("sir_plot"),
        None,
        Some("SIR Plot"),
    );
    assert_modal_titles(
        &ctx,
        Id::new("c_ode"),
        Some("ode_plot"),
        Some("Oscillator"),
        Some("Oscillator"),
    );
    assert_modal_titles(&ctx, Id::new("raw_id_plot"), None, None, None);
}

#[test]
fn test_format_table_value_scientific_precision_and_ieee754_specials() {
    assert_eq!(format_table_value(f64::NAN), "NaN");
    assert_eq!(format_table_value(f64::INFINITY), "inf");
    assert_eq!(format_table_value(f64::NEG_INFINITY), "-inf");
    assert_eq!(format_table_value(0.0), "0.000000");
    assert_eq!(format_table_value(-0.0), "0.000000");
    assert_eq!(format_table_value(3.25e-5), "3.250000e-5");
    assert_eq!(format_table_value(-4.9e-5), "-4.900000e-5");
    assert_eq!(format_table_value(9.99e-4), "9.990000e-4");
    assert_eq!(format_table_value(1e-3), "0.001000");
    assert_eq!(format_table_value(std::f64::consts::E), "2.718282");
    assert_eq!(format_table_value(999_999.25), "999999.250000");
    assert_eq!(format_table_value(1.0e6), "1.000000e6");
}

fn render_dose_image_plot(ctx: &Context, plot_id: Id) {
    egui::CentralPanel::default().show(ctx, |ui| {
        Plot::new("dose_heatmap").id(plot_id).show(ui, |p| {
            p.image(PlotImage::new(
                "dose",
                TextureId::Managed(0),
                PlotPoint::new(0.0, 0.0),
                [20.0, 20.0],
            ));
        });
    });
}

fn access_state(ctx: &Context, id: Id) -> AccessState {
    ctx.data(|d| d.get_temp::<AccessState>(id))
        .unwrap_or_default()
}

#[test]
fn test_plot_image_only_disables_table_view_toolbar_and_context_menu() {
    let (ctx, id) = (Context::default(), Id::new("dose_guard"));
    let show = |c: &Context| render_dose_image_plot(c, id);
    let out = ctx.run(screen_input(), show);
    let pos = find_text_center(&out, "Table View").unwrap();
    let _ = click_at(&ctx, pos, PointerButton::Primary, show);
    assert!(!access_state(&ctx, id).show_table);

    let menu = click_at(&ctx, pos2(640.0, 400.0), PointerButton::Secondary, show);
    if let Some((_, p)) = collect_texts(&menu)
        .into_iter()
        .filter(|(t, _)| t == "Table View")
        .nth(1)
    {
        let _ = click_at(&ctx, p, PointerButton::Primary, show);
    }
    assert!(!access_state(&ctx, id).show_table);
}

#[test]
fn test_open_table_handles_empty_plot_and_escape_dismissal() {
    let (ctx, id) = (Context::default(), Id::new("empty_escape"));
    let show = |c: &Context| render_dose_image_plot(c, id);
    let mut seed = AccessState {
        show_table: true,
        ..Default::default()
    };
    (
        seed.data_table.selected_series,
        seed.data_table.selected_row,
    ) = (5, 99);
    ctx.data_mut(|d| d.insert_temp(id, seed));
    let _ = ctx.run(screen_input(), show);
    let out = ctx.run(screen_input(), show);
    assert!(find_text_center(&out, "No data points are currently available").is_some());
    let clamped = access_state(&ctx, id).data_table;
    assert_eq!((clamped.selected_series, clamped.selected_row), (0, 0));
    let _ = ctx.run(key_input(Key::Escape), show);
    assert!(!access_state(&ctx, id).show_table);
}

fn sample_sir_datasets() -> Vec<(String, Vec<[f64; 2]>, Color32)> {
    let c = Color32::from_rgb(70, 140, 240);
    [
        ("Susceptible", [[0.0, 9990.0], [0.1, 9985.5], [0.2, 4.2e-5]]),
        ("Infected", [[0.0, 10.0], [0.1, 13.25], [0.2, 16.5]]),
        ("Recovered", [[0.0, 0.0], [0.1, 1.25], [0.2, 2.75]]),
    ]
    .into_iter()
    .map(|(n, pts)| (n.to_owned(), pts.to_vec(), c))
    .collect()
}

#[test]
fn test_shared_x_multi_column_arrow_keys_and_tsv_format() {
    let (ctx, id, ds) = (
        Context::default(),
        Id::new("sir_shared_x_test"),
        sample_sir_datasets(),
    );
    let refs: Vec<_> = ds
        .iter()
        .map(|(n, p, c)| TableSeriesRef {
            name: n,
            points: p,
            color: *c,
        })
        .collect();
    assert!(series_share_x(&refs));
    let tsv = format_tsv(&refs);
    assert!(
        tsv.starts_with("X\tSusceptible\tInfected\tRecovered\n")
            && tsv.contains("0.1\t9985.5\t13.25\t1.25\n")
    );

    let (mut open, mut st) = (true, DataTableState::default());
    let _ = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ds);
    let out = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ds);
    for col in ["Susceptible", "Infected", "Recovered", "4.200000e-5"] {
        assert!(find_text_center(&out, col).is_some(), "Missing {col}");
    }
    let _ = run_modal(&ctx, key_input(Key::ArrowDown), id, &mut open, &mut st, &ds);
    let _ = run_modal(&ctx, key_input(Key::ArrowDown), id, &mut open, &mut st, &ds);
    assert_eq!(st.selected_row, 2);
    let _ = run_modal(&ctx, key_input(Key::ArrowUp), id, &mut open, &mut st, &ds);
    assert_eq!(st.selected_row, 1);
}

#[test]
fn test_csv_copy_clipboard_status_and_close_button() {
    let (ctx, id, ds) = (
        Context::default(),
        Id::new("sir_copy_close_test"),
        sample_sir_datasets(),
    );
    let (mut open, mut st) = (true, DataTableState::default());
    let _ = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ds);
    let out1 = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ds);

    let csv_pos = find_text_center(&out1, "Copy CSV").unwrap();
    let out_csv = click_at(&ctx, csv_pos, PointerButton::Primary, |c| {
        show_data_table_modal(c, id, "Data Table - SIR Plot", &mut open, &mut st, &ds);
    });
    assert!(out_csv.platform_output.commands.iter().any(
        |cmd| matches!(cmd, OutputCommand::CopyText(t) if t.starts_with("Dataset,X,Y\n") && t.contains("Susceptible,0.1,9985.5"))
    ));

    ctx.data_mut(|d| {
        d.insert_temp(
            data_table_clipboard_status_id(id),
            ("Copied TSV.".to_owned(), false),
        )
    });
    let out_status = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ds);
    assert!(find_text_center(&out_status, "Copied TSV.").is_some());

    let close_pos = find_text_center(&out_status, "Close").unwrap();
    let _ = click_at(&ctx, close_pos, PointerButton::Primary, |c| {
        show_data_table_modal(c, id, "Data Table - SIR Plot", &mut open, &mut st, &ds);
    });
    assert!(!open);
}

#[test]
fn test_unaligned_series_tabs_and_index_clamping_on_reset() {
    let (ctx, id, c) = (
        Context::default(),
        Id::new("unaligned_tabs_test"),
        Color32::from_rgb(90, 160, 250),
    );
    let full = vec![
        ("Live".to_owned(), vec![[0.0, 1.0], [0.5, 2.0]], c),
        (
            "Snap".to_owned(),
            vec![[0.0, 1.0], [0.25, 1.5], [0.5, 2.1], [0.75, 2.8]],
            c,
        ),
    ];
    let mut st = DataTableState {
        selected_series: 1,
        selected_row: 3,
        ..Default::default()
    };
    let mut open = true;
    let _ = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &full);
    let out = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &full);
    assert!(find_text_center(&out, "Snap (4)").is_some());
    assert!(find_text_center(&out, "Live (2)").is_some());
    assert_eq!((st.selected_series, st.selected_row), (1, 3));

    let shrunk = vec![("Live".to_owned(), vec![[0.0, 1.0]], c)];
    let _ = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &shrunk);
    assert_eq!((st.selected_series, st.selected_row), (0, 0));
}

#[test]
fn test_virtualizes_600006_points_in_bounded_shapes() {
    let (ctx, id, c) = (
        Context::default(),
        Id::new("ode_600k_test"),
        Color32::from_rgb(100, 140, 220),
    );
    let ode_ds: Vec<(String, Vec<[f64; 2]>, Color32)> = (0..6)
        .map(|s| {
            let pts = (0..100_001)
                .map(|i| {
                    [
                        i as f64 * 0.001,
                        (i as f64 * 0.001 + s as f64 * 0.1).exp() * 1e-5,
                    ]
                })
                .collect();
            (format!("Solver {} y(t)", s + 1), pts, c)
        })
        .collect();
    assert_eq!(
        ode_ds.iter().map(|(_, p, _)| p.len()).sum::<usize>(),
        600_006
    );
    let (mut open, mut st) = (true, DataTableState::default());
    let _ = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ode_ds);
    let out = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ode_ds);
    let n = collect_texts(&out).len();
    assert!(n < 300, "Expected O(visible_rows) text shapes, got {n}");
}

#[test]
fn test_scrolls_wide_headers_horizontally_across_48_series() {
    let (ctx, id, c) = (
        Context::default(),
        Id::new("trace_48_test"),
        Color32::from_rgb(100, 160, 210),
    );
    let ds: Vec<_> = (0..48)
        .map(|i| {
            (
                format!("Snapshot #{} Trace", i + 1),
                vec![[0.0, i as f64], [1.0, 1.0]],
                c,
            )
        })
        .collect();
    let (mut open, mut st) = (true, DataTableState::default());
    let _ = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ds);
    let initial = run_modal(&ctx, screen_input(), id, &mut open, &mut st, &ds);
    let pos = find_text_center(&initial, "Snapshot #1 Trace").expect("first series header visible");
    assert!(find_text_center(&initial, "Snapshot #48 Trace").is_none());

    let mut wheel = screen_input();
    wheel.events = vec![
        Event::PointerMoved(pos),
        Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: vec2(-8000.0, 0.0),
            modifiers: Modifiers::NONE,
        },
    ];
    let _ = run_modal(&ctx, wheel, id, &mut open, &mut st, &ds);
    let mut scrolled = initial;
    for _ in 0..15 {
        let mut step = screen_input();
        step.events.push(Event::PointerMoved(pos));
        scrolled = run_modal(&ctx, step, id, &mut open, &mut st, &ds);
    }
    assert!(find_text_center(&scrolled, "Snapshot #48 Trace").is_some());
}

fn sample_pois() -> Vec<crate::poi::CriticalPoint> {
    use crate::poi::{CriticalPoint, CriticalPointKind};
    vec![
        CriticalPoint {
            kind: CriticalPointKind::LocalMax,
            point: [0.1, 13.25],
            primary_series: "Infected".to_owned(),
            secondary_series: None,
            item_id: Id::new("poi_infected"),
        },
        CriticalPoint {
            kind: CriticalPointKind::Intersection,
            point: [0.2, 2.75],
            primary_series: "Susceptible Trajectory".to_owned(),
            secondary_series: Some("Recovered Envelope".to_owned()),
            item_id: Id::new("poi_intersect"),
        },
    ]
}

fn run_modal_with_pois(
    ctx: &Context,
    raw: RawInput,
    id: Id,
    open: &mut bool,
    st: &mut DataTableState,
    ds: &[(String, Vec<[f64; 2]>, Color32)],
    pois: &[crate::poi::CriticalPoint],
) -> FullOutput {
    ctx.run(raw, |c| {
        crate::data_table::show_data_table_modal_with_pois(
            c,
            id,
            "Data Table - SIR Plot",
            open,
            st,
            ds,
            pois,
        );
    })
}

fn high_contrast_visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();
    v.window_fill = Color32::BLACK;
    v.panel_fill = Color32::BLACK;
    v.extreme_bg_color = Color32::BLACK;
    v.faint_bg_color = Color32::from_rgb(16, 16, 16);
    v.selection.bg_fill = Color32::from_rgb(0, 90, 158);
    v.selection.stroke = egui::Stroke::new(2.0_f32, Color32::from_rgb(255, 255, 0));
    v
}

fn collect_circles_and_strokes(output: &FullOutput) -> (Vec<(Pos2, Color32)>, Vec<egui::Stroke>) {
    let (mut circles, mut strokes) = (Vec::new(), Vec::new());
    let mut stack: Vec<&egui::epaint::Shape> = output.shapes.iter().map(|c| &c.shape).collect();
    while let Some(shape) = stack.pop() {
        match shape {
            egui::epaint::Shape::Circle(c) if c.fill != Color32::TRANSPARENT => {
                circles.push((c.center, c.fill));
            }
            egui::epaint::Shape::Rect(r) if r.stroke.width >= 2.0 => {
                strokes.push(r.stroke);
            }
            egui::epaint::Shape::Vec(v) => stack.extend(v.iter().rev()),
            _ => {}
        }
    }
    (circles, strokes)
}

fn tab_to_ak_node(
    ctx: &Context,
    role: egui::accesskit::Role,
    needle: &str,
    mut step: impl FnMut(RawInput) -> FullOutput,
) -> (Id, FullOutput) {
    let initial = step(screen_input());
    let target_raw = initial
        .platform_output
        .accesskit_update
        .as_ref()
        .and_then(|ak| {
            ak.nodes.iter().find_map(|(nid, node)| {
                (node.role() == role && node.label().is_some_and(|l| l.contains(needle)))
                    .then_some(nid.0)
            })
        })
        .unwrap_or_else(|| panic!("Missing AccessKit {role:?} matching '{needle}'"));
    for _ in 0..32 {
        if let Some(fid) = ctx.memory(|m| m.focused())
            && fid.value() == target_raw
        {
            let settled = step(screen_input());
            return (fid, settled);
        }
        let _ = step(key_input(Key::Tab));
    }
    panic!("Tab navigation never reached {role:?} '{needle}'");
}

fn polite_live(ctx: &Context) -> Option<String> {
    ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_message")))
}

fn assertive_live(ctx: &Context) -> Option<String> {
    ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_assertive_message")))
}

fn assert_table_hierarchy(
    ak: &egui::accesskit::TreeUpdate,
    table_label: &str,
    expected_rows: usize,
    expected_cols: usize,
    expected_first_row_sub: &str,
) {
    use egui::accesskit::Role;
    let map: std::collections::HashMap<_, _> = ak.nodes.iter().map(|(id, n)| (*id, n)).collect();
    let (_, tbl) = ak
        .nodes
        .iter()
        .find(|(_, n)| n.role() == Role::Table && n.label() == Some(table_label))
        .expect("Role::Table node");
    assert_eq!(
        (tbl.row_count(), tbl.column_count()),
        (Some(expected_rows), Some(expected_cols))
    );
    let rows: Vec<_> = tbl
        .children()
        .iter()
        .filter_map(|cid| map.get(cid).copied())
        .filter(|n| n.role() == Role::Row)
        .collect();
    assert_eq!(rows.len(), expected_rows);
    let hdr_roles: Vec<_> = rows[0]
        .children()
        .iter()
        .filter_map(|cid| map.get(cid).map(|n| n.role()))
        .collect();
    assert_eq!(hdr_roles, vec![Role::ColumnHeader; expected_cols]);
    assert_eq!(rows[1].is_selected(), Some(true));
    assert!(
        rows[1]
            .label()
            .is_some_and(|l| l.contains(expected_first_row_sub))
    );
    let mut expected_data_roles = vec![Role::Cell; expected_cols];
    expected_data_roles[0] = Role::RowHeader;
    let data_roles: Vec<_> = rows[1]
        .children()
        .iter()
        .filter_map(|cid| map.get(cid).map(|n| n.role()))
        .collect();
    assert_eq!(data_roles, expected_data_roles);
}

#[test]
fn test_series_and_poi_tables_expose_accesskit_roles_and_parent_hierarchy() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let (id, ds, pois) = (
        Id::new("ak_hierarchy_test"),
        sample_sir_datasets(),
        sample_pois(),
    );
    let (mut open, mut st) = (true, DataTableState::default());
    let _ = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let out = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree");
    assert_table_hierarchy(
        ak,
        "Series data table",
        4,
        5,
        "Row 1 of 3: X = 0.000000, Susceptible = 9990.000000, Infected = 10.000000, Recovered = 0.000000",
    );

    st.view_mode = crate::data_table::DataTableViewMode::CriticalPoints;
    let out_poi = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let ak_poi = out_poi.platform_output.accesskit_update.as_ref().unwrap();
    assert_table_hierarchy(
        ak_poi,
        "Critical points table",
        3,
        5,
        "Critical point 1 of 2: Local Max (Infected)",
    );
}

#[test]
fn test_high_contrast_poi_dots_and_row_focus_rings_meet_3_to_1_contrast() {
    use crate::a11y::contrast_ratio;
    let visuals = high_contrast_visuals();
    let sel_bg = visuals.selection.bg_fill;
    assert!(contrast_ratio(Color32::from_rgb(128, 134, 145), sel_bg) < 2.0);

    let ctx = Context::default();
    ctx.enable_accesskit();
    ctx.set_visuals(visuals.clone());
    let (id, ds, pois) = (
        Id::new("hc_contrast_test"),
        sample_sir_datasets(),
        sample_pois(),
    );
    let mut st = DataTableState {
        view_mode: crate::data_table::DataTableViewMode::CriticalPoints,
        ..Default::default()
    };
    let mut open = true;
    let _ = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let (_, out_focused) = tab_to_ak_node(
        &ctx,
        egui::accesskit::Role::Row,
        "Critical point 1 of 2",
        |raw| run_modal_with_pois(&ctx, raw, id, &mut open, &mut st, &ds, &pois),
    );
    let (mut circles, strokes) = collect_circles_and_strokes(&out_focused);
    circles.sort_by(|a, b| a.0.y.total_cmp(&b.0.y));
    assert!(circles.len() >= 2);
    assert!(
        contrast_ratio(circles[0].1, sel_bg) >= 3.0,
        "Selected HighContrast POI dot {:?} failed 3:1 on {:?}",
        circles[0].1,
        sel_bg
    );
    assert!(contrast_ratio(circles[1].1, visuals.faint_bg_color) >= 3.0);
    assert!(
        strokes
            .iter()
            .any(|s| contrast_ratio(s.color, sel_bg) >= 3.0),
        "Focused POI row must paint >= 3:1 focus ring"
    );
}

#[test]
fn test_shared_x_and_poi_live_announcements_on_keys_clicks_and_clipboard() {
    use egui::accesskit::{Live, Role};
    let ctx = Context::default();
    ctx.enable_accesskit();
    let (id, ds, pois) = (
        Id::new("live_announce_test"),
        sample_sir_datasets(),
        sample_pois(),
    );
    let (mut open, mut st) = (true, DataTableState::default());
    let _ = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let out = run_modal_with_pois(
        &ctx,
        key_input(Key::ArrowDown),
        id,
        &mut open,
        &mut st,
        &ds,
        &pois,
    );
    assert_eq!(
        polite_live(&ctx).as_deref(),
        Some(
            "Row 2 of 3: X = 0.100000, Susceptible = 9985.500000, Infected = 13.250000, Recovered = 1.250000"
        )
    );
    let row3_pos = find_text_center(&out, "4.200000e-5").unwrap();
    let _ = click_at(&ctx, row3_pos, PointerButton::Primary, |c| {
        crate::data_table::show_data_table_modal_with_pois(
            c,
            id,
            "Data Table",
            &mut open,
            &mut st,
            &ds,
            &pois,
        );
    });
    assert_eq!(st.selected_row, 2);
    assert!(
        polite_live(&ctx)
            .as_deref()
            .is_some_and(|m| m.contains("Row 3 of 3")
                && m.contains("Infected = 16.500000")
                && m.contains("Recovered = 2.750000"))
    );

    let csv_pos = find_text_center(&out, "Copy CSV").unwrap();
    let out_copy = click_at(&ctx, csv_pos, PointerButton::Primary, |c| {
        crate::data_table::show_data_table_modal_with_pois(
            c,
            id,
            "Data Table",
            &mut open,
            &mut st,
            &ds,
            &pois,
        );
    });
    assert!(
        polite_live(&ctx)
            .as_deref()
            .is_some_and(|m| m.contains("CSV dataset"))
    );
    let ak_copy = out_copy.platform_output.accesskit_update.as_ref().unwrap();
    assert!(
        ak_copy
            .nodes
            .iter()
            .any(|(_, n)| n.role() == Role::Status && n.live() == Some(Live::Polite))
    );

    ctx.data_mut(|d| {
        d.insert_temp(
            data_table_clipboard_status_id(id),
            ("Clipboard copy failed: denied".to_owned(), true),
        );
    });
    let out_err = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    assert_eq!(
        assertive_live(&ctx).as_deref(),
        Some("Clipboard copy failed: denied")
    );
    let ak_err = out_err.platform_output.accesskit_update.as_ref().unwrap();
    assert!(
        ak_err
            .nodes
            .iter()
            .any(|(_, n)| n.role() == Role::Status && n.live() == Some(Live::Assertive))
    );

    st.view_mode = crate::data_table::DataTableViewMode::CriticalPoints;
    let out_poi = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let poi2_pos = find_text_center(&out_poi, "Intersection").unwrap();
    let _ = click_at(&ctx, poi2_pos, PointerButton::Primary, |c| {
        crate::data_table::show_data_table_modal_with_pois(
            c,
            id,
            "Data Table",
            &mut open,
            &mut st,
            &ds,
            &pois,
        );
    });
    assert_eq!(st.selected_poi_row, 1);
    assert!(
        polite_live(&ctx)
            .as_deref()
            .is_some_and(|m| m.contains("Critical point 2 of 2: Intersection"))
    );
}

#[test]
fn test_mode_and_series_tablists_arrow_keys_and_live_announcements() {
    use egui::accesskit::Role;
    let ctx = Context::default();
    ctx.enable_accesskit();
    let c = Color32::from_rgb(90, 160, 250);
    let ds = vec![
        ("Live".to_owned(), vec![[0.0, 1.0], [0.5, 2.0]], c),
        (
            "Snap".to_owned(),
            vec![[0.0, 1.0], [0.25, 1.5], [0.5, 2.1], [0.75, 2.8]],
            c,
        ),
    ];
    let pois = sample_pois();
    let (id, mut open, mut st) = (
        Id::new("tablist_a11y_test"),
        true,
        DataTableState::default(),
    );
    let _ = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let out = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let map: std::collections::HashMap<_, _> = ak.nodes.iter().map(|(id, n)| (*id, n)).collect();
    let tablists: Vec<_> = ak
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == Role::TabList)
        .collect();
    assert_eq!(tablists.len(), 2, "Expected mode and series TabLists");
    for (_, tl) in &tablists {
        assert_eq!(tl.children().len(), 2);
        assert!(
            tl.children()
                .iter()
                .all(|cid| map.get(cid).is_some_and(|n| n.role() == Role::Tab))
        );
    }

    let _ = tab_to_ak_node(&ctx, Role::Tab, "Live (2)", |raw| {
        run_modal_with_pois(&ctx, raw, id, &mut open, &mut st, &ds, &pois)
    });
    let _ = run_modal_with_pois(
        &ctx,
        key_input(Key::ArrowRight),
        id,
        &mut open,
        &mut st,
        &ds,
        &pois,
    );
    assert_eq!(st.selected_series, 1);
    assert_eq!(
        polite_live(&ctx).as_deref(),
        Some("Selected series tab: Snap (4)")
    );

    let _ = tab_to_ak_node(&ctx, Role::Tab, "Series Data (2)", |raw| {
        run_modal_with_pois(&ctx, raw, id, &mut open, &mut st, &ds, &pois)
    });
    let _ = run_modal_with_pois(
        &ctx,
        key_input(Key::ArrowRight),
        id,
        &mut open,
        &mut st,
        &ds,
        &pois,
    );
    assert_eq!(
        st.view_mode,
        crate::data_table::DataTableViewMode::CriticalPoints
    );
    assert_eq!(
        polite_live(&ctx).as_deref(),
        Some("Selected Critical Points (2) tab")
    );
}

#[test]
fn test_disabled_table_view_and_empty_modal_copy_buttons_stay_focusable_and_escape_safe() {
    use egui::{Tooltip, accesskit::Role};
    let (ctx, plot_id) = (Context::default(), Id::new("disabled_table_btn_a11y"));
    ctx.enable_accesskit();
    let show = |c: &Context| render_dose_image_plot(c, plot_id);
    let out = ctx.run(screen_input(), show);
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let (_, tbl_node) = ak
        .nodes
        .iter()
        .find(|(_, n)| n.role() == Role::Button && n.label() == Some("Table View"))
        .expect("Disabled Table View button node");
    assert!(tbl_node.is_disabled());
    let (tbl_btn_id, _) =
        tab_to_ak_node(&ctx, Role::Button, "Table View", |raw| ctx.run(raw, show));
    assert!(Tooltip::was_tooltip_open_last_frame(&ctx, tbl_btn_id));
    let _ = ctx.run(key_input(Key::Escape), show);
    assert!(!Tooltip::was_tooltip_open_last_frame(&ctx, tbl_btn_id));

    let (modal_id, mut open, mut st) =
        (Id::new("empty_copy_a11y"), true, DataTableState::default());
    let _ = run_modal_with_pois(&ctx, screen_input(), modal_id, &mut open, &mut st, &[], &[]);
    let out_empty =
        run_modal_with_pois(&ctx, screen_input(), modal_id, &mut open, &mut st, &[], &[]);
    let copy_pos = find_text_center(&out_empty, "Copy TSV").unwrap();
    let clicked = click_at(&ctx, copy_pos, PointerButton::Primary, |c| {
        crate::data_table::show_data_table_modal_with_pois(
            c,
            modal_id,
            "Data Table",
            &mut open,
            &mut st,
            &[],
            &[],
        );
    });
    assert!(
        !clicked
            .platform_output
            .commands
            .iter()
            .any(|cmd| matches!(cmd, OutputCommand::CopyText(_)))
    );
    let (copy_id, _) = tab_to_ak_node(&ctx, Role::Button, "Copy TSV", |raw| {
        run_modal_with_pois(&ctx, raw, modal_id, &mut open, &mut st, &[], &[])
    });
    assert!(Tooltip::was_tooltip_open_last_frame(&ctx, copy_id));
    let _ = run_modal_with_pois(
        &ctx,
        key_input(Key::Escape),
        modal_id,
        &mut open,
        &mut st,
        &[],
        &[],
    );
    assert!(!Tooltip::was_tooltip_open_last_frame(&ctx, copy_id));
    assert!(open, "First Escape must dismiss disabled-copy tooltip only");
    let _ = run_modal_with_pois(
        &ctx,
        key_input(Key::Escape),
        modal_id,
        &mut open,
        &mut st,
        &[],
        &[],
    );
    assert!(!open, "Second Escape must close Data Table modal");
}

#[test]
fn test_header_and_poi_series_cell_tooltips_dismiss_on_escape_without_closing_modal() {
    use egui::{Tooltip, accesskit::Role};
    let ctx = Context::default();
    ctx.enable_accesskit();
    let (id, ds, pois) = (
        Id::new("cell_tooltip_escape_test"),
        sample_sir_datasets(),
        sample_pois(),
    );
    let (mut open, mut st) = (true, DataTableState::default());
    let _ = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let (hdr_id, _) = tab_to_ak_node(&ctx, Role::ColumnHeader, "Susceptible", |raw| {
        run_modal_with_pois(&ctx, raw, id, &mut open, &mut st, &ds, &pois)
    });
    assert!(Tooltip::was_tooltip_open_last_frame(&ctx, hdr_id));
    let _ = run_modal_with_pois(
        &ctx,
        key_input(Key::Escape),
        id,
        &mut open,
        &mut st,
        &ds,
        &pois,
    );
    assert!(!Tooltip::was_tooltip_open_last_frame(&ctx, hdr_id));
    assert!(open, "Escape must dismiss ColumnHeader tooltip first");

    st.view_mode = crate::data_table::DataTableViewMode::CriticalPoints;
    let _ = run_modal_with_pois(&ctx, screen_input(), id, &mut open, &mut st, &ds, &pois);
    let (series_cell, _) = tab_to_ak_node(&ctx, Role::Cell, "Susceptible Trajectory", |raw| {
        run_modal_with_pois(&ctx, raw, id, &mut open, &mut st, &ds, &pois)
    });
    assert!(Tooltip::was_tooltip_open_last_frame(&ctx, series_cell));
    let _ = run_modal_with_pois(
        &ctx,
        key_input(Key::Escape),
        id,
        &mut open,
        &mut st,
        &ds,
        &pois,
    );
    assert!(!Tooltip::was_tooltip_open_last_frame(&ctx, series_cell));
    assert!(open, "Escape must dismiss POI Series cell tooltip first");
    let _ = run_modal_with_pois(
        &ctx,
        key_input(Key::Escape),
        id,
        &mut open,
        &mut st,
        &ds,
        &pois,
    );
    assert!(!open, "Next Escape with no open tooltip closes modal");
}
