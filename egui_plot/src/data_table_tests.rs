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
