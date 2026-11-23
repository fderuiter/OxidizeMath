use super::{key_input, screen_input};
use crate::ModalLifecycle;
use egui::{Context, Event, Id, Key, RawInput};

fn run_masked_stacked_modals(
    ctx: &Context,
    raw: RawInput,
    lower_open: &mut bool,
    upper_open: &mut bool,
    bg_text: &mut String,
    lower_text: &mut String,
    upper_text: &mut String,
) {
    let _ = ctx.run(raw, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            crate::with_masked_pane_input(ctx, false, || {
                let bg = ui.text_edit_singleline(bg_text);
                if !*lower_open && !*upper_open {
                    bg.request_focus();
                }
                ModalLifecycle::new("lower_modal")
                    .title("Lower Modal")
                    .show(ctx, lower_open, |ui, fctx| {
                        let r = ui.text_edit_singleline(lower_text);
                        if fctx.just_opened {
                            r.request_focus();
                        }
                    });
            });
        });
        ModalLifecycle::new("upper_modal")
            .title("Upper Modal")
            .show(ctx, upper_open, |ui, fctx| {
                let r = ui.text_edit_singleline(upper_text);
                if fctx.just_opened {
                    r.request_focus();
                }
            });
    });
}

#[test]
fn test_with_masked_pane_input_unmasks_strictly_topmost_foreground_modal() {
    let ctx = Context::default();
    let (mut lower_open, mut upper_open) = (true, false);
    let (mut bg, mut lower, mut upper) = (String::new(), String::new(), String::new());

    run_masked_stacked_modals(
        &ctx,
        screen_input(),
        &mut lower_open,
        &mut upper_open,
        &mut bg,
        &mut lower,
        &mut upper,
    );
    let mut typed = screen_input();
    typed.events = vec![
        Event::Text("alpha".into()),
        Event::Ime(egui::ImeEvent::Enabled),
        Event::Ime(egui::ImeEvent::Commit("!".into())),
        Event::Paste("+paste".into()),
    ];
    run_masked_stacked_modals(
        &ctx,
        typed,
        &mut lower_open,
        &mut upper_open,
        &mut bg,
        &mut lower,
        &mut upper,
    );
    assert_eq!(lower, "alpha!+paste");
    assert!(bg.is_empty() && upper.is_empty());

    upper_open = true;
    run_masked_stacked_modals(
        &ctx,
        screen_input(),
        &mut lower_open,
        &mut upper_open,
        &mut bg,
        &mut lower,
        &mut upper,
    );
    let mut covered_events = screen_input();
    covered_events.events = vec![Event::Text("TOP".into()), Event::Paste("_CLIP".into())];
    run_masked_stacked_modals(
        &ctx,
        covered_events,
        &mut lower_open,
        &mut upper_open,
        &mut bg,
        &mut lower,
        &mut upper,
    );
    assert_eq!(upper, "TOP_CLIP");
    assert_eq!(lower, "alpha!+paste", "Covered modal must remain masked");
    assert!(bg.is_empty(), "Background pane widget must remain masked");
}

type SampleTableSeries2D = Vec<(String, Vec<[f64; 2]>, egui::Color32)>;

fn sample_2d_and_3d_table_data() -> (
    SampleTableSeries2D,
    Vec<crate::poi::CriticalPoint>,
    Vec<crate::Plot3DSeries>,
) {
    let ds_2d = vec![(
        "S".to_string(),
        vec![[0.0, 1.0], [1.0, 2.0], [2.0, 0.5]],
        egui::Color32::RED,
    )];
    let pois = vec![
        crate::poi::CriticalPoint {
            kind: crate::poi::CriticalPointKind::LocalMax,
            point: [1.0, 2.0],
            primary_series: "S".to_string(),
            secondary_series: None,
            item_id: Id::new("s_item"),
        },
        crate::poi::CriticalPoint {
            kind: crate::poi::CriticalPointKind::LocalMin,
            point: [2.0, 0.5],
            primary_series: "S".to_string(),
            secondary_series: None,
            item_id: Id::new("s_item"),
        },
    ];
    let ds_3d = vec![crate::Plot3DSeries {
        name: "T".to_string(),
        projected: vec![[0.0, 1.0], [1.0, 2.0]],
        world: vec![[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]],
        color: egui::Color32::BLUE,
    }];
    (ds_2d, pois, ds_3d)
}

#[derive(Default)]
struct CoveredTablesHarness {
    open_2d: bool,
    state_2d: crate::data_table::DataTableState,
    open_3d: bool,
    state_3d: crate::data_table::DataTableState,
    cover_open: bool,
}

impl CoveredTablesHarness {
    fn step(&mut self, ctx: &Context, raw: RawInput) {
        let (ds_2d, pois, ds_3d) = sample_2d_and_3d_table_data();
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |_ui| {
                crate::with_masked_pane_input(ctx, false, || {
                    crate::data_table::show_data_table_modal_with_pois(
                        ctx,
                        Id::new("dt_2d"),
                        "Data Table 2D",
                        &mut self.open_2d,
                        &mut self.state_2d,
                        &ds_2d,
                        &pois,
                    );
                    crate::data_table::show_data_table_modal_3d(
                        ctx,
                        Id::new("dt_3d"),
                        "Data Table 3D",
                        &mut self.open_3d,
                        &mut self.state_3d,
                        &ds_3d,
                    );
                });
            });
            ModalLifecycle::new("covering_modal").title("Cover").show(
                ctx,
                &mut self.cover_open,
                |ui, _| {
                    let _ = ui.button("Foreground Cover");
                },
            );
        });
    }
}

#[test]
fn test_data_table_2d_poi_and_3d_ignore_arrows_and_copy_when_covered() {
    let ctx = Context::default();
    let mut h = CoveredTablesHarness {
        open_2d: true,
        ..Default::default()
    };

    h.step(&ctx, screen_input());
    h.step(&ctx, key_input(Key::ArrowDown));
    assert_eq!(h.state_2d.selected_row, 1);

    h.cover_open = true;
    h.step(&ctx, screen_input());
    let mut arrow_and_copy = key_input(Key::ArrowDown);
    arrow_and_copy.events.push(Event::Copy);
    h.step(&ctx, arrow_and_copy);
    assert_eq!(
        h.state_2d.selected_row, 1,
        "Covered 2D Data Table must ignore ArrowDown"
    );

    h.state_2d.view_mode = crate::data_table::DataTableViewMode::CriticalPoints;
    h.step(&ctx, key_input(Key::ArrowDown));
    assert_eq!(
        h.state_2d.selected_poi_row, 0,
        "Covered POI Data Table must ignore ArrowDown"
    );

    h.open_2d = false;
    h.open_3d = true;
    h.cover_open = false;
    h.step(&ctx, screen_input());
    h.step(&ctx, key_input(Key::ArrowDown));
    assert_eq!(h.state_3d.selected_row, 1);

    h.cover_open = true;
    h.step(&ctx, screen_input());
    h.step(&ctx, key_input(Key::ArrowUp));
    assert_eq!(
        h.state_3d.selected_row, 1,
        "Covered 3D Data Table must ignore ArrowUp"
    );
}
