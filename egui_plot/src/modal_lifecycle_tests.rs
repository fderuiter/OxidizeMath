use crate::{AccessState, Legend, Line, ModalLifecycle, Plot, PlotMemory, items::SavedCallout};
use egui::{
    Context, Event, FullOutput, Id, Key, Modifiers, PointerButton, Pos2, RawInput, Rect, vec2,
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

fn click_inputs(pos: Pos2) -> (RawInput, RawInput) {
    let mut down = screen_input();
    down.events = vec![
        Event::PointerMoved(pos),
        Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed: true,
            modifiers: Modifiers::NONE,
        },
    ];
    let mut up = screen_input();
    up.events = vec![Event::PointerButton {
        pos,
        button: PointerButton::Primary,
        pressed: false,
        modifiers: Modifiers::NONE,
    }];
    (down, up)
}

fn find_text_center(output: &FullOutput, needle: &str) -> Option<Pos2> {
    let mut stack: Vec<&egui::epaint::Shape> = output.shapes.iter().map(|c| &c.shape).collect();
    while let Some(shape) = stack.pop() {
        match shape {
            egui::epaint::Shape::Text(text_shape) => {
                if text_shape.galley.job.text.contains(needle) {
                    return Some(text_shape.pos + text_shape.galley.rect.center().to_vec2());
                }
            }
            egui::epaint::Shape::Vec(shapes) => {
                for inner in shapes.iter().rev() {
                    stack.push(inner);
                }
            }
            _ => {}
        }
    }
    None
}

#[test]
fn test_modal_lifecycle_escape_tab_containment_and_focus_restore() {
    let ctx = Context::default();
    let mut open = false;
    let mut trigger_id = Id::NULL;
    let mut bg_other_id = Id::NULL;

    let run = |raw: RawInput, open: &mut bool, trigger_id: &mut Id, bg_other_id: &mut Id| {
        ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                *trigger_id = ui.button("Open Modal Trigger").id;
                *bg_other_id = ui.button("Background Sibling").id;
                ModalLifecycle::new("unit_test_modal")
                    .title("Test Dialog")
                    .default_width(280.0)
                    .show(ctx, open, |ui, frame_ctx| {
                        let _ = ui.button("Action One");
                        if ui.button("Close").clicked() {
                            frame_ctx.close();
                        }
                    });
            });
        })
    };

    let _ = run(screen_input(), &mut open, &mut trigger_id, &mut bg_other_id);
    ctx.memory_mut(|m| m.request_focus(trigger_id));
    let _ = run(screen_input(), &mut open, &mut trigger_id, &mut bg_other_id);
    assert_eq!(ctx.memory(|m| m.focused()), Some(trigger_id));

    open = true;
    let _ = run(screen_input(), &mut open, &mut trigger_id, &mut bg_other_id);
    let _ = run(screen_input(), &mut open, &mut trigger_id, &mut bg_other_id);
    let initial_modal_focus = ctx.memory(|m| m.focused());
    assert!(initial_modal_focus.is_some());
    assert_ne!(initial_modal_focus, Some(trigger_id));
    assert_ne!(initial_modal_focus, Some(bg_other_id));

    for _ in 0..6 {
        let _ = run(
            key_input(Key::Tab),
            &mut open,
            &mut trigger_id,
            &mut bg_other_id,
        );
        let focused = ctx.memory(|m| m.focused());
        assert_ne!(
            focused,
            Some(trigger_id),
            "Tab must not escape modal to trigger button"
        );
        assert_ne!(
            focused,
            Some(bg_other_id),
            "Tab must not escape modal to background sibling"
        );
    }

    let _ = run(
        key_input(Key::Escape),
        &mut open,
        &mut trigger_id,
        &mut bg_other_id,
    );
    assert!(!open, "Escape must dismiss ModalLifecycle");
    assert_eq!(
        ctx.memory(|m| m.focused()),
        Some(trigger_id),
        "Focus must return to trigger after Escape dismissal"
    );
}

#[derive(Clone, Copy)]
enum PlotDialogKind {
    DataTable,
    ExportPlot,
    PlotSettings,
    AddCallout,
    EditCallout,
}

fn set_plot_dialog_open(ctx: &Context, plot_id: Id, kind: PlotDialogKind, is_open: bool) {
    if matches!(kind, PlotDialogKind::EditCallout)
        && is_open
        && let Some(mut mem) = PlotMemory::load(ctx, plot_id)
        && mem.callouts.is_empty()
    {
        mem.callouts.push(SavedCallout {
            point: [1.0, 2.0],
            offset: [20.0, -20.0],
            label: "Peak".to_string(),
        });
        mem.store(ctx, plot_id);
    }
    ctx.data_mut(|d| {
        let mut state = d.get_temp::<AccessState>(plot_id).unwrap_or_default();
        match kind {
            PlotDialogKind::DataTable => state.show_table = is_open,
            PlotDialogKind::ExportPlot => state.export_modal.open = is_open,
            PlotDialogKind::PlotSettings => state.show_settings_modal = is_open,
            PlotDialogKind::AddCallout => state.show_add_callout_modal = is_open,
            PlotDialogKind::EditCallout => {
                state.show_edit_callout_modal = is_open;
                state.editing_callout_idx = Some(0);
                state.edit_callout_text = "Peak".to_string();
            }
        }
        d.insert_temp(plot_id, state);
    });
}

fn is_plot_dialog_open(ctx: &Context, plot_id: Id, kind: PlotDialogKind) -> bool {
    let state = ctx
        .data(|d| d.get_temp::<AccessState>(plot_id))
        .unwrap_or_default();
    match kind {
        PlotDialogKind::DataTable => state.show_table,
        PlotDialogKind::ExportPlot => state.export_modal.open,
        PlotDialogKind::PlotSettings => state.show_settings_modal,
        PlotDialogKind::AddCallout => state.show_add_callout_modal,
        PlotDialogKind::EditCallout => state.show_edit_callout_modal,
    }
}

fn verify_plot_dialog_lifecycle(kind: PlotDialogKind, salt: &str) {
    let ctx = Context::default();
    let plot_id = Id::new(salt);
    let mut ext_trigger = Id::NULL;

    let run = |raw: RawInput, ext_trigger: &mut Id| {
        ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                *ext_trigger = ui.button("External Trigger").id;
                Plot::new(plot_id)
                    .id(plot_id)
                    .legend(Legend::default())
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("Curve", vec![[0.0, 1.0], [1.0, 3.0]]));
                    });
            });
        })
    };

    let _ = run(screen_input(), &mut ext_trigger);
    ctx.memory_mut(|m| m.request_focus(ext_trigger));
    let _ = run(screen_input(), &mut ext_trigger);
    assert_eq!(ctx.memory(|m| m.focused()), Some(ext_trigger));

    set_plot_dialog_open(&ctx, plot_id, kind, true);
    let _ = run(screen_input(), &mut ext_trigger);
    let _ = run(screen_input(), &mut ext_trigger);
    assert!(is_plot_dialog_open(&ctx, plot_id, kind));
    let initial_focus = ctx.memory(|m| m.focused());
    assert!(
        initial_focus.is_some(),
        "{salt} must place initial keyboard focus inside the modal"
    );
    assert_ne!(initial_focus, Some(ext_trigger));

    for _ in 0..8 {
        let _ = run(key_input(Key::Tab), &mut ext_trigger);
        assert_ne!(
            ctx.memory(|m| m.focused()),
            Some(ext_trigger),
            "Tab leaked outside {salt} modal to background button"
        );
    }

    let _ = run(key_input(Key::Escape), &mut ext_trigger);
    assert!(
        !is_plot_dialog_open(&ctx, plot_id, kind),
        "Escape must close {salt}"
    );
    assert_eq!(
        ctx.memory(|m| m.focused()),
        Some(ext_trigger),
        "Focus must return to trigger on {salt} close"
    );
}

#[test]
fn test_all_five_egui_plot_dialogs_escape_tab_and_focus_restore() {
    verify_plot_dialog_lifecycle(PlotDialogKind::DataTable, "lifecycle_data_table");
    verify_plot_dialog_lifecycle(PlotDialogKind::ExportPlot, "lifecycle_export_plot");
    verify_plot_dialog_lifecycle(PlotDialogKind::PlotSettings, "lifecycle_plot_settings");
    verify_plot_dialog_lifecycle(PlotDialogKind::AddCallout, "lifecycle_add_callout");
    verify_plot_dialog_lifecycle(PlotDialogKind::EditCallout, "lifecycle_edit_callout");
}

#[test]
fn test_data_table_close_button_pointer_dismissal_and_focus_return() {
    let ctx = Context::default();
    let plot_id = Id::new("data_table_close_btn_plot");
    let mut trigger_btn = Id::NULL;

    let run = |raw: RawInput, trigger_btn: &mut Id| {
        ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                *trigger_btn = ui.button("Toolbar Trigger").id;
                Plot::new(plot_id).id(plot_id).show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Wave", vec![[0.0, 0.0], [2.0, 4.0]]));
                });
            });
        })
    };

    let _ = run(screen_input(), &mut trigger_btn);
    ctx.memory_mut(|m| m.request_focus(trigger_btn));
    let _ = run(screen_input(), &mut trigger_btn);

    set_plot_dialog_open(&ctx, plot_id, PlotDialogKind::DataTable, true);
    let _ = run(screen_input(), &mut trigger_btn);
    let out = run(screen_input(), &mut trigger_btn);
    let close_pos = find_text_center(&out, "Close").expect("Data Table must render Close button");

    let (down, up) = click_inputs(close_pos);
    let _ = run(down, &mut trigger_btn);
    let _ = run(up, &mut trigger_btn);

    assert!(
        !is_plot_dialog_open(&ctx, plot_id, PlotDialogKind::DataTable),
        "Clicking Close must dismiss Data Table"
    );
    assert_eq!(
        ctx.memory(|m| m.focused()),
        Some(trigger_btn),
        "Clicking Close must restore focus to trigger"
    );
}

#[test]
fn test_legend_checkbox_toggles_via_space_and_enter_keys() {
    let ctx = Context::default();
    let plot_id = Id::new("legend_keyboard_toggle_plot");

    let run = |raw: RawInput| {
        ctx.run(raw, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                Plot::new(plot_id)
                    .id(plot_id)
                    .legend(Legend::default())
                    .show_navigation_controls(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(Line::new("Alpha Series", vec![[0.0, 1.0], [1.0, 2.0]]));
                    });
            });
        })
    };

    let _ = run(screen_input());
    // Tab order inside Plot::show: 1=plot_rect, 2=x_axis, 3=y_axis, 4=wrench, 5=LegendEntry.
    for _ in 0..5 {
        let _ = run(key_input(Key::Tab));
    }
    let legend_id = ctx
        .memory(|m| m.focused())
        .expect("5th Tab must focus LegendEntry checkbox");

    // Space hides the series.
    let _ = run(key_input(Key::Space));
    let mem = PlotMemory::load(&ctx, plot_id).unwrap();
    assert!(
        !mem.hidden_items.is_empty(),
        "Space must toggle LegendEntry checkbox off"
    );
    assert_eq!(ctx.memory(|m| m.focused()), Some(legend_id));

    // Enter re-enables the series.
    let _ = run(key_input(Key::Enter));
    let mem_after_enter = PlotMemory::load(&ctx, plot_id).unwrap();
    assert!(
        mem_after_enter.hidden_items.is_empty(),
        "Enter must re-enable the hidden legend series"
    );
}
