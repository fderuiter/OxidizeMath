use std::sync::Arc;

use egui::{
    Context, DroppedFile, Event, FullOutput, Id, Key, Modifiers, PointerButton, Pos2, RawInput,
    Rect, Shape, accesskit::Role, pos2, vec2,
};

use crate::{
    AccessState, Line, Plot, PlotMemory,
    commands::CommandRegistryData,
    csv_import::{MAX_CSV_IMPORT_BYTES, MAX_CSV_IMPORT_ROWS},
    export::{set_test_clipboard_read_override, trigger_import_csv_for_plot},
    file_io::take_test_import_triggers,
    get_last_plot_datasets,
};

fn wide_input() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn narrow_input() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(360.0, 480.0))),
        ..Default::default()
    }
}

fn collect_texts_in_shape(shape: &Shape, out: &mut Vec<(String, Pos2)>) {
    match shape {
        Shape::Text(t) => out.push((t.galley.text().to_owned(), t.pos + t.galley.size() * 0.5)),
        Shape::Vec(v) => v.iter().for_each(|s| collect_texts_in_shape(s, out)),
        _ => {}
    }
}

fn find_text(output: &FullOutput, exact: &str) -> Option<Pos2> {
    let mut texts = Vec::new();
    for clipped in &output.shapes {
        collect_texts_in_shape(&clipped.shape, &mut texts);
    }
    texts
        .into_iter()
        .find(|(t, _)| t == exact)
        .map(|(_, pos)| pos)
}

fn click_pos(ctx: &Context, pos: Pos2, mut show: impl FnMut(&Context)) -> FullOutput {
    let down = RawInput {
        events: vec![
            Event::PointerMoved(pos),
            Event::PointerButton {
                pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: Modifiers::NONE,
            },
        ],
        ..wide_input()
    };
    let up = RawInput {
        events: vec![Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed: false,
            modifiers: Modifiers::NONE,
        }],
        ..wide_input()
    };
    let _ = ctx.run(down, &mut show);
    let _ = ctx.run(up, &mut show);
    ctx.run(wide_input(), &mut show)
}

#[test]
fn toolbar_import_csv_and_empty_plot_table_view_enabled() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let _ = take_test_import_triggers();
    let plot_id = Id::new("empty_2d_csv_toolbar");
    let show = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            Plot::new("Empty 2D").id(plot_id).show(ui, |_| {});
        });
    };

    let out = ctx.run(wide_input(), show);
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let import_node = ak
        .nodes
        .iter()
        .find(|(_, n)| n.role() == Role::Button && n.label() == Some("Import CSV"))
        .expect("Import CSV button node");
    assert!(!import_node.1.is_disabled());
    let table_node = ak
        .nodes
        .iter()
        .find(|(_, n)| n.role() == Role::Button && n.label() == Some("Table View"))
        .expect("Table View button node");
    assert!(!table_node.1.is_disabled());

    let import_pos = find_text(&out, "Import CSV").expect("Import CSV visible");
    let _ = click_pos(&ctx, import_pos, show);
    assert_eq!(take_test_import_triggers(), vec![plot_id]);

    let tbl_pos = find_text(&out, "Table View").expect("Table View visible");
    let modal_out = click_pos(&ctx, tbl_pos, show);
    let st = ctx.data(|d| d.get_temp::<AccessState>(plot_id)).unwrap();
    assert!(st.show_table);
    assert!(find_text(&modal_out, "Paste Clipboard").is_some());
}

#[test]
fn ctrl_shift_i_registers_in_command_registry_and_triggers_import() {
    let ctx = Context::default();
    let _ = take_test_import_triggers();
    let plot_id = Id::new("shortcut_import_plot");
    let plot_resp_id = std::cell::Cell::new(Id::NULL);
    let show = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            let pr = Plot::new("Shortcut Plot").id(plot_id).show(ui, |p| {
                p.line(Line::new("Base", vec![[0.0, 0.0], [1.0, 1.0]]));
            });
            plot_resp_id.set(pr.response.id);
        });
    };

    let _ = ctx.run(wide_input(), show);
    let reg = ctx
        .data(|d| d.get_temp::<CommandRegistryData>(Id::new("CMD_REGISTRY")))
        .expect("CMD_REGISTRY populated");
    assert!(reg.commands.iter().any(|c| c.name == "Import CSV"));
    assert!(reg.commands.iter().any(|c| c.name == "Paste CSV / TSV"));

    ctx.memory_mut(|m| m.request_focus(plot_resp_id.get()));
    let mods = Modifiers::COMMAND | Modifiers::SHIFT;
    let raw = RawInput {
        modifiers: mods,
        events: vec![Event::Key {
            key: Key::I,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: mods,
        }],
        ..wide_input()
    };
    let _ = ctx.run(raw, show);
    assert_eq!(take_test_import_triggers(), vec![plot_id]);
}

#[test]
fn focused_paste_overlays_points_lines_zoom_fit_and_delete() {
    let (ctx, plot_id) = (Context::default(), Id::new("paste_and_overlay_controls"));
    let resp_id = std::cell::Cell::new(Id::NULL);
    let show = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            let pr = Plot::new("Overlay Plot").id(plot_id).show(ui, |p| {
                p.line(Line::new("Sim", vec![[0.0, 0.0], [5.0, 5.0]]));
            });
            resp_id.set(pr.response.id);
        });
    };
    let _ = ctx.run(wide_input(), show);
    let mut mem = PlotMemory::load(&ctx, plot_id).unwrap();
    mem.auto_bounds = false.into();
    mem.transform
        .set_bounds(crate::PlotBounds::from_min_max([0.0, 0.0], [10.0, 10.0]));
    mem.store(&ctx, plot_id);

    ctx.memory_mut(|m| m.request_focus(resp_id.get()));
    let csv = "id,x,y\n1,100.0,200.0\n2,120.0,240.0\n3,140.0,280.0\n".to_owned();
    let _ = ctx.run(
        RawInput {
            events: vec![Event::Paste(csv)],
            ..wide_input()
        },
        show,
    );
    let out = ctx.run(wide_input(), show);
    let live = ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_message")));
    assert_eq!(
        live.as_deref(),
        Some("Imported 1 dataset(s) from clipboard")
    );
    assert_eq!(
        PlotMemory::load(&ctx, plot_id).unwrap().imported_datasets[0]
            .points
            .len(),
        3
    );

    let menu_out = click_pos(&ctx, find_text(&out, "Overlays (1)").unwrap(), show);
    let _ = click_pos(&ctx, find_text(&menu_out, "Lines").unwrap(), show);
    assert!(PlotMemory::load(&ctx, plot_id).unwrap().imported_datasets[0].show_lines);
    crate::clear_last_plot_datasets();
    let menu_after_lines = ctx.run(wide_input(), show);
    assert_eq!(get_last_plot_datasets().len(), 2);

    let menu_after_fit = click_pos(
        &ctx,
        find_text(&menu_after_lines, "Zoom Fit").unwrap(),
        show,
    );
    let fitted = *PlotMemory::load(&ctx, plot_id).unwrap().transform.bounds();
    assert!(fitted.min[0] >= 90.0 && fitted.max[0] <= 150.0);
    assert!(fitted.min[1] >= 190.0 && fitted.max[1] <= 290.0);

    let _ = click_pos(&ctx, find_text(&menu_after_fit, "Delete").unwrap(), show);
    assert!(
        PlotMemory::load(&ctx, plot_id)
            .unwrap()
            .imported_datasets
            .is_empty()
    );
}

#[test]
fn overlays_delete_all_clears_multiple_imported_series() {
    let ctx = Context::default();
    let plot_id = Id::new("overlays_delete_all_plot");
    let resp_id = std::cell::Cell::new(Id::NULL);
    let show = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            let pr = Plot::new("Delete All Plot").id(plot_id).show(ui, |_| {});
            resp_id.set(pr.response.id);
        });
    };
    let _ = ctx.run(wide_input(), show);
    ctx.memory_mut(|m| m.request_focus(resp_id.get()));
    let paste_multi = RawInput {
        events: vec![Event::Paste("X\tA\tB\n0\t1\t2\n1\t3\t4\n".to_owned())],
        ..wide_input()
    };
    let _ = ctx.run(paste_multi, show);
    let out_multi = ctx.run(wide_input(), show);
    let ov2_pos = find_text(&out_multi, "Overlays (2)").expect("Overlays (2) button");
    let menu2_out = click_pos(&ctx, ov2_pos, show);
    let del_all_pos = find_text(&menu2_out, "🗑 Delete All").expect("Delete All button");
    let _ = click_pos(&ctx, del_all_pos, show);
    assert!(
        PlotMemory::load(&ctx, plot_id)
            .unwrap()
            .imported_datasets
            .is_empty()
    );
}

#[test]
fn data_table_paste_and_clipboard_button_import_up_to_10k_rows() {
    let ctx = Context::default();
    let plot_id = Id::new("table_modal_paste_plot");
    let show = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            Plot::new("Modal Paste").id(plot_id).show(ui, |_| {});
        });
    };

    let out0 = ctx.run(wide_input(), show);
    let tbl_pos = find_text(&out0, "Table View").unwrap();
    let modal_out = click_pos(&ctx, tbl_pos, show);

    set_test_clipboard_read_override(Some("X\t'-2.0V\n0.0\t-2.0\n1.0\t-1.5\n".to_owned()));
    let paste_btn = find_text(&modal_out, "Paste Clipboard").expect("Paste Clipboard button");
    let _ = click_pos(&ctx, paste_btn, show);
    set_test_clipboard_read_override(None);

    let mem1 = PlotMemory::load(&ctx, plot_id).unwrap();
    assert_eq!(mem1.imported_datasets.len(), 1);
    assert_eq!(mem1.imported_datasets[0].name, "-2.0V");

    let mut big_tsv = String::from("X\tBulk\n");
    for i in 0..(MAX_CSV_IMPORT_ROWS + 25) {
        big_tsv.push_str(&format!("{i}\t{i}\n"));
    }
    let _ = ctx.run(
        RawInput {
            events: vec![Event::Paste(big_tsv)],
            ..wide_input()
        },
        show,
    );
    let _ = ctx.run(wide_input(), show);
    let mem2 = PlotMemory::load(&ctx, plot_id).unwrap();
    let bulk = mem2
        .imported_datasets
        .iter()
        .find(|d| d.name == "Bulk")
        .expect("Bulk imported");
    assert_eq!(bulk.points.len(), MAX_CSV_IMPORT_ROWS);
}

#[test]
fn dropped_files_and_oversized_2mib_payloads_fail_closed_assertively() {
    let ctx = Context::default();
    let plot_id = Id::new("drop_and_2mib_guard");
    let show = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            Plot::new("Drop Plot").id(plot_id).show(ui, |_| {});
        });
    };
    let _ = ctx.run(wide_input(), show);

    let drop_ok = RawInput {
        events: vec![Event::PointerMoved(pos2(400.0, 300.0))],
        dropped_files: vec![
            DroppedFile {
                name: "ignored.png".to_owned(),
                bytes: Some(Arc::from(&b"not_csv"[..])),
                ..Default::default()
            },
            DroppedFile {
                name: "sensor.tsv".to_owned(),
                bytes: Some(Arc::from(&b"X\tSensor\n0,0\t1,25\n1,0\t2,5\n"[..])),
                ..Default::default()
            },
        ],
        ..wide_input()
    };
    let _ = ctx.run(drop_ok, show);
    let mem = PlotMemory::load(&ctx, plot_id).unwrap();
    assert_eq!(mem.imported_datasets.len(), 1);
    assert_eq!(mem.imported_datasets[0].name, "Sensor");

    let oversized = vec![b'1'; MAX_CSV_IMPORT_BYTES + 16];
    let drop_huge = RawInput {
        events: vec![Event::PointerMoved(pos2(400.0, 300.0))],
        dropped_files: vec![DroppedFile {
            name: "huge.csv".to_owned(),
            bytes: Some(Arc::from(oversized)),
            ..Default::default()
        }],
        ..wide_input()
    };
    let _ = ctx.run(drop_huge, show);
    let err_live = ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_assertive_message")));
    assert!(
        err_live
            .as_deref()
            .is_some_and(|m| m.contains("exceeds maximum allowed size"))
    );
    let toast = ctx
        .data(|d| d.get_temp::<AccessState>(plot_id))
        .unwrap()
        .import_toast
        .expect("error toast");
    assert!(toast.1);
}

#[test]
fn allow_csv_import_false_and_narrow_pane_overflow_menu() {
    let ctx = Context::default();
    let locked_id = Id::new("locked_3d_opt_out");
    let out_locked = ctx.run(wide_input(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            Plot::new("Locked")
                .id(locked_id)
                .allow_csv_import(false)
                .show(ui, |p| {
                    p.line(Line::new("Proj", vec![[0.0, 0.0], [1.0, 1.0]]));
                });
        });
    });
    assert!(find_text(&out_locked, "Import CSV").is_none());
    trigger_import_csv_for_plot(&ctx, locked_id);
    let _ = take_test_import_triggers();

    let ctx_narrow = Context::default();
    let narrow_id = Id::new("narrow_split_plot");
    let show_narrow = |c: &Context| {
        egui::CentralPanel::default().show(c, |ui| {
            Plot::new("Narrow").id(narrow_id).show(ui, |p| {
                p.line(Line::new("Curve", vec![[0.0, 0.0], [2.0, 4.0]]));
            });
        });
    };
    let out_narrow = ctx_narrow.run(narrow_input(), show_narrow);
    assert!(find_text(&out_narrow, "Import CSV").is_some());
    assert!(find_text(&out_narrow, "Save PNG").is_none());
    let more_pos = find_text(&out_narrow, "⋯ More").expect("Overflow More menu");
    let down = RawInput {
        events: vec![
            Event::PointerMoved(more_pos),
            Event::PointerButton {
                pos: more_pos,
                button: PointerButton::Primary,
                pressed: true,
                modifiers: Modifiers::NONE,
            },
            Event::PointerButton {
                pos: more_pos,
                button: PointerButton::Primary,
                pressed: false,
                modifiers: Modifiers::NONE,
            },
        ],
        ..narrow_input()
    };
    let _ = ctx_narrow.run(down, show_narrow);
    let out_menu = ctx_narrow.run(narrow_input(), show_narrow);
    assert!(find_text(&out_menu, "Save PNG").is_some());
}
