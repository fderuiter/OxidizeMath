#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use eframe::egui;
use eframe::App;

fn run_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    screen: egui::Rect,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut frame = eframe::Frame::_new_kittest();
    ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events,
            ..Default::default()
        },
        |c| app.update(c, &mut frame),
    )
}

fn key_press(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

#[cfg(all(feature = "pure_math", feature = "ai"))]
fn find_pane0_formula_input_center(out: &egui::FullOutput) -> Option<egui::Pos2> {
    let ak = out.platform_output.accesskit_update.as_ref()?;
    for (_, node) in &ak.nodes {
        if node.role() == egui::accesskit::Role::TextInput {
            let val = node.value().unwrap_or_default();
            if val.contains("sin(t)") {
                if let Some(bounds) = node.bounds() {
                    let cx = ((bounds.x0 + bounds.x1) * 0.5) as f32;
                    let cy = ((bounds.y0 + bounds.y1) * 0.5) as f32;
                    return Some(egui::pos2(cx, cy));
                }
            }
        }
    }
    None
}

#[cfg(all(feature = "pure_math", feature = "ai"))]
fn click_pos(ctx: &egui::Context, app: &mut MathExplorerApp, screen: egui::Rect, pos: egui::Pos2) {
    let _ = run_frame(
        ctx,
        app,
        screen,
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = run_frame(
        ctx,
        app,
        screen,
        vec![egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
}

#[cfg(all(feature = "pure_math", feature = "ai"))]
fn setup_ode_custom_and_training_monitor(app: &mut MathExplorerApp) {
    let analysis_idx = app
        .tabs
        .iter()
        .position(|t| t.framework().is_some_and(|fw| fw.domain == "analysis"))
        .expect("Analysis tab");
    let ai_idx = app
        .tabs
        .iter()
        .position(|t| t.framework().is_some_and(|fw| fw.domain == "ai"))
        .expect("AI domain tab");

    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, analysis_idx);
    app.set_pane_tab(1, ai_idx);
    app.focus_pane(0);

    {
        let fw0 = app.pane_tab_mut(0).unwrap().framework_mut().unwrap();
        assert!(fw0.select_tool_by_name("OdeSolverTool"));
        let state_json = fw0.active_tool.as_ref().unwrap().save_state().unwrap();
        let custom_json = state_json.replace("\"Exponential\"", "\"Custom\"");
        fw0.active_tool.as_mut().unwrap().load_state(&custom_json);
    }
    {
        let fw1 = app.pane_tab_mut(1).unwrap().framework_mut().unwrap();
        assert!(fw1.select_tool_by_name("TrainingMonitorTool"));
    }
}

#[cfg(all(feature = "pure_math", feature = "ai"))]
fn has_rendered_text(out: &egui::FullOutput, needle: &str) -> bool {
    fn walk(shapes: &[egui::epaint::Shape], needle: &str) -> bool {
        shapes.iter().any(|s| match s {
            egui::epaint::Shape::Text(ts) => ts.galley.text().contains(needle),
            egui::epaint::Shape::Vec(nested) => walk(nested, needle),
            _ => false,
        })
    }
    let shapes: Vec<_> = out.shapes.iter().map(|cs| cs.shape.clone()).collect();
    walk(&shapes, needle)
}

#[cfg(all(feature = "pure_math", feature = "ai"))]
fn pane1_is_training(app: &MathExplorerApp) -> bool {
    app.pane_tab(1)
        .and_then(|t| t.framework())
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|tool| tool.save_state())
        .is_some_and(|s| s.contains("\"is_training\":true"))
}

#[cfg(all(feature = "pure_math", feature = "ai"))]
fn pane0_tool_state_json(app: &MathExplorerApp) -> String {
    app.pane_tab(0)
        .unwrap()
        .framework()
        .unwrap()
        .active_tool
        .as_ref()
        .unwrap()
        .save_state()
        .unwrap()
}

#[test]
#[cfg(all(feature = "pure_math", feature = "ai"))]
fn test_typing_in_pane_0_while_hovering_pane_1_or_pointer_gone_isolates_keys_and_clipboard() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 768.0));
    let mut app = MathExplorerApp::default();
    setup_ode_custom_and_training_monitor(&mut app);

    let out0 = run_frame(&ctx, &mut app, screen, Vec::new());
    let input_pos =
        find_pane0_formula_input_center(&out0).expect("Pane 0 MathExpressionInputBar TextEdit");

    click_pos(&ctx, &mut app, screen, input_pos);
    let out_init = run_frame(
        &ctx,
        &mut app,
        screen,
        vec![key_press(egui::Key::End, egui::Modifiers::NONE)],
    );
    assert!(ctx.wants_keyboard_input());
    assert_eq!(app.focused_pane, 0);
    assert!(!pane1_is_training(&app) && has_rendered_text(&out_init, "Epoch: 0"));

    let out_hover = run_frame(
        &ctx,
        &mut app,
        screen,
        vec![
            egui::Event::PointerMoved(egui::pos2(820.0, 420.0)),
            egui::Event::Text(" + y".to_string()),
            key_press(egui::Key::Space, egui::Modifiers::NONE),
            egui::Event::Text(".".to_string()),
            key_press(egui::Key::Period, egui::Modifiers::NONE),
            egui::Event::Paste("0".to_string()),
        ],
    );
    assert_eq!(app.focused_pane, 0);
    assert!(
        !pane1_is_training(&app) && has_rendered_text(&out_hover, "Epoch: 0"),
        "Hovering Pane 1 while typing in Pane 0 must never trigger Pane 1 Space/Period playback hotkeys"
    );

    let out_gone = run_frame(
        &ctx,
        &mut app,
        screen,
        vec![
            egui::Event::PointerGone,
            egui::Event::Text(" + 2".to_string()),
            egui::Event::Text(".".to_string()),
            key_press(egui::Key::Period, egui::Modifiers::NONE),
            egui::Event::Text("5".to_string()),
        ],
    );
    assert_eq!(app.focused_pane, 0);
    assert!(!pane1_is_training(&app) && has_rendered_text(&out_gone, "Epoch: 0"));

    let pane0_state = pane0_tool_state_json(&app);
    assert!(
        pane0_state.contains("+ y.0 + 2.5"),
        "Pane 0 equation bar should receive typed and pasted text across hover/PointerGone: {pane0_state}"
    );
}

#[test]
#[cfg(all(feature = "pure_math", feature = "ai"))]
fn test_switching_focused_pane_surrenders_stale_widget_focus_in_inactive_pane() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 768.0));
    let mut app = MathExplorerApp::default();
    setup_ode_custom_and_training_monitor(&mut app);

    let out0 = run_frame(&ctx, &mut app, screen, Vec::new());
    let input_pos = find_pane0_formula_input_center(&out0).unwrap();
    click_pos(&ctx, &mut app, screen, input_pos);
    let pane0_focused_widget = ctx
        .memory(|m| m.focused())
        .expect("Pane 0 TextEdit focused");

    let pane1_pos = egui::pos2(740.0, 420.0);
    click_pos(&ctx, &mut app, screen, pane1_pos);
    assert_eq!(app.focused_pane, 1);
    assert!(!ctx.memory(|m| m.has_focus(pane0_focused_widget)));
    assert!(!ctx.wants_keyboard_input());

    let before_pane0 = pane0_tool_state_json(&app);
    let _ = run_frame(
        &ctx,
        &mut app,
        screen,
        vec![
            egui::Event::Text(".".to_string()),
            key_press(egui::Key::Period, egui::Modifiers::NONE),
        ],
    );
    let out_after = run_frame(&ctx, &mut app, screen, Vec::new());
    let after_pane0 = pane0_tool_state_json(&app);
    assert_eq!(before_pane0, after_pane0);
    assert!(
        has_rendered_text(&out_after, "Epoch: 1"),
        "Pressing Period with Pane 1 focused should step Pane 1 TrainingMonitorTool to Epoch 1"
    );
}

#[test]
fn test_f6_and_shift_f6_cycle_focused_pane_with_pointer_none_and_announce_aria() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 768.0));
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    if !app.tabs.is_empty() {
        for p in 0..4 {
            app.set_pane_tab(p, p % app.tabs.len());
        }
    }

    let _ = run_frame(&ctx, &mut app, screen, vec![egui::Event::PointerGone]);
    assert!(ctx.input(|i| i.pointer.latest_pos().is_none()));
    assert_eq!(app.focused_pane, 0);
    clear_test_announcements();

    for expected in [1usize, 2, 3, 0] {
        let _ = run_frame(
            &ctx,
            &mut app,
            screen,
            vec![key_press(egui::Key::F6, egui::Modifiers::NONE)],
        );
        assert_eq!(app.focused_pane, expected);
    }

    for expected in [3usize, 2, 1, 0] {
        let _ = run_frame(
            &ctx,
            &mut app,
            screen,
            vec![key_press(egui::Key::F6, egui::Modifiers::SHIFT)],
        );
        assert_eq!(app.focused_pane, expected);
    }

    let announcements = all_announcements();
    assert!(
        announcements.len() >= 8,
        "Expected ARIA live-region announcements for each F6 / Shift+F6 transition, got {announcements:?}"
    );
    assert!(announcements.iter().any(|(msg, _)| msg.contains("Pane 2")));
    assert!(announcements.iter().any(|(msg, _)| msg.contains("Pane 4")));
}
