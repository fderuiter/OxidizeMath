#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use eframe::App;

fn run_app_events(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
) -> egui::FullOutput {
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 900.0));
    let mut frame = eframe::Frame::_new_kittest();
    let input = egui::RawInput {
        screen_rect: Some(screen),
        modifiers,
        events,
        ..Default::default()
    };
    ctx.run(input, |c| app.update(c, &mut frame))
}

fn shortcut_events(modifiers: egui::Modifiers, key: egui::Key) -> Vec<egui::Event> {
    vec![
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers,
        },
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: false,
            repeat: false,
            modifiers,
        },
    ]
}

fn primary_cmd_modifiers() -> egui::Modifiers {
    if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    }
}

fn configure_sir_or_first_framework_tab(app: &mut MathExplorerApp) -> Option<(usize, String)> {
    if let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") {
        app.set_pane_tab(0, epi_idx);
        let fw = app.pane_tab_mut(0)?.framework_mut()?;
        assert!(fw.select_tool_by_name("SirTool"));
        fw.active_tool
            .as_mut()?
            .load_state(r#"{"n":1000.0,"i0":15.0,"beta":2.75,"gamma":0.25,"duration":120.0}"#);
        let snap = app.pane_tab(0)?.save_state()?;
        return Some((epi_idx, snap));
    }
    let idx = app.tabs.iter().position(|t| t.save_state().is_some())?;
    app.set_pane_tab(0, idx);
    let snap = app.pane_tab(0)?.save_state()?;
    Some((idx, snap))
}

#[test]
fn test_duplicate_focused_pane_shortcut_clones_tool_and_registers_command() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some((src_idx, expected_state)) = configure_sir_or_first_framework_tab(&mut app) else {
        return;
    };

    let _ = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    let dup_mods = primary_cmd_modifiers() | egui::Modifiers::SHIFT;
    let _ = run_app_events(
        &ctx,
        &mut app,
        shortcut_events(dup_mods, egui::Key::Num2),
        dup_mods,
    );

    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(app.focused_pane, 1);
    assert_eq!(app.pane_tabs[0], src_idx);
    assert_eq!(app.pane_tabs[1], src_idx);
    assert_eq!(
        app.pane_tab(1).and_then(|t| t.save_state()),
        Some(expected_state)
    );

    let aria = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(
        aria.as_deref()
            .is_some_and(|m| m.contains("Duplicated") && m.contains("Right Pane")),
        "Expected polite ARIA clone announcement, got {aria:?}"
    );
    let registry = ctx.data(|d| {
        d.get_temp::<egui_plot::commands::CommandRegistryData>(egui::Id::new("CMD_REGISTRY"))
            .unwrap_or_default()
    });
    assert!(registry
        .commands
        .iter()
        .any(|c| c.name == "Duplicate Focused Pane" && !c.description.is_empty()));
}

#[test]
fn test_initial_split_and_quad_clone_primary_simulation_state() {
    let mut app = MathExplorerApp::default();
    let Some((src_idx, expected_state)) = configure_sir_or_first_framework_tab(&mut app) else {
        return;
    };

    app.set_layout(ViewportLayout::SplitHorizontal);
    assert_eq!(app.pane_tabs[1], src_idx);
    assert_eq!(app.secondary_tab, src_idx);
    let cloned_split = app.pane_tab_mut(1).and_then(|t| t.save_state());
    assert_eq!(cloned_split, Some(expected_state.clone()));

    app.set_layout(ViewportLayout::QuadGrid);
    for pane in 2..4 {
        assert_eq!(app.pane_tabs[pane], src_idx);
        let cloned_quad = app.pane_tab_mut(pane).and_then(|t| t.save_state());
        assert_eq!(cloned_quad, Some(expected_state.clone()));
    }
}

#[test]
fn test_ctrl_z_restores_pre_clone_layout_and_persists_sync_toggles() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some((src_idx, _)) = configure_sir_or_first_framework_tab(&mut app) else {
        return;
    };

    let _ = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    assert_eq!(app.viewport_layout, ViewportLayout::Single);

    let dup_mods = primary_cmd_modifiers() | egui::Modifiers::SHIFT;
    let _ = run_app_events(
        &ctx,
        &mut app,
        shortcut_events(dup_mods, egui::Key::Num2),
        dup_mods,
    );
    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);

    let undo_mods = primary_cmd_modifiers();
    let _ = run_app_events(
        &ctx,
        &mut app,
        shortcut_events(undo_mods, egui::Key::Z),
        undo_mods,
    );
    assert_eq!(app.viewport_layout, ViewportLayout::Single);
    assert_eq!(app.selected_tab, src_idx);

    app.redo();
    assert_eq!(app.viewport_layout, ViewportLayout::SplitHorizontal);
    assert_eq!(app.pane_tabs[1], src_idx);
}

#[test]
fn test_session_envelope_recovery_and_undo_preserve_comparison_sync_toggles() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let _ = app.duplicate_focused_pane(Some(&ctx));
    let _ = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);

    app.set_link_plot_axes(Some(&ctx), true);
    app.set_link_plot_cursors(Some(&ctx), true);
    app.set_sync_playback(Some(&ctx), true);
    let _ = run_app_events(
        &ctx,
        &mut app,
        vec![egui::Event::PointerButton {
            pos: egui::pos2(10.0, 10.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        }],
        egui::Modifiers::NONE,
    );

    let envelope = app.session_manager.create_envelope(app.get_app_state());
    let mut restored = MathExplorerApp::default();
    restored.load_state_from_json(&envelope);
    assert_eq!(
        restored.compare_sync,
        ComparisonSyncConfig {
            link_axes: true,
            link_cursors: true,
            sync_playback: true,
        }
    );

    let snap = app.capture_recovery_snapshot();
    let mut recovered = MathExplorerApp::default();
    recovered.restore_recovery_snapshot(&snap, &ctx);
    assert_eq!(recovered.compare_sync, app.compare_sync);

    app.undo();
    assert_eq!(app.compare_sync, ComparisonSyncConfig::default());
    app.redo();
    assert!(app.compare_sync.link_axes && app.compare_sync.sync_playback);
}

fn collect_pane_texts(
    shapes: &[egui::epaint::ClippedShape],
    x_range: std::ops::Range<f32>,
    out: &mut Vec<String>,
) {
    for cs in shapes {
        collect_shape_texts(&cs.shape, &x_range, out);
    }
}

fn collect_shape_texts(
    shape: &egui::epaint::Shape,
    x_range: &std::ops::Range<f32>,
    out: &mut Vec<String>,
) {
    match shape {
        egui::epaint::Shape::Text(t) if x_range.contains(&t.pos.x) => {
            out.push(t.galley.text().to_string());
        }
        egui::epaint::Shape::Vec(inner) => {
            for s in inner {
                collect_shape_texts(s, x_range, out);
            }
        }
        _ => {}
    }
}

#[test]
fn test_multi_pane_sir_linked_plot_axes_share_union_y_bounds() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    app.set_pane_tab(0, epi_idx);
    {
        let fw0 = app.pane_tab_mut(0).unwrap().framework_mut().unwrap();
        assert!(fw0.select_tool_by_name("SirTool"));
        fw0.active_tool
            .as_mut()
            .unwrap()
            .load_state(r#"{"n":200.0,"i0":5.0,"beta":0.25,"gamma":0.2,"duration":100.0}"#);
    }
    let _ = app.duplicate_focused_pane(Some(&ctx));
    {
        let fw1 = app.pane_tab_mut(1).unwrap().framework_mut().unwrap();
        fw1.active_tool
            .as_mut()
            .unwrap()
            .load_state(r#"{"n":5000.0,"i0":50.0,"beta":2.5,"gamma":0.1,"duration":100.0}"#);
    }

    // Before linking: Left Pane (n=200) does not show 4000/5000 Y-axis ticks.
    let _ = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    let unlinked = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    let mut left_unlinked = Vec::new();
    collect_pane_texts(&unlinked.shapes, 0.0..640.0, &mut left_unlinked);
    assert!(
        !left_unlinked.iter().any(|t| t == "4000" || t == "5000"),
        "Unlinked Left Pane should stay near n=200 scale, got {left_unlinked:?}"
    );

    // Enable Link Plot Axes: Left Pane automatically adopts Right Pane's 5000-scale envelope.
    app.set_link_plot_axes(Some(&ctx), true);
    app.set_link_plot_cursors(Some(&ctx), true);
    let _ = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    let linked = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    let mut left_linked = Vec::new();
    let mut right_linked = Vec::new();
    collect_pane_texts(&linked.shapes, 0.0..640.0, &mut left_linked);
    collect_pane_texts(&linked.shapes, 640.0..1280.0, &mut right_linked);
    assert!(
        left_linked.iter().any(|t| t == "4000" || t == "5000")
            && right_linked.iter().any(|t| t == "4000" || t == "5000"),
        "Both panes must share union Y-axis scale ticks: left={left_linked:?}, right={right_linked:?}"
    );
}

#[cfg(feature = "epidemiology")]
#[test]
fn test_multi_pane_network_propagation_sync_playback_lockstep_period_key() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let epi_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Epidemiology")
        .unwrap();
    app.set_pane_tab(0, epi_idx);
    {
        let fw = app.pane_tab_mut(0).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("NetworkPropagationTool"));
        fw.active_tool
            .as_mut()
            .unwrap()
            .load_state(r#"{"num_nodes":40,"beta":0.6,"gamma":0.1,"is_running":false}"#);
    }
    let _ = app.duplicate_focused_pane(Some(&ctx));
    app.set_sync_playback(Some(&ctx), true);
    app.focus_pane(1);
    let _ = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    crate::accessibility::clear_test_announcements();

    let _ = run_app_events(
        &ctx,
        &mut app,
        shortcut_events(egui::Modifiers::NONE, egui::Key::Period),
        egui::Modifiers::NONE,
    );
    let announcements = crate::accessibility::all_announcements();
    assert_eq!(announcements.len(), 1);
    assert_eq!(announcements[0].1, "polite");
    for pane in [0, 1] {
        let tool = app
            .pane_tab(pane)
            .unwrap()
            .framework()
            .unwrap()
            .active_tool
            .as_ref()
            .unwrap();
        let net = tool
            .as_any()
            .and_then(|a| {
                a.downcast_ref::<crate::tabs::epidemiology::network_propagation::NetworkPropagationTool>()
            })
            .unwrap();
        assert_eq!(net.sim_steps(), 1, "Pane {pane} must step once in lockstep");
    }
}

fn run_dual_sync_transport_frame(
    ctx: &egui::Context,
    app: &MathExplorerApp,
    key: egui::Key,
    steps: &mut [usize; 2],
    playing: &mut [bool; 2],
) {
    let _ = ctx.run(
        egui::RawInput {
            events: shortcut_events(egui::Modifiers::NONE, key),
            ..Default::default()
        },
        |c| {
            c.data_mut(|d| {
                d.insert_temp(egui::Id::new("ACTIVE_FOCUSED_PANE"), 1_usize);
                d.insert_temp(egui::Id::new("SHELL_MODAL_OR_SASH_FOCUSED"), false);
            });
            super::viewport::compare_sync::publish_multi_pane_sync_context(app, c);
            egui::CentralPanel::default().show(c, |ui| {
                for (pane_idx, (step_cnt, is_playing)) in
                    steps.iter_mut().zip(playing.iter_mut()).enumerate()
                {
                    c.data_mut(|d| {
                        d.insert_temp(egui::Id::new("CURRENT_RENDERING_PANE"), pane_idx)
                    });
                    ui.push_id(("pane_content", pane_idx), |pane_ui| {
                        let act = PlaybackTransport::in_thread(
                            "sync_test_transport",
                            "Wave",
                            *is_playing,
                        )
                        .show(pane_ui);
                        if act == Some(PlaybackAction::Step) {
                            *step_cnt += 1;
                        } else if act == Some(PlaybackAction::Play) {
                            *is_playing = true;
                        }
                    });
                }
            });
        },
    );
}

#[test]
fn test_sync_playback_steps_and_toggles_dual_transports_in_lockstep() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_sync_playback(None, true);
    app.focus_pane(1);
    crate::accessibility::clear_test_announcements();

    let mut steps = [0_usize, 0_usize];
    let mut playing = [false, false];

    run_dual_sync_transport_frame(&ctx, &app, egui::Key::Period, &mut steps, &mut playing);
    assert_eq!(steps, [1, 1], "Both panes must step in lockstep on Period");
    assert_eq!(crate::accessibility::all_announcements().len(), 1);
    crate::accessibility::clear_test_announcements();

    run_dual_sync_transport_frame(&ctx, &app, egui::Key::Space, &mut steps, &mut playing);
    assert_eq!(playing, [true, true], "Both panes must Play in lockstep");
    assert_eq!(crate::accessibility::all_announcements().len(), 1);
}

fn find_galley_center(shapes: &[egui::epaint::ClippedShape], needle: &str) -> Option<egui::Pos2> {
    for cs in shapes {
        if let Some(pos) = find_shape_galley(&cs.shape, needle) {
            return Some(pos);
        }
    }
    None
}

fn find_shape_galley(shape: &egui::epaint::Shape, needle: &str) -> Option<egui::Pos2> {
    match shape {
        egui::epaint::Shape::Text(t) if t.galley.text().contains(needle) => {
            Some(t.pos + t.galley.rect.center().to_vec2())
        }
        egui::epaint::Shape::Vec(inner) => inner.iter().find_map(|s| find_shape_galley(s, needle)),
        _ => None,
    }
}

#[test]
fn test_pane_header_duplicate_button_and_sync_flyout_controls() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    if app.tabs.len() < 2 {
        return;
    }
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, 0);
    app.set_pane_tab(1, 1);

    let out = run_app_events(&ctx, &mut app, Vec::new(), egui::Modifiers::NONE);
    let dup_pos = find_galley_center(&out.shapes, "Duplicate").expect("Header Duplicate button");
    let click = |pos| {
        vec![
            egui::Event::PointerMoved(pos),
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ]
    };
    let _ = run_app_events(&ctx, &mut app, click(dup_pos), egui::Modifiers::NONE);
    assert_eq!(app.pane_tabs[1], 0, "Header Duplicate must clone Left Pane");

    let sync_pos = find_galley_center(&out.shapes, "Sync").expect("Header Sync menu button");
    let _ = run_app_events(&ctx, &mut app, click(sync_pos), egui::Modifiers::NONE);
    let menu_out = run_app_events(
        &ctx,
        &mut app,
        vec![egui::Event::PointerMoved(sync_pos)],
        egui::Modifiers::NONE,
    );
    let axes_pos =
        find_galley_center(&menu_out.shapes, "Link Plot Axes").expect("Link Plot Axes checkbox");
    let _ = run_app_events(&ctx, &mut app, click(axes_pos), egui::Modifiers::NONE);
    assert!(app.compare_sync.link_axes);
}
