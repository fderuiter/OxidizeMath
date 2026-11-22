#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use eframe::egui;
use eframe::App;

fn shortcut_mods() -> egui::Modifiers {
    if cfg!(target_os = "macos") {
        egui::Modifiers::MAC_CMD
    } else {
        egui::Modifiers::CTRL
    }
}

fn redo_shortcut_events() -> Vec<egui::Event> {
    let mods = shortcut_mods();
    if cfg!(target_os = "macos") {
        key_event(egui::Key::Z, mods | egui::Modifiers::SHIFT)
    } else {
        key_event(egui::Key::Y, mods)
    }
}

fn key_event(key: egui::Key, modifiers: egui::Modifiers) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }]
}

fn step_events(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1200.0, 800.0),
        )),
        predicted_dt: 1.0 / 60.0,
        events,
        ..Default::default()
    };
    ctx.run(raw_input, |c| app.update(c, frame))
}

fn click_point(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    center: egui::Pos2,
) -> egui::FullOutput {
    let _ = step_events(
        ctx,
        app,
        frame,
        vec![
            egui::Event::PointerMoved(center),
            egui::Event::PointerButton {
                pos: center,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    step_events(
        ctx,
        app,
        frame,
        vec![egui::Event::PointerButton {
            pos: center,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    )
}

fn click_accesskit_button_by_label(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    last_out: &egui::FullOutput,
    label_substr: &str,
) -> egui::FullOutput {
    let bounds = last_out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit must be enabled")
        .nodes
        .iter()
        .find_map(|(_, node)| {
            (node.role() == egui::accesskit::Role::Button && node.label()?.contains(label_substr))
                .then(|| node.bounds())
                .flatten()
        })
        .unwrap_or_else(|| panic!("Expected AccessKit button matching '{label_substr}'"));
    let center = egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    );
    click_point(ctx, app, frame, center)
}

#[cfg(feature = "pure_math")]
fn ode_param_k_and_runs(app: &MathExplorerApp, analysis_idx: usize) -> (f64, usize) {
    let fw = app.tabs[analysis_idx].framework().unwrap();
    let tool = fw
        .active_tool
        .as_ref()
        .unwrap()
        .as_any()
        .unwrap()
        .downcast_ref::<crate::tabs::analysis::ode::OdeSolverTool>()
        .unwrap();
    (tool.param_k, tool.snapshot_manager.history_runs().len())
}

#[cfg(feature = "pure_math")]
fn seed_ode_history(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    analysis_idx: usize,
) -> egui::FullOutput {
    for k in [0.5_f64, 0.75, 1.0, 1.25, 1.5] {
        let json = format!(
            r#"{{"preset":"Exponential","method":"RungeKutta4","dt":0.05,"total_time":10.0,"param_k":{k},"param_r":1.0,"param_cap_k":10.0,"ic_y0":1.0,"ic_v0":0.0,"custom_expr":"-k * y * sin(t)","comparison_mode":false,"show_annotations":true}}"#
        );
        app.tabs[analysis_idx]
            .framework_mut()
            .unwrap()
            .active_tool
            .as_mut()
            .unwrap()
            .try_load_state(&json)
            .unwrap();
        let _ = step_events(ctx, app, frame, Vec::new());
    }
    step_events(ctx, app, frame, Vec::new())
}

#[cfg(feature = "pure_math")]
#[test]
fn test_ode_60_frame_sweep_coalesces_workspace_undo_and_trace_runs() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mods = shortcut_mods();

    let analysis_idx = find_tab_index(&app, "Analysis & Calculus");
    app.select_tab(&ctx, analysis_idx);
    assert!(app.tabs[analysis_idx].select_tool_by_name("OdeSolverTool"));
    let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());

    let out = seed_ode_history(&ctx, &mut app, &mut frame, analysis_idx);
    let (pre_sweep_k, pre_sweep_runs) = ode_param_k_and_runs(&app, analysis_idx);
    assert!((pre_sweep_k - 1.5).abs() < 1e-9);

    let mut last_out = click_accesskit_button_by_label(
        &ctx,
        &mut app,
        &mut frame,
        &out,
        "Play animation for k (Rate / Parameter)",
    );
    for _ in 0..60 {
        last_out = step_events(&ctx, &mut app, &mut frame, Vec::new());
        assert!(crate::reflective_ui::slider::has_active_sweeps(&ctx));
        assert_eq!(
            app.session_manager.lifecycle_state(),
            SessionLifecycleState::DragEditing
        );
        assert_eq!(ode_param_k_and_runs(&app, analysis_idx).1, pre_sweep_runs);
    }

    let _ = click_accesskit_button_by_label(
        &ctx,
        &mut app,
        &mut frame,
        &last_out,
        "Pause animation for k (Rate / Parameter)",
    );
    assert_eq!(
        app.session_manager.lifecycle_state(),
        SessionLifecycleState::Idle
    );
    let (swept_k, post_stop_runs) = ode_param_k_and_runs(&app, analysis_idx);
    assert!((swept_k - pre_sweep_k).abs() > 0.1);
    assert_eq!(post_stop_runs, pre_sweep_runs + 1);

    let _ = step_events(&ctx, &mut app, &mut frame, key_event(egui::Key::Z, mods));
    assert!((ode_param_k_and_runs(&app, analysis_idx).0 - 1.5).abs() < 1e-9);

    for expected_k in [1.25_f64, 1.0, 0.75, 0.5] {
        let _ = step_events(&ctx, &mut app, &mut frame, key_event(egui::Key::Z, mods));
        assert!((ode_param_k_and_runs(&app, analysis_idx).0 - expected_k).abs() < 1e-9);
    }
}

#[cfg(feature = "physics")]
fn parse_bifurcation_bounds_from_framework_json(fw_json: &str) -> (f64, f64) {
    let v: serde_json::Value = serde_json::from_str(fw_json).unwrap();
    let tool_entry = &v["tool_states"]["Bifurcation Diagram"];
    let tool_val = if let Some(s) = tool_entry.as_str() {
        serde_json::from_str::<serde_json::Value>(s).unwrap()
    } else {
        tool_entry.clone()
    };
    (
        tool_val["r_min"].as_f64().unwrap(),
        tool_val["r_max"].as_f64().unwrap(),
    )
}

#[cfg(feature = "physics")]
#[test]
fn test_bifurcation_60_frame_sweep_updates_save_state_json_and_secondary_slot() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let chaos_idx = find_tab_index(&app, "Chaos Theory");
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, chaos_idx);
    app.set_pane_tab(1, chaos_idx);
    for pane in [0_usize, 1] {
        let tab = app.pane_tab_mut(pane).unwrap();
        assert!(tab.select_tool_by_name("BifurcationDiagram"));
        tab.load_state(
            r#"{"selected_tool_index":0,"selected_tool_name":"BifurcationDiagram","show_theory_portal":false,"show_starter_gallery":false,"tool_states":{"Bifurcation Diagram":{"r_min":1.0,"r_max":3.2,"steps":200}}}"#,
        );
    }
    let out = step_events(&ctx, &mut app, &mut frame, Vec::new());

    for pane in [0_usize, 1] {
        let target_x = if pane == 0 { 0.0..600.0 } else { 600.0..1200.0 };
        for label in ["Play animation for Min r", "Play animation for Max r"] {
            let bounds = out
                .platform_output
                .accesskit_update
                .as_ref()
                .unwrap()
                .nodes
                .iter()
                .find_map(|(_, n)| {
                    (n.role() == egui::accesskit::Role::Button
                        && n.label()? == label
                        && target_x.contains(&n.bounds()?.x0))
                    .then(|| n.bounds())
                    .flatten()
                })
                .unwrap();
            let c = egui::pos2(
                ((bounds.x0 + bounds.x1) * 0.5) as f32,
                ((bounds.y0 + bounds.y1) * 0.5) as f32,
            );
            let _ = click_point(&ctx, &mut app, &mut frame, c);
        }
    }

    for _ in 0..60 {
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
        assert_eq!(crate::reflective_ui::slider::active_sweep_count(&ctx), 4);
        assert!(crate::reflective_ui::slider::has_sweep_mutation(&ctx));
    }

    let saved_json = app.save_state_to_json();
    let parsed: AppState = serde_json::from_str(&saved_json).unwrap();
    let (p0_rmin, p0_rmax) =
        parse_bifurcation_bounds_from_framework_json(&parsed.tab_states["Chaos Theory"]);
    let (p1_rmin, p1_rmax) =
        parse_bifurcation_bounds_from_framework_json(&parsed.pane_tab_states[1]["Chaos Theory"]);
    assert!((p0_rmin - 1.0).abs() > 0.1 && (p0_rmax - 3.2).abs() > 0.1);
    assert!((p1_rmin - 1.0).abs() > 0.1 && (p1_rmax - 3.2).abs() > 0.1);
}

fn assert_file_shortcuts_stripped(ctx: &egui::Context, mods: egui::Modifiers) {
    let sm = mods | egui::Modifiers::SHIFT;
    for (key, m, expected) in [
        (egui::Key::O, mods, (true, false, false)),
        (egui::Key::S, mods, (false, false, true)),
        (egui::Key::S, sm, (false, true, false)),
    ] {
        let raw = egui::RawInput {
            events: key_event(key, m),
            ..Default::default()
        };
        let _ = ctx.run(raw, |c| {
            assert_eq!(
                WorkspaceSessionManager::check_file_shortcuts(c, mods, sm),
                expected
            );
            c.input(|i| {
                assert!(!i.keys_down.contains(&key));
                assert!(i.events.is_empty() && i.raw.events.is_empty());
            });
        });
    }
}

#[test]
fn test_single_layout_ctrl_z_and_ctrl_y_across_all_explorer_tabs_and_strip_shortcuts() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mods = shortcut_mods();

    #[cfg(all(feature = "pure_math", feature = "physics"))]
    assert_eq!(app.tabs.len(), 21);

    app.set_layout(ViewportLayout::Single);
    let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());

    for tab_idx in 0..app.tabs.len() {
        app.select_tab(&ctx, tab_idx);
        app.split_ratio = 0.50;
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());

        app.split_ratio = 0.65;
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
        assert_eq!(app.selected_tab, tab_idx);

        let _ = step_events(&ctx, &mut app, &mut frame, key_event(egui::Key::Z, mods));
        assert!((app.split_ratio - 0.50).abs() < 1e-5);
        ctx.input(|i| {
            let z_ev = |e: &egui::Event| {
                matches!(
                    e,
                    egui::Event::Key {
                        key: egui::Key::Z,
                        pressed: true,
                        ..
                    }
                )
            };
            assert!(!i.keys_down.contains(&egui::Key::Z));
            assert!(!i.events.iter().any(z_ev) && !i.raw.events.iter().any(z_ev));
        });

        let _ = step_events(&ctx, &mut app, &mut frame, redo_shortcut_events());
        assert!((app.split_ratio - 0.65).abs() < 1e-5);
        ctx.input(|i| {
            assert!(!i.keys_down.contains(&egui::Key::Y) && !i.keys_down.contains(&egui::Key::Z));
        });
    }
    assert_file_shortcuts_stripped(&ctx, mods);
}

#[cfg(feature = "pure_math")]
fn graph_next_node_id(app: &MathExplorerApp, pane_idx: usize) -> usize {
    let saved = app.pane_tab(pane_idx).unwrap().save_state().unwrap();
    let v: serde_json::Value = serde_json::from_str(&saved).unwrap();
    let entry = &v["tool_states"]["Graph Editor"];
    let g = if let Some(s) = entry.as_str() {
        serde_json::from_str::<serde_json::Value>(s).unwrap()
    } else {
        entry.clone()
    };
    g["next_node_id"].as_u64().unwrap() as usize
}

#[cfg(feature = "pure_math")]
fn insert_node_in_pane(app: &mut MathExplorerApp, pane_idx: usize) {
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
    let fw = app.pane_tab_mut(pane_idx).unwrap().framework_mut().unwrap();
    let tool = fw
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<crate::tabs::graph_theory::graph_editor::GraphEditorTool>()
        .unwrap();
    tool.insert_node_from_keyboard(rect);
}

#[cfg(feature = "pure_math")]
#[test]
fn test_graph_editor_mixing_edit_menu_sidebar_buttons_and_ctrl_z_preserves_stacks() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mods = shortcut_mods();

    let graph_idx = find_tab_index(&app, "Graph Theory");
    app.select_tab(&ctx, graph_idx);
    assert!(app.tabs[graph_idx].select_tool_by_name("GraphEditorTool"));
    let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());

    for expected in 1..=3 {
        insert_node_in_pane(&mut app, 0);
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
        assert_eq!(graph_next_node_id(&app, 0), expected);
    }

    app.undo();
    let out = step_events(&ctx, &mut app, &mut frame, Vec::new());
    assert_eq!(graph_next_node_id(&app, 0), 2);
    assert!(app.can_undo() && app.can_redo());

    let _ = click_accesskit_button_by_label(&ctx, &mut app, &mut frame, &out, "↩ Undo");
    let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
    assert_eq!(graph_next_node_id(&app, 0), 1);
    assert!(app.can_undo() && app.can_redo());

    let out = step_events(&ctx, &mut app, &mut frame, key_event(egui::Key::Z, mods));
    assert_eq!(graph_next_node_id(&app, 0), 0);
    assert!(app.can_redo());

    let _ = click_accesskit_button_by_label(&ctx, &mut app, &mut frame, &out, "↪ Redo");
    let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
    assert_eq!(graph_next_node_id(&app, 0), 1);
    assert!(app.can_undo() && app.can_redo());

    app.redo();
    let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
    assert_eq!(graph_next_node_id(&app, 0), 2);

    let _ = step_events(&ctx, &mut app, &mut frame, redo_shortcut_events());
    assert_eq!(graph_next_node_id(&app, 0), 3);
    assert!(!app.can_redo() && app.can_undo());
}

#[cfg(feature = "pure_math")]
#[test]
fn test_ctrl_z_in_split_and_quad_layouts_undoes_one_edit_in_focused_pane() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let mods = shortcut_mods();

    for (layout, target_pane) in [
        (ViewportLayout::SplitHorizontal, 1_usize),
        (ViewportLayout::SplitVertical, 1_usize),
        (ViewportLayout::QuadGrid, 3_usize),
    ] {
        let ctx = egui::Context::default();
        let mut app = MathExplorerApp::default();
        let mut frame = eframe::Frame::_new_kittest();
        let graph_idx = find_tab_index(&app, "Graph Theory");

        app.set_layout(layout);
        app.set_pane_tab(target_pane, graph_idx);
        app.focus_pane(target_pane);
        assert!(app
            .pane_tab_mut(target_pane)
            .unwrap()
            .select_tool_by_name("GraphEditorTool"));
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());

        insert_node_in_pane(&mut app, target_pane);
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
        insert_node_in_pane(&mut app, target_pane);
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
        assert_eq!(graph_next_node_id(&app, target_pane), 2);

        let _ = step_events(&ctx, &mut app, &mut frame, key_event(egui::Key::Z, mods));
        assert_eq!(graph_next_node_id(&app, target_pane), 1);
    }
}

#[cfg(feature = "physics")]
#[test]
fn test_continuous_physics_trajectories_excluded_from_undo_history() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let play = key_event(egui::Key::Space, egui::Modifiers::NONE);

    for (tab, tool) in [("MRI Physics", ""), ("Quantum Mechanics", "SpinVisualizer")] {
        let idx = find_tab_index(&app, tab);
        app.selected_tab = idx;
        assert!(tool.is_empty() || app.tabs[idx].select_tool_by_name(tool));
        let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
        if !tool.is_empty() {
            let _ = step_events(&ctx, &mut app, &mut frame, play.clone());
        }
        let baseline_undo = app.session_manager.undo_stack.len();
        for _ in 0..20 {
            let _ = step_events(&ctx, &mut app, &mut frame, Vec::new());
        }
        assert_eq!(app.session_manager.undo_stack.len(), baseline_undo);
    }
}
