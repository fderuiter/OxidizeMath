#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::viewport::focus::{focus_pane_interactive, pane_focus_border_color};
use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements, with_live_region};
use eframe::egui;
use eframe::egui::accesskit::{NodeId, Role};
use eframe::App;
use std::collections::{HashMap, HashSet};

fn run_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 900.0));
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

fn has_ancestor(parents: &HashMap<NodeId, NodeId>, mut cur: NodeId, target: NodeId) -> bool {
    for _ in 0..64 {
        if cur == target {
            return true;
        }
        let Some(&next) = parents.get(&cur) else {
            return false;
        };
        cur = next;
    }
    false
}

fn shape_has_stroke_color(shape: &egui::Shape, target: egui::Color32) -> bool {
    match shape {
        egui::Shape::Rect(r) => r.stroke.color == target && r.stroke.width >= 2.0,
        egui::Shape::Vec(v) => v.iter().any(|s| shape_has_stroke_color(s, target)),
        _ => false,
    }
}

fn verify_layout_regions_and_unique_ids(layout: ViewportLayout) {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    app.set_layout(layout);
    for p in 0..layout.pane_count() {
        app.set_pane_tab(p, 0);
    }

    let out = run_frame(&ctx, &mut app, Vec::new());
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update expected");

    let mut seen_ids = HashSet::new();
    let mut parents = HashMap::new();
    for (id, node) in &ak.nodes {
        assert!(
            seen_ids.insert(*id),
            "Duplicate NodeId {id:?} in {layout:?}"
        );
        for &child in node.children() {
            assert!(
                parents.insert(child, *id).is_none(),
                "Child {child:?} claimed by multiple parents in {layout:?}"
            );
        }
    }

    let region_ids: Vec<NodeId> = layout
        .pane_labels()
        .iter()
        .map(|&expected| {
            ak.nodes
                .iter()
                .find(|(_, n)| n.role() == Role::Region && n.label() == Some(expected))
                .unwrap_or_else(|| panic!("Missing Role::Region '{expected}' in {layout:?}"))
                .0
        })
        .collect();

    let pane_widgets: Vec<NodeId> = ak
        .nodes
        .iter()
        .filter(|(_, n)| {
            matches!(
                n.role(),
                Role::Button | Role::Slider | Role::TextInput | Role::CheckBox
            ) && n.bounds().is_some_and(|b| b.y0 >= 62.0)
        })
        .map(|(id, _)| *id)
        .collect();
    assert!(!pane_widgets.is_empty());
    for wid in pane_widgets {
        assert!(
            region_ids
                .iter()
                .any(|&rid| has_ancestor(&parents, wid, rid)),
            "Widget {wid:?} missing Role::Region ancestor in {layout:?}"
        );
    }
}

#[test]
fn test_accesskit_named_regions_and_zero_node_id_collisions() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for layout in [
        ViewportLayout::SplitHorizontal,
        ViewportLayout::SplitVertical,
        ViewportLayout::QuadGrid,
    ] {
        verify_layout_regions_and_unique_ids(layout);
    }
}

#[test]
fn test_f6_shift_f6_and_focus_pane_menu_cycle_without_pointer_hover() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);
    for p in 0..4 {
        app.set_pane_tab(p, p % app.tabs.len().max(1));
    }

    let _ = run_frame(
        &ctx,
        &mut app,
        vec![egui::Event::PointerMoved(egui::pos2(120.0, 200.0))],
    );
    assert_eq!(app.focused_pane, 0);
    clear_test_announcements();

    let central_rect = egui::Rect::from_min_max(egui::pos2(0.0, 62.0), egui::pos2(1280.0, 900.0));
    let pane_rects = viewport::calculate_pane_rects(ViewportLayout::QuadGrid, central_rect);

    for expected_pane in [1usize, 2, 3, 0] {
        let _ = run_frame(
            &ctx,
            &mut app,
            vec![key_press(egui::Key::F6, egui::Modifiers::NONE)],
        );
        assert_eq!(app.focused_pane, expected_pane);
        let fid = ctx.memory(|m| m.focused()).expect("widget focus expected");
        let resp = ctx.read_response(fid).expect("focused widget response");
        assert!(pane_rects[expected_pane].contains(resp.rect.center()));
    }

    let _ = run_frame(
        &ctx,
        &mut app,
        vec![key_press(egui::Key::F6, egui::Modifiers::SHIFT)],
    );
    assert_eq!(app.focused_pane, 3);

    focus_pane_interactive(&mut app, &ctx, 2);
    let _ = run_frame(&ctx, &mut app, Vec::new());
    assert_eq!(app.focused_pane, 2);
    let fid = ctx.memory(|m| m.focused()).expect("widget focus expected");
    assert!(pane_rects[2].contains(ctx.read_response(fid).unwrap().rect.center()));

    let msgs = all_announcements();
    assert!(msgs
        .iter()
        .any(|(m, _)| m.contains("Top-Right Pane (Pane 2 of 4)")));
    assert!(msgs
        .iter()
        .any(|(m, _)| m.contains("Bottom-Left Pane (Pane 3 of 4)")));
    assert!(msgs
        .iter()
        .any(|(m, _)| m.contains("Bottom-Right Pane (Pane 4 of 4)")));
    with_live_region(|r| {
        assert!(r.slots[r.active_slot].contains("Bottom-Left Pane (Pane 3 of 4)"));
    });
}

#[test]
fn test_menu_modal_and_sash_focus_never_activate_simulation_controls() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(gt_idx) = app.tabs.iter().position(|t| t.name() == "Graph Theory") else {
        return;
    };
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, gt_idx);
    app.pane_tab_mut(0)
        .unwrap()
        .select_tool_by_name("Algorithm Visualizer");
    let _ = run_frame(&ctx, &mut app, Vec::new());

    let sash_id = viewport::sash::split_sash_id();
    ctx.memory_mut(|m| m.request_focus(sash_id));
    let _ = run_frame(&ctx, &mut app, Vec::new());
    clear_test_announcements();
    for key in [egui::Key::Space, egui::Key::Enter, egui::Key::Tab] {
        let _ = run_frame(&ctx, &mut app, vec![key_press(key, egui::Modifiers::NONE)]);
    }
    assert!(!all_announcements()
        .iter()
        .any(|(m, _)| m.contains("Simulation playing")));

    app.show_help_menu = true;
    let _ = run_frame(&ctx, &mut app, Vec::new());
    clear_test_announcements();
    for key in [egui::Key::Space, egui::Key::Enter, egui::Key::Tab] {
        let _ = run_frame(&ctx, &mut app, vec![key_press(key, egui::Modifiers::NONE)]);
    }
    assert!(!all_announcements()
        .iter()
        .any(|(m, _)| m.contains("Simulation playing")));
    app.show_help_menu = false;

    app.command_palette.open();
    let _ = run_frame(&ctx, &mut app, Vec::new());
    clear_test_announcements();
    for key in [egui::Key::Space, egui::Key::Tab] {
        let _ = run_frame(&ctx, &mut app, vec![key_press(key, egui::Modifiers::NONE)]);
    }
    assert!(!all_announcements()
        .iter()
        .any(|(m, _)| m.contains("Simulation playing")));
}

fn setup_dual_sir_app(ctx: &egui::Context) -> Option<(MathExplorerApp, egui::FullOutput)> {
    let mut app = MathExplorerApp::default();
    let epi_idx = app.tabs.iter().position(|t| t.name() == "Epidemiology")?;
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, epi_idx);
    app.set_pane_tab(1, epi_idx);
    for p in 0..2 {
        let fw = app.pane_tab_mut(p).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("SirTool"));
    }
    focus_pane_interactive(&mut app, ctx, 1);
    let out = run_frame(ctx, &mut app, Vec::new());
    Some((app, out))
}

#[test]
fn test_secondary_pane_reflective_copy_emits_root_copy_text() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let Some((mut app, out)) = setup_dual_sir_app(&ctx) else {
        return;
    };
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let pos = ak
        .nodes
        .iter()
        .find_map(|(_, n)| {
            if n.role() == Role::Button
                && n.label()
                    .is_some_and(|l| l.contains("📋") || l.contains("Copy"))
            {
                let b = n.bounds()?;
                (b.x0 >= 640.0)
                    .then(|| egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32))
            } else {
                None
            }
        })
        .expect("Secondary pane must expose a reflective_ui copy button");

    let _ = run_frame(
        &ctx,
        &mut app,
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
    let click_out = run_frame(
        &ctx,
        &mut app,
        vec![egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert!(click_out
        .platform_output
        .commands
        .iter()
        .any(|cmd| matches!(cmd, egui::OutputCommand::CopyText(_))));
}

#[test]
fn test_secondary_pane_plot_navigation_and_csv_error_live_region() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let Some((mut app, out)) = setup_dual_sir_app(&ctx) else {
        return;
    };
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let target = ak
        .nodes
        .iter()
        .find_map(|(id, n)| {
            (n.role() == Role::GraphicsDocument && n.bounds()?.x0 >= 640.0).then_some(*id)
        })
        .expect("Secondary pane plot node expected");

    let _ = run_frame(
        &ctx,
        &mut app,
        vec![egui::Event::AccessKitActionRequest(
            egui::accesskit::ActionRequest {
                action: egui::accesskit::Action::Focus,
                target,
                data: None,
            },
        )],
    );
    clear_test_announcements();
    let _ = run_frame(
        &ctx,
        &mut app,
        vec![key_press(egui::Key::ArrowRight, egui::Modifiers::NONE)],
    );
    assert!(all_announcements()
        .iter()
        .any(|(m, p)| m.contains("Point ") && p == "polite"));

    clear_test_announcements();
    egui_plot::emit_plot_live_message(
        &ctx,
        "CSV Import Error: Invalid numeric column".into(),
        true,
    );
    assert!(all_announcements()
        .iter()
        .any(|(m, p)| m.contains("CSV Import Error") && p == "assertive"));
}

#[test]
fn test_scoped_temp_keys_and_wcag_g195_focus_border_contrast() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let Some(epi_idx) = app.tabs.iter().position(|t| t.name() == "Epidemiology") else {
        return;
    };
    app.set_layout(ViewportLayout::QuadGrid);
    for (p, beta) in [(0, 0.25), (1, 2.75), (2, 0.9), (3, 1.5)] {
        app.set_pane_tab(p, epi_idx);
        let fw = app.pane_tab_mut(p).unwrap().framework_mut().unwrap();
        assert!(fw.select_tool_by_name("SirTool"));
        fw.active_tool.as_mut().unwrap().load_state(&format!(
            r#"{{"n":1000.0,"i0":10.0,"beta":{beta},"gamma":0.1,"duration":100.0}}"#
        ));
    }

    app.pane_tab_mut(0)
        .unwrap()
        .framework_mut()
        .unwrap()
        .show_theory_portal = true;
    app.pane_tab_mut(1)
        .unwrap()
        .framework_mut()
        .unwrap()
        .show_starter_gallery = true;
    let _ = run_frame(&ctx, &mut app, Vec::new());
    let _ = run_frame(&ctx, &mut app, Vec::new());

    let fw0 = app.pane_tab(0).unwrap().framework().unwrap();
    let fw1 = app.pane_tab(1).unwrap().framework().unwrap();
    assert!(fw0.show_theory_portal && !fw0.show_starter_gallery);
    assert!(!fw1.show_theory_portal && fw1.show_starter_gallery);
    assert_ne!(
        fw0.export_datasets_with_fallback(),
        fw1.export_datasets_with_fallback()
    );

    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        app.theme = theme;
        let out = run_frame(&ctx, &mut app, Vec::new());
        let visuals = theme.visuals();
        let border = pane_focus_border_color(&visuals);
        assert!(contrast_ratio(border, visuals.panel_fill) >= 4.5);
        assert!(contrast_ratio(border, visuals.extreme_bg_color) >= 4.5);
        assert!(out
            .shapes
            .iter()
            .any(|cs| shape_has_stroke_color(&cs.shape, border)));
    }
}
