#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::domain_bar::AcademicDomain;
use eframe::egui;
use eframe::App;
use std::collections::HashSet;

fn collect_text_rects(shapes: &[egui::epaint::ClippedShape], out: &mut Vec<(String, egui::Rect)>) {
    for cs in shapes {
        collect_shape_text(&cs.shape, out);
    }
}

fn collect_shape_text(shape: &egui::Shape, out: &mut Vec<(String, egui::Rect)>) {
    match shape {
        egui::Shape::Text(text_shape) => {
            let rect = egui::Rect::from_min_size(text_shape.pos, text_shape.galley.size());
            out.push((text_shape.galley.text().to_string(), rect));
        }
        egui::Shape::Vec(nested) => {
            for s in nested {
                collect_shape_text(s, out);
            }
        }
        _ => {}
    }
}

fn screen_input(events: Vec<egui::Event>, time: f64) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        time: Some(time),
        events,
        ..Default::default()
    }
}

fn next_sim_time(ctx: &egui::Context) -> f64 {
    ctx.data_mut(|d| {
        let id = egui::Id::new("sim_test_clock");
        let t = d.get_temp::<f64>(id).unwrap_or(0.0) + 1.0;
        d.insert_temp(id, t);
        t
    })
}

fn run_menu_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    events: Vec<egui::Event>,
) -> Vec<(String, egui::Rect)> {
    let t = next_sim_time(ctx);
    let output = ctx.run(screen_input(events, t), |ctx| {
        egui::TopBottomPanel::top("main_menu").show(ctx, |ui| {
            ui_panels::render_main_menu(app, ctx, ui);
        });
    });
    let mut texts = Vec::new();
    collect_text_rects(&output.shapes, &mut texts);
    texts
}

fn click_events(pos: egui::Pos2) -> Vec<egui::Event> {
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
}

fn find_text_pos(texts: &[(String, egui::Rect)], needle: &str) -> egui::Pos2 {
    texts
        .iter()
        .find(|(t, _)| t == needle || t == &format!("● {}", needle))
        .unwrap_or_else(|| {
            panic!(
                "Expected text '{}' in rendered UI; found: {:?}",
                needle,
                texts.iter().map(|(t, _)| t).collect::<Vec<_>>()
            )
        })
        .1
        .center()
}

#[test]
fn test_view_menu_height_reduction_and_primary_module_domains() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    let texts = run_menu_frame(&ctx, &mut app, Vec::new());
    let view_pos = find_text_pos(&texts, "View");
    let _ = run_menu_frame(&ctx, &mut app, click_events(view_pos));
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(view_pos)]);

    // Flat 21-module View menu previously exceeded 500px; hierarchical View menu must reduce by >=50% (<220px)
    let cmd_y = find_text_pos(&texts, "🔍 Command Palette...").y;
    let diag_y = find_text_pos(&texts, "Issues & Diagnostics Panel").y;
    let view_menu_span = (diag_y - cmd_y).abs() + 32.0;
    assert!(
        view_menu_span < 220.0,
        "View dropdown span {} must decrease by >=50% compared to ~540px flat list",
        view_menu_span
    );

    // Hover Primary Module and verify all 6 academic domain submenus appear
    let primary_pos = find_text_pos(&texts, "Primary Module");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);

    for domain in AcademicDomain::ALL {
        let present = texts
            .iter()
            .any(|(t, _)| t == domain.name() || t == domain.active_name());
        assert!(
            present,
            "Primary Module submenu must contain academic domain '{}'",
            domain.name()
        );
    }
}

#[test]
fn test_primary_module_each_domain_shows_only_belonging_modules() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    let texts = run_menu_frame(&ctx, &mut app, Vec::new());
    let view_pos = find_text_pos(&texts, "View");
    let _ = run_menu_frame(&ctx, &mut app, click_events(view_pos));
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(view_pos)]);
    let primary_pos = find_text_pos(&texts, "Primary Module");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);
    let domain_texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);

    let mut discovered_modules = HashSet::new();
    for &domain in AcademicDomain::ALL {
        let dom_rect = domain_texts
            .iter()
            .find(|(t, _)| t == domain.name() || t == domain.active_name())
            .expect("Domain submenu item must exist")
            .1;
        let dom_pos = dom_rect.center();
        let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(dom_pos)]);
        let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(dom_pos)]);
        let sub_texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(dom_pos)]);

        for tab in &app.tabs {
            let belongs = AcademicDomain::from_str(tab.academic_domain()) == Some(domain);
            // Scope lookup to child flyout column to the right of the domain button
            let shown = sub_texts
                .iter()
                .any(|(t, r)| t == tab.name() && r.min.x > dom_rect.max.x);
            assert_eq!(
                shown,
                belongs,
                "Module '{}' visibility in domain '{}' must be {}",
                tab.name(),
                domain.name(),
                belongs
            );
            if shown {
                discovered_modules.insert(tab.name());
            }
        }
    }

    assert_eq!(
        discovered_modules.len(),
        21,
        "All 21 simulation modules must appear under their academic domain"
    );
}

#[test]
fn test_primary_module_selection_indicator_and_aria_announcement() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    crate::accessibility::clear_test_announcements();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    let texts = run_menu_frame(&ctx, &mut app, Vec::new());
    let view_pos = find_text_pos(&texts, "View");
    let _ = run_menu_frame(&ctx, &mut app, click_events(view_pos));
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(view_pos)]);
    let primary_pos = find_text_pos(&texts, "Primary Module");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);
    let physics_pos = find_text_pos(&texts, "Physics");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(physics_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(physics_pos)]);

    let fluid_pos = find_text_pos(&texts, "Fluid Dynamics");
    let _ = run_menu_frame(&ctx, &mut app, click_events(fluid_pos));

    let expected_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Fluid Dynamics")
        .unwrap();
    assert_eq!(app.selected_tab, expected_idx);
    assert_eq!(app.pane_tabs[0], expected_idx);

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg.as_deref(), Some("Switched to tab Fluid Dynamics"));

    // Re-open menu and verify active domain ("● Physics") and active module ("✓") indicators
    let texts = run_menu_frame(&ctx, &mut app, Vec::new());
    let view_pos = find_text_pos(&texts, "View");
    let _ = run_menu_frame(&ctx, &mut app, click_events(view_pos));
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(view_pos)]);
    let primary_pos = find_text_pos(&texts, "Primary Module");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(primary_pos)]);
    assert!(
        texts.iter().any(|(t, _)| t == "● Physics"),
        "Active domain must display selection indicator"
    );

    let physics_pos = find_text_pos(&texts, "Physics");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(physics_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(physics_pos)]);
    assert!(
        texts.iter().any(|(t, _)| t == "✓"),
        "Active module must display checkmark selection indicator"
    );
}

#[test]
fn test_pane_assignments_quad_grid_categorized_by_domain() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::QuadGrid);

    let texts = run_menu_frame(&ctx, &mut app, Vec::new());
    let view_pos = find_text_pos(&texts, "View");
    let _ = run_menu_frame(&ctx, &mut app, click_events(view_pos));
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(view_pos)]);

    let assign_pos = find_text_pos(&texts, "Pane Assignments");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(assign_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(assign_pos)]);

    for prefix in ViewportLayout::QuadGrid.pane_label_prefixes() {
        assert!(
            texts.iter().any(|(t, _)| t == *prefix),
            "Pane Assignments must list configured pane '{}'",
            prefix
        );
    }

    // Open Bottom-Right Pane (index 3) -> Biology -> Morphogenesis
    let br_pos = find_text_pos(&texts, "Bottom-Right Pane:");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(br_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(br_pos)]);
    for domain in AcademicDomain::ALL {
        assert!(
            texts
                .iter()
                .any(|(t, _)| t == domain.name() || t == domain.active_name()),
            "Pane assignment submenu must show domain '{}'",
            domain.name()
        );
    }

    let bio_pos = find_text_pos(&texts, "Biology");
    let _ = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(bio_pos)]);
    let texts = run_menu_frame(&ctx, &mut app, vec![egui::Event::PointerMoved(bio_pos)]);
    let morph_pos = find_text_pos(&texts, "Morphogenesis");
    let _ = run_menu_frame(&ctx, &mut app, click_events(morph_pos));

    let morph_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .unwrap();
    assert_eq!(app.pane_tabs[3], morph_idx);
    assert_eq!(app.pane_tabs[0], 0);
    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(
        aria_msg.as_deref(),
        Some("Assigned Morphogenesis to Bottom-Right Pane")
    );
}

fn run_full_app_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    events: Vec<egui::Event>,
) -> Vec<(String, egui::Rect)> {
    let t = next_sim_time(ctx);
    let output = ctx.run(screen_input(events, t), |ctx| {
        app.update(ctx, frame);
    });
    let mut texts = Vec::new();
    collect_text_rects(&output.shapes, &mut texts);
    texts
}

fn hover_full_app_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    pos: egui::Pos2,
) -> Vec<(String, egui::Rect)> {
    run_full_app_frame(ctx, app, frame, vec![egui::Event::PointerMoved(pos)])
}

#[test]
fn test_split_viewport_pane_header_flyout_assigns_biology_module() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.set_layout(ViewportLayout::SplitHorizontal);
    app.set_pane_tab(0, 0);
    app.set_pane_tab(1, 1);
    let right_initial_name = app.tabs[1].name();

    let texts = run_full_app_frame(&ctx, &mut app, &mut frame, Vec::new());
    // Locate Right Pane header selector button in the right half of the 1280px viewport
    let right_selector_pos = texts
        .iter()
        .find(|(t, r)| t == right_initial_name && r.min.x >= 600.0 && r.min.y < 120.0)
        .expect("Right pane header module selector must render")
        .1
        .center();

    // Click 1: open Right Pane header flyout menu and verify domain categories (no flat list)
    let _ = run_full_app_frame(&ctx, &mut app, &mut frame, click_events(right_selector_pos));
    let texts = hover_full_app_frame(&ctx, &mut app, &mut frame, right_selector_pos);
    let bio_pos = texts
        .iter()
        .find(|(t, r)| t == "Biology" && r.min.x >= 600.0)
        .expect("Right pane flyout must display Biology domain submenu")
        .1
        .center();
    assert!(
        !texts
            .iter()
            .any(|(t, r)| t == "Morphogenesis" && r.min.x >= 600.0),
        "Pane header must not show flat module list before opening domain submenu"
    );

    // Hover Biology domain submenu, then Click 2 on Morphogenesis
    let _ = hover_full_app_frame(&ctx, &mut app, &mut frame, bio_pos);
    let texts = hover_full_app_frame(&ctx, &mut app, &mut frame, bio_pos);
    let morph_pos = find_text_pos(&texts, "Morphogenesis");
    let _ = run_full_app_frame(&ctx, &mut app, &mut frame, click_events(morph_pos));

    let morph_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .unwrap();
    assert_eq!(app.pane_tabs[1], morph_idx);
    assert_eq!(app.secondary_tab, morph_idx);
    assert_eq!(app.pane_tabs[0], 0, "Left pane must remain unchanged");
    assert_eq!(app.selected_tab, 0, "Primary tab must remain unchanged");

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(
        aria_msg.as_deref(),
        Some("Assigned Morphogenesis to Right Pane")
    );
}

#[test]
fn test_academic_domain_from_str_variants_and_empty_domain_graceful_skip() {
    assert_eq!(
        AcademicDomain::from_str("Pure Math"),
        Some(AcademicDomain::PureMath)
    );
    assert_eq!(
        AcademicDomain::from_str("  pure_math "),
        Some(AcademicDomain::PureMath)
    );
    assert_eq!(
        AcademicDomain::from_str("applied-math"),
        Some(AcademicDomain::AppliedMath)
    );
    assert_eq!(
        AcademicDomain::from_str("Artificial Intelligence"),
        Some(AcademicDomain::Ai)
    );
    assert_eq!(AcademicDomain::from_str("unknown_category"), None);

    // Simulate disabled optional features where tabs list is empty
    let ctx = egui::Context::default();
    let empty_tabs: Vec<Box<dyn crate::tabs::ExplorerTab>> = Vec::new();
    let output = ctx.run(screen_input(Vec::new(), 1.0), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let res = domain_bar::render_domain_module_submenus(ui, &empty_tabs, 0);
            assert_eq!(res, None);
        });
    });
    assert!(!output.shapes.is_empty());
}
