#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui::{self, vec2, Event, Id, Key, Modifiers, Pos2, RawInput, Rect};
use eframe::App;

fn screen_raw() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn key_raw(key: Key, modifiers: Modifiers) -> RawInput {
    let mut raw = screen_raw();
    raw.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    });
    raw
}

fn run_app_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    raw: RawInput,
) -> egui::FullOutput {
    ctx.run(raw, |ctx| {
        app.update(ctx, frame);
    })
}

fn find_node_by_label(
    out: &egui::FullOutput,
    label_substr: &str,
) -> Option<egui::accesskit::NodeId> {
    let ak = out.platform_output.accesskit_update.as_ref()?;
    ak.nodes
        .iter()
        .find(|(_, node)| {
            node.label().unwrap_or("").contains(label_substr)
                || node.value().unwrap_or("").contains(label_substr)
        })
        .map(|(id, _)| *id)
}

fn click_accesskit_node(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    frame: &mut eframe::Frame,
    target: egui::accesskit::NodeId,
) -> egui::FullOutput {
    let mut raw = screen_raw();
    raw.events.push(Event::AccessKitActionRequest(
        egui::accesskit::ActionRequest {
            action: egui::accesskit::Action::Click,
            target,
            data: None,
        },
    ));
    run_app_frame(ctx, app, frame, raw)
}

fn twin_slot_active_text() -> String {
    crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone())
}

fn live_message(ctx: &egui::Context) -> Option<String> {
    ctx.data(|d| d.get_temp::<String>(Id::new("aria_live_message")))
}

#[test]
fn test_cold_boot_opens_global_starter_gallery_and_session_or_url_dismisses_it() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    assert!(app.starter_gallery.is_open);
    assert!(app.starter_gallery.is_cold_boot);

    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    assert!(find_node_by_label(&out, "Starter Template Gallery").is_some());
    assert!(find_node_by_label(&out, "All Domains (97)").is_some());
    assert!(find_node_by_label(&out, "Inverse Challenges (5)").is_some());
    for domain in AcademicDomain::ALL {
        assert!(
            find_node_by_label(&out, domain.name()).is_some(),
            "Missing domain chip '{}' on cold boot",
            domain.name()
        );
    }

    // Restoring saved session JSON dismisses cold-boot gallery
    let saved_json = app.save_state_to_json();
    let mut restored_app = MathExplorerApp::default();
    assert!(restored_app.starter_gallery.is_open);
    restored_app.load_state_from_json(&saved_json);
    assert!(!restored_app.starter_gallery.is_open);

    // Hydrating a shared URL fragment also dismisses cold-boot gallery
    let share_url = app.encode_current_share_url().unwrap();
    let hash_idx = share_url.find('#').unwrap();
    let mut url_app = MathExplorerApp::default();
    assert!(url_app
        .apply_share_location_strings(&share_url[hash_idx..], "", None)
        .unwrap());
    assert!(!url_app.starter_gallery.is_open);
}

#[test]
fn test_global_starter_gallery_escape_and_close_button_restore_focus_and_announce() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Dismiss cold-boot first so we can focus a background control cleanly
    app.starter_gallery.close_silently();
    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let trigger_id = Id::new("focus_probe_before_catalog");
    let _ = ctx.run(screen_raw(), |ctx| {
        egui::Area::new(Id::new("probe_area")).show(ctx, |ui| {
            let r = ui.add(egui::Button::new("Probe"));
            ctx.memory_mut(|m| m.request_focus(r.id));
            ctx.data_mut(|d| d.insert_temp(trigger_id, r.id));
        });
    });
    let expected_focus = ctx.data(|d| d.get_temp::<Id>(trigger_id)).unwrap();

    // Open gallery and dismiss via Escape
    app.open_explore_catalog(&ctx);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Starter template gallery opened")
    );
    let _ = ctx.run(screen_raw(), |ctx| {
        egui::Area::new(Id::new("probe_area")).show(ctx, |ui| {
            let _ = ui.add(egui::Button::new("Probe"));
        });
        app.render_global_starter_gallery(ctx);
    });
    let _ = ctx.run(key_raw(Key::Escape, Modifiers::NONE), |ctx| {
        egui::Area::new(Id::new("probe_area")).show(ctx, |ui| {
            let _ = ui.add(egui::Button::new("Probe"));
        });
        app.render_global_starter_gallery(ctx);
    });
    assert!(!app.starter_gallery.is_open);
    assert_eq!(ctx.memory(|m| m.focused()), Some(expected_focus));
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Starter template gallery closed")
    );

    // Re-open gallery in full app and dismiss via Close button click
    app.open_explore_catalog(&ctx);
    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let close_node = find_node_by_label(&out, "Close Gallery").expect("Close Gallery button");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, close_node);
    assert!(!app.starter_gallery.is_open);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Starter template gallery closed")
    );
}

#[test]
fn test_starter_gallery_live_search_and_domain_chips_filter_and_announce_counts() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());

    // Click Biology domain chip -> filters to Biology presets and announces count
    let bio_node = find_node_by_label(&out, "Biology (").expect("Biology filter chip");
    let out = click_accesskit_node(&ctx, &mut app, &mut frame, bio_node);
    assert_eq!(
        app.starter_gallery.filter.selected_domain,
        Some(AcademicDomain::Biology)
    );
    let msg = live_message(&ctx).unwrap_or_default();
    assert!(
        msg.contains("starter presets shown in Biology"),
        "Expected Biology filter announcement, got '{msg}'"
    );
    assert_eq!(twin_slot_active_text(), msg);
    assert!(find_node_by_label(&out, "Launch Turing Spots").is_some());

    // Enter search query "kermack" -> narrows live and announces count
    app.starter_gallery.filter.search_query = "kermack".to_string();
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    assert!(find_node_by_label(&out, "Kermack-McKendrick SIR Model").is_some());
    let msg = live_message(&ctx).unwrap_or_default();
    assert!(
        msg.contains("starter preset") && msg.contains("Biology"),
        "Expected live search announcement, got '{msg}'"
    );

    // Enter unmatched query "klein" -> 0 matches + Show All Presets recovery button
    app.starter_gallery.filter.search_query = "klein".to_string();
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("No matching starter presets in Biology")
    );
    let reset_node = find_node_by_label(&out, "Show All Presets").expect("Show All Presets button");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, reset_node);
    assert!(app.starter_gallery.filter.search_query.is_empty());
    assert!(app.starter_gallery.filter.selected_domain.is_none());
}

#[test]
fn test_explore_catalog_from_standalone_traceability_tab_via_shortcut_and_top_bar() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let trace_idx = app
        .find_tab_index_for_domain("traceability")
        .expect("Traceability tab index");
    assert!(app.tabs[trace_idx].framework().is_none());
    app.select_tab(&ctx, trace_idx);
    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    assert!(!app.starter_gallery.is_open);

    // Ctrl+Shift+G / Cmd+Shift+G opens global catalog from TraceabilityTab
    let shift_mods = if cfg!(target_os = "macos") {
        Modifiers::MAC_CMD | Modifiers::SHIFT
    } else {
        Modifiers::CTRL | Modifiers::SHIFT
    };
    let _ = run_app_frame(&ctx, &mut app, &mut frame, key_raw(Key::G, shift_mods));
    assert!(app.starter_gallery.is_open);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Starter template gallery opened")
    );

    // Pressing shortcut again closes it cleanly
    let _ = run_app_frame(&ctx, &mut app, &mut frame, key_raw(Key::G, shift_mods));
    assert!(!app.starter_gallery.is_open);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Starter template gallery closed")
    );

    // Clicking "🧭 Explore Catalog" in top domain switcher bar opens global catalog
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let explore_btn = find_node_by_label(&out, "Explore Catalog").expect("Explore Catalog button");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, explore_btn);
    assert!(app.starter_gallery.is_open);
}

#[test]
fn test_per_tab_starter_gallery_prefilters_active_domain_and_expands_to_all_domains() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let epi_idx = app
        .find_tab_index_for_domain("epidemiology")
        .expect("Epidemiology tab");
    app.select_tab(&ctx, epi_idx);
    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    assert!(!app.starter_gallery.is_open);

    // Click per-tab "🎨 Starter Gallery" button
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let per_tab_btn = find_node_by_label(&out, "Starter Gallery").expect("Per-tab Starter Gallery");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, per_tab_btn);
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());

    // Pre-filtered to Biology (with Kermack-McKendrick SIR Model visible)
    assert!(find_node_by_label(&out, "of 97)").is_some());
    assert!(find_node_by_label(&out, "(97 of 97)").is_none());
    assert!(find_node_by_label(&out, "Kermack-McKendrick SIR Model").is_some());

    // One-click expansion to "All Domains (97)" without closing modal
    let all_chip = find_node_by_label(&out, "All Domains (97)").expect("All Domains chip");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, all_chip);
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    assert!(find_node_by_label(&out, "(97 of 97)").is_some());
    assert!(
        app.tabs[epi_idx].framework().unwrap().show_starter_gallery,
        "Per-tab modal must stay open after expanding to All Domains"
    );
}

#[test]
fn test_modal_and_ctrl_k_preset_selection_switches_focused_secondary_pane_and_loads_preset() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.starter_gallery.close_silently();
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.focus_pane(1);
    let primary_tab_before = app.pane_tabs[0];

    // 1. Launch "Turing Spots" from GlobalStarterGallery into focused secondary pane 1
    app.open_explore_catalog(&ctx);
    app.starter_gallery.filter.search_query = "Turing Spots".to_string();
    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let launch_btn = find_node_by_label(&out, "Launch Turing Spots").expect("Launch Turing Spots");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, launch_btn);

    let morph_idx = app.find_tab_index_for_domain("biology").unwrap();
    assert_eq!(app.pane_tabs[0], primary_tab_before);
    assert_eq!(app.pane_tabs[1], morph_idx);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Launched starter preset Turing Spots in Morphogenesis")
    );

    // 2. Launch "Mandelbrot Explorer" from Ctrl+K into focused secondary pane 1
    app.command_palette.open();
    app.command_palette.query = "Mandelbrot Explorer".to_string();
    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let _ = run_app_frame(
        &ctx,
        &mut app,
        &mut frame,
        key_raw(Key::Enter, Modifiers::NONE),
    );

    let chaos_idx = app.find_tab_index_for_domain("chaos").unwrap();
    assert_eq!(app.pane_tabs[0], primary_tab_before);
    assert_eq!(app.pane_tabs[1], chaos_idx);
    assert_eq!(
        app.pane_tab_mut(1).unwrap().active_tool_id(),
        Some("FractalViewer")
    );
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Launched starter preset Mandelbrot Explorer in Chaos Theory")
    );
}

#[test]
fn test_command_palette_workspace_actions_update_layout_theme_catalog_and_exports() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    let mut exec_query = |app: &mut MathExplorerApp, query: &str| {
        app.command_palette.open();
        app.command_palette.query = query.to_string();
        let _ = run_app_frame(&ctx, app, &mut frame, screen_raw());
        let _ = run_app_frame(&ctx, app, &mut frame, key_raw(Key::Enter, Modifiers::NONE));
    };

    // Layout action
    exec_query(&mut app, "2x2 Quad Grid");
    assert_eq!(app.viewport_layout, ViewportLayout::QuadGrid);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Switched viewport layout to 2x2 Quad Grid")
    );
    assert_eq!(
        twin_slot_active_text(),
        "Switched viewport layout to 2x2 Quad Grid"
    );

    // Theme action
    exec_query(&mut app, "High Contrast");
    assert_eq!(app.theme, ThemeMode::HighContrast);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Switched theme to High Contrast")
    );

    // Explore Catalog action
    exec_query(&mut app, "Explore Catalog");
    assert!(app.starter_gallery.is_open);
    assert_eq!(
        live_message(&ctx).as_deref(),
        Some("Starter template gallery opened")
    );
    app.starter_gallery.close_silently();

    // Export Preset JSON action on an active simulation tool
    let epi_idx = app.find_tab_index_for_domain("epidemiology").unwrap();
    app.select_tab(&ctx, epi_idx);
    exec_query(&mut app, "Save Parameter Preset JSON");
    let msg = live_message(&ctx).unwrap_or_default();
    assert!(
        msg.contains("preset JSON"),
        "Expected polite export announcement for preset JSON, got '{msg}'"
    );
}

#[test]
fn test_global_starter_gallery_inverse_challenges_tab_launches_in_focused_pane() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    app.starter_gallery.close_silently();
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.focus_pane(1);
    let primary_tab_before = app.pane_tabs[0];

    app.open_explore_catalog(&ctx);
    let _ = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let challenges_tab =
        find_node_by_label(&out, "Inverse Challenges (5)").expect("Inverse Challenges sub-tab");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, challenges_tab);

    app.starter_gallery.filter.search_query = "therapeutic".to_string();
    let out = run_app_frame(&ctx, &mut app, &mut frame, screen_raw());
    let start_btn = find_node_by_label(&out, "Start Multi-Dose Therapeutic Window")
        .expect("Start challenge button");
    let _ = click_accesskit_node(&ctx, &mut app, &mut frame, start_btn);

    let trials_idx = app.find_tab_index_for_domain("clinical_trials").unwrap();
    assert_eq!(app.pane_tabs[0], primary_tab_before);
    assert_eq!(app.pane_tabs[1], trials_idx);
    assert_eq!(
        app.pane_tab_mut(1).unwrap().active_challenge_id(),
        Some("pk_therapeutic_window")
    );
}
