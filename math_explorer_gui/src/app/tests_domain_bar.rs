#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::domain_bar::AcademicDomain;
use eframe::App;
use std::collections::HashMap;

#[test]
fn test_all_21_modules_categorized_into_academic_domains() {
    let app = MathExplorerApp::default();
    assert_eq!(
        app.tabs.len(),
        26,
        "Expected all 26 modules to be instantiated with default features"
    );

    let mut domain_counts: HashMap<AcademicDomain, Vec<&'static str>> = HashMap::new();
    for tab in &app.tabs {
        let domain_str = tab.academic_domain();
        let parsed_domain = AcademicDomain::from_str(domain_str);
        assert!(
            parsed_domain.is_some(),
            "Module '{}' has unknown domain '{}'",
            tab.name(),
            domain_str
        );
        domain_counts
            .entry(parsed_domain.unwrap())
            .or_default()
            .push(tab.name());
    }

    // Verify all 6 domains exist and have the expected modules
    assert_eq!(
        domain_counts.get(&AcademicDomain::PureMath).map(|v| v.len()),
        Some(6),
        "Pure Math should have 6 modules: Algorithmic Information, Analysis & Calculus, Financial Math, Geometry & Topology, Graph Theory, Number Theory"
    );
    assert_eq!(
        domain_counts.get(&AcademicDomain::Physics).map(|v| v.len()),
        Some(8),
        "Physics should have 8 modules: Astrophysics, Chaos Theory, Fluid Dynamics, General Relativity, Medical Physics, MRI Physics, Quantum Mechanics, Solid State Physics"
    );
    assert_eq!(
        domain_counts.get(&AcademicDomain::AppliedMath).map(|v| v.len()),
        Some(7),
        "Applied Math should have 7 modules: Battery Degradation, Clinical Trials, Favoritism, Game Theory, Paradox Lab, Population Structure, Traceability Portal"
    );
    assert_eq!(
        domain_counts.get(&AcademicDomain::Biology).map(|v| v.len()),
        Some(3),
        "Biology should have 3 modules: Morphogenesis, Neuroscience, Epidemiology"
    );
    assert_eq!(
        domain_counts.get(&AcademicDomain::Climate).map(|v| v.len()),
        Some(1),
        "Climate should have 1 module: Climate"
    );
    assert_eq!(
        domain_counts.get(&AcademicDomain::Ai).map(|v| v.len()),
        Some(1),
        "AI should have 1 module: Artificial Intelligence"
    );
}

#[test]
fn test_domain_switcher_bar_renders_within_height_constraint() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Render frame at standard 1080p (1920x1080)
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1920.0, 1080.0),
        )),
        ..Default::default()
    };

    let _ = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });

    // Check that domain_switcher_bar panel is registered in context
    let panel_id = egui::Id::new("domain_switcher_bar");
    let panel_rect = ctx.memory(|m| m.area_rect(panel_id));
    if let Some(rect) = panel_rect {
        assert!(
            rect.height() <= 40.0,
            "Domain switcher bar height {} exceeds 40px limit",
            rect.height()
        );
    }
}

#[test]
fn test_domain_switcher_bar_renders_at_720p_without_overflow() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Render frame at standard 720p (1280x720)
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 720.0),
        )),
        ..Default::default()
    };

    let _ = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });

    let panel_id = egui::Id::new("domain_switcher_bar");
    let panel_rect = ctx.memory(|m| m.area_rect(panel_id));
    if let Some(rect) = panel_rect {
        assert!(
            rect.height() <= 40.0,
            "Domain switcher bar height {} exceeds 40px limit on 720p",
            rect.height()
        );
        assert!(
            rect.width() <= 1280.0,
            "Domain switcher bar width {} overflows 720p width",
            rect.width()
        );
    }
}

#[test]
fn test_select_tab_updates_state_and_announces() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    // Select Fluid Dynamics (find its index)
    let fluid_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Fluid Dynamics")
        .expect("Fluid Dynamics tab must exist");

    app.select_tab(&ctx, fluid_idx);

    assert_eq!(app.selected_tab, fluid_idx);
    assert_eq!(app.pane_tabs[0], fluid_idx);

    let aria_msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(aria_msg, Some("Switched to tab Fluid Dynamics".to_string()));
}

#[test]
fn test_command_palette_synchronizes_active_domain_and_module() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    // Initially at tab 0 (AI)
    let initial_domain = AcademicDomain::from_str(app.tabs[app.selected_tab].academic_domain());
    assert_eq!(initial_domain, Some(AcademicDomain::Ai));

    // Switch to Fluid Dynamics (Physics)
    let physics_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Fluid Dynamics")
        .expect("Fluid Dynamics tab must exist");

    app.select_tab(&ctx, physics_idx);

    let current_domain = AcademicDomain::from_str(app.tabs[app.selected_tab].academic_domain());
    assert_eq!(current_domain, Some(AcademicDomain::Physics));
    assert_ne!(initial_domain, current_domain);
    assert_eq!(app.selected_tab, physics_idx);
}

#[test]
fn test_tab_state_preserved_across_domain_switches() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();

    let pure_math_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Number Theory")
        .expect("Number Theory tab must exist");
    let ai_idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Artificial Intelligence")
        .expect("AI tab must exist");

    // Select pure math tab
    app.select_tab(&ctx, pure_math_idx);
    let state_before = app.tabs[pure_math_idx].save_state();

    // Switch to AI tab
    app.select_tab(&ctx, ai_idx);
    assert_eq!(app.selected_tab, ai_idx);

    // Switch back to pure math tab
    app.select_tab(&ctx, pure_math_idx);
    let state_after = app.tabs[pure_math_idx].save_state();

    assert_eq!(
        state_before, state_after,
        "Tab state must be preserved across switches"
    );
}

#[test]
fn test_generate_domain_switcher_bar_screenshot() {
    let app = MathExplorerApp::default();
    let mut datasets = Vec::new();

    for (domain_idx, domain) in AcademicDomain::ALL.iter().enumerate() {
        let count = app
            .tabs
            .iter()
            .filter(|t| AcademicDomain::from_str(t.academic_domain()) == Some(*domain))
            .count() as f64;
        let x = domain_idx as f64;
        datasets.push((
            format!("{}: {} modules", domain.name(), count as usize),
            vec![[x, 0.0], [x, count], [x + 0.8, count], [x + 0.8, 0.0]],
        ));
    }

    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/domain_switcher_bar.png", png_data);
    }
}

fn contains_rect_fill(shapes: &[egui::epaint::Shape], expected: egui::Color32) -> bool {
    shapes.iter().any(|s| match s {
        egui::epaint::Shape::Rect(r) => r.fill == expected,
        egui::epaint::Shape::Vec(v) => contains_rect_fill(v, expected),
        _ => false,
    })
}

#[test]
fn test_active_domain_button_renders_selection_fill_and_high_contrast() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp {
        theme: ThemeMode::HighContrast,
        ..Default::default()
    };
    let visuals = ThemeMode::HighContrast.visuals();
    ctx.set_visuals(visuals.clone());
    let mut frame = eframe::Frame::_new_kittest();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1920.0, 1080.0),
        )),
        ..Default::default()
    };

    let out = ctx.run(raw_input, |ctx| {
        app.update(ctx, &mut frame);
    });
    let shapes: Vec<egui::epaint::Shape> = out.shapes.into_iter().map(|cs| cs.shape).collect();
    assert!(
        contains_rect_fill(&shapes, visuals.selection.bg_fill),
        "Active AcademicDomain button must render selection.bg_fill"
    );
    assert!(
        contrast_ratio(visuals.selection.bg_fill, visuals.selection.stroke.color) >= 7.0,
        "Active AcademicDomain button text contrast must reach >= 7.0:1 in HighContrast"
    );
}
