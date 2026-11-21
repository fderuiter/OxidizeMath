#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::sash::{focus_outline_color, split_sash_id};
use super::ViewportLayout;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::app::{contrast_ratio, MathExplorerApp, ThemeMode};
use eframe::egui;
use eframe::App;

fn run_app_frame(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    events: Vec<egui::Event>,
) -> egui::FullOutput {
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0));
    ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            events,
            ..Default::default()
        },
        |c| app.update(c, &mut frame),
    )
}

fn key_event(key: egui::Key, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

fn tab_until_sash_focused(
    ctx: &egui::Context,
    app: &mut MathExplorerApp,
    max_tabs: usize,
) -> Option<usize> {
    let sash_id = split_sash_id();
    for step in 1..=max_tabs {
        let _ = run_app_frame(
            ctx,
            app,
            vec![key_event(egui::Key::Tab, egui::Modifiers::NONE)],
        );
        if ctx.memory(|m| m.has_focus(sash_id)) {
            return Some(step);
        }
    }
    None
}

fn has_stroke_color(shapes: &[egui::epaint::ClippedShape], target: egui::Color32) -> bool {
    shapes.iter().any(|cs| shape_uses_stroke(&cs.shape, target))
}

fn shape_uses_stroke(shape: &egui::Shape, target: egui::Color32) -> bool {
    match shape {
        egui::Shape::Rect(rect_shape) => rect_shape.stroke.color == target,
        egui::Shape::Vec(nested) => nested.iter().any(|s| shape_uses_stroke(s, target)),
        _ => false,
    }
}

#[test]
fn test_tab_navigation_reaches_and_leaves_split_sash_in_dual_layouts() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    for layout in [
        ViewportLayout::SplitHorizontal,
        ViewportLayout::SplitVertical,
    ] {
        let ctx = egui::Context::default();
        let mut app = MathExplorerApp::default();
        app.set_layout(layout);
        let _ = run_app_frame(&ctx, &mut app, Vec::new());

        let steps = tab_until_sash_focused(&ctx, &mut app, 40);
        assert!(
            steps.is_some(),
            "Tab navigation must reach the split sash in {layout:?}"
        );

        let sash_id = split_sash_id();
        assert!(ctx.memory(|m| m.has_focus(sash_id)));

        // Pressing Tab again must advance focus to the next control without trapping focus
        let _ = run_app_frame(
            &ctx,
            &mut app,
            vec![key_event(egui::Key::Tab, egui::Modifiers::NONE)],
        );
        assert!(
            !ctx.memory(|m| m.has_focus(sash_id)),
            "Tab must advance past the split sash to child/header controls"
        );
        assert!(ctx.memory(|m| m.focused().is_some()));

        // Shift+Tab must navigate focus back to the split sash (egui settles Previous focus on next pass)
        let _ = run_app_frame(
            &ctx,
            &mut app,
            vec![key_event(egui::Key::Tab, egui::Modifiers::SHIFT)],
        );
        if !ctx.memory(|m| m.has_focus(sash_id)) {
            let _ = run_app_frame(&ctx, &mut app, Vec::new());
        }
        assert!(
            ctx.memory(|m| m.has_focus(sash_id)),
            "Shift+Tab must return focus to the split sash"
        );
    }
}

#[test]
fn test_quad_grid_and_single_layouts_do_not_expose_split_sash() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    for layout in [ViewportLayout::Single, ViewportLayout::QuadGrid] {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = MathExplorerApp::default();
        app.set_layout(layout);
        let out = run_app_frame(&ctx, &mut app, Vec::new());

        let steps = tab_until_sash_focused(&ctx, &mut app, 30);
        assert!(
            steps.is_none(),
            "Layout {layout:?} must not allocate a dual-pane split sash"
        );
        let splitter_count = out
            .platform_output
            .accesskit_update
            .as_ref()
            .map_or(0, |ak| {
                ak.nodes
                    .iter()
                    .filter(|(_, n)| n.role() == egui::accesskit::Role::Splitter)
                    .count()
            });
        match layout {
            ViewportLayout::Single => assert_eq!(splitter_count, 0),
            ViewportLayout::QuadGrid => assert_eq!(splitter_count, 2),
            _ => unreachable!(),
        }
    }
}

#[test]
fn test_arrow_keys_step_five_percent_and_clamp_strictly() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    let _ = run_app_frame(&ctx, &mut app, Vec::new());

    assert!(tab_until_sash_focused(&ctx, &mut app, 40).is_some());
    clear_test_announcements();

    // ArrowRight -> +5% (55%), sash must retain focus
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowRight, egui::Modifiers::NONE)],
    );
    assert!((app.split_ratio - 0.55).abs() < 1e-4);
    assert!(ctx.memory(|m| m.has_focus(split_sash_id())));

    // ArrowDown -> +5% (60%)
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowDown, egui::Modifiers::NONE)],
    );
    assert!((app.split_ratio - 0.60).abs() < 1e-4);

    // ArrowLeft -> -5% (55%), ArrowUp -> -5% (50%)
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowLeft, egui::Modifiers::NONE)],
    );
    assert!((app.split_ratio - 0.55).abs() < 1e-4);
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowUp, egui::Modifiers::NONE)],
    );
    assert!((app.split_ratio - 0.50).abs() < 1e-4);

    let announcements = all_announcements();
    assert_eq!(announcements.len(), 4);
    assert_eq!(
        announcements[0],
        ("Split ratio 55%".to_string(), "polite".to_string())
    );
    assert_eq!(
        announcements[3],
        ("Split ratio 50%".to_string(), "polite".to_string())
    );

    // Clamp at upper bound 80%
    for _ in 0..10 {
        let _ = run_app_frame(
            &ctx,
            &mut app,
            vec![key_event(egui::Key::ArrowRight, egui::Modifiers::NONE)],
        );
    }
    assert!((app.split_ratio - 0.80).abs() < 1e-4);

    // Clamp at lower bound 20%
    for _ in 0..20 {
        let _ = run_app_frame(
            &ctx,
            &mut app,
            vec![key_event(egui::Key::ArrowLeft, egui::Modifiers::NONE)],
        );
    }
    assert!((app.split_ratio - 0.20).abs() < 1e-4);
}

#[test]
fn test_shift_arrow_precision_steps_and_home_enter_reset() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitVertical);
    let _ = run_app_frame(&ctx, &mut app, Vec::new());

    assert!(tab_until_sash_focused(&ctx, &mut app, 40).is_some());
    clear_test_announcements();

    // Shift + ArrowRight -> +1% (51%)
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowRight, egui::Modifiers::SHIFT)],
    );
    assert!((app.split_ratio - 0.51).abs() < 1e-4);

    // Shift + ArrowDown -> +1% (52%)
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowDown, egui::Modifiers::SHIFT)],
    );
    assert!((app.split_ratio - 0.52).abs() < 1e-4);

    // Shift + ArrowLeft -> -1% (51%)
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowLeft, egui::Modifiers::SHIFT)],
    );
    assert!((app.split_ratio - 0.51).abs() < 1e-4);

    // Home resets to 50%
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::Home, egui::Modifiers::NONE)],
    );
    assert!((app.split_ratio - 0.50).abs() < 1e-4);

    // Move to 35% via Shift+ArrowUp + ArrowUp, then Enter resets to 50%
    app.split_ratio = 0.36;
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::ArrowUp, egui::Modifiers::SHIFT)],
    );
    assert!((app.split_ratio - 0.35).abs() < 1e-4);

    clear_test_announcements();
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![key_event(egui::Key::Enter, egui::Modifiers::NONE)],
    );
    assert!((app.split_ratio - 0.50).abs() < 1e-4);
    assert_eq!(
        all_announcements(),
        vec![("Split ratio 50%".to_string(), "polite".to_string())]
    );
}

#[test]
fn test_accesskit_splitter_role_value_and_widget_info() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    app.split_ratio = 0.65;

    let out = run_app_frame(&ctx, &mut app, Vec::new());
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update must be emitted");

    let (_, node) = ak
        .nodes
        .iter()
        .find(|(_, n)| n.role() == egui::accesskit::Role::Splitter)
        .expect("Split sash AccessKit Splitter node must exist");

    assert_eq!(node.role(), egui::accesskit::Role::Splitter);
    assert_eq!(node.label(), Some("Horizontal viewport split divider"));
    assert_eq!(node.value(), Some("65%"));
    assert_eq!(node.numeric_value(), Some(65.0));
    assert_eq!(node.min_numeric_value(), Some(20.0));
    assert_eq!(node.max_numeric_value(), Some(80.0));
    assert_eq!(node.numeric_value_step(), Some(5.0));
    assert_eq!(
        node.orientation(),
        Some(egui::accesskit::Orientation::Vertical)
    );
    assert!(node.supports_action(egui::accesskit::Action::Focus));
    assert!(node.supports_action(egui::accesskit::Action::SetValue));
    assert!(node.supports_action(egui::accesskit::Action::Increment));
    assert!(node.supports_action(egui::accesskit::Action::Decrement));

    let info: egui::WidgetInfo = ctx
        .data(|d| d.get_temp(split_sash_id().with("sash_widget_info")))
        .expect("WidgetInfo must be registered for split sash");
    assert_eq!(info.value, Some(65.0));
    assert_eq!(info.current_text_value.as_deref(), Some("65%"));
}

#[test]
fn test_accesskit_splitter_actions_update_ratio_and_announce() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitVertical);
    app.split_ratio = 0.65;

    let out = run_app_frame(&ctx, &mut app, Vec::new());
    let sash_ak_id = out
        .platform_output
        .accesskit_update
        .as_ref()
        .and_then(|ak| {
            ak.nodes
                .iter()
                .find(|(_, n)| n.role() == egui::accesskit::Role::Splitter)
                .map(|(id, _)| *id)
        })
        .expect("Splitter node must exist");

    clear_test_announcements();
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![egui::Event::AccessKitActionRequest(
            egui::accesskit::ActionRequest {
                action: egui::accesskit::Action::Increment,
                target: sash_ak_id,
                data: None,
            },
        )],
    );
    assert!((app.split_ratio - 0.70).abs() < 1e-4);
    assert_eq!(
        all_announcements(),
        vec![("Split ratio 70%".to_string(), "polite".to_string())]
    );

    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![egui::Event::AccessKitActionRequest(
            egui::accesskit::ActionRequest {
                action: egui::accesskit::Action::SetValue,
                target: sash_ak_id,
                data: Some(egui::accesskit::ActionData::NumericValue(40.0)),
            },
        )],
    );
    assert!((app.split_ratio - 0.40).abs() < 1e-4);
}

#[test]
fn test_focus_outline_rendering_and_theme_contrast() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let visuals = theme.visuals();
        let outline = focus_outline_color(&visuals);
        let ratio = contrast_ratio(outline, visuals.panel_fill);
        assert!(
            ratio >= 4.5,
            "Focus outline contrast ({ratio:.2}:1) must exceed 4.5:1 in {theme:?}"
        );
    }

    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);

    let unfocused_out = run_app_frame(&ctx, &mut app, Vec::new());
    let expected_outline = focus_outline_color(&app.theme.visuals());
    assert!(!has_stroke_color(&unfocused_out.shapes, expected_outline));

    assert!(tab_until_sash_focused(&ctx, &mut app, 40).is_some());
    let focused_out = run_app_frame(&ctx, &mut app, Vec::new());
    assert!(
        has_stroke_color(&focused_out.shapes, expected_outline),
        "Focused sash must render high-contrast focus outline"
    );
}

#[test]
fn test_pointer_drag_and_cursor_icons_preserved() {
    let _guard = crate::reflective_ui::tests::TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.set_layout(ViewportLayout::SplitHorizontal);
    let _ = run_app_frame(&ctx, &mut app, Vec::new());

    let hover_out = run_app_frame(
        &ctx,
        &mut app,
        vec![egui::Event::PointerMoved(egui::pos2(600.0, 400.0))],
    );
    assert_eq!(
        hover_out.platform_output.cursor_icon,
        egui::CursorIcon::ResizeColumn
    );

    clear_test_announcements();
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![
            egui::Event::PointerMoved(egui::pos2(600.0, 400.0)),
            egui::Event::PointerButton {
                pos: egui::pos2(600.0, 400.0),
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
    );
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![egui::Event::PointerMoved(egui::pos2(720.0, 400.0))],
    );
    let _ = run_app_frame(
        &ctx,
        &mut app,
        vec![egui::Event::PointerButton {
            pos: egui::pos2(720.0, 400.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    );
    assert!((app.split_ratio - 0.60).abs() < 1e-2);
    assert!(!all_announcements().is_empty());
}
