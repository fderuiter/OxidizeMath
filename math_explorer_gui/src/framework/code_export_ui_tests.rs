//! UI modal and inline copy-button tests for unified `.tex`, `.rs`, and `.py` code exports.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use crate::framework::SimulationFramework;
use crate::framework::code_export_modal::CodeExportTab;
use crate::framework::math_input::MathExpressionInputBar;
use crate::tabs::analysis::ode::OdeSolverTool;
use eframe::egui;
use std::collections::HashMap;

#[test]
fn test_code_export_modal_open_tab_switch_and_escape_close() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut fw = SimulationFramework::new("non_existent_domain_xyz");
    fw.active_tool = Some(Box::new(OdeSolverTool::default()));

    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1400.0, 800.0),
    ));
    let raw = || egui::RawInput {
        screen_rect,
        ..Default::default()
    };
    let _ = ctx.run(raw(), |ctx| fw.show(ctx, "modal_code_test"));
    assert!(!fw.show_code_export_modal);

    fw.show_code_export_modal = true;
    fw.code_export_tab = CodeExportTab::Rust;
    let out_open = ctx.run(raw(), |ctx| fw.show(ctx, "modal_code_test"));
    assert!(fw.show_code_export_modal);
    let ak = out_open.platform_output.accesskit_update.as_ref().unwrap();
    assert!(
        ak.nodes
            .iter()
            .any(|(_, n)| n.label().is_some_and(|l| l.contains("Copy Rust")))
    );

    let esc = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let _ = ctx.run(
        egui::RawInput {
            events: vec![esc],
            ..raw()
        },
        |ctx| fw.show(ctx, "modal_code_test"),
    );
    assert!(!fw.show_code_export_modal);
}

#[test]
fn test_math_input_with_params_inline_copy_buttons() {
    let mut bar = MathExpressionInputBar::new("y'", "-k * y * sin(t)", vec![]);
    let mut params = HashMap::new();
    params.insert("k".to_string(), 1.85);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(800.0, 300.0),
    ));
    let out = ctx.run(
        egui::RawInput {
            screen_rect,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                bar.ui_with_params(ui, &params);
            });
        },
    );
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    let py_pos = ak
        .nodes
        .iter()
        .find_map(|(_, n)| {
            (n.role() == egui::accesskit::Role::Button && n.label() == Some("📋 Python"))
                .then(|| n.bounds().unwrap())
        })
        .map(|b| egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32))
        .expect("📋 Python button");

    crate::accessibility::clear_test_announcements();
    let mut copied_py = String::new();
    for pressed in [true, false] {
        let step_out = ctx.run(
            egui::RawInput {
                screen_rect,
                events: vec![
                    egui::Event::PointerMoved(py_pos),
                    egui::Event::PointerButton {
                        pos: py_pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    bar.ui_with_params(ui, &params);
                });
            },
        );
        for cmd in &step_out.platform_output.commands {
            if let egui::OutputCommand::CopyText(txt) = cmd {
                copied_py.clone_from(txt);
            }
        }
    }
    assert_eq!(copied_py, "(-1.85 * y) * np.sin(t)");
}
