#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use scientific_metadata::theory::TheoryDescribable;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct MockTheory;

impl TheoryDescribable for MockTheory {
    fn theory_description(&self) -> String {
        "Mock simulation theory".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Mock simulation theory".to_string()
    }

    fn theory_citation(&self) -> String {
        "Mock (2026)".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

struct CustomExportTool {
    theory: MockTheory,
    steps: Arc<AtomicUsize>,
}

impl InteractiveTool for CustomExportTool {
    fn name(&self) -> &'static str {
        "Custom Wave Solver"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let step = self.steps.load(Ordering::SeqCst) as f64;
        vec![(
            "Wave".to_string(),
            vec![[0.0, step], [1.0, step + 1.0], [2.0, step + 0.5]],
        )]
    }

    fn export_figure_options(&self) -> Option<egui_plot::export::ExportOptions> {
        Some(egui_plot::export::ExportOptions {
            width: 640,
            height: 480,
            aspect_ratio: egui_plot::export::AspectRatioOption::Free,
            background_color: egui::Color32::WHITE,
            ..Default::default()
        })
    }

    fn step_simulation(&mut self) {
        self.steps.fetch_add(1, Ordering::SeqCst);
    }
}

fn find_ak_button_center(output: &egui::FullOutput, label_substr: &str) -> egui::Pos2 {
    let ak_update = output
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must be emitted");
    let bounds = ak_update
        .nodes
        .iter()
        .find_map(|(_, node)| {
            let label = node.label().unwrap_or("");
            if node.role() == egui::accesskit::Role::Button && label.contains(label_substr) {
                node.bounds()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("Button containing '{label_substr}' must exist"));

    egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    )
}

fn has_ak_text(output: &egui::FullOutput, needle: &str) -> bool {
    output
        .platform_output
        .accesskit_update
        .as_ref()
        .is_some_and(|ak| {
            ak.nodes.iter().any(|(_, node)| {
                node.label().is_some_and(|l| l.contains(needle))
                    || node.value().is_some_and(|v| v.contains(needle))
            })
        })
}

fn click_framework_button(
    ctx: &egui::Context,
    framework: &mut SimulationFramework,
    pos: egui::Pos2,
    id_source: &str,
) -> egui::FullOutput {
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1280.0, 800.0),
    ));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| framework.show(ctx, id_source),
    );
    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        },
        |ctx| framework.show(ctx, id_source),
    );
    ctx.run(
        egui::RawInput {
            screen_rect,
            ..Default::default()
        },
        |ctx| framework.show(ctx, id_source),
    )
}

#[test]
fn test_framework_custom_figure_options_and_slug_filenames() {
    let mut framework = SimulationFramework::new("non_existent_domain_xyz");
    let steps = Arc::new(AtomicUsize::new(0));
    framework.active_tool = Some(Box::new(CustomExportTool {
        theory: MockTheory,
        steps,
    }));

    assert!(!framework.show_export_modal);
    assert!(framework.export_bg_dark);

    framework.open_export_modal_for_active_tool();
    assert!(framework.show_export_modal);
    assert_eq!(framework.export_options.width, 640);
    assert_eq!(framework.export_options.height, 480);
    assert_eq!(
        framework.export_options.aspect_ratio,
        egui_plot::export::AspectRatioOption::Free
    );
    assert!(!framework.export_bg_dark);

    let (png, svg, gif) = SimulationFramework::export_filenames_for_tool("Custom Wave Solver");
    assert_eq!(png, "custom_wave_solver_figure.png");
    assert_eq!(svg, "custom_wave_solver_figure.svg");
    assert_eq!(gif, "custom_wave_solver_animation.gif");
}

#[test]
fn test_framework_unified_modal_gif_sampling_and_close_button() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let mut framework = SimulationFramework::new("non_existent_domain_xyz");
    let steps = Arc::new(AtomicUsize::new(0));
    framework.active_tool = Some(Box::new(CustomExportTool {
        theory: MockTheory,
        steps: Arc::clone(&steps),
    }));
    framework.export_gif_frames = 5;

    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1280.0, 800.0),
    ));
    let out1 = ctx.run(
        egui::RawInput {
            screen_rect,
            ..Default::default()
        },
        |ctx| framework.show(ctx, "sim_modal_test"),
    );

    let open_btn = find_ak_button_center(&out1, "Export PNG Image");
    let out_open = click_framework_button(&ctx, &mut framework, open_btn, "sim_modal_test");
    assert!(framework.show_export_modal);
    assert!(has_ak_text(&out_open, egui_plot::MODAL_HEADER_TEXT));
    assert!(has_ak_text(&out_open, "Target Canvas Size: 640 x 480 px"));

    let gif_btn = find_ak_button_center(&out_open, "Save GIF");
    let out_gif = click_framework_button(&ctx, &mut framework, gif_btn, "sim_modal_test");
    assert_eq!(steps.load(Ordering::SeqCst), 5);
    assert_eq!(
        framework.export_status_message.as_deref(),
        Some("GIF animation saved successfully.")
    );

    let close_btn = find_ak_button_center(&out_gif, "Close");
    let _ = click_framework_button(&ctx, &mut framework, close_btn, "sim_modal_test");
    assert!(!framework.show_export_modal);
}

#[test]
fn test_framework_polls_scoped_clipboard_status_when_modal_closed() {
    let ctx = egui::Context::default();
    let mut framework = SimulationFramework::new("non_existent_domain_xyz");
    assert!(!framework.show_export_modal);

    let scope_a = SimulationFramework::export_modal_scope_id("pane_a");
    let scope_b = SimulationFramework::export_modal_scope_id("pane_b");
    ctx.data_mut(|d| {
        d.insert_temp(
            egui_plot::clipboard_status_id(scope_a),
            "Copied plot image to clipboard!".to_string(),
        );
        d.insert_temp(
            egui_plot::clipboard_status_id(scope_b),
            "Pane B status".to_string(),
        );
    });

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        framework.show(ctx, "pane_a");
    });

    assert_eq!(
        framework.export_status_message.as_deref(),
        Some("Copied plot image to clipboard!")
    );
    assert!(ctx
        .data(|d| d.get_temp::<String>(egui_plot::clipboard_status_id(scope_a)))
        .is_none());
    assert_eq!(
        ctx.data(|d| d.get_temp::<String>(egui_plot::clipboard_status_id(scope_b)))
            .as_deref(),
        Some("Pane B status")
    );
}
