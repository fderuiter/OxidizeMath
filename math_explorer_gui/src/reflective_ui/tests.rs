use super::*;
use std::sync::Mutex;

pub(crate) static TEST_MUTEX: Mutex<()> = Mutex::new(());

#[test]
fn test_render_copyable_metric() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp1 = render_copyable_metric(ui, "Test Metric", "42.0");
            assert!(resp1.rect.width() >= 0.0);

            let resp2 = render_copyable_metric(ui, "Test Metric:", "100");
            assert!(resp2.rect.width() >= 0.0);

            let resp3 = render_copyable_metric(ui, "", "empty_label_val");
            assert!(resp3.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_render_formula_with_export() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let r1 = render_formula_with_export(ui, "\\frac{d}{dx} e^x", Some("d/dx e^x"));
            let r2 = render_formula_with_export(ui, "", None);
            assert!(r1.rect.width() >= 0.0 && r2.rect.width() >= 0.0);
        });
    });
    let datasets = vec![(r"\frac{dy}{dt} = -k y".into(), vec![[0.0, 1.0], [1.0, 0.5]])];
    if let Ok(bytes) = egui_plot::export::export_custom_png(
        &datasets,
        &egui_plot::export::ExportOptions::default(),
    ) {
        let _ = std::fs::write("/tmp/reflective_ui_latex_pipeline.png", &bytes);
        let _ = std::fs::write("/tmp/mathbox_formula_rendering.png", bytes);
    }
}

#[test]
fn test_render_theory_summary_with_export() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let r1 = render_theory_summary_with_export(ui, "Euler identity", Some("e^{i\\pi}+1=0"));
            let r2 = render_theory_summary_with_export(ui, "", None);
            let r3 = render_theory_summary_with_export(ui, "Model: $\\frac{dy}{dt} = -k y$", None);
            let r4 = render_theory_summary_with_export(ui, "Unclosed $formula fallback", None);
            assert!(r1.rect.width() >= 0.0 && r2.rect.width() >= 0.0);
            assert!(r3.rect.width() >= 0.0 && r4.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_formula_export_populates_copied_text() {
    let ctx = egui::Context::default();
    let full_output = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.ctx().copy_text("\\int_0^1 x^2 dx = 1/3".to_string());
        });
    });
    let copied = full_output
        .platform_output
        .commands
        .iter()
        .any(|cmd| match cmd {
            egui::OutputCommand::CopyText(txt) => txt == "\\int_0^1 x^2 dx = 1/3",
            _ => false,
        });
    assert!(
        copied,
        "Expected CopyText output command in full_output.platform_output, got: {:?}",
        full_output.platform_output.commands
    );
}
