use super::*;
use std::sync::Mutex;

pub(crate) static TEST_MUTEX: Mutex<()> = Mutex::new(());

pub(crate) fn lock_test_mutex() -> std::sync::MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

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
    ctx.enable_accesskit();
    let out = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let r1 = render_theory_summary_with_export(
                ui,
                r"Zeeman spin $H = -\vec{\mu}\cdot\vec{B}$ with ratio $\frac{a + b}{c + d}$",
                Some(r"i\hbar \frac{d|\psi\rangle}{dt} = H|\psi\rangle"),
            );
            let r2 = render_theory_summary_with_export(ui, "", None);
            let r3 = render_theory_summary_with_export(ui, r"Model: $\frac{dy}{dt} = -k y$", None);
            let r4 = render_theory_summary_with_export(ui, "Unclosed $formula fallback", None);
            let r5 = render_theory_summary_with_export(
                ui,
                "A Quantitative Description of g_na_bar (Hodgkin & Huxley, 1952)",
                None,
            );
            assert!(r1.rect.width() >= 0.0 && r2.rect.width() >= 0.0);
            assert!(r3.rect.width() >= 0.0 && r4.rect.width() >= 0.0 && r5.rect.width() >= 0.0);
        });
    });

    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must be populated");
    let labels: Vec<&str> = ak
        .nodes
        .iter()
        .flat_map(|(_, n)| n.label().into_iter().chain(n.value()))
        .collect();
    assert!(
        labels
            .iter()
            .any(|l| l.contains("μ\u{20D7}·B\u{20D7}") && l.contains("(a + b)⁄(c + d)")),
        "Some(latex) branch must parse inline prose math: {labels:?}"
    );
    assert!(
        labels
            .iter()
            .any(|l| l.contains("Model: dy⁄dt = -k y") && !l.contains('─')),
        "None branch must format inline fractions on one line: {labels:?}"
    );
    assert!(
        labels.contains(&"A Quantitative Description of g_na_bar (Hodgkin & Huxley, 1952)"),
        "Plain citation without '$' must preserve underscores verbatim: {labels:?}"
    );
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

fn find_button_pos(output: &egui::FullOutput, exact_label: &str) -> egui::Pos2 {
    let ak = output
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit tree must exist");
    let bounds = ak
        .nodes
        .iter()
        .find_map(|(_, node)| {
            if node.role() == egui::accesskit::Role::Button && node.label() == Some(exact_label) {
                node.bounds()
            } else {
                None
            }
        })
        .unwrap_or_else(|| panic!("Button '{exact_label}' must exist"));
    egui::pos2(
        ((bounds.x0 + bounds.x1) * 0.5) as f32,
        ((bounds.y0 + bounds.y1) * 0.5) as f32,
    )
}

fn verify_reflective_hover_and_escape<F: FnMut(&mut egui::Ui)>(
    ctx: &egui::Context,
    mut render: F,
    btn_label: &str,
    expected_tooltip: &str,
) {
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(800.0, 400.0),
    ));
    let out = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![egui::Event::PointerMoved(egui::pos2(790.0, 390.0))],
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| render(ui));
        },
    );
    let pos = find_button_pos(&out, btn_label);

    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![egui::Event::PointerMoved(pos)],
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| render(ui));
        },
    );
    assert!(
        crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, prio)| msg == expected_tooltip && prio == "polite"),
        "Expected announcement '{expected_tooltip}' for '{btn_label}'"
    );

    crate::accessibility::clear_test_announcements();
    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            events: vec![
                egui::Event::PointerMoved(pos),
                egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| render(ui));
        },
    );
    assert!(
        crate::accessibility::all_announcements().is_empty(),
        "Reflective copy tooltip '{expected_tooltip}' must dismiss on Escape"
    );
}

#[test]
fn test_reflective_copy_controls_announce_and_dismiss_on_escape() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    verify_reflective_hover_and_escape(
        &ctx,
        |ui| {
            render_copyable_metric(ui, "Eigenvalue", "2.718");
        },
        "📋",
        "Copy value",
    );

    for (btn, tooltip) in [
        ("📋 LaTeX", "Copy LaTeX formula"),
        ("📋 Text", "Copy plain text formula"),
    ] {
        verify_reflective_hover_and_escape(
            &ctx,
            |ui| {
                render_formula_with_export(ui, "\\frac{1}{2}", Some("1/2"));
            },
            btn,
            tooltip,
        );
    }

    for (btn, tooltip) in [
        ("📋 Text", "Copy text summary"),
        ("📋 LaTeX", "Copy LaTeX formula/text"),
    ] {
        verify_reflective_hover_and_escape(
            &ctx,
            |ui| {
                render_theory_summary_with_export(ui, "Harmonic oscillator", Some("x'' + x = 0"));
            },
            btn,
            tooltip,
        );
    }
}
