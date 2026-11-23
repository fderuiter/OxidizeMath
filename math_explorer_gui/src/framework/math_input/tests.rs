#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use std::collections::BTreeMap;

#[test]
fn test_error_and_recovery_announcements() {
    crate::accessibility::clear_test_announcements();

    let mut bar = MathExpressionInputBar::new("y'", "sin(t)", vec![]);
    let ctx = egui::Context::default();

    // Initial valid render: no announcements should be dispatched
    let _ = ctx.run(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            bar.ui(ui);
        });
    });
    assert_eq!(crate::accessibility::all_announcements().len(), 0);

    // Enter invalid syntax: should trigger assertive announcement
    bar.set_expression("1 +");
    let _ = ctx.run(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            bar.ui(ui);
        });
    });
    let announcements = crate::accessibility::all_announcements();
    assert_eq!(announcements.len(), 1);
    assert_eq!(announcements[0].1, "assertive");
    assert!(announcements[0].0.starts_with("Syntax Error:"));

    // Subsequent frame with unchanged error: should NOT re-announce
    let _ = ctx.run(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            bar.ui(ui);
        });
    });
    assert_eq!(crate::accessibility::all_announcements().len(), 1);

    // Correct invalid expression to valid: should trigger polite validation announcement
    bar.set_expression("1 + 2");
    let _ = ctx.run(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            bar.ui(ui);
        });
    });
    let announcements = crate::accessibility::all_announcements();
    assert_eq!(announcements.len(), 2);
    assert_eq!(
        announcements[1],
        (
            "Mathematical expression is valid".to_string(),
            "polite".to_string()
        )
    );

    // Subsequent valid frames: should NOT re-announce
    let _ = ctx.run(Default::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            bar.ui(ui);
        });
    });
    assert_eq!(crate::accessibility::all_announcements().len(), 2);
}

#[test]
fn test_plain_language_error_messages_have_no_debug_spans_or_eof() {
    for invalid in ["1 +", "(1 + 2", "1 + @", "sin(", "foo(x)"] {
        crate::accessibility::clear_test_announcements();
        let mut bar = MathExpressionInputBar::new("y'", invalid, vec![]);
        let msg = bar
            .error_message()
            .expect("expected syntax error message for invalid input");
        assert!(
            !msg.contains("Span {"),
            "leaked raw Span struct in '{msg}' for '{invalid}'"
        );
        assert!(
            !msg.contains("<EOF>"),
            "leaked <EOF> token in '{msg}' for '{invalid}'"
        );
        assert!(
            msg.contains("character ") || msg.contains("end of expression"),
            "expected 1-indexed character position or 'end of expression' in '{msg}'"
        );

        let ctx = egui::Context::default();
        let _ = ctx.run(Default::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                bar.ui(ui);
            });
        });
        for (announced, _) in crate::accessibility::all_announcements() {
            assert!(!announced.contains("Span {"));
            assert!(!announced.contains("<EOF>"));
        }
    }
}

fn collect_text_colors(shapes: &[egui::epaint::Shape], out: &mut Vec<(String, egui::Color32)>) {
    for shape in shapes {
        match shape {
            egui::epaint::Shape::Text(t) => {
                let color = t
                    .override_text_color
                    .or_else(|| t.galley.job.sections.first().map(|s| s.format.color))
                    .unwrap_or(egui::Color32::TRANSPARENT);
                out.push((t.galley.job.text.clone(), color));
            }
            egui::epaint::Shape::Vec(inner) => collect_text_colors(inner, out),
            _ => {}
        }
    }
}

#[test]
fn test_syntax_error_uses_semantic_palette_error_across_themes() {
    for mode in [
        crate::app::ThemeMode::Dark,
        crate::app::ThemeMode::Light,
        crate::app::ThemeMode::HighContrast,
    ] {
        let mut bar = MathExpressionInputBar::new("f(x)", "foo(x)", vec![]).with_axes(&["x"]);
        let ctx = egui::Context::default();
        ctx.set_visuals(mode.visuals());
        let out = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                bar.ui(ui);
            });
        });
        let shapes: Vec<_> = out.shapes.into_iter().map(|cs| cs.shape).collect();
        let mut rendered = Vec::new();
        collect_text_colors(&shapes, &mut rendered);
        let err_entry = rendered
            .iter()
            .find(|(text, _)| text.contains("Syntax Error:"))
            .expect("expected rendered syntax error label");
        assert!(err_entry.0.contains("character 1"));
        assert!(err_entry.0.contains("unknown function 'foo'"));
        assert!(!err_entry.0.contains("<EOF>") && !err_entry.0.contains("Span {"));
        assert_eq!(err_entry.1, mode.palette().error);
        assert!(crate::app::contrast_ratio(err_entry.1, mode.visuals().panel_fill) >= 4.5);
    }
}

#[test]
fn test_parameter_slider_synthesis_persistence_and_last_valid_fallback() {
    let mut bar = MathExpressionInputBar::new("f(x)", "a * sin(k * x)", vec![]).with_axes(&["x"]);
    assert!(bar.is_valid());
    assert_eq!(bar.param_slice(), &[1.0, 1.0]);

    let mut saved = BTreeMap::new();
    saved.insert("a".to_string(), 2.5);
    saved.insert("k".to_string(), 0.5);
    bar.set_params_map(&saved);
    assert_eq!(bar.param_slice(), &[2.5, 0.5]);
    assert_eq!(bar.params_map(), saved);

    let val_before = bar
        .active_compiled()
        .unwrap()
        .eval_1d(1.0, bar.param_slice());
    assert!((val_before - 2.5 * 0.5_f64.sin()).abs() < 1e-12);

    // Transient syntax edit keeps last valid compiled expression and synchronized parameter slice
    bar.set_expression("a * sin(k * x) +");
    assert!(!bar.is_valid());
    assert!(bar.compiled().is_none());
    let val_during_edit = bar
        .active_compiled()
        .unwrap()
        .eval_1d(1.0, bar.param_slice());
    assert!((val_during_edit - val_before).abs() < 1e-12);

    // Completing edit preserves 'a' and 'k' while defaulting newly discovered 'b' to 1.0
    bar.set_expression("a * sin(k * x) + b");
    assert!(bar.is_valid());
    assert_eq!(bar.param_slice(), &[2.5, 0.5, 1.0]);

    // Exceeding 6 free parameters produces clear validation error while retaining last valid plot
    bar.set_expression("p1 + p2 + p3 + p4 + p5 + p6 + p7 + x");
    assert!(!bar.is_valid());
    assert!(bar.error_message().unwrap().contains("maximum 6"));
    assert!(bar.active_compiled().is_some());
}

#[test]
fn test_cached_expr_persists_across_incomplete_keystrokes() {
    let mut bar = MathExpressionInputBar::new("y'", "-k * y * sin(t)", vec![]);
    assert!(bar.is_valid());
    let valid_ast = bar.expr().cloned().expect("initial expr valid");

    bar.set_expression("cos(");
    assert!(!bar.is_valid());
    assert!(bar.expr().is_none());
    assert!(bar.error_message().is_some());
    assert_eq!(bar.cached_expr(), Some(&valid_ast));

    bar.set_expression("sin(t, y)");
    assert!(!bar.is_valid());
    assert!(bar.expr().is_none());
    assert_eq!(bar.cached_expr(), Some(&valid_ast));

    bar.set_expression("cos(t) - y");
    assert!(bar.is_valid());
    assert_eq!(bar.cached_expr(), bar.expr());
    assert_ne!(bar.cached_expr(), Some(&valid_ast));
}

#[test]
fn test_coupled_expression_bank_shared_params_transient_edits_and_zero_alloc() {
    let mut bank = CoupledExpressionBank::new(
        &["t", "y", "v"],
        &[("y'", "a * y - b * y * v"), ("v'", "d * y * v - g * v")],
    )
    .with_initial_params(&[("a", 1.5), ("b", 1.0), ("d", 0.75), ("g", 1.25)]);

    assert!(bank.is_valid());
    assert_eq!(bank.param_names(), &["a", "b", "d", "g"]);
    assert_eq!(bank.param_slice(), &[1.5, 1.0, 0.75, 1.25]);

    let eval_pair = |b: &CoupledExpressionBank| -> (f64, f64) {
        let c = b.active_compiled();
        let p = b.param_slice();
        (
            c[0].eval_3d(0.0, 2.0, 3.0, p),
            c[1].eval_3d(0.0, 2.0, 3.0, p),
        )
    };

    let baseline = eval_pair(&bank);
    assert!((baseline.0 - (1.5 * 2.0 - 1.0 * 2.0 * 3.0)).abs() < 1e-12);
    assert!((baseline.1 - (0.75 * 2.0 * 3.0 - 1.25 * 3.0)).abs() < 1e-12);

    // Transient syntax edit on second equation preserves last valid bytecode and shared sliders
    bank.set_expression(1, "d * y * v -");
    assert!(!bank.is_valid());
    assert_eq!(eval_pair(&bank), baseline);

    // Restoring valid equation and exceeding 6 union parameters retains valid bytecode
    bank.set_expressions(&["p1 * y + p2 + p3 + p4", "p5 * v + p6 + p7"]);
    assert!(!bank.is_valid());
    assert!(bank.bank_error().unwrap().contains("maximum 6"));
    assert_eq!(eval_pair(&bank), baseline);

    // Zero heap allocations under VerifiedAllocator during inner-loop evaluation
    bank.set_expressions(&["a * y - b * y * v", "d * y * v - g * v"]);
    assert!(bank.is_valid());
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let compiled = bank.active_compiled();
    let params = bank.param_slice();
    let (acc, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut sum = 0.0;
        for step in 0..64 {
            let y = 1.0 + f64::from(step) * 0.01;
            let v = 0.5 + f64::from(step) * 0.02;
            sum += compiled[0].eval_3d(0.0, y, v, params) + compiled[1].eval_3d(0.0, y, v, params);
        }
        sum
    });
    assert!(acc.is_finite());
}
