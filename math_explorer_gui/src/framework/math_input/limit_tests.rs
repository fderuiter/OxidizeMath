#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_math_input_enforces_byte_and_instruction_caps_while_keeping_last_valid() {
    crate::accessibility::clear_test_announcements();
    let mut bar = MathExpressionInputBar::new("y'", "-k * y + sin(t)", vec![])
        .with_axes(&["t", "y"])
        .with_max_instructions(16);
    assert!(bar.is_valid());
    let initial_eval = bar
        .active_compiled()
        .expect("compiled")
        .eval_2d(0.0, 2.0, &[3.0]);
    assert!((initial_eval - (-6.0)).abs() < 1e-12);

    // 1. Oversized input (> MAX_EXPR_BYTES) is rejected in O(1), retains bounded raw_text,
    //    and preserves last_valid_expr & last_valid_compiled.
    let huge = format!("{}1", "1+".repeat(50_000));
    bar.set_expression(&huge);
    assert!(!bar.is_valid());
    assert!(bar.raw_text.len() > MAX_EXPR_BYTES);
    assert!(bar.raw_text.len() <= MAX_EXPR_BYTES + 4);
    let len_err = bar.error_message().expect("length error");
    assert!(len_err.contains("exceeds maximum limit"));
    assert!(bar.last_valid_expr.is_some());
    assert_eq!(
        bar.active_compiled()
            .expect("last valid compiled preserved")
            .eval_2d(0.0, 2.0, &[3.0]),
        initial_eval
    );

    // 2. Formula exceeding max_instructions (16) is rejected while keeping last_valid_compiled.
    let many_terms = ["sin(t)"; 10].join(" + "); // 10*2 + 9 = 29 instructions > 16
    bar.set_expression(&many_terms);
    assert!(!bar.is_valid());
    let inst_err = bar.error_message().expect("instruction error");
    assert!(inst_err.contains("instructions"));
    assert_eq!(
        bar.active_compiled()
            .expect("last valid compiled still preserved")
            .eval_2d(0.0, 2.0, &[3.0]),
        initial_eval
    );

    // Render UI and verify inline error row + assertive live-region announcement.
    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            bar.ui(ui);
        });
    });
    let announcements = crate::accessibility::all_announcements();
    assert!(
        announcements
            .iter()
            .any(|(msg, prio)| prio == "assertive" && msg.contains("instructions"))
    );
    assert!(!out.shapes.is_empty());
}

#[test]
fn test_coupled_bank_enforces_row_and_total_instruction_caps_with_announcements() {
    crate::accessibility::clear_test_announcements();
    let mut bank = CoupledExpressionBank::new(&["t", "y", "v"], &[("y'", "v"), ("v'", "-k * y")])
        .with_initial_params(&[("k", 2.0)])
        .with_max_total_instructions(20);
    assert!(bank.is_valid());
    assert_eq!(bank.total_instruction_count(), 5);

    // Exceed combined instruction cap (two 13-instruction formulas = 26 > 20, each <= 256):
    let expr_13 = "sin(t) + cos(y) + sin(v) + k * y"; // 2 + 1 + 2 + 1 + 2 + 1 + 3 + 1 = 13
    bank.set_expressions(&[expr_13, expr_13]);
    assert!(!bank.is_valid());
    assert!(
        bank.bank_error()
            .expect("bank error")
            .contains("total instructions")
    );
    // Previous valid bytecode (total 5 instructions) must remain active!
    assert_eq!(bank.total_instruction_count(), 5);
    assert_eq!(
        bank.active_compiled()[1].eval_3d(0.0, 3.0, 0.0, bank.param_slice()),
        -6.0
    );

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            bank.ui(ui);
        });
    });
    assert!(
        crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, prio)| prio == "assertive" && msg.contains("total instructions"))
    );

    // Restore valid expressions -> polite recovery announcement.
    bank.set_expressions(&["v", "-k * y"]);
    let _ = ctx.run(egui::RawInput::default(), |c| {
        egui::CentralPanel::default().show(c, |ui| {
            bank.ui(ui);
        });
    });
    assert!(
        crate::accessibility::all_announcements()
            .iter()
            .any(|(msg, prio)| prio == "polite" && msg.contains("valid"))
    );
}
