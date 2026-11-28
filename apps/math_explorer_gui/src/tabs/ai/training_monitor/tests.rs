use super::canvas::{
    boundary_colors_for_theme, lerp_class_bg, BOUNDARY_X_MAX, BOUNDARY_X_MIN, BOUNDARY_Y_MAX,
    BOUNDARY_Y_MIN,
};
use super::*;
use crate::app::{SemanticPalette, ThemeMode};
use eframe::egui::Color32;

fn relative_luminance(c: Color32) -> f64 {
    fn channel(v: u8) -> f64 {
        let s = v as f64 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }
    0.2126 * channel(c.r()) + 0.7152 * channel(c.g()) + 0.0722 * channel(c.b())
}

fn contrast_ratio(a: Color32, b: Color32) -> f64 {
    let la = relative_luminance(a);
    let lb = relative_luminance(b);
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

#[test]
fn test_training_monitor_reset_and_step_once() {
    let mut tool = TrainingMonitorTool {
        learning_rate: 0.05,
        hidden_dim: 32,
        epoch: 10,
        is_training: true,
        ..Default::default()
    };
    tool.loss_history.push([1.0, 0.5]);
    tool.accuracy_history.push([1.0, 0.8]);

    tool.reset();

    assert_eq!(tool.learning_rate, 0.05);
    assert_eq!(tool.hidden_dim, 32);
    assert_eq!(tool.epoch, 0);
    assert!(!tool.is_training);
    assert!(tool.loss_history.is_empty());
    assert!(tool.accuracy_history.is_empty());

    tool.step_once();
    assert!(!tool.is_training);
    assert_eq!(tool.epoch, 1);
    assert_eq!(tool.loss_history.len(), 1);
}

#[test]
fn test_training_monitor_learning_rate_reinit_applies_immediately() {
    let mut slow = TrainingMonitorTool {
        learning_rate: 0.001,
        ..Default::default()
    };
    slow.reinit_model();
    slow.step_once();
    slow.step_once();

    let mut fast = TrainingMonitorTool {
        learning_rate: 0.08,
        ..Default::default()
    };
    fast.reinit_model();
    fast.step_once();
    fast.step_once();

    assert_ne!(slow.loss_history[1][1], fast.loss_history[1][1]);
}

#[test]
fn test_training_monitor_predict_contour_and_optimizer_weight_init_controls() {
    let mut tool = TrainingMonitorTool {
        learning_rate: 0.02,
        hidden_dim: 32,
        optimizer_choice: OptimizerChoice::Sgd,
        weight_init: WeightInitChoice::HeRelu,
        ..Default::default()
    };
    tool.reinit_model();
    for _ in 0..12 {
        tool.step_once();
    }
    assert!(tool.last_grad_norm.is_finite() && tool.last_grad_norm > 0.0);

    tool.ensure_boundary_cache();
    assert!(
        !tool.boundary_cache.contour_segments.is_empty(),
        "Expected live p = 0.50 contour segments evaluated from TrainingLoop::predict"
    );
    for seg in tool.boundary_cache.contour_segments.iter() {
        for pt in seg {
            assert!((BOUNDARY_X_MIN..=BOUNDARY_X_MAX).contains(&pt.x));
            assert!((BOUNDARY_Y_MIN..=BOUNDARY_Y_MAX).contains(&pt.y));
        }
    }

    let min_p = tool
        .boundary_cache
        .prob_class_0
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_p = tool
        .boundary_cache
        .prob_class_0
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        min_p < 0.5 && max_p > 0.5,
        "Expected decision surface probabilities to straddle p = 0.50 (min={min_p}, max={max_p})"
    );

    // Verify state roundtrip with OptimizerChoice::Sgd and WeightInitChoice::XavierGlorot
    tool.weight_init = WeightInitChoice::XavierGlorot;
    let saved = tool.save_state().expect("save state");
    let mut restored = TrainingMonitorTool::default();
    restored.try_load_state(&saved).expect("load state");
    assert_eq!(restored.optimizer_choice, OptimizerChoice::Sgd);
    assert_eq!(restored.weight_init, WeightInitChoice::XavierGlorot);
}

#[test]
fn test_training_monitor_wcag_aa_contrast_across_all_themes() {
    for theme in [ThemeMode::Light, ThemeMode::Dark, ThemeMode::HighContrast] {
        let palette = SemanticPalette::for_theme(theme);
        let b_colors = boundary_colors_for_theme(theme);
        let mid_bg = lerp_class_bg(&b_colors, 0.5);
        for bg in [b_colors.class_0_bg, b_colors.class_1_bg, mid_bg] {
            let c_contour = contrast_ratio(b_colors.contour, bg);
            let c_class0 = contrast_ratio(palette.axis_x, bg);
            let c_class1 = contrast_ratio(palette.axis_z, bg);
            assert!(
                c_contour >= 4.5,
                "Contour contrast {c_contour:.2} < 4.5 in {theme:?}"
            );
            assert!(
                c_class0 >= 4.5,
                "Class 0 scatter contrast {c_class0:.2} < 4.5 in {theme:?}"
            );
            assert!(
                c_class1 >= 4.5,
                "Class 1 scatter contrast {c_class1:.2} < 4.5 in {theme:?}"
            );
        }
    }
}

// [cite:graph_parameters_rust]
