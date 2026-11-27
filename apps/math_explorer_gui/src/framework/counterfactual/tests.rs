#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::palette::{DARK_PALETTE, HIGH_CONTRAST_PALETTE, LIGHT_PALETTE};
use crate::app::theme::ThemeMode;
use eframe::egui;

fn relative_luminance(c: egui::Color32) -> f64 {
    fn chan(v: u8) -> f64 {
        let s = f64::from(v) / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }
    0.2126 * chan(c.r()) + 0.7152 * chan(c.g()) + 0.0722 * chan(c.b())
}

fn contrast_ratio(a: egui::Color32, b: egui::Color32) -> f64 {
    let la = relative_luminance(a);
    let lb = relative_luminance(b);
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

#[test]
fn test_perturbation_delta_sanitization_and_engine_bounds() {
    assert_eq!(sanitize_perturbation_delta(0.0), 0.0);
    assert_eq!(sanitize_perturbation_delta(-1.0), 0.0);
    assert_eq!(sanitize_perturbation_delta(f64::NAN), 0.0);
    assert_eq!(sanitize_perturbation_delta(1e-12), MIN_PERTURBATION_DELTA);
    assert_eq!(sanitize_perturbation_delta(5.0), MAX_PERTURBATION_DELTA);
    assert!((sanitize_perturbation_delta(1e-5) - 1e-5).abs() < 1e-15);

    let mut engine = CounterfactualBranchEngine::default();
    assert!((engine.effective_delta() - 1e-5).abs() < 1e-15);
    engine.set_effective_delta(0.0);
    assert_eq!(engine.effective_delta(), 0.0);
    engine.set_effective_delta(1e-3);
    assert!((engine.effective_delta() - 1e-3).abs() < 1e-15);
}

#[test]
fn test_separation_tracker_zero_parity_and_positive_exponential_slope() {
    let mut parity = SeparationTracker::new(1.0, 0.0);
    for step in 1..=1000 {
        parity.observe(1.0 + step as f64 * 0.01, 0.0);
    }
    assert_eq!(parity.l2_separation(), 0.0);
    assert_eq!(parity.peak_separation(), 0.0);
    assert_eq!(parity.log_separation_slope(), 0.0);

    let mut chaotic = SeparationTracker::new(1.0, 1e-5);
    let expected_lambda = 0.9;
    for step in 1..=200 {
        let dt = step as f64 * 0.01;
        chaotic.observe(1.0 + dt, 1e-5 * (expected_lambda * dt).exp());
    }
    assert!((chaotic.log_separation_slope() - expected_lambda).abs() < 1e-9);
    assert!(chaotic.peak_separation() > 1e-5);
}

#[test]
fn test_divergence_strip_and_twin_strokes_meet_wcag_across_all_themes() {
    for (mode, palette) in [
        (ThemeMode::Dark, DARK_PALETTE),
        (ThemeMode::Light, LIGHT_PALETTE),
        (ThemeMode::HighContrast, HIGH_CONTRAST_PALETTE),
    ] {
        let visuals = mode.visuals();
        assert!(
            contrast_ratio(palette.accent, visuals.extreme_bg_color) >= 4.5,
            "Accent token failed 4.5:1 contrast in {mode:?}"
        );
        assert!(
            contrast_ratio(palette.muted_text, palette.surface_fill) >= 4.5,
            "Divergence strip muted text failed 4.5:1 contrast in {mode:?}"
        );
        for series_color in palette.series {
            assert!(
                contrast_ratio(series_color, visuals.extreme_bg_color) >= 3.0,
                "Twin curve stroke failed 3.0:1 contrast in {mode:?}"
            );
        }
    }
}
