#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_confounded_default_triggers_simpson_rr_reversal_and_callout() {
    let lab = SimpsonParadoxLab::default();
    assert!(lab.validation_error.is_none());
    let mild = lab.mild_eval.as_ref().expect("mild eval");
    let sev = lab.severe_eval.as_ref().expect("severe eval");
    let pool = lab.pooled_eval.as_ref().expect("pooled eval");

    assert!(mild.risk.relative_risk < 1.0, "Treatment must win Mild");
    assert!(sev.risk.relative_risk < 1.0, "Treatment must win Severe");
    assert!(pool.risk.relative_risk > 1.0, "Treatment must lose Pooled");
    assert!(pool.chi_sq.statistic > 3.84 && pool.chi_sq.is_significant);
    assert!(lab.is_simpson_reversal);

    let reversal_callout = lab
        .callout_manager
        .callouts()
        .iter()
        .find(|c| c.tag.as_deref() == Some("simpson_reversal"))
        .expect("simpson_reversal callout must exist");
    assert!(reversal_callout.text.contains("Simpson Reversal"));
}

#[test]
fn test_stratified_randomizer_resolves_confounding_reversal() {
    let mut lab = SimpsonParadoxLab::default();
    let mut seed_counter = 0_u64;
    lab.balance_via_stratified_randomizer(&mut seed_counter);

    assert_eq!(seed_counter, 1);
    assert!(lab.validation_error.is_none());
    assert!((lab.config.mild_treatment_share - 0.50).abs() < 1e-6);
    assert!((lab.config.severe_treatment_share - 0.50).abs() < 1e-6);

    let pool = lab.pooled_eval.as_ref().expect("balanced pooled eval");
    assert!(
        pool.risk.relative_risk < 1.0,
        "Stratified balancing must restore RR_pooled < 1.0, got {}",
        pool.risk.relative_risk
    );
    assert!(!lab.is_simpson_reversal);
}

#[test]
fn test_zero_count_guard_surfaces_clear_validation_error() {
    let mut lab = SimpsonParadoxLab::default();
    lab.config.mild_counts.treat_event = 0;
    lab.recalculate();

    assert!(lab.validation_error.is_some());
    assert!(lab.mild_eval.is_none() && lab.pooled_eval.is_none());
    assert!(!lab.is_simpson_reversal);
    assert!(lab.config.validate().is_err());
}

// [cite:clinical_trials_statistics]
