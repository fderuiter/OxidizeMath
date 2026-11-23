//! Unit, zero-allocation, and inverse challenge tests for `HiddenMarkovParadoxTool`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::challenge::CHALLENGE_HMM_FORBIDDEN_FORK;
use super::{HiddenMarkovParadoxTool, HmmParadoxConfig, STARTER_PRESET_JSON};
use crate::framework::InteractiveTool;
use crate::starter_gallery::find_challenge_metadata;

fn assert_close(actual: f64, expected: f64, eps: f64) {
    assert!(
        (actual - expected).abs() <= eps,
        "expected {expected}, got {actual} (eps = {eps})"
    );
}

#[test]
fn test_starter_preset_reproduces_forbidden_marginal_viterbi_and_forward_likelihood() {
    let mut tool = HiddenMarkovParadoxTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset must load");

    let b = &tool.baseline;
    assert_eq!(b.marginal_path, vec![0, 1, 1]);
    assert_close(b.marginal_joint_prob, 0.0, 1e-15);
    assert!(b.has_forbidden_marginal_step);
    assert_eq!(b.first_forbidden_step, Some(1));

    assert_eq!(b.viterbi_path, vec![0, 2, 1]);
    assert_close(b.viterbi_joint_prob, 0.028_262_5, 1e-12);
    assert_close(b.forward_likelihood, 0.069_376_5, 1e-12);
    assert_close(b.forward_log_likelihood, (0.069_376_5_f64).ln(), 1e-12);

    let ((eval_base, eval_branch), _cert) =
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            let eb = tool
                .hmm
                .evaluate_with_scratch(&tool.config.observations, &mut tool.baseline_scratch)
                .unwrap();
            let ef = tool
                .hmm
                .evaluate_with_scratch(&tool.branched_observations, &mut tool.branched_scratch)
                .unwrap();
            (eb, ef)
        });
    assert_close(eval_base.likelihood, 0.069_376_5, 1e-12);
    assert!(eval_branch.likelihood > 0.0);
}

#[test]
fn test_seeded_rollout_determinism_and_hmm_challenge_solve() {
    let mut tool_a = HiddenMarkovParadoxTool::default();
    let mut tool_b = HiddenMarkovParadoxTool::default();
    tool_a.simulate_seeded_rollout();
    tool_b.simulate_seeded_rollout();
    assert_eq!(tool_a.config.observations, tool_b.config.observations);
    assert_eq!(tool_a.baseline, tool_b.baseline);

    let meta = find_challenge_metadata(CHALLENGE_HMM_FORBIDDEN_FORK)
        .expect("hmm_trellis_branch_rescue registered");
    let baseline = meta.generate_baseline_state();

    let mut tool = HiddenMarkovParadoxTool::default();
    tool.apply_challenge(Some(CHALLENGE_HMM_FORBIDDEN_FORK))
        .unwrap();
    tool.try_load_state(&baseline).unwrap();
    let base_eval = tool.evaluate_challenge().unwrap();
    assert!(!base_eval.passed);

    // Tamper locked observations & A[1,1], while setting valid fork t=0 -> token 1 ([1, 1, 1])!
    let tampered = serde_json::json!({
        "observations": [2, 2, 2, 2],
        "fork_step": 0,
        "fork_token": 1,
        "shock_self_loop_prob": 0.45,
        "bridge_to_shock_prob": 0.95,
        "shock_emission_focus": 0.85,
        "seed_counter": 42
    })
    .to_string();
    let clamped_str = meta.clamp_state_json(&tampered).unwrap();
    let clamped: HmmParadoxConfig = serde_json::from_str(&clamped_str).unwrap();
    assert_eq!(clamped.observations, vec![0, 1, 1]);
    assert_eq!(clamped.shock_self_loop_prob, 0.0);
    assert_eq!(clamped.seed_counter, 0);
    assert_eq!(clamped.fork_step, 0);
    assert_eq!(clamped.fork_token, 1);

    tool.try_load_state(&clamped_str).unwrap();
    let solved = tool.evaluate_challenge().unwrap();
    assert!(solved.passed);
    assert_close(solved.score, 100.0, 1e-9);
}

// [cite:clinical_trials_statistics]
