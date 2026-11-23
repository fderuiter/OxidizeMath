//! Cross-platform deterministic tests for `domain_ai` stochastic modules with pinned SHA-256 digests.

use domain_ai::ai::deep_learning_theory::linear_algebra::DenseLayer;
use domain_ai::ai::reinforcement_learning::{Action, QLearningAgent, State, TabularQFunction};
use domain_ai::ai::sds::diffusion::{
    NoiseSchedule, generate_noise_with_rng, inject_noise, sample_timestep_with_rng,
};
use math_commons::primitives::UnitInterval;
use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::{OxidizeRng, TrajectoryDigest};

const PINNED_SDS_DIGEST: &str = "79f96ed7b116f1bfbab0916922ef86d9c53dbda72bb07396617d7831433ef02e";
const PINNED_DENSE_DIGEST: &str =
    "ff1cb7e748ea559e52434b714e91552003cf5bd52f409e2230bf1ae741ff74c7";
const PINNED_QLEARNING_DIGEST: &str =
    "e0cf14dd4bdf272f0dca28c0db075c0a296c12ac434567226ffa04ee5ef6e778";

#[test]
#[verified_engine::verified]
fn test_sds_diffusion_schedule_and_noise_pinned_digest() {
    let schedule = NoiseSchedule::new(40);
    let base_img = DMatrix::from_fn(4, 4, |r, c| (r * 4 + c) as f64 / 16.0);

    let run = |seed: u64| -> (Vec<usize>, DMatrix<f64>, String) {
        let mut rng = OxidizeRng::new(seed);
        let mut digest = TrajectoryDigest::new();
        let mut steps = Vec::with_capacity(8);
        for _ in 0..8 {
            let t = sample_timestep_with_rng(40, &mut rng);
            let (sig, noi) = schedule.get_scales(t);
            digest.update_usize(t);
            digest.update_f64(sig);
            digest.update_f64(noi);
            steps.push(t);
        }
        let noise = generate_noise_with_rng(4, 4, &mut rng).unwrap();
        let (sig, noi) = schedule.get_scales(steps[0]);
        let latent = inject_noise(&base_img, &noise, sig, noi);
        digest.update_f64_slice(latent.as_slice());
        (steps, latent, digest.finish_hex())
    };

    let (s1, l1, h1) = run(31415);
    let (s2, l2, h2) = run(31415);
    let (s3, l3, h3) = run(27182);
    assert_eq!(s1, s2);
    assert_eq!(l1, l2);
    assert_eq!(h1, h2);
    assert_ne!(s1, s3);
    assert_ne!(l1, l3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_SDS_DIGEST);
}

#[test]
#[verified_engine::verified]
fn test_dense_layer_threaded_rng_pinned_digest() {
    let run = |seed: u64| -> (DVector<f64>, String) {
        let mut rng = OxidizeRng::new(seed);
        let layer1 = DenseLayer::new_with_rng(5, 4, &mut rng);
        let layer2 = DenseLayer::new_with_rng(5, 4, &mut rng);
        assert_ne!(layer1.weights, layer2.weights);
        let x = DVector::from_fn(5, |i, _| (i + 1) as f64 * 0.25);
        let y = layer2.forward(&DVector::from_fn(5, |i, _| layer1.forward(&x)[i % 4]));
        let mut digest = TrajectoryDigest::new();
        digest.update_f64_slice(layer1.weights.as_slice());
        digest.update_f64_slice(layer1.bias.as_slice());
        digest.update_f64_slice(layer2.weights.as_slice());
        digest.update_f64_slice(layer2.bias.as_slice());
        digest.update_f64_slice(y.as_slice());
        (y, digest.finish_hex())
    };

    let (y1, h1) = run(101);
    let (y2, h2) = run(101);
    let (y3, h3) = run(202);
    assert_eq!(y1, y2);
    assert_eq!(h1, h2);
    assert_ne!(y1, y3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_DENSE_DIGEST);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct LineState(i32);
impl State for LineState {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StepAction {
    Left,
    Right,
}
impl Action for StepAction {}

#[test]
#[verified_engine::verified]
fn test_q_learning_rollout_pinned_digest() {
    let actions = [StepAction::Left, StepAction::Right];
    let run = |seed: u64| -> (Vec<f64>, String) {
        let mut agent =
            QLearningAgent::<LineState, StepAction, TabularQFunction<LineState, StepAction>>::new(
                UnitInterval::new(0.2).unwrap(),
                UnitInterval::new(0.9).unwrap(),
                UnitInterval::new(0.25).unwrap(),
                Some(seed),
            );
        let mut pos = 0_i32;
        for _ in 0..40 {
            let s = LineState(pos);
            let act = agent.select_action(&s, &actions).unwrap();
            let next_pos = match act {
                StepAction::Left => (pos - 1).max(-2),
                StepAction::Right => (pos + 1).min(3),
            };
            let reward = if next_pos == 3 { 5.0 } else { -0.1 };
            agent.update(&s, &act, reward, &LineState(next_pos), &actions);
            pos = if next_pos == 3 { 0 } else { next_pos };
        }
        let mut q_vals = Vec::with_capacity(12);
        let mut digest = TrajectoryDigest::new();
        for p in -2..=3 {
            for &act in &actions {
                let q = agent.get_q_value(&LineState(p), &act);
                digest.update_f64(q);
                q_vals.push(q);
            }
        }
        (q_vals, digest.finish_hex())
    };

    let (q1, h1) = run(777);
    let (q2, h2) = run(777);
    let (q3, h3) = run(888);
    assert_eq!(q1, q2);
    assert_eq!(h1, h2);
    assert_ne!(q1, q3);
    assert_ne!(h1, h3);
    assert_eq!(h1, PINNED_QLEARNING_DIGEST);
}
