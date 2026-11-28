//! Integration tests for test dl theory.
use domain_ai::ai::deep_learning_theory::cycle::TrainingLoop;
use domain_ai::ai::deep_learning_theory::model::TwoLayerMLP;
use domain_ai::ai::optimization::SGD;
use nalgebra::DVector;

#[test]
#[verified_engine::verified]
fn test_deep_learning_cycle() {
    // Xor problem-ish (non-linear)
    // 2 inputs, 2 hidden units, 2 output classes
    // Use the new Strategy Pattern with SGD

    let mut rng = oxidize_core::rng::OxidizeRng::new(1337);
    let model = TwoLayerMLP::new_with_rng(2, 4, 2, &mut rng);
    let mut network = TrainingLoop::new_with_model(model, Box::new(SGD::new(0.5)));

    // Dummy data: Input [1, 0] -> Class 0 ([1, 0])
    let x = DVector::from_vec(vec![1.0, 0.0]);
    let y_true = DVector::from_vec(vec![1.0, 0.0]);

    // Initial prediction should be close to uniform or random
    let _initial_pred = network.predict(&x);
    let initial_loss = network.train_step(&x, &y_true).unwrap();

    // Train for a few steps
    let mut final_loss = 0.0;
    for _ in 0..500 {
        final_loss = network.train_step(&x, &y_true).unwrap();
    }

    // Prediction should improve (loss should decrease)
    println!("Initial Loss: {}, Final Loss: {}", initial_loss, final_loss);
    assert!(final_loss < initial_loss, "Loss did not decrease!");

    let final_pred = network.predict(&x);
    // Probability of class 0 should be high
    assert!(
        final_pred[0] > final_pred[1],
        "Did not learn to predict class 0"
    );
    assert!(final_pred[0] > 0.8, "Prediction confidence too low");
}

#[test]
#[verified_engine::verified]
fn test_explicit_model_construction() {
    // Explicitly construct TwoLayerMLP and inject it

    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let model = TwoLayerMLP::new_with_rng(2, 4, 2, &mut rng);
    let network = TrainingLoop::new_with_model(model, Box::new(SGD::new(0.1)));

    let x = DVector::from_vec(vec![1.0, 0.0]);
    let pred = network.predict(&x);
    assert_eq!(pred.len(), 2);
}

#[test]
#[verified_engine::verified]
fn test_backward_compatibility_layer_access() {
    // Verify we can still access layer1/layer2 like before (via Deref)

    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let model = TwoLayerMLP::new_with_rng(2, 4, 2, &mut rng);
    let network = TrainingLoop::new_with_model(model, Box::new(SGD::new(0.1)));

    // This access works because of Deref<Target=TwoLayerMLP>
    let _w1 = &network.layer1.weights;
    let _b2 = &network.layer2.bias;

    assert_eq!(network.layer1.weights.ncols(), 2);
}

#[test]
#[verified_engine::verified]
fn test_linear_algebra_basics() {
    use domain_ai::ai::deep_learning_theory::linear_algebra::*;

    let v1 = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    let v2 = DVector::from_vec(vec![4.0, 5.0, 6.0]);

    // Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(dot_product(&v1, &v2), 32.0);
}

fn sample_standard_normal_pair(rng: &mut oxidize_core::rng::OxidizeRng) -> (f64, f64) {
    use rand::Rng;
    let u1 = rng.r#gen::<f64>().clamp(1.0e-12, 1.0 - 1.0e-12);
    let u2 = rng.r#gen::<f64>();
    let mag = (-2.0 * u1.ln()).sqrt();
    let angle = std::f64::consts::TAU * u2;
    (mag * angle.cos(), mag * angle.sin())
}

#[test]
#[verified_engine::verified]
fn test_two_layer_mlp_distinct_streams_and_he_second_layer_variance() {
    use domain_ai::ai::deep_learning_theory::calculus::relu;
    use domain_ai::ai::deep_learning_theory::linear_algebra::DenseLayer;

    let square_mlp = TwoLayerMLP::new(16, 16, 16);
    assert_ne!(
        square_mlp.layer1.weights, square_mlp.layer2.weights,
        "TwoLayerMLP::new(16, 16, 16) must initialize distinct weights across layers"
    );

    let default_mlp = TwoLayerMLP::new(2, 16, 2);
    let mut input_rng = oxidize_core::rng::OxidizeRng::new(2026_1127);
    let sample_count = 4_000usize;
    let mut z2_sq_sum_single = 0.0_f64;
    for _ in 0..sample_count {
        let (x0, x1) = sample_standard_normal_pair(&mut input_rng);
        let x = DVector::from_vec(vec![x0, x1]);
        let a1 = relu(&default_mlp.layer1.forward(&x));
        let z2 = default_mlp.layer2.forward(&a1);
        z2_sq_sum_single += 0.5 * (z2[0] * z2[0] + z2[1] * z2[1]);
    }
    let single_second_moment = z2_sq_sum_single / (sample_count as f64);
    assert!(
        (0.8..3.5).contains(&single_second_moment),
        "Single TwoLayerMLP::new(2, 16, 2) z2 second moment {single_second_moment} far from 2.0"
    );

    // He et al. (2015) pre-activation variance Var(z^(2)) across independent streams = 2 * Var(x) = 2.0.
    let num_models = 256usize;
    let samples_per_model = 64usize;
    let mut sum_z = 0.0_f64;
    let mut sum_z_sq = 0.0_f64;
    let total_scalars = (num_models * samples_per_model * 2) as f64;
    let base_rng = oxidize_core::rng::OxidizeRng::new(0x5EED_2015);
    for m in 0..num_models {
        let mut r1 = base_rng.split_stream((m as u64) * 2);
        let mut r2 = base_rng.split_stream((m as u64) * 2 + 1);
        let mlp = TwoLayerMLP {
            layer1: DenseLayer::new_with_rng(2, 16, &mut r1),
            layer2: DenseLayer::new_with_rng(16, 2, &mut r2),
        };
        for _ in 0..samples_per_model {
            let (x0, x1) = sample_standard_normal_pair(&mut input_rng);
            let x = DVector::from_vec(vec![x0, x1]);
            let z2 = mlp.layer2.forward(&relu(&mlp.layer1.forward(&x)));
            sum_z += z2[0] + z2[1];
            sum_z_sq += z2[0] * z2[0] + z2[1] * z2[1];
        }
    }
    let mean_z = sum_z / total_scalars;
    let pre_activation_var = (sum_z_sq / total_scalars) - mean_z * mean_z;
    assert!(
        (pre_activation_var - 2.0).abs() < 0.25,
        "Expected He second-layer pre-activation variance near 2.0, got {pre_activation_var}"
    );
}
