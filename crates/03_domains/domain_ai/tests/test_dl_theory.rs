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
