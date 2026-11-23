//! Zero-allocation runtime verification tests for initialized `Adam` and `SGD`
//! dynamic matrix and vector parameter updates under `VerifiedAllocator`.

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::analysis::evolution::EvolutionEngine;
use pure_math::pure_math::analysis::optimization::{Adam, AdamState, Optimizer, ParamType, SGD};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_initialized_adam_and_sgd_updates_zero_heap_allocations() {
    let mut rng = OxidizeRng::new(0x0A11_0C00);
    let mut adam = Adam::<f64, (usize, ParamType)>::new(0.05).expect("valid Adam");
    let mut sgd = SGD::new(0.25_f64);

    let mut adam_mat = DMatrix::from_element(3, 4, 1.0_f64);
    let adam_mat_grad = DMatrix::from_element(3, 4, 0.2_f64);
    let mut adam_vec = DVector::from_element(5, -0.5_f64);
    let adam_vec_grad = DVector::from_element(5, -0.1_f64);

    // Pre-initialize HashMap entries and moment buffers outside run_verified.
    let _ = adam.get_state((0, ParamType::Weight), (3, 4));
    let _ = adam.get_state((0, ParamType::Bias), (5, 1));

    let mut evo_adam_mat = DMatrix::from_element(2, 3, 0.75_f64);
    let mut evo_adam_mat_state = AdamState {
        m: DMatrix::zeros(2, 3),
        v: DMatrix::zeros(2, 3),
        t: 0,
    };
    let mut evo_adam_mat_aux = (
        DMatrix::from_element(2, 3, 0.15_f64),
        &mut evo_adam_mat_state,
    );

    let mut evo_adam_vec = DVector::from_element(4, 0.4_f64);
    let mut evo_adam_vec_state = AdamState {
        m: DMatrix::zeros(4, 1),
        v: DMatrix::zeros(4, 1),
        t: 0,
    };
    let mut evo_adam_vec_aux = (DVector::from_element(4, 0.05_f64), &mut evo_adam_vec_state);

    let mut sgd_mat = DMatrix::from_element(3, 4, 2.0_f64);
    let sgd_mat_grad = DMatrix::from_element(3, 4, 0.5_f64);
    let mut sgd_vec = DVector::from_element(5, 1.5_f64);
    let sgd_vec_grad = DVector::from_element(5, 0.4_f64);

    let mut evo_sgd_mat = DMatrix::from_element(2, 2, 1.0_f64);
    let mut evo_sgd_mat_grad = DMatrix::from_element(2, 2, 0.2_f64);
    let mut evo_sgd_vec = DVector::from_element(3, 1.0_f64);
    let mut evo_sgd_vec_grad = DVector::from_element(3, 0.2_f64);

    // Keep all VerifiedEngine::run_verified checks in a single test to avoid MEMORY_LOCK contention.
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..16 {
            adam.update_matrix((0, ParamType::Weight), &mut adam_mat, &adam_mat_grad)
                .expect("adam matrix step");
            adam.update_vector((0, ParamType::Bias), &mut adam_vec, &adam_vec_grad)
                .expect("adam vector step");
            adam.step(&mut evo_adam_mat, &mut evo_adam_mat_aux, &mut rng, 0.02)
                .expect("adam evo matrix step");
            adam.step(&mut evo_adam_vec, &mut evo_adam_vec_aux, &mut rng, 0.02)
                .expect("adam evo vector step");

            Optimizer::<f64, usize>::update_matrix(&mut sgd, 0, &mut sgd_mat, &sgd_mat_grad)
                .expect("sgd matrix step");
            Optimizer::<f64, usize>::update_vector(&mut sgd, 1, &mut sgd_vec, &sgd_vec_grad)
                .expect("sgd vector step");
            sgd.step(&mut evo_sgd_mat, &mut evo_sgd_mat_grad, &mut rng, 0.05)
                .expect("sgd evo matrix step");
            sgd.step(&mut evo_sgd_vec, &mut evo_sgd_vec_grad, &mut rng, 0.05)
                .expect("sgd evo vector step");
        }
    });

    assert_eq!(metrics.memory_loads, 0);
    assert!(adam_mat[(0, 0)] < 1.0 && adam_vec[0] > -0.5);
    assert!(evo_adam_mat[(0, 0)] < 0.75 && evo_adam_vec[0] < 0.4);

    // Verify SGD::update_matrix and SGD::update_vector preserve the exact 0.01 step rate:
    // 16 steps * 0.01 * grad.
    assert_relative_eq!(
        sgd_mat[(0, 0)],
        2.0 - 16.0 * 0.01 * 0.5,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        sgd_vec[0],
        1.5 - 16.0 * 0.01 * 0.4,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        evo_sgd_mat[(0, 0)],
        1.0 - 16.0 * 0.05 * 0.2,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
    assert_relative_eq!(
        evo_sgd_vec[0],
        1.0 - 16.0 * 0.05 * 0.2,
        epsilon = math_commons::registry::TOLERANCE_FAST
    );
}
