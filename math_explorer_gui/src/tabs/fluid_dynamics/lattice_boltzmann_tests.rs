#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_lbm_unified_propagates_fluid_across_grid_without_allocation() {
    let mut params = HashMap::new();
    params.insert("viscosity".to_string(), 0.02);
    let mut model = LbmUnified::new(&params);

    model.step(&params);
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..90 {
                model.step(&params);
            }
        });
    }

    assert!(!model.diverged());
    let wake_speed = model.solver.get_velocity_magnitude(45, 25);
    assert!(
        wake_speed > 1e-4,
        "LbmUnified::step must stream fluid past cylinder obstacle to x=45, got {wake_speed}"
    );
}
