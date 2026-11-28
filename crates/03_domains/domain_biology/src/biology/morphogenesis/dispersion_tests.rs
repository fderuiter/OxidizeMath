use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_schnakenberg_and_gray_scott_dispersion_classifiers_zero_alloc() {
    let u_field = [1.0, 1.0, 1.0, 1.0];
    let ((sch, gs, var_u), _) = VerifiedEngine::run_verified(|| {
        (
            classify_schnakenberg_dispersion(0.1, 0.9, 1.0, 40.0, 0.005),
            classify_gray_scott_dispersion(0.0367, 0.0649, 0.2097, 0.105, 0.8),
            grid_u_variance(&u_field),
        )
    });
    assert_eq!(
        MorphogenesisRegime::from_u8(sch.regime),
        MorphogenesisRegime::TuringPattern
    );
    assert!(is_gray_scott_pattern_viable(
        0.0367, 0.0649, 0.2097, 0.105, 0.8
    ));
    assert_ne!(
        GrayScottRegime::from_u8(gs.regime),
        GrayScottRegime::EulerCflUnstable
    );
    assert_eq!(var_u, 0.0);
}
