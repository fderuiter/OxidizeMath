use crate::ai::sds::score::classifier_free_guidance;
use approx::assert_relative_eq;
use nalgebra::DMatrix;

#[test]
#[verified_engine::verified]
fn test_cfg() {
    let rows = 2;
    let cols = 2;
    let uncond = DMatrix::from_element(rows, cols, 0.5);
    let cond = DMatrix::from_element(rows, cols, 0.8);
    let scale = 7.5;

    // output = uncond + scale * (cond - uncond)
    // 0.5 + 7.5 * (0.8 - 0.5) = 0.5 + 7.5 * 0.3 = 0.5 + 2.25 = 2.75

    let out = classifier_free_guidance(&cond, &uncond, scale);
    assert_relative_eq!(out[(0, 0)], 2.75);
}
