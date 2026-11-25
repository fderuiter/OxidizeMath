// [cite:math_commons]
use super::{
    Scalar, SolverWorkspace, fill_standard_normal, sample_open_unit, sample_standard_normal,
    sample_standard_normal_pair,
};
use oxidize_core::rng::OxidizeRng;

#[test]
#[verified_engine::verified]
fn test_scalar_f64_and_f32_identities() {
    let x64 = <f64 as Scalar>::from_f64(4.0);
    assert!((x64.sqrt() - 2.0).abs() < 1e-12);
    assert!((x64.ln().exp() - 4.0).abs() < 1e-12);
    let trig64 = <f64 as Scalar>::PI.cos();
    assert!((trig64 + 1.0).abs() < 1e-12);

    let x32 = <f32 as Scalar>::from_f64(9.0);
    assert!((x32.sqrt() - 3.0).abs() < 1e-6);
    assert!((x32.ln().exp() - 9.0).abs() < 1e-5);
    let trig32 = <f32 as Scalar>::HALF * <f32 as Scalar>::PI;
    assert!((trig32.sin() - 1.0).abs() < 1e-6);
}

#[test]
#[verified_engine::verified]
fn test_solver_workspace_reborrow_and_partition() {
    let mut backing = [1.0_f64; 10];
    let mut ws = SolverWorkspace::new(&mut backing);
    assert_eq!(ws.len(), 10);
    ws.zero_fill();
    assert!(ws.as_slice().iter().all(|&v| v == 0.0));

    for iter in 0..3 {
        let mut step_ws = ws.reborrow();
        let a = step_ws.take_slice(4);
        let b = step_ws.take_slice(6);
        assert_eq!(step_ws.len(), 0);
        assert!(step_ws.is_empty());
        a.fill(iter as f64 + 1.0);
        b.fill(iter as f64 + 10.0);
        assert_eq!(a[3] + b[5], 2.0 * (iter as f64) + 11.0);
    }
    assert_eq!(ws.len(), 10);
}

#[test]
#[verified_engine::verified]
fn test_solver_workspace_try_take_slice_preserves_buffer_on_overflow() {
    let mut backing = [0.0_f64; 6];
    let mut ws = SolverWorkspace::new(&mut backing);
    assert_eq!(
        ws.try_take_slice(8),
        Err(crate::error::MathError::WorkspaceTooSmall {
            required: 8,
            available: 6,
        })
    );
    assert_eq!(ws.len(), 6);
    let head = ws.try_take_slice(4).expect("4 <= 6");
    head.fill(3.0);
    assert_eq!(ws.len(), 2);
    assert_eq!(
        ws.try_take_slice(3),
        Err(crate::error::MathError::WorkspaceTooSmall {
            required: 3,
            available: 2,
        })
    );
    assert_eq!(ws.len(), 2);
}

#[test]
#[verified_engine::verified]
fn test_box_muller_moments_and_open_unit_bounds() {
    let mut rng = OxidizeRng::new(0xDEC0_DED0_1234_5678);
    for _ in 0..256 {
        let u: f64 = sample_open_unit(&mut rng);
        assert!(u > 0.0 && u <= 1.0);
    }

    let mut samples = [0.0_f64; 4095];
    fill_standard_normal(&mut samples, &mut rng);
    let n = samples.len() as f64;
    let mean = samples.iter().copied().sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&z| (z - mean) * (z - mean))
        .sum::<f64>()
        / n;
    assert!(mean.abs() < 0.06, "unexpected Box-Muller mean {mean}");
    assert!(
        (var - 1.0).abs() < 0.08,
        "unexpected Box-Muller variance {var}"
    );

    let (p0, p1): (f32, f32) = sample_standard_normal_pair(&mut rng);
    let s0: f32 = sample_standard_normal(&mut rng);
    assert!(p0.is_finite() && p1.is_finite() && s0.is_finite());
}
