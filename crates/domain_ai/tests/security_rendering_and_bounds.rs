//! Tier 5 security tests for `domain_ai`: `checked_mul` overflow, zero-dimension
//! guards, `NaN`/infinity rejection, subnormal volumetric stability, and zero-allocation rendering.

use domain_ai::ai::reinforcement_learning::maze::SculptedGridMdp;
use domain_ai::ai::sds::rendering::{
    NeRFModel, RayBundle, RenderContext, render_image_with_rng, stratified_sampling_with_rng,
    try_generate_ray_bundle, try_positional_encoding, volume_integration,
};
use domain_ai::error::AIError;
use math_commons::error::MathError;
use nalgebra::{Matrix4, Vector3};
use oxidize_core::rng::OxidizeRng;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static TEST_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct SubnormalFieldModel {
    sigma: f64,
}

impl NeRFModel for SubnormalFieldModel {
    fn query(&self, _pos: &Vector3<f64>, _dir: &Vector3<f64>) -> (f64, Vector3<f64>) {
        (self.sigma, Vector3::new(0.25, 0.5, 0.75))
    }
}

#[test]
fn ray_bundle_context_and_maze_reject_overflow_zero_and_non_finite_inputs() {
    let mut ctx = RenderContext::new();
    let resize_err = ctx
        .try_resize(usize::MAX, 2, 4)
        .expect_err("width * height overflow must fail");
    assert!(matches!(
        resize_err,
        AIError::Math(MathError::CapacityOverflow { .. })
    ));

    let pose = Matrix4::identity();
    assert!(try_generate_ray_bundle(&pose, 0, 8, 1.0).is_err());
    assert!(try_generate_ray_bundle(&pose, 8, 0, 1.0).is_err());
    assert!(try_generate_ray_bundle(&pose, 4, 4, 0.0).is_err());
    assert!(try_generate_ray_bundle(&pose, 4, 4, f64::NAN).is_err());
    assert!(try_generate_ray_bundle(&pose, 4, 4, f64::INFINITY).is_err());
    assert!(matches!(
        try_generate_ray_bundle(&pose, usize::MAX, 2, 1.0),
        Err(AIError::Math(MathError::CapacityOverflow { .. }))
    ));

    let mut nan_pose = pose;
    nan_pose[(0, 3)] = f64::NAN;
    assert!(try_generate_ray_bundle(&nan_pose, 4, 4, 1.0).is_err());

    assert!(try_positional_encoding(f64::NAN, 4).is_err());
    assert!(try_positional_encoding(f64::NEG_INFINITY, 4).is_err());
    assert!(matches!(
        try_positional_encoding(0.25, usize::MAX),
        Err(AIError::Math(MathError::CapacityOverflow { .. }))
    ));

    assert!(SculptedGridMdp::try_new(0, 4, 0.9, 0.0).is_err());
    assert!(SculptedGridMdp::try_new(4, -1, 0.9, 0.0).is_err());
}

#[test]
fn stratified_sampling_and_subnormal_volume_rendering_remain_finite_and_zero_alloc() {
    let subnormal = f64::MIN_POSITIVE * 0.5;
    let enc = try_positional_encoding(subnormal, 6).expect("subnormal positional encoding");
    assert_eq!(enc.len(), 12);
    assert!(enc.iter().all(|v| v.is_finite()));

    let pose = Matrix4::identity();
    let bundle: RayBundle =
        try_generate_ray_bundle(&pose, 2, 2, 0.8).expect("valid 2x2 ray bundle");
    let mut ctx = RenderContext::new();
    ctx.try_resize(2, 2, 4).expect("pre-allocate context");
    let mut rng = OxidizeRng::from_seed(42);
    let model = SubnormalFieldModel { sigma: subnormal };

    // Zero samples must return finite black image without division by zero.
    let zero_img = render_image_with_rng(&mut ctx, &bundle, &model, 0.1, 2.0, 0, &mut rng);
    assert_eq!(zero_img.shape(), (2, 2));
    assert!(zero_img.iter().all(|px| px.iter().all(|c| *c == 0.0)));

    // Re-prime context for 4 samples and verify hot kernels allocate 0 bytes inside run_verified.
    ctx.try_resize(2, 2, 4).expect("re-prime context");
    let mut out_ts = [1.0_f64; 4];
    let densities = [subnormal, f64::NAN, subnormal * 2.0, 0.0];
    let colors = [Vector3::new(1.0, 0.5, 0.25); 4];
    let deltas = [0.25, 0.25, f64::INFINITY, 0.25];

    let (integrated, metrics) = VerifiedEngine::run_verified(|| {
        stratified_sampling_with_rng(0.1, 1.0, 0, &mut rng, &mut out_ts);
        stratified_sampling_with_rng(f64::NAN, 1.0, 4, &mut rng, &mut out_ts);
        stratified_sampling_with_rng(0.1, 1.0, 4, &mut rng, &mut out_ts);
        volume_integration(&densities, &colors, &deltas)
    });

    assert_eq!(metrics.allocations, 0);
    assert_eq!(metrics.allocated_bytes, 0);
    assert!(
        out_ts
            .iter()
            .all(|t| t.is_finite() && *t >= 0.1 && *t <= 1.0)
    );
    assert!(integrated.iter().all(|c| c.is_finite() && *c >= 0.0));

    let img = render_image_with_rng(&mut ctx, &bundle, &model, 0.1, 2.0, 4, &mut rng);
    assert_eq!(img.shape(), (2, 2));
    assert!(img.iter().all(|px| px.iter().all(|c| c.is_finite())));
}
