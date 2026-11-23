use crate::ai::sds::diffusion::{generate_noise_with_rng, inject_noise};
use crate::ai::sds::gradient::compute_sds_gradient;
use crate::ai::sds::models::{
    AnalyticalScorePrior, AnalyticalVoxelNeRF, PromptPrototypeEncoder, rgb_image_to_latent,
};
use crate::ai::sds::rendering::{
    NeRFModel, RenderContext, generate_ray_bundle, render_image_with_rng,
};
use crate::ai::sds::score::{DiffusionModel, TextEncoder, classifier_free_guidance};
use crate::ai::sds::training::{AdamOptimizer, DifferentiableNeRF, Optimizer, SgdOptimizer};
use math_commons::primitives::{PositiveFloat, UnitInterval};
use nalgebra::{Matrix4, Point3, Vector3};
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

fn camera_pose() -> Matrix4<f64> {
    let eye = Point3::new(0.0, 0.0, 2.7);
    let target = Point3::origin();
    let up = Vector3::y();
    Matrix4::look_at_rh(&eye, &target, &up)
        .try_inverse()
        .expect("invertible camera pose")
}

#[test]
#[verified_engine::verified]
fn test_analytical_voxel_nerf_query_and_render_zero_alloc() {
    let nerf = AnalyticalVoxelNeRF::new(8, 1.0);
    let pos = Vector3::new(0.15, -0.2, 0.05);
    let dir = Vector3::new(0.0, 0.0, -1.0);

    let ((sigma, color), _metrics) = VerifiedEngine::run_verified(|| {
        let mut last = (0.0, Vector3::zeros());
        for _ in 0..64 {
            last = nerf.query(&pos, &dir);
        }
        last
    });
    assert!(sigma > 0.0);
    assert!(color.x >= 0.0 && color.x <= 1.0);

    // Outside bounding box returns zero density
    let (out_sigma, _) = nerf.query(&Vector3::new(2.5, 0.0, 0.0), &dir);
    assert_eq!(out_sigma, 0.0);

    // render_image_with_rng explicitly calls lock_allocations() across all ray queries
    let pose = camera_pose();
    let bundle = generate_ray_bundle(&pose, 8, 8, std::f64::consts::FRAC_PI_3);
    let mut ctx = RenderContext::new();
    let mut rng = OxidizeRng::from_seed(42);
    let img = render_image_with_rng(&mut ctx, &bundle, &nerf, 1.2, 4.2, 12, &mut rng);
    assert_eq!(img.nrows(), 8);
    assert_eq!(img.ncols(), 8);
}

fn run_sds_distillation_loop<O: Optimizer>(opt: &mut O, steps: usize) -> (f64, f64) {
    let width = 10;
    let height = 10;
    let pose = camera_pose();
    let bundle = generate_ray_bundle(&pose, width, height, std::f64::consts::FRAC_PI_3);
    let mut nerf = AnalyticalVoxelNeRF::new(6, 1.0).with_ray_march_bounds(1.2, 4.2, 12);
    let encoder = PromptPrototypeEncoder::new();
    let prior = AnalyticalScorePrior::new(width, height, 100);
    let e_cond = encoder.encode("glowing amber torus");
    let e_uncond = encoder.encode("");
    let target = prior.target_latent(&e_cond);

    let mut ctx = RenderContext::new();
    let mut rng = OxidizeRng::from_seed(1337);

    let eval_mse = |model: &AnalyticalVoxelNeRF, render_ctx: &mut RenderContext| -> f64 {
        let mut eval_rng = OxidizeRng::from_seed(99);
        let img = render_image_with_rng(render_ctx, &bundle, model, 1.2, 4.2, 12, &mut eval_rng);
        let lat = rgb_image_to_latent(&img);
        (&lat - &target).norm_squared() / (lat.len() as f64)
    };

    let initial_mse = eval_mse(&nerf, &mut ctx);
    for step_idx in 0..steps {
        let mut render_rng = OxidizeRng::from_seed(500 + step_idx as u64);
        let rendered =
            render_image_with_rng(&mut ctx, &bundle, &nerf, 1.2, 4.2, 12, &mut render_rng);
        let x0 = rgb_image_to_latent(&rendered);
        let t = 25 + (step_idx % 40);
        let (sig_s, noise_s) = prior.schedule().get_scales(t);
        let eps = generate_noise_with_rng(x0.nrows(), 3, &mut rng).expect("noise");
        let z_t = inject_noise(&x0, &eps, sig_s, noise_s);
        let eps_cond = prior.predict_noise(&z_t, t, &e_cond);
        let eps_uncond = prior.predict_noise(&z_t, t, &e_uncond);
        let eps_guided = classifier_free_guidance(&eps_cond, &eps_uncond, 6.0);
        let sds_grad = compute_sds_gradient(&eps_guided, &eps, 0.15);
        let param_grads = nerf.backward(&bundle, &sds_grad);
        let next_params = opt.step(nerf.params(), &param_grads).expect("opt step");
        nerf.set_params(next_params);
    }
    let final_mse = eval_mse(&nerf, &mut ctx);
    (initial_mse, final_mse)
}

#[test]
#[verified_engine::verified]
fn test_sds_distillation_reduces_loss_with_adam_and_sgd() {
    let mut adam = AdamOptimizer::new(PositiveFloat::new(0.12).unwrap());
    let (init_adam, final_adam) = run_sds_distillation_loop(&mut adam, 18);
    assert!(
        final_adam < init_adam,
        "Adam SDS distillation should decrease target MSE ({final_adam} < {init_adam})"
    );

    let mut sgd = SgdOptimizer::new(
        PositiveFloat::new(0.35).unwrap(),
        UnitInterval::new(0.8).unwrap(),
    );
    let (init_sgd, final_sgd) = run_sds_distillation_loop(&mut sgd, 18);
    assert!(
        final_sgd < init_sgd,
        "SGD SDS distillation should decrease target MSE ({final_sgd} < {init_sgd})"
    );
}
