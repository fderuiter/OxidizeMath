//! Integration tests verifying zero-allocation 3D Gaussian splatting projection and blending.
use domain_ai::ai::gaussian_splatting::{
    Gaussian2D, Gaussian3D, Scene, UnitQuaternion,
    optimization::{DensityAction, determine_density_action},
    projection::{
        LensedProjectionOutcome, LensedSplatImageSpec, SplatCameraProjection, project_scene_into,
        project_scene_lensed_into,
    },
    rendering::blend_gaussians,
};
use nalgebra::{Matrix2, Matrix4, Point2, Point3, Vector3};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_project_scene_into_and_blend_zero_allocation_hot_path() {
    let scene = Scene {
        gaussians: vec![
            Gaussian3D {
                mean: Point3::new(-0.4, 0.2, -0.5),
                scale: Vector3::new(0.25, 0.18, 0.12),
                rotation: UnitQuaternion::identity(),
                opacity: 0.85,
                color: Vector3::new(0.9, 0.3, 0.2),
            },
            Gaussian3D {
                mean: Point3::new(0.3, -0.1, 0.4),
                scale: Vector3::new(0.55, 0.30, 0.20),
                rotation: UnitQuaternion::from_scaled_axis([0.1, 0.2, -0.3]),
                opacity: 0.75,
                color: Vector3::new(0.2, 0.7, 0.95),
            },
        ],
    };
    let mut view = Matrix4::identity();
    view[(2, 3)] = 5.0;
    let cam = SplatCameraProjection {
        view_matrix: view,
        focal: (5.0, 5.0),
        scale_multiplier: 1.1,
    };
    let mut projected: Vec<Gaussian2D> = Vec::with_capacity(8);
    let mut blended = Vector3::zeros();
    let mut action = DensityAction::Keep;

    VerifiedEngine::run_verified(|| {
        project_scene_into(&scene, &cam, &mut projected);
        blended = blend_gaussians(&projected, &Point2::new(0.0, 0.0));
        action = determine_density_action(0.22, 0.6, 0.75, 0.1, 0.45, 0.05);
    });

    assert_eq!(projected.len(), 2);
    assert!(projected[0].depth <= projected[1].depth);
    assert!(blended.iter().all(|c| c.is_finite() && *c >= 0.0));
    assert_eq!(action, DensityAction::Split);
}

#[test]
fn test_project_scene_lensed_into_dual_images_384_splats_zero_alloc() {
    let mut gaussians = Vec::with_capacity(384);
    for i in 0..384 {
        let angle = (i as f64) * 0.049;
        gaussians.push(Gaussian3D {
            mean: Point3::new(angle.cos(), 0.25 * angle.sin(), 4.0 + 0.3 * angle.sin()),
            scale: Vector3::new(0.18, 0.09, 0.12),
            rotation: UnitQuaternion::identity(),
            opacity: 0.72,
            color: Vector3::new(0.85, 0.52, 0.22),
        });
    }
    let scene = Scene { gaussians };
    let cam = SplatCameraProjection {
        view_matrix: Matrix4::identity(),
        focal: (3.2, 3.2),
        scale_multiplier: 1.0,
    };
    let mut projected = Vec::with_capacity(2 * 384);
    let mut sample_color = Vector3::zeros();

    VerifiedEngine::run_verified(|| {
        project_scene_lensed_into(
            &scene,
            &cam,
            |_idx, _g, _cam_vec, beta| LensedProjectionOutcome {
                primary: Some(LensedSplatImageSpec {
                    mean: Point2::new(beta.x * 1.25, beta.y * 1.25),
                    lens_jacobian: Matrix2::new(1.6, 0.2, 0.2, 0.7),
                    radiance_weight: 1.4,
                    opacity_weight: 1.0,
                }),
                secondary: Some(LensedSplatImageSpec {
                    mean: Point2::new(-beta.x * 0.55, -beta.y * 0.55),
                    lens_jacobian: Matrix2::new(-0.5, 0.0, 0.0, 1.3),
                    radiance_weight: 0.65,
                    opacity_weight: 0.75,
                }),
            },
            &mut projected,
        );
        sample_color = blend_gaussians(&projected, &Point2::new(0.3, 0.0));
    });

    assert_eq!(projected.len(), 768);
    assert!(sample_color.iter().all(|c| c.is_finite() && *c >= 0.0));
}
