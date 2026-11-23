#![allow(missing_docs)]
use domain_ai::ai::gaussian_splatting::{
    Gaussian2D, Gaussian3D, Scene,
    optimization::{DensityAction, determine_density_action},
    projection::{SplatCameraProjection, project_scene_into},
    rendering::blend_gaussians,
};
use nalgebra::{Matrix4, Point2, Point3, UnitQuaternion, Vector3};
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
                rotation: UnitQuaternion::new(Vector3::new(0.1, 0.2, -0.3)),
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
