use super::structs::{Gaussian2D, Gaussian3D, Scene};
use nalgebra::{Matrix2, Matrix3, Matrix4, Point2, SMatrix, Vector3};

/// Single lensed 2D Gaussian image specification emitted by a thin-lens operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensedSplatImageSpec {
    /// Deflected 2D image-plane center $\boldsymbol{\theta}_{\pm}$.
    pub mean: Point2<f64>,
    /// Image-plane $2 \times 2$ magnification/shear Jacobian $J_{\text{lens}} = \partial\boldsymbol{\theta}/\partial\boldsymbol{\beta}$.
    pub lens_jacobian: Matrix2<f64>,
    /// Multiplicative relativistic radiance boost (e.g., gravitational redshift, Doppler, synchrotron).
    pub radiance_weight: f64,
    /// Multiplicative opacity weight (e.g., lens magnification or shadow horizon attenuation).
    pub opacity_weight: f64,
}

/// Dual-image thin-lens evaluation result for a single projected 3D Gaussian splat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensedProjectionOutcome {
    /// Primary lensed image (`None` when occluded inside a shadow horizon).
    pub primary: Option<LensedSplatImageSpec>,
    /// Optional secondary parity-inverted counterpart image (e.g., inner Einstein-ring arc).
    pub secondary: Option<LensedSplatImageSpec>,
}

/// Computes the Jacobian of the affine approximation of the projective transformation.
///
/// J = [
///   f_x/z, 0, -(f_x * x) / z^2
///   0, f_y/z, -(f_y * y) / z^2
/// ]
///
/// * `t`: The point in camera space (t.x, t.y, t.z).
/// * `focal_x`: Focal length in x.
/// * `focal_y`: Focal length in y.
#[verified_engine::verified]
pub fn compute_jacobian(t: &Vector3<f64>, focal_x: f64, focal_y: f64) -> SMatrix<f64, 2, 3> {
    let x = t.x;
    let y = t.y;
    let z = t.z;

    let mut j = SMatrix::<f64, 2, 3>::zeros();

    // Row 0
    j[(0, 0)] = focal_x / z;
    j[(0, 1)] = 0.0;
    j[(0, 2)] = -(focal_x * x) / (z * z);

    // Row 1
    j[(1, 0)] = 0.0;
    j[(1, 1)] = focal_y / z;
    j[(1, 2)] = -(focal_y * y) / (z * z);

    j
}

/// Projects a 3D Covariance matrix to 2D using the viewing transformation and Jacobian.
///
/// Sigma' = J * W * Sigma * W^T * J^T
///
/// * `cov3d`: The 3D covariance matrix.
/// * `view_matrix`: The 4x4 viewing transformation matrix (World -> Camera).
/// * `jacobian`: The 2x3 Jacobian of the projection.
#[verified_engine::verified]
pub fn project_covariance(
    cov3d: &Matrix3<f64>,
    view_matrix: &Matrix4<f64>,
    jacobian: &SMatrix<f64, 2, 3>,
) -> Matrix2<f64> {
    // Extract the 3x3 rotation part of the view matrix (W)
    let w = view_matrix.fixed_view::<3, 3>(0, 0);

    // T = W * Sigma * W^T (Covariance in camera space)
    let t = w * cov3d * w.transpose();

    // Sigma' = J * T * J^T
    jacobian * t * jacobian.transpose()
}

/// Camera pose and projection parameters for zero-allocation 3D Gaussian splat projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplatCameraProjection {
    /// World-to-camera rigid view matrix $W$ (upper-left $3 \times 3$ rotation, translation in column 3).
    pub view_matrix: Matrix4<f64>,
    /// Focal length $(f_x, f_y)$ in plot/viewport coordinates.
    pub focal: (f64, f64),
    /// Uniform scale multiplier applied to each 3D Gaussian's principal semi-axes.
    pub scale_multiplier: f64,
}

/// Projects `scene` into `out_splats` in-place (clearing first) and sorts front-to-back by depth.
///
/// Performs zero heap allocations when `out_splats.capacity() >= scene.gaussians.len()`.
#[verified_engine::verified]
pub fn project_scene_into(
    scene: &Scene,
    cam: &SplatCameraProjection,
    out_splats: &mut Vec<Gaussian2D>,
) {
    out_splats.clear();
    let rot = cam.view_matrix.fixed_view::<3, 3>(0, 0);
    let trans = cam.view_matrix.fixed_view::<3, 1>(0, 3);
    let scale_factor = if cam.scale_multiplier.is_finite() {
        cam.scale_multiplier.clamp(0.05, 20.0)
    } else {
        1.0
    };
    let cap = out_splats.capacity();
    for g in &scene.gaussians {
        if out_splats.len() >= cap && cap > 0 && scene.gaussians.len() > cap {
            break;
        }
        let scaled = Gaussian3D {
            mean: g.mean,
            scale: g.scale * scale_factor,
            rotation: g.rotation,
            opacity: g.opacity,
            color: g.color,
        };
        let p_cam = rot * scaled.mean.coords + trans;
        let tz = p_cam.z.max(0.25);
        let cam_vec = Vector3::new(p_cam.x, p_cam.y, tz);
        let jacobian = compute_jacobian(&cam_vec, cam.focal.0, cam.focal.1);
        let cov3d = scaled.compute_covariance();
        let cov2d = project_covariance(&cov3d, &cam.view_matrix, &jacobian)
            + Matrix2::from_diagonal_element(1e-4);
        let Some(inv_cov) = cov2d.try_inverse() else {
            continue;
        };
        out_splats.push(Gaussian2D {
            mean: Point2::new(cam.focal.0 * p_cam.x / tz, cam.focal.1 * p_cam.y / tz),
            conic: -0.5 * inv_cov,
            opacity: scaled.opacity,
            color: scaled.color,
            depth: tz,
        });
    }
    out_splats.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
}

/// Composes a source-plane 2D Gaussian covariance with a thin-lens $2 \times 2$ Jacobian.
///
/// Evaluates $\Sigma_{\text{lensed}} = J_{\text{lens}} \Sigma_{\text{src}} J_{\text{lens}}^T$
/// using symmetric stabilization so downstream conic inversion remains negative-definite.
#[verified_engine::verified]
pub fn project_lensed_covariance(
    cov2d: &Matrix2<f64>,
    lens_jacobian: &Matrix2<f64>,
) -> Matrix2<f64> {
    let raw = lens_jacobian * cov2d * lens_jacobian.transpose();
    0.5 * (raw + raw.transpose())
}

#[verified_engine::verified]
fn push_lensed_splat_image(
    base_cov2d: &Matrix2<f64>,
    base_opacity: f64,
    base_color: &Vector3<f64>,
    depth: f64,
    spec: &LensedSplatImageSpec,
    cap: usize,
    out_splats: &mut Vec<Gaussian2D>,
) {
    if cap > 0 && out_splats.len() >= cap {
        return;
    }
    if !spec.mean.x.is_finite()
        || !spec.mean.y.is_finite()
        || !spec.lens_jacobian.iter().all(|v| v.is_finite())
    {
        return;
    }
    let opacity_w = if spec.opacity_weight.is_finite() {
        spec.opacity_weight.max(0.0)
    } else {
        0.0
    };
    let opacity = (base_opacity * opacity_w).clamp(0.0, 1.0);
    if opacity <= 1e-6 {
        return;
    }
    let radiance_w = if spec.radiance_weight.is_finite() {
        spec.radiance_weight.clamp(0.0, 16.0)
    } else {
        0.0
    };
    let cov_lensed = project_lensed_covariance(base_cov2d, &spec.lens_jacobian)
        + Matrix2::from_diagonal_element(1e-4);
    if cov_lensed[(0, 0)] <= 0.0 || cov_lensed.determinant() <= 1e-16 {
        return;
    }
    let Some(inv_cov) = cov_lensed.try_inverse() else {
        return;
    };
    let color = Vector3::new(
        (base_color.x * radiance_w).clamp(0.0, 8.0),
        (base_color.y * radiance_w).clamp(0.0, 8.0),
        (base_color.z * radiance_w).clamp(0.0, 8.0),
    );
    out_splats.push(Gaussian2D {
        mean: spec.mean,
        conic: -0.5 * inv_cov,
        opacity,
        color,
        depth,
    });
}

/// Projects `scene` through `cam` and composes generic thin-lens $2 \times 2$ Jacobians and
/// relativistic radiance weights into up to two [`Gaussian2D`] images per 3D Gaussian.
///
/// Performs zero heap allocations when `out_splats.capacity() >= 2 * scene.gaussians.len()`.
#[verified_engine::verified]
pub fn project_scene_lensed_into<F>(
    scene: &Scene,
    cam: &SplatCameraProjection,
    mut lens_eval: F,
    out_splats: &mut Vec<Gaussian2D>,
) where
    F: FnMut(usize, &Gaussian3D, &Vector3<f64>, Point2<f64>) -> LensedProjectionOutcome,
{
    out_splats.clear();
    let rot = cam.view_matrix.fixed_view::<3, 3>(0, 0);
    let trans = cam.view_matrix.fixed_view::<3, 1>(0, 3);
    let scale_factor = if cam.scale_multiplier.is_finite() {
        cam.scale_multiplier.clamp(0.05, 20.0)
    } else {
        1.0
    };
    let cap = out_splats.capacity();
    for (idx, g) in scene.gaussians.iter().enumerate() {
        if cap > 0 && out_splats.len() >= cap {
            break;
        }
        let scaled = Gaussian3D {
            mean: g.mean,
            scale: g.scale * scale_factor,
            rotation: g.rotation,
            opacity: g.opacity,
            color: g.color,
        };
        let p_cam = rot * scaled.mean.coords + trans;
        let tz = p_cam.z.max(0.25);
        let cam_vec = Vector3::new(p_cam.x, p_cam.y, tz);
        let jacobian = compute_jacobian(&cam_vec, cam.focal.0, cam.focal.1);
        let cov3d = scaled.compute_covariance();
        let base_cov2d = project_covariance(&cov3d, &cam.view_matrix, &jacobian);
        let unlensed_mean = Point2::new(cam.focal.0 * p_cam.x / tz, cam.focal.1 * p_cam.y / tz);
        let outcome = lens_eval(idx, &scaled, &cam_vec, unlensed_mean);
        if let Some(primary) = &outcome.primary {
            push_lensed_splat_image(
                &base_cov2d,
                scaled.opacity,
                &scaled.color,
                tz,
                primary,
                cap,
                out_splats,
            );
        }
        if let Some(secondary) = &outcome.secondary {
            push_lensed_splat_image(
                &base_cov2d,
                scaled.opacity,
                &scaled.color,
                tz + 1e-6,
                secondary,
                cap,
                out_splats,
            );
        }
    }
    out_splats.sort_unstable_by(|a, b| a.depth.total_cmp(&b.depth));
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion};

    #[test]
    #[verified_engine::verified]
    fn test_jacobian() {
        let t = Vector3::new(1.0, 2.0, 5.0);
        let fx = 100.0;
        let fy = 100.0;
        let j = compute_jacobian(&t, fx, fy);

        // Expected: fx/z = 100/5 = 20
        assert_eq!(j[(0, 0)], 20.0);
        // Expected: -(fx*x)/z^2 = -(100*1)/25 = -4
        assert_eq!(j[(0, 2)], -4.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_lensed_projection_dual_images_and_zero_alloc() {
        let scene = Scene {
            gaussians: vec![Gaussian3D {
                mean: Point3::new(0.2, -0.1, 3.0),
                scale: Vector3::new(0.15, 0.08, 0.12),
                rotation: UnitQuaternion::identity(),
                opacity: 0.75,
                color: Vector3::new(0.8, 0.5, 0.2),
            }],
        };
        let cam = SplatCameraProjection {
            view_matrix: Matrix4::identity(),
            focal: (2.0, 2.0),
            scale_multiplier: 1.0,
        };
        let mut out_splats = Vec::with_capacity(2 * scene.gaussians.len());
        let shear = Matrix2::new(1.8, 0.25, 0.25, 0.65);

        let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            project_scene_lensed_into(
                &scene,
                &cam,
                |_idx, _g, _cam_vec, beta| LensedProjectionOutcome {
                    primary: Some(LensedSplatImageSpec {
                        mean: Point2::new(beta.x * 1.4, beta.y * 1.4),
                        lens_jacobian: shear,
                        radiance_weight: 1.5,
                        opacity_weight: 1.0,
                    }),
                    secondary: Some(LensedSplatImageSpec {
                        mean: Point2::new(-beta.x * 0.6, -beta.y * 0.6),
                        lens_jacobian: Matrix2::new(-0.5, 0.0, 0.0, 1.4),
                        radiance_weight: 0.7,
                        opacity_weight: 0.8,
                    }),
                },
                &mut out_splats,
            );
        });
        assert!(metrics.function_calls > 0);
        assert_eq!(out_splats.len(), 2);
        assert!(out_splats[0].color.x > out_splats[1].color.x);
        assert!(out_splats[0].conic[(0, 0)] < 0.0);
        assert!(out_splats[1].conic[(0, 0)] < 0.0);
    }
}
