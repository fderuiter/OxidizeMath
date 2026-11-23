use super::structs::{Gaussian2D, Gaussian3D, Scene};
use nalgebra::{Matrix2, Matrix3, Matrix4, Point2, SMatrix, Vector3};

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
