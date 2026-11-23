#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Parallel and pinhole perspective 3D projection mathematics for [`super::Camera3D`].

use super::{Camera3D, ProjectionMode};

/// Default pinhole perspective camera distance (in world units) matching Gaussian Splatting.
pub const CAMERA_3D_PERSPECTIVE_DISTANCE: f64 = 5.5;

/// Minimum positive pinhole near-plane depth preventing division-by-zero singularities.
pub const CAMERA_3D_NEAR_DEPTH: f64 = 0.25;

impl Camera3D {
    /// Returns the active pinhole camera distance used in [`ProjectionMode::Perspective`].
    #[must_use]
    pub fn perspective_distance(&self) -> f64 {
        if self.perspective_distance.is_finite() && self.perspective_distance > 0.5 {
            self.perspective_distance
        } else {
            CAMERA_3D_PERSPECTIVE_DISTANCE
        }
    }

    /// Rotates a 3D world point `[x, y, z]` into viewer-aligned `[right, up, viewer_z]` space.
    #[must_use]
    pub fn rotate_to_camera_space(&self, point: &[f64; 3]) -> [f64; 3] {
        let (sy, cy) = (self.yaw as f64).sin_cos();
        let (sp, cp) = (self.pitch as f64).sin_cos();

        let x1 = point[0] * cy - point[1] * sy;
        let y1 = point[0] * sy + point[1] * cy;
        let z1 = point[2];

        let y2 = y1 * cp - z1 * sp;
        let z2 = y1 * sp + z1 * cp;

        [x1, y2, z2]
    }

    /// Builds a 4x4 pinhole world-to-camera matrix (`[right, up, depth]` with `depth = camera_distance - viewer_z`).
    #[must_use]
    pub fn view_matrix(&self, camera_distance: f64) -> nalgebra::Matrix4<f64> {
        let c0 = self.rotate_to_camera_space(&[1.0, 0.0, 0.0]);
        let c1 = self.rotate_to_camera_space(&[0.0, 1.0, 0.0]);
        let c2 = self.rotate_to_camera_space(&[0.0, 0.0, 1.0]);
        nalgebra::Matrix4::new(
            c0[0],
            c1[0],
            c2[0],
            0.0,
            c0[1],
            c1[1],
            c2[1],
            0.0,
            -c0[2],
            -c1[2],
            -c2[2],
            camera_distance,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    pub(super) fn project_unit(&self, point: &[f64; 3]) -> (f32, f32, f32) {
        let [x2, y2, z2] = self.rotate_to_camera_space(point);
        (x2 as f32, y2 as f32, z2 as f32)
    }

    /// Projects a 3D point `[x, y, z]` to `([proj_x, proj_y], viewer_z)` using the active
    /// [`ProjectionMode`], `zoom`, and `(pan_x, pan_y)` translation.
    #[must_use]
    pub fn project_with_depth(&self, point: &[f64; 3]) -> ([f64; 2], f64) {
        let [tx, ty, vz] = self.rotate_to_camera_space(point);
        let zoom = f64::from(self.zoom);
        let px = f64::from(self.pan_x);
        let py = f64::from(self.pan_y);
        let proj = match self.projection {
            ProjectionMode::Parallel => [tx * zoom + px, ty * zoom + py],
            ProjectionMode::Perspective => {
                let dist = self.perspective_distance();
                let focal = dist * zoom;
                let tz = (dist - vz).max(CAMERA_3D_NEAR_DEPTH);
                [focal * tx / tz + px, focal * ty / tz + py]
            }
        };
        (proj, vz)
    }

    /// Projects a 3D point `[x, y, z]` into 2D plot coordinates `[proj_x, proj_y]`.
    #[must_use]
    pub fn project(&self, point: &[f64; 3]) -> [f64; 2] {
        self.project_with_depth(point).0
    }

    /// Projects a `nalgebra::Point3<f64>` into `([proj_x, proj_y], viewer_z)`.
    #[must_use]
    pub fn project_point_with_depth(&self, point: &nalgebra::Point3<f64>) -> ([f64; 2], f64) {
        self.project_with_depth(&[point.x, point.y, point.z])
    }
}
