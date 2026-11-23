#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Shared interactive 3D camera, viewport, projection modes, and depth-sorted scene builder.

use std::hash::{Hash, Hasher};

#[cfg(test)]
use eframe::egui;

#[path = "../camera_gizmo.rs"]
mod gizmo;
mod interaction;
mod projection;
pub mod scene;
mod viewport;

pub use projection::CAMERA_3D_PERSPECTIVE_DISTANCE;
pub use scene::Scene3DBuilder;
pub use viewport::Plot3DViewport;

const fn default_angle() -> f32 {
    0.5_f32
}

const fn default_zoom() -> f32 {
    1.0_f32
}

const fn default_perspective_distance() -> f64 {
    CAMERA_3D_PERSPECTIVE_DISTANCE
}

const fn default_pan_units_per_px() -> f32 {
    0.02_f32
}

fn is_zero_f32(value: &f32) -> bool {
    *value == 0.0_f32
}

/// Minimum shared zoom scale across all 3D camera controls and gestures.
pub const CAMERA_3D_ZOOM_MIN: f32 = 0.01_f32;

/// Maximum shared zoom scale across all 3D camera controls and gestures.
pub const CAMERA_3D_ZOOM_MAX: f32 = 100.0_f32;

/// Maximum finite camera angle magnitude (radians) safe from `f32::to_degrees()` overflow.
pub const CAMERA_3D_MAX_ANGLE_RAD: f32 = 1_000.0_f32 * std::f32::consts::TAU;

/// Maximum finite 3D viewport pan offset in projected plot coordinates.
pub const CAMERA_3D_MAX_PAN: f32 = 1_000.0_f32;

/// Canonical canvas helper text displayed below interactive 3D viewports.
pub const CAMERA_3D_HELPER_TEXT: &str =
    "Drag canvas or use WASD to orbit, Shift+Drag or Shift+WASD to pan, scroll or Q/E to zoom, P to switch projection, double-click or press R to reset view.";

/// 3D viewport projection mode matching GeoGebra 3D style-bar projection switching.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum ProjectionMode {
    /// Orthographic parallel projection preserving parallel lines regardless of depth.
    #[default]
    Parallel,
    /// Pinhole perspective projection scaling apparent size inversely with camera depth.
    Perspective,
}

impl ProjectionMode {
    /// Returns the opposite projection mode (`Parallel <-> Perspective`).
    #[must_use]
    pub const fn toggle(self) -> Self {
        match self {
            Self::Parallel => Self::Perspective,
            Self::Perspective => Self::Parallel,
        }
    }

    /// Human-readable label for UI controls and accessibility descriptions.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Parallel => "Parallel",
            Self::Perspective => "Perspective",
        }
    }

    const fn is_parallel(&self) -> bool {
        matches!(self, Self::Parallel)
    }
}

/// Shared 3D orbital camera state with pan offsets and parallel/perspective projection support.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct Camera3D {
    pub pitch: f32,
    pub yaw: f32,
    pub zoom: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub pan_x: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub pan_y: f32,
    #[serde(default, skip_serializing_if = "ProjectionMode::is_parallel")]
    pub projection: ProjectionMode,
    #[serde(skip, default = "default_angle")]
    reset_pitch: f32,
    #[serde(skip, default = "default_angle")]
    reset_yaw: f32,
    #[serde(skip, default = "default_zoom")]
    reset_zoom: f32,
    #[serde(skip, default)]
    reset_projection: ProjectionMode,
    #[serde(skip, default = "default_perspective_distance")]
    pub(crate) perspective_distance: f64,
    #[serde(skip, default = "default_pan_units_per_px")]
    pub(crate) pan_units_per_px: f32,
}

impl PartialEq for Camera3D {
    fn eq(&self, other: &Self) -> bool {
        self.pitch == other.pitch
            && self.yaw == other.yaw
            && self.zoom == other.zoom
            && self.pan_x == other.pan_x
            && self.pan_y == other.pan_y
            && self.projection == other.projection
    }
}

impl Default for Camera3D {
    fn default() -> Self {
        Self::new(default_angle(), default_angle(), default_zoom())
    }
}

impl Camera3D {
    /// Constructs a 3D camera with configured default reset `pitch`, `yaw`, and `zoom`.
    #[must_use]
    pub fn new(pitch: f32, yaw: f32, zoom: f32) -> Self {
        Self {
            pitch,
            yaw,
            zoom,
            pan_x: 0.0_f32,
            pan_y: 0.0_f32,
            projection: ProjectionMode::Parallel,
            reset_pitch: pitch,
            reset_yaw: yaw,
            reset_zoom: zoom,
            reset_projection: ProjectionMode::Parallel,
            perspective_distance: default_perspective_distance(),
            pan_units_per_px: default_pan_units_per_px(),
        }
    }

    /// Configures a custom pinhole perspective camera distance for large-extent 3D scenes.
    #[must_use]
    pub fn with_perspective_distance(mut self, distance: f64) -> Self {
        if distance.is_finite() && distance > 0.5 {
            self.perspective_distance = distance;
        }
        self
    }

    /// Returns true if orientation, zoom, and pan offsets are finite and within safe bounds.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.pitch.is_finite()
            && self.yaw.is_finite()
            && self.zoom.is_finite()
            && self.pan_x.is_finite()
            && self.pan_y.is_finite()
            && self.pitch.abs() <= CAMERA_3D_MAX_ANGLE_RAD
            && self.yaw.abs() <= CAMERA_3D_MAX_ANGLE_RAD
            && (CAMERA_3D_ZOOM_MIN..=CAMERA_3D_ZOOM_MAX).contains(&self.zoom)
            && self.pan_x.abs() <= CAMERA_3D_MAX_PAN
            && self.pan_y.abs() <= CAMERA_3D_MAX_PAN
    }

    /// Applies orientation, zoom, pan, and projection from `pose` while preserving reset defaults.
    pub fn apply_pose(&mut self, pose: Self) {
        self.pitch = pose.pitch;
        self.yaw = pose.yaw;
        self.zoom = pose.zoom;
        self.pan_x = pose.pan_x;
        self.pan_y = pose.pan_y;
        self.projection = pose.projection;
    }

    /// Restores camera pitch, yaw, zoom, pan offsets, and projection mode to configured defaults.
    pub fn reset(&mut self) {
        self.pitch = self.reset_pitch;
        self.yaw = self.reset_yaw;
        self.zoom = self.reset_zoom;
        self.pan_x = 0.0_f32;
        self.pan_y = 0.0_f32;
        self.projection = self.reset_projection;
    }

    /// Feeds all persisted camera parameters into `hasher` for `InteractiveTool::state_revision`.
    pub fn hash_state(&self, hasher: &mut impl Hasher) {
        self.pitch.to_bits().hash(hasher);
        self.yaw.to_bits().hash(hasher);
        self.zoom.to_bits().hash(hasher);
        self.pan_x.to_bits().hash(hasher);
        self.pan_y.to_bits().hash(hasher);
        self.projection.hash(hasher);
    }

    /// Returns a deterministic 64-bit fingerprint of all camera pose and projection fields.
    #[must_use]
    pub fn pose_fingerprint(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash_state(&mut hasher);
        hasher.finish()
    }

    /// Resolves either a nested `camera` object or legacy `pitch`/`yaw`/`zoom` preset fields.
    pub fn from_preset_fields(
        camera: Option<Self>,
        pitch: Option<f32>,
        yaw: Option<f32>,
        zoom: Option<f32>,
        fallback: Option<Self>,
    ) -> Result<Self, String> {
        let has_legacy = pitch.is_some() || yaw.is_some() || zoom.is_some();
        if let Some(cam) = camera {
            if has_legacy {
                return Err("Conflicting camera and legacy pitch/yaw/zoom fields".to_string());
            }
            if !cam.is_valid() {
                return Err("Camera parameters out of valid bounds".to_string());
            }
            return Ok(cam);
        }
        match (pitch, yaw, zoom) {
            (Some(p), Some(y), Some(z)) => {
                let cam = Self::new(p, y, z);
                if !cam.is_valid() {
                    return Err("Camera parameters out of valid bounds".to_string());
                }
                Ok(cam)
            }
            (None, None, None) => {
                fallback.ok_or_else(|| "Missing camera configuration in preset state".to_string())
            }
            _ => Err("Incomplete legacy camera fields: pitch, yaw, and zoom required".to_string()),
        }
    }

    /// Snaps camera orientation to Top view (XY plane).
    pub fn snap_to_xy(&mut self) {
        self.pitch = 0.0_f32;
        self.yaw = 0.0_f32;
    }

    /// Snaps camera orientation to Front view (XZ plane).
    pub fn snap_to_xz(&mut self) {
        self.pitch = -std::f32::consts::FRAC_PI_2;
        self.yaw = 0.0_f32;
    }

    /// Snaps camera orientation to Side view (YZ plane).
    pub fn snap_to_yz(&mut self) {
        self.pitch = -std::f32::consts::FRAC_PI_2;
        self.yaw = -std::f32::consts::FRAC_PI_2;
    }

    /// Snaps camera orientation to Isometric view.
    pub fn snap_to_iso(&mut self) {
        self.pitch = 0.5_f32;
        self.yaw = 0.5_f32;
    }
}

#[cfg(all(test, feature = "physics", feature = "pure_math"))]
mod scene_tests;
#[cfg(test)]
#[path = "../camera_tests.rs"]
mod tests;
#[cfg(all(test, feature = "ai", feature = "physics", feature = "pure_math"))]
mod world_readout_tests;
