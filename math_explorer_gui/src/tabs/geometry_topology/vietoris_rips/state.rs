#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{VietorisRipsTool, MAX_VIETORIS_RIPS_POINTS};
use crate::framework::{Camera3D, InteractiveTool};
use math_explorer::pure_math::statistics::tda::{Point2D, Point3D};

/// Maximum valid filtration radius accepted by `VietorisRipsTool`.
pub const MAX_VR_RADIUS: f64 = 30.0;
/// Maximum absolute coordinate magnitude accepted for 2D/3D point clouds.
pub const MAX_VR_COORD_ABS: f64 = 200.0;

/// Data source mode for `VietorisRipsTool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum VrSourceMode {
    /// Planar 2D point cloud (legacy-compatible default).
    #[default]
    #[serde(rename = "points_2d")]
    Points2D,
    /// Custom or sculpted 3D landmark cloud rendered in `Plot3DViewport`.
    #[serde(rename = "points_3d")]
    Points3D,
    /// Live 3D Lorenz strange attractor subsampled via farthest-point landmarks.
    #[serde(rename = "lorenz_3d")]
    Lorenz3D,
    /// 3D Takens delay-coordinate embedding `(x(t), x(t - τ), x(t - 2τ))` from scalar `x(t)`.
    #[serde(rename = "takens_3d")]
    Takens3D,
}

impl VrSourceMode {
    /// Returns `true` when the active point cloud lives in 3D space.
    pub fn is_3d(self) -> bool {
        !matches!(self, Self::Points2D)
    }
}

/// Interactive canvas gesture / vertex sculpting brush mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SculptBrush {
    /// Standard 2D plot pan/zoom or 3D `Camera3D` orbit/zoom.
    #[default]
    OrbitOrPan,
    /// Drag the nearest landmark within the brush radius.
    Move,
    /// Insert a new landmark (capped at `MAX_VIETORIS_RIPS_POINTS`).
    Add,
    /// Delete landmarks inside the brush radius (preserving at least 1 vertex).
    Erase,
}

/// Serializable Lorenz & Takens configuration for deterministic 3D preset/URL round-trips.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VrChaosConfig {
    /// Prandtl number `σ` in `[0.0, 50.0]`.
    pub sigma: f64,
    /// Rayleigh number `ρ` in `[0.0, 100.0]`.
    pub rho: f64,
    /// Geometric factor `β` in `[0.0, 10.0]`.
    pub beta: f64,
    /// Integration step `dt` in `[0.001, 0.05]`.
    pub dt: f64,
    /// Subsampled landmark budget in `[4, 64]`.
    pub landmark_count: usize,
    /// Takens delay lag `τ` in `[1, 50]`.
    pub takens_lag: usize,
}

impl VrChaosConfig {
    /// Validates finite numeric bounds on Lorenz and Takens parameters.
    pub fn validate(&self) -> Result<(), String> {
        if !self.sigma.is_finite() || !(0.0..=50.0).contains(&self.sigma) {
            return Err(format!("sigma out of [0.0, 50.0]: {}", self.sigma));
        }
        if !self.rho.is_finite() || !(0.0..=100.0).contains(&self.rho) {
            return Err(format!("rho out of [0.0, 100.0]: {}", self.rho));
        }
        if !self.beta.is_finite() || !(0.0..=10.0).contains(&self.beta) {
            return Err(format!("beta out of [0.0, 10.0]: {}", self.beta));
        }
        if !self.dt.is_finite() || !(0.001..=0.05).contains(&self.dt) {
            return Err(format!("dt out of [0.001, 0.05]: {}", self.dt));
        }
        if !(4..=MAX_VIETORIS_RIPS_POINTS).contains(&self.landmark_count) {
            return Err(format!(
                "landmark_count out of [4, {MAX_VIETORIS_RIPS_POINTS}]: {}",
                self.landmark_count
            ));
        }
        if !(1..=50).contains(&self.takens_lag) {
            return Err(format!("takens_lag out of [1, 50]: {}", self.takens_lag));
        }
        Ok(())
    }
}

struct BorrowedPoints2D<'a>(&'a [Point2D]);

impl serde::Serialize for BorrowedPoints2D<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for pt in self.0 {
            seq.serialize_element(&[pt.x, pt.y])?;
        }
        seq.end()
    }
}

struct BorrowedPoints3D<'a>(&'a [Point3D]);

impl serde::Serialize for BorrowedPoints3D<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for pt in self.0 {
            seq.serialize_element(&[pt.x, pt.y, pt.z])?;
        }
        seq.end()
    }
}

#[derive(serde::Serialize)]
struct Legacy2DStateRef<'a> {
    radius: f64,
    points: BorrowedPoints2D<'a>,
}

#[derive(serde::Serialize)]
struct Extended3DStateRef<'a> {
    mode: VrSourceMode,
    radius: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    points_3d: Option<BorrowedPoints3D<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chaos: Option<VrChaosConfig>,
    camera: Camera3D,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct WireState {
    #[serde(default)]
    mode: Option<VrSourceMode>,
    radius: f64,
    #[serde(default)]
    points: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    points_3d: Option<Vec<[f64; 3]>>,
    #[serde(default)]
    chaos: Option<VrChaosConfig>,
    #[serde(default)]
    camera: Option<Camera3D>,
}

impl VietorisRipsTool {
    pub(super) fn serialize_tool_state(&self) -> Option<String> {
        match self.mode {
            VrSourceMode::Points2D => serde_json::to_string(&Legacy2DStateRef {
                radius: self.radius,
                points: BorrowedPoints2D(&self.points),
            })
            .ok(),
            VrSourceMode::Points3D => serde_json::to_string(&Extended3DStateRef {
                mode: self.mode,
                radius: self.radius,
                points_3d: Some(BorrowedPoints3D(&self.points_3d)),
                chaos: None,
                camera: self.camera,
            })
            .ok(),
            VrSourceMode::Lorenz3D | VrSourceMode::Takens3D => {
                serde_json::to_string(&Extended3DStateRef {
                    mode: self.mode,
                    radius: self.radius,
                    points_3d: None,
                    chaos: Some(self.chaos.to_config()),
                    camera: self.camera,
                })
                .ok()
            }
        }
    }

    pub(super) fn deserialize_tool_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let wire: WireState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Vietoris-Rips state: {e}"))?;
        let mode = wire.mode.unwrap_or(VrSourceMode::Points2D);
        let max_r = if mode.is_3d() { MAX_VR_RADIUS } else { 15.0 };
        if !wire.radius.is_finite() || !(0.0..=max_r).contains(&wire.radius) {
            return Err(format!("radius out of [0.0, {max_r:.1}]: {}", wire.radius));
        }
        if let Some(cam) = wire.camera {
            if !cam.is_valid() {
                return Err("camera parameters out of valid bounds".to_string());
            }
        }
        match mode {
            VrSourceMode::Points2D => self.apply_wire_2d(wire),
            VrSourceMode::Points3D => self.apply_wire_3d(wire),
            VrSourceMode::Lorenz3D | VrSourceMode::Takens3D => self.apply_wire_chaos(mode, wire),
        }
    }

    fn apply_wire_2d(&mut self, wire: WireState) -> Result<(), String> {
        let raw = wire
            .points
            .ok_or_else(|| "missing 'points' for 2D Vietoris-Rips state".to_string())?;
        if !(1..=MAX_VIETORIS_RIPS_POINTS).contains(&raw.len())
            || !raw.iter().all(|p| {
                p[0].is_finite()
                    && p[1].is_finite()
                    && p[0].abs() <= MAX_VR_COORD_ABS
                    && p[1].abs() <= MAX_VR_COORD_ABS
            })
        {
            return Err(format!(
                "points must contain 1..={MAX_VIETORIS_RIPS_POINTS} finite bounded coordinates"
            ));
        }
        let pts: Vec<Point2D> = raw.into_iter().map(|[x, y]| Point2D::new(x, y)).collect();
        self.load_validated_2d_points(pts, wire.radius)
    }

    fn apply_wire_3d(&mut self, wire: WireState) -> Result<(), String> {
        let raw = wire
            .points_3d
            .ok_or_else(|| "missing 'points_3d' for 3D Vietoris-Rips state".to_string())?;
        if !(1..=MAX_VIETORIS_RIPS_POINTS).contains(&raw.len())
            || !raw.iter().all(|p| {
                p.iter()
                    .all(|c| c.is_finite() && c.abs() <= MAX_VR_COORD_ABS)
            })
        {
            return Err(format!(
                "points_3d must contain 1..={MAX_VIETORIS_RIPS_POINTS} finite bounded coordinates"
            ));
        }
        let cam = wire
            .camera
            .unwrap_or_else(|| Camera3D::new(0.35, 0.55, 1.0));
        let pts: Vec<Point3D> = raw
            .into_iter()
            .map(|[x, y, z]| Point3D::new(x, y, z))
            .collect();
        self.camera.apply_pose(cam);
        self.load_validated_3d_points(pts, wire.radius)
    }

    fn apply_wire_chaos(&mut self, mode: VrSourceMode, wire: WireState) -> Result<(), String> {
        let cfg = wire
            .chaos
            .ok_or_else(|| "missing 'chaos' configuration for chaotic 3D mode".to_string())?;
        cfg.validate()?;
        let cam = wire
            .camera
            .unwrap_or_else(|| Camera3D::new(0.35, 0.55, 1.0));
        self.camera.apply_pose(cam);
        self.mode = mode;
        self.radius = wire.radius;
        self.chaos.apply_config(cfg);
        self.chaos.playing = false;
        self.rebuild_chaos_landmarks()?;
        Ok(())
    }
}
