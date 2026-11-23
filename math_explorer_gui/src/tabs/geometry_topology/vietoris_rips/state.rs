//! State enums and bounded JSON/URL serialization for 2D, N-D, and 3D `VietorisRipsTool` modes.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::plots::PersistencePlotMode;
use super::presets::{generate_preset_points, GeneratorDescriptor};
use super::{CloudSource, VietorisRipsTool, MAX_CHAOS_LANDMARKS, MAX_VIETORIS_RIPS_POINTS};
use crate::framework::Camera3D;
use math_explorer::pure_math::statistics::tda::core::{NdPointCloud, Point2D};
use math_explorer::pure_math::statistics::tda::Point3D;

/// Maximum valid filtration radius in 3D chaos/sculpt modes.
pub const MAX_VR_RADIUS: f64 = 30.0;
/// Maximum absolute coordinate magnitude accepted for sculpted/serialized 3D landmarks.
pub const MAX_VR_COORD_ABS: f64 = 200.0;

const MAX_ALLOWED_RADIUS: f64 = 1_000_000.0;

/// Data source mode for `VietorisRipsTool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum VrSourceMode {
    /// Planar 2D or projected N-D workbench cloud (default).
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
    #[must_use]
    pub const fn is_3d(self) -> bool {
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
    /// Insert a new landmark.
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
        if !(4..=MAX_CHAOS_LANDMARKS).contains(&self.landmark_count) {
            return Err(format!(
                "landmark_count out of [4, {MAX_CHAOS_LANDMARKS}]: {}",
                self.landmark_count
            ));
        }
        if !(1..=50).contains(&self.takens_lag) {
            return Err(format!("takens_lag out of [1, 50]: {}", self.takens_lag));
        }
        Ok(())
    }
}

struct SerializedPoints2D<'a> {
    pts: &'a [Point2D],
    quantize: bool,
}

impl serde::Serialize for SerializedPoints2D<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(self.pts.len()))?;
        for pt in self.pts {
            let pair = if self.quantize {
                [
                    super::io::quantize_url_coord(pt.x),
                    super::io::quantize_url_coord(pt.y),
                ]
            } else {
                [pt.x, pt.y]
            };
            seq.serialize_element(&pair)?;
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

fn is_zero(val: &f64) -> bool {
    *val == 0.0
}

#[derive(serde::Serialize)]
struct VietorisRipsSaveState2D<'a> {
    radius: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    points: Option<SerializedPoints2D<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generator: Option<&'a GeneratorDescriptor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    points_nd: Option<&'a [Vec<f64>]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    axis_x: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    axis_y: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plot_mode: Option<PersistencePlotMode>,
    #[serde(skip_serializing_if = "is_zero")]
    min_persistence: f64,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    plot_mode: Option<PersistencePlotMode>,
    #[serde(skip_serializing_if = "is_zero")]
    min_persistence: f64,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct VietorisRipsLoadState {
    #[serde(default)]
    mode: Option<VrSourceMode>,
    radius: f64,
    #[serde(default)]
    points: Option<Vec<[f64; 2]>>,
    #[serde(default)]
    generator: Option<GeneratorDescriptor>,
    #[serde(default)]
    points_nd: Option<Vec<Vec<f64>>>,
    #[serde(default)]
    axis_x: Option<usize>,
    #[serde(default)]
    axis_y: Option<usize>,
    #[serde(default)]
    points_3d: Option<Vec<[f64; 3]>>,
    #[serde(default)]
    chaos: Option<VrChaosConfig>,
    #[serde(default)]
    camera: Option<Camera3D>,
    #[serde(default)]
    plot_mode: PersistencePlotMode,
    #[serde(default)]
    min_persistence: f64,
}

pub(super) fn serialize_tool_state(tool: &VietorisRipsTool) -> Option<String> {
    let mode_opt = (!tool.plot_mode.is_default()).then_some(tool.plot_mode);
    match tool.mode {
        VrSourceMode::Points2D => serialize_2d_or_nd(tool, mode_opt),
        VrSourceMode::Points3D => serde_json::to_string(&Extended3DStateRef {
            mode: tool.mode,
            radius: tool.radius,
            points_3d: Some(BorrowedPoints3D(&tool.points_3d)),
            chaos: None,
            camera: tool.camera,
            plot_mode: mode_opt,
            min_persistence: tool.min_persistence,
        })
        .ok(),
        VrSourceMode::Lorenz3D | VrSourceMode::Takens3D => {
            serde_json::to_string(&Extended3DStateRef {
                mode: tool.mode,
                radius: tool.radius,
                points_3d: None,
                chaos: Some(tool.chaos.to_config()),
                camera: tool.camera,
                plot_mode: mode_opt,
                min_persistence: tool.min_persistence,
            })
            .ok()
        }
    }
}

fn serialize_2d_or_nd(
    tool: &VietorisRipsTool,
    mode_opt: Option<PersistencePlotMode>,
) -> Option<String> {
    let state = match &tool.source {
        CloudSource::Custom2D(pts) => VietorisRipsSaveState2D {
            radius: tool.radius,
            points: Some(SerializedPoints2D {
                pts,
                quantize: pts.len() <= super::io::MAX_URL_CUSTOM_POINTS_2D,
            }),
            generator: None,
            points_nd: None,
            axis_x: None,
            axis_y: None,
            plot_mode: mode_opt,
            min_persistence: tool.min_persistence,
        },
        CloudSource::Generator(desc) => VietorisRipsSaveState2D {
            radius: tool.radius,
            points: None,
            generator: Some(desc),
            points_nd: None,
            axis_x: None,
            axis_y: None,
            plot_mode: mode_opt,
            min_persistence: tool.min_persistence,
        },
        CloudSource::CustomNd {
            cloud,
            axis_x,
            axis_y,
        } => VietorisRipsSaveState2D {
            radius: tool.radius,
            points: None,
            generator: None,
            points_nd: Some(cloud.points()),
            axis_x: Some(*axis_x),
            axis_y: Some(*axis_y),
            plot_mode: mode_opt,
            min_persistence: tool.min_persistence,
        },
    };
    serde_json::to_string(&state).ok()
}

pub(super) fn deserialize_tool_state(
    tool: &mut VietorisRipsTool,
    state: &str,
) -> Result<(), String> {
    let loaded: VietorisRipsLoadState =
        serde_json::from_str(state).map_err(|e| format!("Invalid Vietoris-Rips state: {e}"))?;
    let mode = loaded.mode.unwrap_or(VrSourceMode::Points2D);
    let max_r = if mode.is_3d() {
        MAX_VR_RADIUS
    } else {
        MAX_ALLOWED_RADIUS
    };
    if !loaded.radius.is_finite() || !(0.0..=max_r).contains(&loaded.radius) {
        return Err(format!("radius out of [0.0, {max_r}]: {}", loaded.radius));
    }
    if !loaded.min_persistence.is_finite()
        || !(0.0..=MAX_ALLOWED_RADIUS).contains(&loaded.min_persistence)
    {
        return Err(format!(
            "invalid min_persistence: {}",
            loaded.min_persistence
        ));
    }
    if let Some(cam) = loaded.camera {
        if !cam.is_valid() {
            return Err("camera parameters out of valid bounds".to_string());
        }
    }
    let (plot_mode, min_p) = (loaded.plot_mode, loaded.min_persistence);
    match mode {
        VrSourceMode::Points2D => apply_loaded_2d_or_nd(tool, loaded)?,
        VrSourceMode::Points3D => apply_loaded_3d(tool, loaded)?,
        VrSourceMode::Lorenz3D | VrSourceMode::Takens3D => apply_loaded_chaos(tool, mode, loaded)?,
    }
    tool.plot_mode = plot_mode;
    tool.min_persistence = min_p;
    tool.refresh_url_budget_flag();
    Ok(())
}

fn apply_loaded_2d_or_nd(
    tool: &mut VietorisRipsTool,
    loaded: VietorisRipsLoadState,
) -> Result<(), String> {
    let sources = usize::from(loaded.points.is_some())
        + usize::from(loaded.generator.is_some())
        + usize::from(loaded.points_nd.is_some());
    if sources != 1 {
        return Err(
            "Vietoris-Rips state must specify one of 'points', 'generator', or 'points_nd'"
                .to_string(),
        );
    }
    if let Some(raw_pts) = loaded.points {
        if !(1..=MAX_VIETORIS_RIPS_POINTS).contains(&raw_pts.len())
            || !raw_pts.iter().all(|p| p[0].is_finite() && p[1].is_finite())
        {
            return Err(format!(
                "points must contain 1..={MAX_VIETORIS_RIPS_POINTS} finite points"
            ));
        }
        let quantize = raw_pts.len() <= super::io::MAX_URL_CUSTOM_POINTS_2D;
        let pts: Vec<Point2D> = raw_pts
            .into_iter()
            .map(|[x, y]| {
                if quantize {
                    Point2D::new(
                        super::io::quantize_url_coord(x),
                        super::io::quantize_url_coord(y),
                    )
                } else {
                    Point2D::new(x, y)
                }
            })
            .collect();
        tool.rebuild_from_2d(CloudSource::Custom2D(pts), Vec::new(), loaded.radius)?;
    } else if let Some(desc) = loaded.generator {
        desc.validate()?;
        let pts = generate_preset_points(&desc);
        tool.rebuild_from_2d(CloudSource::Generator(desc), pts, loaded.radius)?;
    } else if let Some(nd_rows) = loaded.points_nd {
        let nd = NdPointCloud::new(nd_rows).map_err(|e| format!("{e}"))?;
        let ax = loaded.axis_x.unwrap_or(0);
        let ay = loaded
            .axis_y
            .unwrap_or(1.min(nd.dimension().saturating_sub(1)));
        tool.rebuild_from_nd(nd, ax, ay, loaded.radius)?;
    }
    Ok(())
}

fn apply_loaded_3d(
    tool: &mut VietorisRipsTool,
    loaded: VietorisRipsLoadState,
) -> Result<(), String> {
    let raw = loaded
        .points_3d
        .ok_or_else(|| "missing 'points_3d' for 3D Vietoris-Rips state".to_string())?;
    if !(1..=MAX_CHAOS_LANDMARKS).contains(&raw.len())
        || !raw.iter().all(|p| {
            p.iter()
                .all(|c| c.is_finite() && c.abs() <= MAX_VR_COORD_ABS)
        })
    {
        return Err(format!(
            "points_3d must contain 1..={MAX_CHAOS_LANDMARKS} finite bounded coordinates"
        ));
    }
    let cam = loaded
        .camera
        .unwrap_or_else(|| Camera3D::new(0.35, 0.55, 1.0));
    let pts: Vec<Point3D> = raw
        .into_iter()
        .map(|[x, y, z]| Point3D::new(x, y, z))
        .collect();
    tool.camera.apply_pose(cam);
    tool.load_validated_3d_points(pts, loaded.radius)
}

fn apply_loaded_chaos(
    tool: &mut VietorisRipsTool,
    mode: VrSourceMode,
    loaded: VietorisRipsLoadState,
) -> Result<(), String> {
    let cfg = loaded
        .chaos
        .ok_or_else(|| "missing 'chaos' configuration for chaotic 3D mode".to_string())?;
    cfg.validate()?;
    let cam = loaded
        .camera
        .unwrap_or_else(|| Camera3D::new(0.35, 0.55, 1.0));
    tool.camera.apply_pose(cam);
    tool.mode = mode;
    tool.radius = loaded.radius;
    tool.chaos.apply_config(cfg);
    tool.chaos.playing = false;
    tool.rebuild_chaos_landmarks()
}
