#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Validated configuration, bounds, and enums for [`super::VolumetricFieldStudioTool`].
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]

use super::ast_field::DEFAULT_CUSTOM_3D_EXPR;
use super::webgpu_runner::ComputeBackend;
use crate::camera::Camera3D;
use crate::volumetric_scene::VolumeCutFilter;
use math_explorer::physics::quantum::{OrbitalQuantumNumbers, QuantumFieldObservable};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Minimum interactive 3D voxel grid dimension (`16^3 = 4,096` voxels).
pub const MIN_VOLUMETRIC_GRID: usize = 16;
/// Maximum CPU zero-alloc 3D voxel grid dimension (`48^3 = 110,592` voxels).
pub const MAX_VOLUMETRIC_GRID: usize = 48;
/// Maximum WebGPU compute 3D voxel grid dimension (`128^3 = 2,097,152` voxels).
pub const MAX_WEBGPU_VOLUMETRIC_GRID: usize = 128;
/// World-space half-extent of the normalized 3D studio box `[-2.5, +2.5]^3`.
pub const STUDIO_VIEWPORT_EXTENT: f64 = 2.5;
/// Camera distance used for perspective splat covariance projection.
pub const STUDIO_CAMERA_DISTANCE: f64 = 5.5;
/// Default implicit formula (`Gyroid` triply periodic minimal surface).
pub const DEFAULT_IMPLICIT_EXPR: &str = "sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)";
/// Minimum coordinate bound allowed on implicit X, Y, and Z axes.
pub const IMPLICIT_DOMAIN_MIN: f64 = -50.0;
/// Maximum coordinate bound allowed on implicit X, Y, and Z axes.
pub const IMPLICIT_DOMAIN_MAX: f64 = 50.0;
/// Minimum span `max - min` enforced on each implicit coordinate axis.
pub const IMPLICIT_DOMAIN_MIN_SPAN: f64 = 0.1;
/// Default periodic coordinate interval `[-PI, +PI]` for implicit surfaces.
pub const DEFAULT_IMPLICIT_BOUNDS: [f64; 2] = [-std::f64::consts::PI, std::f64::consts::PI];
/// Minimum thickened implicit shell width `w`.
pub const MIN_IMPLICIT_WALL: f64 = 0.05;
/// Maximum thickened implicit shell width `w`.
pub const MAX_IMPLICIT_WALL: f64 = 4.0;
/// Default thickened implicit shell width `w`.
pub const DEFAULT_IMPLICIT_WALL: f64 = 0.45;
/// Minimum physical export size in millimeters (`mm`).
pub const MIN_EXPORT_SIZE_MM: f64 = 5.0;
/// Maximum physical export size in millimeters (`mm`).
pub const MAX_EXPORT_SIZE_MM: f64 = 250.0;
/// Default physical export size in millimeters (`mm`).
pub const DEFAULT_EXPORT_SIZE_MM: f64 = 50.0;

/// Active 3D volumetric PDE, eigenstate, implicit CAD, or signed implicit AST generator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldSourceMode {
    /// Two-state 3D hydrogenic eigenstate superposition $\Psi = c_1(t)\psi_1 + c_2(t)\psi_2$.
    #[default]
    QuantumSuperposition,
    /// Two-species 3D Schnakenberg reaction-diffusion grid via `FiniteDifference3D`.
    TuringMorphogenesis,
    /// User-authored 3D implicit CAD solid $f(x, y, z) = c$ evaluated via `CompiledExpr::eval_3d`.
    #[serde(alias = "custom_implicit_formula")]
    CustomImplicit,
    /// User-authored signed 3D implicit scalar field $f(x, y, z)$ with dual nodal shells and $20^3$ drag preview.
    CustomImplicitAst,
}

impl FieldSourceMode {
    /// Human-readable UI label.
    pub fn label(self) -> &'static str {
        match self {
            Self::QuantumSuperposition => "3D Hydrogenic Superposition",
            Self::TuringMorphogenesis => "3D Turing Reaction-Diffusion",
            Self::CustomImplicit => "3D Implicit CAD f(x,y,z) = c",
            Self::CustomImplicitAst => "Custom 3D Implicit Formula",
        }
    }
}

/// Solid-closure rule converting $f(x, y, z) = c$ into a printable 3D volume.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImplicitClosureMode {
    /// Half-space solid volume $f(x, y, z) \le c$.
    #[default]
    HalfSpaceSolid,
    /// Thickened two-sided level-set shell $|f(x, y, z) - c| \le w / 2$.
    ThickenedWall,
}

impl ImplicitClosureMode {
    /// Human-readable UI label.
    pub fn label(self) -> &'static str {
        match self {
            Self::HalfSpaceSolid => "Half-Space Solid (f <= c)",
            Self::ThickenedWall => "Thickened Wall (|f - c| <= w/2)",
        }
    }
}

/// Active 3D visualization pipeline on `Plot3DViewport`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VolumetricRenderMode {
    /// Crisp `MarchingCubes` level-set nodal isosurfaces.
    MarchingCubesMesh,
    /// Gradient-aligned `Gaussian3D` EWA radiance cloud.
    GaussianRadianceCloud,
    /// Combined `MarchingCubes` nodal shells embedded inside a `Gaussian3D` radiance cloud.
    #[default]
    HybridShellAndCloud,
}

impl VolumetricRenderMode {
    /// Human-readable UI label.
    pub fn label(self) -> &'static str {
        match self {
            Self::MarchingCubesMesh => "MarchingCubes Nodal Mesh",
            Self::GaussianRadianceCloud => "Gaussian3D Radiance Cloud",
            Self::HybridShellAndCloud => "Hybrid Shell + Cloud",
        }
    }

    /// Returns `true` if `MarchingCubes` surfaces are rendered.
    pub fn uses_mesh(self) -> bool {
        matches!(self, Self::MarchingCubesMesh | Self::HybridShellAndCloud)
    }

    /// Returns `true` if `Gaussian3D` radiance splats are rendered.
    pub fn uses_splats(self) -> bool {
        matches!(
            self,
            Self::GaussianRadianceCloud | Self::HybridShellAndCloud
        )
    }
}

/// Scalar observable mapped into `VoxelGrid` in quantum mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantumObservableChoice {
    /// Signed real wavefunction $\text{Re}\,\Psi(\mathbf{r}, t)$ showing $+/-$ nodal lobes.
    #[default]
    SignedWavefunction,
    /// Born probability density $|\Psi(\mathbf{r}, t)|^2$.
    ProbabilityDensity,
}

impl QuantumObservableChoice {
    /// Converts to the `domain_physics` observable enum.
    pub fn to_physics(self) -> QuantumFieldObservable {
        match self {
            Self::SignedWavefunction => QuantumFieldObservable::SignedRealWavefunction,
            Self::ProbabilityDensity => QuantumFieldObservable::ProbabilityDensity,
        }
    }
}

/// Coordinate axis perpendicular to the 3D cutting plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceAxis {
    /// Cut along world X (`x <= offset` retained).
    #[default]
    X,
    /// Cut along world Y (`y <= offset` retained).
    Y,
    /// Cut along world Z (`z <= offset` retained).
    Z,
}

impl SliceAxis {
    /// Numeric coordinate index (`0 = X`, `1 = Y`, `2 = Z`).
    pub fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    /// Short axis label.
    pub fn label(self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_implicit_expr() -> String {
    DEFAULT_IMPLICIT_EXPR.to_string()
}
fn default_implicit_bounds() -> [f64; 2] {
    DEFAULT_IMPLICIT_BOUNDS
}
fn default_wall_width() -> f64 {
    DEFAULT_IMPLICIT_WALL
}
fn default_export_size_mm() -> f64 {
    DEFAULT_EXPORT_SIZE_MM
}
fn default_custom_3d_expr() -> String {
    DEFAULT_CUSTOM_3D_EXPR.to_string()
}
fn is_default_custom_3d_expr(expr: &str) -> bool {
    expr == DEFAULT_CUSTOM_3D_EXPR
}

/// Complete serializable state for `VolumetricFieldStudioTool` (`#[serde(deny_unknown_fields)]`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolumetricStudioConfig {
    /// Active 3D volumetric simulation, implicit CAD solid, or signed AST field model.
    pub source: FieldSourceMode,
    /// Active 3D surface/splat renderer.
    pub render_mode: VolumetricRenderMode,
    /// Execution backend (`cpu_zero_alloc` up to `48^3` or `web_gpu_compute` up to `128^3`).
    #[serde(default, skip_serializing_if = "ComputeBackend::is_cpu_default")]
    pub compute_backend: ComputeBackend,
    /// Cubic voxel resolution (`16..=48` on CPU, `16..=128` on WebGPU Compute).
    pub grid_resolution: usize,
    /// State 1 principal quantum number (`1..=6`).
    pub n1: u32,
    /// State 1 orbital angular momentum (`0..n1`).
    pub l1: u32,
    /// State 1 magnetic quantum number (`-l1..=l1`).
    pub m1: i32,
    /// State 2 principal quantum number (`1..=6`).
    pub n2: u32,
    /// State 2 orbital angular momentum (`0..n2`).
    pub l2: u32,
    /// State 2 magnetic quantum number (`-l2..=l2`).
    pub m2: i32,
    /// Superposition mixing fraction in `[0.0, 1.0]`.
    pub superposition_mix: f64,
    /// Initial/perturbed relative quantum phase in `[-PI, PI]`.
    pub relative_phase: f64,
    /// Quantum time step `dt` in `[0.005, 0.5]`.
    pub quantum_dt: f64,
    /// Quantum scalar observable choice.
    #[serde(default)]
    pub observable: QuantumObservableChoice,
    /// Isosurface level threshold in `[0.002, 0.8]`.
    pub iso_threshold: f64,
    /// Gaussian3D principal-axis scale multiplier in `[0.2, 4.0]`.
    pub splat_scale: f64,
    /// Schnakenberg rate `a` in `[0.01, 0.5]`.
    pub turing_a: f64,
    /// Schnakenberg rate `b` in `[0.1, 2.0]`.
    pub turing_b: f64,
    /// Activator diffusion rate `D_u` in `[0.1, 5.0]`.
    pub turing_du: f64,
    /// Inhibitor diffusion rate `D_v` in `[5.0, 60.0]`.
    pub turing_dv: f64,
    /// Turing integration step `dt` in `[0.001, 0.05]`.
    pub turing_dt: f64,
    /// User-authored signed 3D AST scalar field expression over `(x, y, z)` (`<= 512` bytes).
    #[serde(
        default = "default_custom_3d_expr",
        skip_serializing_if = "is_default_custom_3d_expr"
    )]
    pub custom_expr: String,
    /// Synthesized free-parameter values paired with `custom_expr`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom_params: BTreeMap<String, f64>,
    /// Implicit CAD equation formula string $f(x, y, z)$.
    #[serde(default = "default_implicit_expr")]
    pub implicit_expr: String,
    /// Free scalar parameter values keyed by identifier name for `implicit_expr`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub implicit_params: BTreeMap<String, f64>,
    /// Implicit level-set constant $c$ in $f(x, y, z) = c$ (`[-20.0, 20.0]`).
    #[serde(default)]
    pub implicit_level: f64,
    /// Solid closure mode (`f <= c` half-space or `|f - c| <= w/2` wall).
    #[serde(default)]
    pub implicit_closure: ImplicitClosureMode,
    /// Wall thickness $w$ for `ImplicitClosureMode::ThickenedWall` (`[0.05, 4.0]`).
    #[serde(default = "default_wall_width")]
    pub wall_thickness: f64,
    /// Ordered X domain bounds `[x_min, x_max]` in `[-50.0, 50.0]`.
    #[serde(default = "default_implicit_bounds")]
    pub x_bounds: [f64; 2],
    /// Ordered Y domain bounds `[y_min, y_max]` in `[-50.0, 50.0]`.
    #[serde(default = "default_implicit_bounds")]
    pub y_bounds: [f64; 2],
    /// Ordered Z domain bounds `[z_min, z_max]` in `[-50.0, 50.0]`.
    #[serde(default = "default_implicit_bounds")]
    pub z_bounds: [f64; 2],
    /// Physical bounding-box target span in millimeters (`[5.0, 250.0]`).
    #[serde(default = "default_export_size_mm")]
    pub export_size_mm: f64,
    /// Enables 3D half-space cutting plane & cross-section nodal contours.
    pub cut_enabled: bool,
    /// Cutting-plane normal axis.
    pub cut_axis: SliceAxis,
    /// Normalized cutting-plane offset in `[-1.0, 1.0]`.
    pub cut_offset: f64,
    /// Enables viewport click probe / activator perturbation brush.
    #[serde(default = "default_true")]
    pub probe_brush_enabled: bool,
    /// Whether live cooperative stepping is running.
    #[serde(default)]
    pub running: bool,
    /// Shared 3D orbit & zoom camera pose.
    pub camera: Camera3D,
}

impl Default for VolumetricStudioConfig {
    fn default() -> Self {
        Self {
            source: FieldSourceMode::QuantumSuperposition,
            render_mode: VolumetricRenderMode::HybridShellAndCloud,
            compute_backend: ComputeBackend::CpuZeroAlloc,
            grid_resolution: 28,
            n1: 1,
            l1: 0,
            m1: 0,
            n2: 2,
            l2: 1,
            m2: 0,
            superposition_mix: 0.5,
            relative_phase: 0.0,
            quantum_dt: 0.08,
            observable: QuantumObservableChoice::SignedWavefunction,
            iso_threshold: 0.032,
            splat_scale: 1.0,
            turing_a: 0.10,
            turing_b: 0.90,
            turing_du: 1.0,
            turing_dv: 18.0,
            turing_dt: 0.008,
            custom_expr: default_custom_3d_expr(),
            custom_params: BTreeMap::new(),
            implicit_expr: default_implicit_expr(),
            implicit_params: BTreeMap::new(),
            implicit_level: 0.0,
            implicit_closure: ImplicitClosureMode::HalfSpaceSolid,
            wall_thickness: DEFAULT_IMPLICIT_WALL,
            x_bounds: DEFAULT_IMPLICIT_BOUNDS,
            y_bounds: DEFAULT_IMPLICIT_BOUNDS,
            z_bounds: DEFAULT_IMPLICIT_BOUNDS,
            export_size_mm: DEFAULT_EXPORT_SIZE_MM,
            cut_enabled: false,
            cut_axis: SliceAxis::X,
            cut_offset: 0.0,
            probe_brush_enabled: true,
            running: false,
            camera: Camera3D::new(0.45, 0.60, 1.0),
        }
    }
}

impl VolumetricStudioConfig {
    /// Maximum cubic grid dimension permitted by `self.compute_backend` (`48` CPU, `128` WebGPU).
    #[must_use]
    pub const fn max_grid_resolution(&self) -> usize {
        match self.compute_backend {
            ComputeBackend::CpuZeroAlloc => MAX_VOLUMETRIC_GRID,
            ComputeBackend::WebGpuCompute => MAX_WEBGPU_VOLUMETRIC_GRID,
        }
    }

    /// Physical Bohr-radius box half-extent auto-scaled to fit principal quantum numbers `max(n1, n2)`.
    pub fn physical_quantum_extent(&self) -> f64 {
        let n_max = self.n1.max(self.n2).clamp(1, 6) as f64;
        (1.8 * n_max * n_max + 2.0).clamp(5.5, 36.0)
    }

    /// Isotropically normalized viewport half-extents `[hx, hy, hz]` within `STUDIO_VIEWPORT_EXTENT`.
    pub fn viewport_half_extents(&self) -> [f64; 3] {
        if self.source != FieldSourceMode::CustomImplicit {
            return [STUDIO_VIEWPORT_EXTENT; 3];
        }
        let sx = (self.x_bounds[1] - self.x_bounds[0]).max(IMPLICIT_DOMAIN_MIN_SPAN);
        let sy = (self.y_bounds[1] - self.y_bounds[0]).max(IMPLICIT_DOMAIN_MIN_SPAN);
        let sz = (self.z_bounds[1] - self.z_bounds[0]).max(IMPLICIT_DOMAIN_MIN_SPAN);
        let smax = sx.max(sy).max(sz);
        [
            STUDIO_VIEWPORT_EXTENT * (sx / smax),
            STUDIO_VIEWPORT_EXTENT * (sy / smax),
            STUDIO_VIEWPORT_EXTENT * (sz / smax),
        ]
    }

    /// World-space cutting plane filter when `cut_enabled` is active.
    pub fn active_cut_filter(&self) -> Option<VolumeCutFilter> {
        let ext = self.viewport_half_extents();
        let axis = self.cut_axis.index();
        self.cut_enabled.then(|| VolumeCutFilter {
            axis,
            max_coord: self.cut_offset.clamp(-1.0, 1.0) * ext[axis],
        })
    }

    /// Strictly validates quantum numbers, PDE rates, implicit CAD/AST formulas, and `Camera3D` pose.
    pub fn validate(&self) -> Result<(), String> {
        let max_grid = self.max_grid_resolution();
        if !(MIN_VOLUMETRIC_GRID..=max_grid).contains(&self.grid_resolution) {
            return Err(format!(
                "grid_resolution={} must be in {MIN_VOLUMETRIC_GRID}..={max_grid}",
                self.grid_resolution
            ));
        }
        OrbitalQuantumNumbers::new(self.n1, self.l1, self.m1).map_err(|err| err.to_string())?;
        OrbitalQuantumNumbers::new(self.n2, self.l2, self.m2).map_err(|err| err.to_string())?;
        self.validate_scalar_ranges()?;
        super::implicit::validate_implicit_config(self)?;
        super::ast_field::validate_custom_ast_config(self)?;
        if !self.camera.is_valid() {
            return Err("Invalid Camera3D pose in VolumetricFieldStudioTool".into());
        }
        Ok(())
    }

    fn validate_scalar_ranges(&self) -> Result<(), String> {
        let ranges = [
            ("superposition_mix", self.superposition_mix, 0.0, 1.0),
            ("relative_phase", self.relative_phase, -4.0, 4.0),
            ("quantum_dt", self.quantum_dt, 0.005, 0.5),
            ("iso_threshold", self.iso_threshold, 0.002, 0.8),
            ("splat_scale", self.splat_scale, 0.2, 4.0),
            ("turing_a", self.turing_a, 0.01, 0.5),
            ("turing_b", self.turing_b, 0.1, 2.0),
            ("turing_du", self.turing_du, 0.1, 5.0),
            ("turing_dv", self.turing_dv, 5.0, 60.0),
            ("turing_dt", self.turing_dt, 0.001, 0.05),
            ("implicit_level", self.implicit_level, -20.0, 20.0),
            (
                "wall_thickness",
                self.wall_thickness,
                MIN_IMPLICIT_WALL,
                MAX_IMPLICIT_WALL,
            ),
            (
                "export_size_mm",
                self.export_size_mm,
                MIN_EXPORT_SIZE_MM,
                MAX_EXPORT_SIZE_MM,
            ),
            ("cut_offset", self.cut_offset, -1.0, 1.0),
        ];
        for (name, val, lo, hi) in ranges {
            if !val.is_finite() || val < lo || val > hi {
                return Err(format!(
                    "Parameter '{name}'={val} must be finite within [{lo}, {hi}]"
                ));
            }
        }
        Ok(())
    }

    /// Clamps any out-of-range or non-finite fields back into safe physical bounds.
    pub fn clamp_in_place(&mut self) {
        let max_grid = self.max_grid_resolution();
        self.grid_resolution = self.grid_resolution.clamp(MIN_VOLUMETRIC_GRID, max_grid);
        let q1 = OrbitalQuantumNumbers::clamped(self.n1, self.l1, self.m1);
        (self.n1, self.l1, self.m1) = (q1.n, q1.l, q1.m);
        let q2 = OrbitalQuantumNumbers::clamped(self.n2, self.l2, self.m2);
        (self.n2, self.l2, self.m2) = (q2.n, q2.l, q2.m);
        let pi = std::f64::consts::PI;
        self.superposition_mix = clamp_finite(self.superposition_mix, 0.0, 1.0, 0.5);
        self.relative_phase = clamp_finite(self.relative_phase, -pi, pi, 0.0);
        self.quantum_dt = clamp_finite(self.quantum_dt, 0.005, 0.5, 0.08);
        self.iso_threshold = clamp_finite(self.iso_threshold, 0.002, 0.8, 0.032);
        self.splat_scale = clamp_finite(self.splat_scale, 0.2, 4.0, 1.0);
        self.turing_a = clamp_finite(self.turing_a, 0.01, 0.5, 0.10);
        self.turing_b = clamp_finite(self.turing_b, 0.1, 2.0, 0.90);
        self.turing_du = clamp_finite(self.turing_du, 0.1, 5.0, 1.0);
        self.turing_dv = clamp_finite(self.turing_dv, 5.0, 60.0, 18.0);
        self.turing_dt = clamp_finite(self.turing_dt, 0.001, 0.05, 0.008);
        super::implicit::clamp_implicit_config_in_place(self);
        super::ast_field::clamp_custom_ast_config_in_place(self);
        self.cut_offset = clamp_finite(self.cut_offset, -1.0, 1.0, 0.0);
        if !self.camera.is_valid() {
            self.camera.reset();
        }
    }
}

pub(super) fn clamp_finite(val: f64, lo: f64, hi: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val.clamp(lo, hi)
    } else {
        fallback
    }
}
