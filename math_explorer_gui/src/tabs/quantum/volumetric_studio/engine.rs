#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Pre-allocated 3D volumetric PDE, quantum eigenstate, `MarchingCubes`, and `Gaussian3D` engine.
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]

#[path = "engine_mesh.rs"]
pub mod mesh_ops;

use super::export_3d::extract_implicit_splats_into;
use super::implicit::ImplicitRuntimeEvaluator;
use super::state::{
    FieldSourceMode, QuantumObservableChoice, VolumetricStudioConfig, MAX_VOLUMETRIC_GRID,
    MIN_VOLUMETRIC_GRID, STUDIO_CAMERA_DISTANCE, STUDIO_VIEWPORT_EXTENT,
};
use crate::framework::recovery::DiagnosticReport;
use crate::volumetric_scene::{
    extract_voxel_splats_into, presets::seed_3d_turing_gyroid, project_and_rasterize_workspace,
    VolumetricSplatWorkspace, MAX_DYNAMIC_SPLATS,
};
use math_explorer::applied::isosurface::{IsosurfaceWorkspace, Point3D, VoxelGrid};
use math_explorer::biology::diffusion::FiniteDifference3D;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use math_explorer::physics::quantum::{HydrogenicSuperposition3D, OrbitalQuantumNumbers};
use math_explorer::pure_math::special_functions::legendre::legendre_p_associated;
use mesh_ops::{
    extract_slice_contours_into, inject_turing_pulse, sample_voxel_nearest, ProjectedFacet,
    MAX_RENDERED_FACETS,
};
use oxidize_core::mesh::Mesh;

const MESH_VERTEX_CAP: usize = 131_072;
const MESH_INDEX_CAP: usize = 393_216;
const SLICE_CONTOUR_CAP: usize = 2_048;
const TURING_SEED: u64 = 0x517C_C1B7_2722_0A95;

/// Scalar & angular harmonic readout captured at the interactive 3D viewport probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeReadout {
    /// World-space coordinates `[x, y, z]` inside `[-2.5, +2.5]^3`.
    pub world_pos: [f64; 3],
    /// Sampled voxel value at `world_pos`.
    pub scalar_value: f32,
    /// Exact associated Legendre angular value $P_{\ell_1}^{m_1}(\cos\theta)$ at `world_pos`.
    pub legendre_polar: f64,
}

type QuantumStructKey = (u32, u32, i32, u32, u32, i32, usize, u64);
type TuringStructKey = (usize, u64, u64, u64, u64);

/// Pre-allocated workspace executing zero-allocation 3D stepping, `MarchingCubes`, and `Gaussian3D` projection.
pub struct VolumetricEngine {
    /// Reusable `VoxelGrid` pre-allocated to `MAX_VOLUMETRIC_GRID^3`.
    pub voxel_grid: VoxelGrid,
    /// Pre-allocated `(MAX_VOLUMETRIC_GRID + 2)^3` apron grid for watertight boundary capping.
    pub padded_grid: VoxelGrid,
    /// Two-state 3D hydrogenic superposition engine.
    pub quantum: HydrogenicSuperposition3D,
    /// Two-species 3D Schnakenberg reaction-diffusion grid over `FiniteDifference3D`.
    pub turing: TuringSystem<2, SchnakenbergKinetics, FiniteDifference3D>,
    /// Zero-allocation `CompiledExpr::eval_3d` implicit CAD evaluator.
    pub implicit_eval: ImplicitRuntimeEvaluator,
    /// Pre-allocated `MarchingCubes` rolling-slice workspace.
    pub iso_ws: IsosurfaceWorkspace,
    /// Positive-phase (`>= +iso_threshold`) manifold mesh.
    pub pos_mesh: Mesh,
    /// Negative-phase (`<= -iso_threshold`) manifold mesh.
    pub neg_mesh: Mesh,
    /// Pre-allocated `Gaussian3D` extraction, projection, and rasterization workspace.
    pub splat_ws: VolumetricSplatWorkspace,
    /// Depth-sorted projected 2D triangles ready for `egui_plot::Polygon`.
    pub projected_facets: Vec<ProjectedFacet>,
    /// Nodal contour segments (`[[p0, p1]; ...]`) on the active 3D cutting plane.
    pub slice_contours: Vec<[[f64; 3]; 2]>,
    /// Latest interactive probe / perturbation sample.
    pub probe: Option<ProbeReadout>,
    /// Cumulative integration steps executed since reset.
    pub step_count: usize,
    /// Numerical divergence flag monitored by `FiniteStateGuard`.
    pub diverged: bool,
    /// Structured divergence report when `diverged` is set.
    pub diagnostic_report: Option<DiagnosticReport>,
    cached_q_key: QuantumStructKey,
    cached_mix_phase: (u64, u64),
    cached_t_key: TuringStructKey,
}

impl VolumetricEngine {
    /// Allocates all 3D voxel, PDE, `MarchingCubes`, and `Gaussian3D` buffers up front.
    pub fn new(cfg: &VolumetricStudioConfig) -> Self {
        let dim = cfg
            .grid_resolution
            .clamp(MIN_VOLUMETRIC_GRID, MAX_VOLUMETRIC_GRID);
        let (voxel_grid, padded_grid) = build_studio_grids(dim, cfg.viewport_half_extents());
        let q1 = OrbitalQuantumNumbers::clamped(cfg.n1, cfg.l1, cfg.m1);
        let q2 = OrbitalQuantumNumbers::clamped(cfg.n2, cfg.l2, cfg.m2);
        let mut quantum = HydrogenicSuperposition3D::new(
            q1,
            q2,
            dim,
            MAX_VOLUMETRIC_GRID,
            cfg.physical_quantum_extent(),
            cfg.quantum_dt,
        );
        quantum.set_superposition(cfg.superposition_mix, cfg.relative_phase);
        let mut turing = build_turing_3d(dim, cfg);
        seed_3d_turing_gyroid(&mut turing, dim, cfg.turing_a, cfg.turing_b, TURING_SEED);
        let mut engine = Self {
            voxel_grid,
            padded_grid,
            quantum,
            turing,
            implicit_eval: ImplicitRuntimeEvaluator::new(cfg),
            iso_ws: IsosurfaceWorkspace::with_locked_capacity(
                MAX_VOLUMETRIC_GRID + 2,
                MAX_VOLUMETRIC_GRID + 2,
            ),
            pos_mesh: Mesh::with_capacity(MESH_VERTEX_CAP, MESH_INDEX_CAP),
            neg_mesh: Mesh::with_capacity(MESH_VERTEX_CAP, MESH_INDEX_CAP),
            splat_ws: VolumetricSplatWorkspace::with_capacity(MAX_DYNAMIC_SPLATS),
            projected_facets: Vec::with_capacity(MAX_RENDERED_FACETS),
            slice_contours: Vec::with_capacity(SLICE_CONTOUR_CAP),
            probe: None,
            step_count: 0,
            diverged: false,
            diagnostic_report: None,
            cached_q_key: quantum_key(cfg, dim),
            cached_mix_phase: (
                cfg.superposition_mix.to_bits(),
                cfg.relative_phase.to_bits(),
            ),
            cached_t_key: turing_key(cfg, dim),
        };
        engine.step_and_extract_zero_alloc(cfg, false);
        engine
    }

    /// Synchronizes basis grids or 3D Turing lattices when structural parameters change.
    pub fn sync_structural_config(&mut self, cfg: &VolumetricStudioConfig) {
        let dim = cfg
            .grid_resolution
            .clamp(MIN_VOLUMETRIC_GRID, MAX_VOLUMETRIC_GRID);
        let (step, orig) = grid_step_and_origin_3d(dim, cfg.viewport_half_extents());
        let _ = self
            .voxel_grid
            .reconfigure_in_place((dim, dim, dim), step, orig);
        self.implicit_eval.sync_zero_alloc(cfg);
        let qk = quantum_key(cfg, dim);
        if self.cached_q_key != qk {
            let q1 = OrbitalQuantumNumbers::clamped(cfg.n1, cfg.l1, cfg.m1);
            let q2 = OrbitalQuantumNumbers::clamped(cfg.n2, cfg.l2, cfg.m2);
            self.quantum
                .reconfigure_basis(q1, q2, dim, cfg.physical_quantum_extent(), 1.0);
            self.quantum.rebuild_propagator(cfg.quantum_dt, 0.0);
            self.quantum
                .set_superposition(cfg.superposition_mix, cfg.relative_phase);
            self.cached_q_key = qk;
            self.cached_mix_phase = (
                cfg.superposition_mix.to_bits(),
                cfg.relative_phase.to_bits(),
            );
        } else {
            let mp = (
                cfg.superposition_mix.to_bits(),
                cfg.relative_phase.to_bits(),
            );
            if self.cached_mix_phase != mp {
                self.quantum
                    .set_superposition(cfg.superposition_mix, cfg.relative_phase);
                self.cached_mix_phase = mp;
            }
        }
        let tk = turing_key(cfg, dim);
        if self.cached_t_key != tk {
            self.turing = build_turing_3d(dim, cfg);
            seed_3d_turing_gyroid(
                &mut self.turing,
                dim,
                cfg.turing_a,
                cfg.turing_b,
                TURING_SEED,
            );
            self.cached_t_key = tk;
        }
    }

    /// Resets quantum phase or re-seeds the 3D Turing gyroid field.
    pub fn reset_simulation(&mut self, cfg: &VolumetricStudioConfig) {
        self.diverged = false;
        self.diagnostic_report = None;
        self.step_count = 0;
        self.probe = None;
        let dim = cfg
            .grid_resolution
            .clamp(MIN_VOLUMETRIC_GRID, MAX_VOLUMETRIC_GRID);
        self.sync_structural_config(cfg);
        match cfg.source {
            FieldSourceMode::QuantumSuperposition => {
                self.quantum
                    .set_superposition(cfg.superposition_mix, cfg.relative_phase);
            }
            FieldSourceMode::TuringMorphogenesis => {
                self.turing = build_turing_3d(dim, cfg);
                seed_3d_turing_gyroid(
                    &mut self.turing,
                    dim,
                    cfg.turing_a,
                    cfg.turing_b,
                    TURING_SEED,
                );
                self.cached_t_key = turing_key(cfg, dim);
            }
            FieldSourceMode::CustomImplicit => {
                self.implicit_eval.sync_zero_alloc(cfg);
            }
        }
        self.step_and_extract_zero_alloc(cfg, false);
    }

    /// Executes one hot-path PDE/quantum step + `MarchingCubes` + `Gaussian3D` projection with zero heap allocations.
    pub fn step_and_extract_zero_alloc(
        &mut self,
        cfg: &VolumetricStudioConfig,
        advance_time: bool,
    ) {
        if self.diverged {
            return;
        }
        if advance_time {
            self.advance_source_in_place(cfg);
            if self.diverged {
                return;
            }
        }
        self.populate_voxel_grid_in_place(cfg);
        if !self.voxel_grid.data().iter().all(|v| v.is_finite()) {
            self.diverged = true;
            self.diagnostic_report = Some(DiagnosticReport::new(
                self.step_count as f64,
                self.step_count,
                "voxel_grid",
                "Non-finite scalar field detected in 3D VolumetricEngine.",
            ));
            return;
        }
        let signed = is_signed_field(cfg);
        let cut = cfg.active_cut_filter();
        let iso = (cfg.iso_threshold as f32).clamp(0.001, 2.0);
        self.extract_isosurfaces_zero_alloc(cfg, iso, signed, cut);
        if cfg.render_mode.uses_splats() {
            if cfg.source == FieldSourceMode::CustomImplicit {
                extract_implicit_splats_into(&self.voxel_grid, cut, &mut self.splat_ws);
            } else {
                extract_voxel_splats_into(
                    &self.voxel_grid,
                    iso * 0.45,
                    signed,
                    cut,
                    &mut self.splat_ws,
                );
            }
            self.splat_ws.recompute_density_counts(cfg.splat_scale);
            project_and_rasterize_workspace(
                &mut self.splat_ws,
                &cfg.camera,
                STUDIO_CAMERA_DISTANCE,
                STUDIO_VIEWPORT_EXTENT,
                cfg.splat_scale,
                64,
            );
        }
        let contour_level = if cfg.source == FieldSourceMode::CustomImplicit {
            0.0
        } else {
            iso
        };
        extract_slice_contours_into(
            &self.voxel_grid,
            cut,
            contour_level,
            signed,
            &mut self.slice_contours,
        );
    }

    fn advance_source_in_place(&mut self, cfg: &VolumetricStudioConfig) {
        self.step_count += 1;
        match cfg.source {
            FieldSourceMode::QuantumSuperposition => {
                self.quantum.step_in_place();
                let ok = self
                    .quantum
                    .state
                    .vector
                    .iter()
                    .all(|c| c.re.is_finite() && c.im.is_finite());
                if !ok {
                    self.diverged = true;
                    self.diagnostic_report = Some(DiagnosticReport::new(
                        self.quantum.elapsed_time(),
                        self.step_count,
                        "psi_3d",
                        "Non-finite amplitude in 3D hydrogenic superposition.",
                    ));
                }
            }
            FieldSourceMode::TuringMorphogenesis => {
                self.turing.step(cfg.turing_dt);
                let ok = self.turing.u().iter().all(|v| v.is_finite())
                    && self.turing.v().iter().all(|v| v.is_finite());
                if !ok {
                    self.diverged = true;
                    self.diagnostic_report = Some(DiagnosticReport::new(
                        (self.step_count as f64) * cfg.turing_dt,
                        self.step_count,
                        "turing_3d",
                        "Non-finite concentration in 3D Schnakenberg reaction-diffusion.",
                    ));
                }
            }
            FieldSourceMode::CustomImplicit => {}
        }
    }

    fn populate_voxel_grid_in_place(&mut self, cfg: &VolumetricStudioConfig) {
        match cfg.source {
            FieldSourceMode::QuantumSuperposition => {
                self.quantum
                    .write_field_into(cfg.observable.to_physics(), self.voxel_grid.data_mut());
            }
            FieldSourceMode::TuringMorphogenesis => {
                let u_ss = (cfg.turing_a + cfg.turing_b) as f32;
                let buf = self.voxel_grid.data_mut();
                self.turing.write_activator_f32_into(buf);
                for slot in buf.iter_mut() {
                    *slot -= u_ss;
                }
            }
            FieldSourceMode::CustomImplicit => {
                self.implicit_eval
                    .populate_grid_in_place(cfg, &mut self.voxel_grid);
            }
        }
    }

    /// Samples `world_pt`, evaluates `legendre_p_associated`, and optionally injects a 3D perturbation pulse.
    pub fn apply_probe_or_pulse(
        &mut self,
        cfg: &mut VolumetricStudioConfig,
        world_pt: [f64; 3],
    ) -> ProbeReadout {
        let ext = cfg.viewport_half_extents();
        let clamped = [
            world_pt[0].clamp(-ext[0], ext[0]),
            world_pt[1].clamp(-ext[1], ext[1]),
            world_pt[2].clamp(-ext[2], ext[2]),
        ];
        if cfg.probe_brush_enabled {
            match cfg.source {
                FieldSourceMode::TuringMorphogenesis => {
                    inject_turing_pulse(&mut self.turing, self.voxel_grid.width(), clamped);
                }
                FieldSourceMode::QuantumSuperposition => {
                    let wrapped = ((cfg.relative_phase + 0.45 + std::f64::consts::PI)
                        .rem_euclid(std::f64::consts::TAU))
                        - std::f64::consts::PI;
                    cfg.relative_phase = wrapped;
                    self.quantum
                        .set_superposition(cfg.superposition_mix, cfg.relative_phase);
                    self.cached_mix_phase = (
                        cfg.superposition_mix.to_bits(),
                        cfg.relative_phase.to_bits(),
                    );
                }
                FieldSourceMode::CustomImplicit => {}
            }
            self.step_and_extract_zero_alloc(cfg, false);
        }
        let scalar_value = sample_voxel_nearest(&self.voxel_grid, clamped);
        let r =
            (clamped[0] * clamped[0] + clamped[1] * clamped[1] + clamped[2] * clamped[2]).sqrt();
        let cos_theta = if r > 1e-9 {
            (clamped[2] / r).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        let q1 = OrbitalQuantumNumbers::clamped(cfg.n1, cfg.l1, cfg.m1);
        let legendre_polar = legendre_p_associated(q1.l as u64, q1.m as i64, cos_theta);
        let readout = ProbeReadout {
            world_pos: clamped,
            scalar_value,
            legendre_polar,
        };
        self.probe = Some(readout);
        readout
    }
}

pub(super) fn is_signed_field(cfg: &VolumetricStudioConfig) -> bool {
    matches!(
        (cfg.source, cfg.observable),
        (
            FieldSourceMode::QuantumSuperposition,
            QuantumObservableChoice::SignedWavefunction
        )
    ) || matches!(cfg.source, FieldSourceMode::TuringMorphogenesis)
}

fn build_studio_grids(dim: usize, half_extents: [f64; 3]) -> (VoxelGrid, VoxelGrid) {
    let mut backing = Vec::with_capacity(MAX_VOLUMETRIC_GRID.pow(3));
    backing.resize(dim * dim * dim, 0.0_f32);
    let (step, orig) = grid_step_and_origin_3d(dim, half_extents);
    let voxel_grid = VoxelGrid::builder()
        .dimensions(dim, dim, dim)
        .data(backing)
        .voxel_size(step)
        .origin(orig)
        .build()
        .expect("valid studio voxel grid");
    let pad_dim = dim + 2;
    let mut pad_backing = Vec::with_capacity((MAX_VOLUMETRIC_GRID + 2).pow(3));
    pad_backing.resize(pad_dim * pad_dim * pad_dim, 1.0_f32);
    let padded_grid = VoxelGrid::builder()
        .dimensions(pad_dim, pad_dim, pad_dim)
        .data(pad_backing)
        .voxel_size(step)
        .origin(orig - step)
        .build()
        .expect("valid studio padded apron grid");
    (voxel_grid, padded_grid)
}

fn grid_step_and_origin_3d(dim: usize, half_extents: [f64; 3]) -> (Point3D, Point3D) {
    let d = (dim.max(2) - 1) as f64;
    let step = Point3D::new(
        (2.0 * half_extents[0] / d) as f32,
        (2.0 * half_extents[1] / d) as f32,
        (2.0 * half_extents[2] / d) as f32,
    );
    let orig = Point3D::new(
        -(half_extents[0] as f32),
        -(half_extents[1] as f32),
        -(half_extents[2] as f32),
    );
    (step, orig)
}

fn quantum_key(cfg: &VolumetricStudioConfig, dim: usize) -> QuantumStructKey {
    (
        cfg.n1,
        cfg.l1,
        cfg.m1,
        cfg.n2,
        cfg.l2,
        cfg.m2,
        dim,
        cfg.quantum_dt.to_bits(),
    )
}

fn turing_key(cfg: &VolumetricStudioConfig, dim: usize) -> TuringStructKey {
    (
        dim,
        cfg.turing_a.to_bits(),
        cfg.turing_b.to_bits(),
        cfg.turing_du.to_bits(),
        cfg.turing_dv.to_bits(),
    )
}

fn build_turing_3d(
    dim: usize,
    cfg: &VolumetricStudioConfig,
) -> TuringSystem<2, SchnakenbergKinetics, FiniteDifference3D> {
    TuringSystem::new_3d(
        (Dimension(dim), Dimension(dim), Dimension(dim)),
        DiffusionCoeff(cfg.turing_du),
        DiffusionCoeff(cfg.turing_dv),
        StepSize(1.0),
        SchnakenbergKinetics::new(cfg.turing_a, cfg.turing_b),
    )
}
