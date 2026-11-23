#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
//! Pre-allocated 3D volumetric PDE, quantum eigenstate, `MarchingCubes`, and `Gaussian3D` engine.
// [cite:quantum_mechanics]
// [cite:isosurface_extraction]
// [cite:gaussian_splatting]

#[path = "engine_mesh.rs"]
pub mod mesh_ops;

use super::formula::{
    build_implicit_formula_bar, default_compiled_gyroid, extract_implicit_splats_into,
    populate_implicit_voxel_grid,
};
use super::state::{
    FieldSourceMode, QuantumObservableChoice, VolumetricStudioConfig, MAX_VOLUMETRIC_GRID,
    MIN_VOLUMETRIC_GRID, STUDIO_CAMERA_DISTANCE, STUDIO_VIEWPORT_EXTENT,
};
use crate::framework::math_input::{MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS};
use crate::framework::recovery::DiagnosticReport;
use crate::volumetric_scene::{
    extract_voxel_splats_into, presets::seed_3d_turing_gyroid, project_and_rasterize_workspace,
    VolumetricSplatWorkspace, MAX_DYNAMIC_SPLATS,
};
use math_commons::ast::CompiledExpr;
use math_explorer::applied::isosurface::{IsosurfaceWorkspace, Point3D, VoxelGrid};
use math_explorer::biology::diffusion::FiniteDifference3D;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use math_explorer::physics::quantum::{HydrogenicSuperposition3D, OrbitalQuantumNumbers};
use mesh_ops::{extract_slice_contours_into, ProjectedFacet, MAX_RENDERED_FACETS};
use oxidize_core::mesh::Mesh;

const MESH_VERTEX_CAP: usize = 98_304;
const MESH_INDEX_CAP: usize = 294_912;
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
    /// Two-state 3D hydrogenic superposition engine.
    pub quantum: HydrogenicSuperposition3D,
    /// Two-species 3D Schnakenberg reaction-diffusion grid over `FiniteDifference3D`.
    pub turing: TuringSystem<2, SchnakenbergKinetics, FiniteDifference3D>,
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
    compiled_formula: CompiledExpr,
    formula_params_buf: [f64; MAX_SYNTHESIZED_PARAMS],
    formula_param_len: usize,
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
        let voxel_grid = build_studio_voxel_grid(dim);
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
        let mut init_bar = build_implicit_formula_bar(&cfg.implicit_expr);
        init_bar.set_params_map(&cfg.formula_params);
        let mut engine = Self {
            voxel_grid,
            quantum,
            turing,
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
            compiled_formula: default_compiled_gyroid(),
            formula_params_buf: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            formula_param_len: 1,
            cached_q_key: quantum_key(cfg, dim),
            cached_mix_phase: (
                cfg.superposition_mix.to_bits(),
                cfg.relative_phase.to_bits(),
            ),
            cached_t_key: turing_key(cfg, dim),
        };
        engine.sync_formula_from_bar(&init_bar);
        engine.step_and_extract_zero_alloc(cfg, false);
        engine
    }

    /// Cold-path synchronization of the compiled 3D formula bytecode and active parameter slice.
    pub fn sync_formula_from_bar(&mut self, bar: &MathExpressionInputBar) {
        if let Some(compiled) = bar.active_compiled() {
            self.compiled_formula = compiled.clone();
        }
        let slice = bar.param_slice();
        let len = slice.len().min(MAX_SYNTHESIZED_PARAMS);
        self.formula_params_buf = [0.0; MAX_SYNTHESIZED_PARAMS];
        self.formula_params_buf[..len].copy_from_slice(&slice[..len]);
        self.formula_param_len = len;
    }

    /// Synchronizes basis grids or 3D Turing lattices when structural parameters change.
    pub fn sync_structural_config(&mut self, cfg: &VolumetricStudioConfig) {
        let dim = cfg
            .grid_resolution
            .clamp(MIN_VOLUMETRIC_GRID, MAX_VOLUMETRIC_GRID);
        let (step, orig) = grid_step_and_origin(dim);
        let _ = self.voxel_grid.reconfigure_in_place(
            (dim, dim, dim),
            Point3D::new(step, step, step),
            Point3D::new(orig, orig, orig),
        );
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
            FieldSourceMode::CustomImplicitFormula => {}
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
            if cfg.source == FieldSourceMode::CustomImplicitFormula {
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
        let contour_iso = if cfg.source == FieldSourceMode::CustomImplicitFormula {
            0.0
        } else {
            iso
        };
        extract_slice_contours_into(
            &self.voxel_grid,
            cut,
            contour_iso,
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
            FieldSourceMode::CustomImplicitFormula => {}
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
            FieldSourceMode::CustomImplicitFormula => {
                let params = &self.formula_params_buf[..self.formula_param_len];
                populate_implicit_voxel_grid(
                    &mut self.voxel_grid,
                    cfg,
                    &self.compiled_formula,
                    params,
                );
            }
        }
    }
}

fn build_studio_voxel_grid(dim: usize) -> VoxelGrid {
    let mut backing = Vec::with_capacity(MAX_VOLUMETRIC_GRID.pow(3));
    backing.resize(dim * dim * dim, 0.0_f32);
    let (step, orig) = grid_step_and_origin(dim);
    VoxelGrid::builder()
        .dimensions(dim, dim, dim)
        .data(backing)
        .voxel_size(Point3D::new(step, step, step))
        .origin(Point3D::new(orig, orig, orig))
        .build()
        .expect("valid studio voxel grid")
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

fn grid_step_and_origin(dim: usize) -> (f32, f32) {
    let step = ((2.0 * STUDIO_VIEWPORT_EXTENT) / ((dim.max(2) - 1) as f64)) as f32;
    (step, -(STUDIO_VIEWPORT_EXTENT as f32))
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
