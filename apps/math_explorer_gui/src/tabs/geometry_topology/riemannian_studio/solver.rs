//! Multi-physics simulation engine over [`ManifoldGeometry`] coupling:
//! - `CotangentLaplacian`, `HeatEquationSolver`, and `laplace_beltrami`
//! - `MeanCurvatureFlow`
//! - `SchnakenbergKinetics` and `GrayScottKinetics` via `CotangentMeshDiffusion` (`biology`)
//! - `CorticalSheetRunner` (`HodgkinHuxleyNeuron` + FreeSurfer thickness, `biology`)
//! - FreeSurfer morphometry and group GLM overlays (`applied`).

use super::super::surface_brush::{apply_geodesic_kernel, GeodesicBrushStroke};
#[cfg(feature = "applied")]
use super::freesurfer_overlay::FreeSurferMorphometryState;
use super::mesh_gen::{recompute_vertex_normals, ManifoldGeometry};
#[cfg(feature = "biology")]
use super::solver_init::{
    build_cortical_sheet, build_gray_scott_system, build_schnakenberg_system,
};
use super::solver_init::{evaluate_analytical_chart_verification, seed_heat_field};
use super::state::{PdeRegime, RiemannianStudioConfig};
use math_explorer::pure_math::differential_geometry::mean_curvature_flow::MeanCurvatureFlow;

#[cfg(feature = "biology")]
use math_explorer::biology::diffusion::CotangentMeshDiffusion;
#[cfg(feature = "biology")]
use math_explorer::biology::morphogenesis::{
    GrayScottKinetics, SchnakenbergKinetics, TuringSystem,
};
#[cfg(feature = "biology")]
use math_explorer::biology::neuroscience::CorticalSheetRunner;

/// Live solver state and analytical verification telemetry for `RiemannianPdeStudio`.
pub struct RiemannianSimulationEngine {
    /// Active vertex scalar field rendered on the 3D manifold.
    pub scalar_field: Vec<f64>,
    /// Previous step's vertex scalar field (used for spike and wavefront flux audio).
    pub prev_field: Vec<f64>,
    scratch_field: Vec<f64>,
    /// Step counter since last reset.
    pub step_count: u64,
    /// Relative L2 discrepancy between `CotangentLaplacian` and analytical `laplace_beltrami` on $Y_1^0$.
    pub chart_beltrami_l2_err: f64,
    /// Analytical `laplace_beltrami` value evaluated at the cursor's `(u, v)` chart coordinate.
    pub cursor_beltrami_val: f64,
    /// Peak temperature from reference `HeatEquationSolver` chart step.
    pub chart_heat_solver_peak: f64,
    /// Transient impulse (`0.0..=1.0`) triggered by brush strokes or neural action potentials.
    pub transient_impulse: f32,
    #[cfg(feature = "biology")]
    hh_pulse_frames_remaining: u8,
    mcf_flow: MeanCurvatureFlow,
    #[cfg(feature = "applied")]
    pub freesurfer: FreeSurferMorphometryState,
    #[cfg(feature = "biology")]
    schnakenberg: TuringSystem<2, SchnakenbergKinetics, CotangentMeshDiffusion>,
    #[cfg(feature = "biology")]
    gray_scott: TuringSystem<2, GrayScottKinetics, CotangentMeshDiffusion>,
    #[cfg(feature = "biology")]
    pub cortical_sheet: CorticalSheetRunner,
}

impl RiemannianSimulationEngine {
    /// Initializes all enabled PDE solvers and morphometry overlays over `geom`.
    #[must_use]
    pub fn new(
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        cursor_vertex: usize,
    ) -> Self {
        let n = geom.positions.len();
        let mut scalar_field = vec![0.0; n];
        seed_heat_field(geom, &mut scalar_field);

        let mcf_flow = MeanCurvatureFlow::new(geom.discrete_surface.clone());
        let (chart_err, cursor_lb, heat_peak) =
            evaluate_analytical_chart_verification(geom, cfg, cursor_vertex);

        #[cfg(feature = "applied")]
        let freesurfer = FreeSurferMorphometryState::build(geom, cfg, cursor_vertex);

        #[cfg(feature = "biology")]
        let schnakenberg = build_schnakenberg_system(geom, cfg);
        #[cfg(feature = "biology")]
        let gray_scott = build_gray_scott_system(geom, cfg);
        #[cfg(feature = "biology")]
        let cortical_sheet = {
            #[cfg(feature = "applied")]
            let thick_slice: &[f64] = &freesurfer.thickness_mm;
            #[cfg(not(feature = "applied"))]
            let thick_slice: &[f64] = &[];
            build_cortical_sheet(geom, cfg, thick_slice)
        };

        let mut engine = Self {
            prev_field: scalar_field.clone(),
            scratch_field: vec![0.0; n],
            scalar_field,
            step_count: 0,
            chart_beltrami_l2_err: chart_err,
            cursor_beltrami_val: cursor_lb,
            chart_heat_solver_peak: heat_peak,
            transient_impulse: 0.0,
            #[cfg(feature = "biology")]
            hh_pulse_frames_remaining: 10,
            mcf_flow,
            #[cfg(feature = "applied")]
            freesurfer,
            #[cfg(feature = "biology")]
            schnakenberg,
            #[cfg(feature = "biology")]
            gray_scott,
            #[cfg(feature = "biology")]
            cortical_sheet,
        };
        engine.sync_active_field(geom, cfg);
        engine.prev_field.copy_from_slice(&engine.scalar_field);
        engine
    }

    /// Synchronizes `scalar_field` from whichever `PdeRegime` / `FreeSurferOverlayKind` is active.
    pub fn sync_active_field(&mut self, geom: &ManifoldGeometry, cfg: &RiemannianStudioConfig) {
        let regime = cfg.regime.coerce_to_enabled_features();
        match regime {
            PdeRegime::HeatBeltrami => {}
            PdeRegime::MeanCurvatureSmoothing => {
                self.scalar_field.copy_from_slice(&geom.mean_curvature);
            }
            PdeRegime::SchnakenbergTuring => {
                #[cfg(feature = "biology")]
                self.scalar_field.copy_from_slice(self.schnakenberg.u());
            }
            PdeRegime::GrayScottTuring => {
                #[cfg(feature = "biology")]
                self.scalar_field.copy_from_slice(self.gray_scott.v());
            }
            PdeRegime::CorticalHodgkinHuxley => {
                #[cfg(feature = "biology")]
                self.scalar_field
                    .copy_from_slice(self.cortical_sheet.potentials());
            }
            PdeRegime::FreeSurferMorphometryGlm => {
                #[cfg(feature = "applied")]
                self.freesurfer
                    .write_overlay(cfg.overlay, &mut self.scalar_field);
            }
        }
    }

    /// Refreshes cursor-level `laplace_beltrami` and FreeSurfer `cortical_thickness` probes.
    pub fn refresh_cursor_probes(
        &mut self,
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        cursor_vertex: usize,
    ) {
        let (_err, lb_val, _peak) =
            evaluate_analytical_chart_verification(geom, cfg, cursor_vertex);
        self.cursor_beltrami_val = lb_val;
        #[cfg(feature = "applied")]
        self.freesurfer.refresh_cursor_probe(cursor_vertex);
    }

    /// Advances the active regime forward by one simulation frame.
    pub fn step(
        &mut self,
        geom: &mut ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        cursor_vertex: usize,
    ) {
        self.prev_field.copy_from_slice(&self.scalar_field);
        self.transient_impulse *= 0.72;
        match cfg.regime.coerce_to_enabled_features() {
            PdeRegime::HeatBeltrami => {
                geom.laplacian.step_heat_euler(
                    &self.scalar_field,
                    &mut self.scratch_field,
                    cfg.heat_alpha,
                    cfg.dt.min(0.01),
                );
                std::mem::swap(&mut self.scalar_field, &mut self.scratch_field);
            }
            PdeRegime::MeanCurvatureSmoothing => self.step_mean_curvature_flow(geom, cfg),
            PdeRegime::SchnakenbergTuring
            | PdeRegime::GrayScottTuring
            | PdeRegime::CorticalHodgkinHuxley => {
                #[cfg(feature = "biology")]
                self.step_biology_regime(cfg);
            }
            PdeRegime::FreeSurferMorphometryGlm => {
                #[cfg(feature = "applied")]
                self.step_freesurfer_regime(geom, cfg, cursor_vertex);
                #[cfg(not(feature = "applied"))]
                let _ = cursor_vertex;
            }
        }
        self.step_count = self.step_count.wrapping_add(1);
    }

    #[cfg(feature = "biology")]
    fn step_biology_regime(&mut self, cfg: &RiemannianStudioConfig) {
        match cfg.regime {
            PdeRegime::SchnakenbergTuring => {
                self.schnakenberg.diffusion_coeffs = [cfg.d_u, cfg.d_v];
                self.schnakenberg.kinetics = SchnakenbergKinetics::new(cfg.schnak_a, cfg.schnak_b);
                for _ in 0..4 {
                    self.schnakenberg.step(cfg.dt.min(0.005));
                }
                self.scalar_field.copy_from_slice(self.schnakenberg.u());
            }
            PdeRegime::GrayScottTuring => {
                self.gray_scott.diffusion_coeffs = [cfg.d_u, cfg.d_v * 0.5];
                self.gray_scott.kinetics = GrayScottKinetics::new(cfg.gs_f, cfg.gs_k);
                for _ in 0..6 {
                    self.gray_scott.step(cfg.dt.min(0.02));
                }
                self.scalar_field.copy_from_slice(self.gray_scott.v());
            }
            PdeRegime::CorticalHodgkinHuxley => self.step_cortical_sheet(cfg),
            _ => {}
        }
    }

    #[cfg(feature = "applied")]
    fn step_freesurfer_regime(
        &mut self,
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        cursor_vertex: usize,
    ) {
        self.freesurfer
            .step_pial_evolution(geom, cfg, cursor_vertex);
        self.freesurfer
            .write_overlay(cfg.overlay, &mut self.scalar_field);
        #[cfg(feature = "biology")]
        self.cortical_sheet
            .update_thickness_map(&self.freesurfer.thickness_mm, 2.5);
    }

    fn step_mean_curvature_flow(
        &mut self,
        geom: &mut ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
    ) {
        self.mcf_flow.step(cfg.dt.min(0.005) * 0.25);
        for (v_idx, &(i, j)) in geom.vertex_grid_ij.iter().enumerate() {
            if let Some(pt) = self.mcf_flow.surface.points.get(i).and_then(|r| r.get(j)) {
                if pt.x.is_finite() && pt.y.is_finite() && pt.z.is_finite() {
                    geom.positions[v_idx] = [
                        pt.x.clamp(-10.0, 10.0),
                        pt.y.clamp(-10.0, 10.0),
                        pt.z.clamp(-10.0, 10.0),
                    ];
                }
            }
        }
        geom.discrete_surface = self.mcf_flow.surface.clone();
        recompute_vertex_normals(&geom.positions, &geom.triangles, &mut geom.normals);
        geom.laplacian =
            math_explorer::pure_math::differential_geometry::CotangentLaplacian::from_indexed(
                &geom.positions,
                &geom.triangles,
            );
        geom.laplacian
            .compute_mean_curvature(&geom.positions, &mut geom.mean_curvature);
        self.scalar_field.copy_from_slice(&geom.mean_curvature);
    }

    #[cfg(feature = "biology")]
    fn step_cortical_sheet(&mut self, cfg: &RiemannianStudioConfig) {
        self.cortical_sheet
            .set_sheet_conductance(cfg.hh_conductance);
        let frame_dt_ms = (cfg.dt * 50.0).clamp(0.05, 0.32);
        self.cortical_sheet.step(frame_dt_ms);
        if self.hh_pulse_frames_remaining > 0 {
            self.hh_pulse_frames_remaining -= 1;
            if self.hh_pulse_frames_remaining == 0 {
                self.cortical_sheet.clear_stimulus();
            }
        }
        self.scalar_field
            .copy_from_slice(self.cortical_sheet.potentials());
        let spiked = self
            .prev_field
            .iter()
            .zip(self.scalar_field.iter())
            .any(|(&prev_v, &cur_v)| prev_v <= 0.0 && cur_v > 0.0);
        if spiked {
            self.transient_impulse = 1.0;
        }
    }

    /// Injects a 3D geodesic brush stroke into the active manifold solver state.
    pub fn apply_brush_stroke(
        &mut self,
        geom: &mut ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        stroke: GeodesicBrushStroke,
    ) {
        self.transient_impulse = (self.transient_impulse + 0.65).min(1.0);
        match cfg.regime.coerce_to_enabled_features() {
            PdeRegime::HeatBeltrami => {
                apply_geodesic_kernel(&geom.laplacian, &geom.positions, stroke, |v, delta| {
                    self.scalar_field[v] = (self.scalar_field[v] + delta * 1.5).clamp(-5.0, 5.0);
                });
            }
            PdeRegime::MeanCurvatureSmoothing => self.sculpt_mcf_vertices(geom, stroke),
            PdeRegime::SchnakenbergTuring
            | PdeRegime::GrayScottTuring
            | PdeRegime::CorticalHodgkinHuxley => {
                #[cfg(feature = "biology")]
                self.apply_biology_brush(geom, cfg, stroke);
            }
            PdeRegime::FreeSurferMorphometryGlm => {
                #[cfg(feature = "applied")]
                self.apply_freesurfer_brush(geom, cfg, stroke);
            }
        }
    }

    fn sculpt_mcf_vertices(&mut self, geom: &mut ManifoldGeometry, stroke: GeodesicBrushStroke) {
        let mut hits = Vec::new();
        apply_geodesic_kernel(&geom.laplacian, &geom.positions, stroke, |v, delta| {
            hits.push((v, delta));
        });
        for (v, delta) in hits {
            let nrm = geom.normals[v];
            let bump = 0.12 * delta;
            for (pos_c, &n_c) in geom.positions[v].iter_mut().zip(nrm.iter()) {
                *pos_c += n_c * bump;
            }
            let (gi, gj) = geom.vertex_grid_ij[v];
            if let Some(slot) = self
                .mcf_flow
                .surface
                .points
                .get_mut(gi)
                .and_then(|r| r.get_mut(gj))
            {
                *slot = nalgebra::Point3::new(
                    geom.positions[v][0],
                    geom.positions[v][1],
                    geom.positions[v][2],
                );
            }
        }
        recompute_vertex_normals(&geom.positions, &geom.triangles, &mut geom.normals);
        geom.laplacian
            .compute_mean_curvature(&geom.positions, &mut geom.mean_curvature);
        self.scalar_field.copy_from_slice(&geom.mean_curvature);
    }

    #[cfg(feature = "biology")]
    fn apply_biology_brush(
        &mut self,
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        stroke: GeodesicBrushStroke,
    ) {
        match cfg.regime {
            PdeRegime::SchnakenbergTuring => {
                apply_geodesic_kernel(&geom.laplacian, &geom.positions, stroke, |v, delta| {
                    let u_slot = &mut self.schnakenberg.u_mut()[v];
                    *u_slot = (*u_slot + 0.8 * delta).clamp(0.0, 5.0);
                });
                self.scalar_field.copy_from_slice(self.schnakenberg.u());
            }
            PdeRegime::GrayScottTuring => {
                apply_geodesic_kernel(&geom.laplacian, &geom.positions, stroke, |v, delta| {
                    let v_slot = &mut self.gray_scott.v_mut()[v];
                    *v_slot = (*v_slot + 0.35 * delta).clamp(0.0, 1.0);
                    let u_slot = &mut self.gray_scott.u_mut()[v];
                    *u_slot = (*u_slot - 0.25 * delta.max(0.0)).clamp(0.0, 1.0);
                });
                self.scalar_field.copy_from_slice(self.gray_scott.v());
            }
            PdeRegime::CorticalHodgkinHuxley => {
                apply_geodesic_kernel(&geom.laplacian, &geom.positions, stroke, |v, delta| {
                    let cur_v = self.cortical_sheet.potentials()[v];
                    self.cortical_sheet
                        .depolarize_vertex(v, cur_v + delta * 24.0);
                    self.cortical_sheet.add_stimulus(v, delta * cfg.hh_stimulus);
                });
                self.hh_pulse_frames_remaining = 8;
                self.scalar_field
                    .copy_from_slice(self.cortical_sheet.potentials());
            }
            _ => {}
        }
    }

    #[cfg(feature = "applied")]
    fn apply_freesurfer_brush(
        &mut self,
        geom: &ManifoldGeometry,
        cfg: &RiemannianStudioConfig,
        stroke: GeodesicBrushStroke,
    ) {
        apply_geodesic_kernel(&geom.laplacian, &geom.positions, stroke, |v, delta| {
            if v < self.freesurfer.pial_surface.vertices.len() {
                let nrm = geom.normals[v];
                for (pial_c, &n_c) in self.freesurfer.pial_surface.vertices[v]
                    .iter_mut()
                    .zip(nrm.iter())
                {
                    *pial_c += n_c * delta * 0.45;
                }
            }
        });
        self.freesurfer
            .recompute_metrics(geom, cfg, stroke.center_vertex);
        self.freesurfer
            .write_overlay(cfg.overlay, &mut self.scalar_field);
    }
}

// [cite:graph_parameters_rust]
