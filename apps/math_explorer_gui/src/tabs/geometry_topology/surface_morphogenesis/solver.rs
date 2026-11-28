// [cite:modular_polynomials_review]
//! Coupled Laplace-Beltrami Turing solver, 3D pulse brush, and `FiniteStateGuard` rollback engine.

use super::{
    MorphogenesisKineticsKind, PulseBrushAction, SurfaceMorphogenesisTool, SurfaceOverlayMode,
};
use crate::tabs::geometry_topology::custom_surface::SurfaceType;
use crate::tabs::geometry_topology::export_utils::{
    has_direct_periodic_seams, has_klein_seam_topology, has_polar_singularities,
    is_standard_tau_domain,
};
use math_commons::math_kernel::types::{flatten_2d_index, Dimension};
use math_explorer::biology::diffusion::LaplaceBeltramiDiffusion2D;
use math_explorer::biology::morphogenesis::reaction::{GrayScottKinetics, SchnakenbergKinetics};
use math_explorer::biology::morphogenesis::{DiffusionCoeff, TuringSystem};
use math_explorer::pure_math::differential_geometry::buckled_flow::{
    BuckledSurfaceFlow, SurfaceTopology,
};
use math_explorer::pure_math::differential_geometry::surface::ParametricSurface;
use nalgebra::Point3;
use std::f64::consts::{PI, TAU};

pub(super) enum KineticsRunner {
    Schnakenberg(TuringSystem<2, SchnakenbergKinetics, LaplaceBeltramiDiffusion2D>),
    GrayScott(TuringSystem<2, GrayScottKinetics, LaplaceBeltramiDiffusion2D>),
}

impl KineticsRunner {
    pub(super) fn u(&self) -> &[f64] {
        match self {
            Self::Schnakenberg(sys) => sys.u(),
            Self::GrayScott(sys) => sys.u(),
        }
    }

    pub(super) fn v(&self) -> &[f64] {
        match self {
            Self::Schnakenberg(sys) => sys.v(),
            Self::GrayScott(sys) => sys.v(),
        }
    }

    pub(super) fn u_mut(&mut self) -> &mut [f64] {
        match self {
            Self::Schnakenberg(sys) => sys.u_mut(),
            Self::GrayScott(sys) => sys.u_mut(),
        }
    }

    pub(super) fn v_mut(&mut self) -> &mut [f64] {
        match self {
            Self::Schnakenberg(sys) => sys.v_mut(),
            Self::GrayScott(sys) => sys.v_mut(),
        }
    }

    pub(super) fn diffusion(&self) -> &LaplaceBeltramiDiffusion2D {
        match self {
            Self::Schnakenberg(sys) => &sys.diffusion,
            Self::GrayScott(sys) => &sys.diffusion,
        }
    }

    fn sync_params(&mut self, tool: &SurfaceMorphogenesisTool) {
        match self {
            Self::Schnakenberg(sys) => {
                sys.kinetics = SchnakenbergKinetics::new(tool.schnak_a, tool.schnak_b);
                sys.diffusion_coeffs = [tool.d_u, tool.d_v];
            }
            Self::GrayScott(sys) => {
                sys.kinetics = GrayScottKinetics::new(tool.gs_feed, tool.gs_kill);
                sys.diffusion_coeffs = [tool.d_u, tool.d_v];
            }
        }
    }

    fn step(&mut self, dt: f64) {
        match self {
            Self::Schnakenberg(sys) => sys.step(dt),
            Self::GrayScott(sys) => sys.step(dt),
        }
    }
}

pub(super) struct MorphogenesisSurfaceEngine {
    pub geometry_key: u64,
    pub kinetics_kind: MorphogenesisKineticsKind,
    pub topology: SurfaceTopology,
    pub range_u: (f64, f64),
    pub range_v: (f64, f64),
    pub nu: usize,
    pub nv: usize,
    pub runner: KineticsRunner,
    pub flow: BuckledSurfaceFlow,
    pub step_count: u64,
    pub effective_dt: f64,
    pub rollback_count: usize,
    last_valid_u: Vec<f64>,
    last_valid_v: Vec<f64>,
    last_valid_points: Vec<Point3<f64>>,
    base_mean_curvature: Vec<f64>,
}

pub(super) fn resolve_topology_and_domain(
    tool: &SurfaceMorphogenesisTool,
    surface: &dyn ParametricSurface,
) -> (SurfaceTopology, (f64, f64), (f64, f64)) {
    match tool.surface_type {
        SurfaceType::Sphere => (SurfaceTopology::PolarSphere, (0.0, TAU), (0.0, PI)),
        SurfaceType::Torus => (SurfaceTopology::PeriodicTorus, (0.0, TAU), (0.0, TAU)),
        SurfaceType::KleinBottle => (SurfaceTopology::TwistedKlein, (0.0, TAU), (0.0, TAU)),
        SurfaceType::CustomHeightfield => {
            let (ru, rv) = tool.custom_sandbox.uv_domain_for(tool.surface_type);
            (SurfaceTopology::OpenNeumann, ru, rv)
        }
        SurfaceType::CustomParametric => {
            let (ru, rv) = tool.custom_sandbox.uv_domain_for(tool.surface_type);
            if is_standard_tau_domain(ru, rv) {
                if has_polar_singularities(surface) {
                    return (SurfaceTopology::PolarSphere, (0.0, TAU), (0.0, PI));
                }
                if has_direct_periodic_seams(surface) {
                    return (SurfaceTopology::PeriodicTorus, ru, rv);
                }
                if has_klein_seam_topology(surface) {
                    return (SurfaceTopology::TwistedKlein, ru, rv);
                }
            }
            (SurfaceTopology::OpenNeumann, ru, rv)
        }
    }
}

fn build_kinetics_runner(
    tool: &SurfaceMorphogenesisTool,
    len: usize,
    diff: LaplaceBeltramiDiffusion2D,
) -> KineticsRunner {
    match tool.kinetics_kind {
        MorphogenesisKineticsKind::Schnakenberg => {
            KineticsRunner::Schnakenberg(TuringSystem::new_with_kinetics(
                Dimension(len),
                DiffusionCoeff(tool.d_u),
                DiffusionCoeff(tool.d_v),
                SchnakenbergKinetics::new(tool.schnak_a, tool.schnak_b),
                diff,
            ))
        }
        MorphogenesisKineticsKind::GrayScott => {
            KineticsRunner::GrayScott(TuringSystem::new_with_kinetics(
                Dimension(len),
                DiffusionCoeff(tool.d_u),
                DiffusionCoeff(tool.d_v),
                GrayScottKinetics::new(tool.gs_feed, tool.gs_kill),
                diff,
            ))
        }
    }
}

impl MorphogenesisSurfaceEngine {
    pub(super) fn build(tool: &SurfaceMorphogenesisTool) -> Self {
        let surface = tool.create_base_surface();
        let (topology, range_u, range_v) = resolve_topology_and_domain(tool, surface.as_ref());
        let nu = tool.clamped_u_res();
        let nv = if topology == SurfaceTopology::TwistedKlein
            && !tool.clamped_v_res().is_multiple_of(2)
        {
            (tool.clamped_v_res() + 1).min(super::MAX_SURFACE_MORPHO_RES)
        } else {
            tool.clamped_v_res()
        };

        let diff = LaplaceBeltramiDiffusion2D::from_surface(
            surface.as_ref(),
            topology,
            range_u,
            range_v,
            nu,
            nv,
        );
        let flow = BuckledSurfaceFlow::from_parametric(
            surface.as_ref(),
            topology,
            range_u,
            range_v,
            nu,
            nv,
        );
        let len = nu * nv;
        let mut runner = build_kinetics_runner(tool, len, diff);

        seed_initial_pattern(&mut runner, flow.reference_points(), tool);
        let last_valid_u = runner.u().to_vec();
        let last_valid_v = runner.v().to_vec();
        let last_valid_points = flow.points().to_vec();
        let max_d = tool.d_u.abs().max(tool.d_v.abs());
        let effective_dt = runner.diffusion().clamp_dt(tool.dt, max_d);

        Self {
            geometry_key: tool.geometry_fingerprint(),
            kinetics_kind: tool.kinetics_kind,
            topology,
            range_u,
            range_v,
            nu,
            nv,
            runner,
            flow,
            step_count: 0,
            effective_dt,
            rollback_count: 0,
            last_valid_u,
            last_valid_v,
            last_valid_points,
            base_mean_curvature: vec![0.0; len],
        }
    }

    pub(super) fn active_morphogen(&self) -> &[f64] {
        match self.kinetics_kind {
            MorphogenesisKineticsKind::Schnakenberg => self.runner.u(),
            MorphogenesisKineticsKind::GrayScott => self.runner.v(),
        }
    }

    pub(super) fn morphogen_baseline(&self, tool: &SurfaceMorphogenesisTool) -> f64 {
        match self.kinetics_kind {
            MorphogenesisKineticsKind::Schnakenberg => (tool.schnak_a + tool.schnak_b).max(0.05),
            MorphogenesisKineticsKind::GrayScott => 0.12,
        }
    }

    pub(super) fn overlay_field(&self, mode: SurfaceOverlayMode) -> &[f64] {
        match mode {
            SurfaceOverlayMode::MorphogenConcentration => self.active_morphogen(),
            SurfaceOverlayMode::GaussianCurvature => self.runner.diffusion().gaussian_curvature(),
            SurfaceOverlayMode::MetricAnisotropy => self.runner.diffusion().metric_anisotropy(),
        }
    }

    pub(super) fn step_coupled(&mut self, tool: &SurfaceMorphogenesisTool) -> bool {
        if !tool.has_finite_parameters_internal() {
            self.rollback_to_last_valid();
            return false;
        }
        self.runner.sync_params(tool);
        let max_d = tool.d_u.abs().max(tool.d_v.abs());
        let diff_dt = self.runner.diffusion().clamp_dt(tool.dt, max_d);
        let flow_dt = self.flow.max_stable_dt(tool.curvature_tension);
        let eff_dt = diff_dt.min(flow_dt);
        self.effective_dt = eff_dt;

        self.runner.step(eff_dt);
        if !self.runner.u().iter().all(|v| v.is_finite())
            || !self.runner.v().iter().all(|v| v.is_finite())
        {
            self.rollback_to_last_valid();
            return false;
        }

        let baseline = self.morphogen_baseline(tool);
        let morph = match self.kinetics_kind {
            MorphogenesisKineticsKind::Schnakenberg => self.runner.u(),
            MorphogenesisKineticsKind::GrayScott => self.runner.v(),
        };
        let buckled_ok = self.flow.step_buckled(
            morph,
            baseline,
            tool.buckling_gain,
            tool.curvature_tension,
            tool.restoring_tether,
            eff_dt,
        );
        if !buckled_ok {
            self.rollback_to_last_valid();
            return false;
        }

        if self.step_count == 0 {
            self.base_mean_curvature
                .copy_from_slice(self.flow.mean_curvatures());
        }
        self.last_valid_u.copy_from_slice(self.runner.u());
        self.last_valid_v.copy_from_slice(self.runner.v());
        self.last_valid_points.copy_from_slice(self.flow.points());
        self.step_count += 1;
        true
    }

    pub(super) fn rollback_to_last_valid(&mut self) {
        self.runner.u_mut().copy_from_slice(&self.last_valid_u);
        self.runner.v_mut().copy_from_slice(&self.last_valid_v);
        let _ = self.flow.restore_points(&self.last_valid_points);
        self.rollback_count += 1;
    }

    pub(super) fn reseed(&mut self, tool: &SurfaceMorphogenesisTool, phase_offset: f64) {
        self.flow.reset_to_reference();
        seed_pattern_with_phase(
            &mut self.runner,
            self.flow.reference_points(),
            tool,
            phase_offset,
        );
        if self.topology == SurfaceTopology::PolarSphere {
            sync_polar_caps(&mut self.runner, self.nu, self.nv);
        }
        self.last_valid_u.copy_from_slice(self.runner.u());
        self.last_valid_v.copy_from_slice(self.runner.v());
        self.last_valid_points.copy_from_slice(self.flow.points());
        self.step_count = 0;
    }

    pub(super) fn apply_3d_pulse(
        &mut self,
        center: Point3<f64>,
        radius: f64,
        strength: f64,
        action: PulseBrushAction,
        tool: &SurfaceMorphogenesisTool,
    ) {
        let r_sq = radius.max(0.05).powi(2);
        let pts = self.flow.points();
        let (u0, v0) = steady_baseline(tool);
        for (idx, pt) in pts.iter().enumerate() {
            let d2 = (pt - center).norm_squared();
            if d2 > r_sq * 2.25 {
                continue;
            }
            let weight = (-d2 / r_sq).exp();
            match action {
                PulseBrushAction::ActivatorPulse => {
                    self.runner.u_mut()[idx] =
                        (self.runner.u()[idx] + strength * weight).clamp(0.0, 6.0);
                }
                PulseBrushAction::InhibitorPulse => {
                    self.runner.v_mut()[idx] =
                        (self.runner.v()[idx] + strength * weight).clamp(0.0, 6.0);
                }
                PulseBrushAction::ResetPatch => {
                    let keep = 1.0 - weight;
                    self.runner.u_mut()[idx] = keep * self.runner.u()[idx] + weight * u0;
                    self.runner.v_mut()[idx] = keep * self.runner.v()[idx] + weight * v0;
                }
            }
        }
        if self.topology == SurfaceTopology::PolarSphere {
            sync_polar_caps(&mut self.runner, self.nu, self.nv);
        }
        self.last_valid_u.copy_from_slice(self.runner.u());
        self.last_valid_v.copy_from_slice(self.runner.v());
    }

    pub(super) fn sonification_metrics(&self) -> (f64, f64, f64, f64) {
        let morph = self.active_morphogen();
        let n = morph.len().max(1) as f64;
        let mean = morph.iter().sum::<f64>() / n;
        let var = morph
            .iter()
            .map(|&c| {
                let d = c - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let curv_shift = self
            .flow
            .mean_curvatures()
            .iter()
            .zip(&self.base_mean_curvature)
            .map(|(&h, &h0)| (h - h0).abs())
            .sum::<f64>()
            / n;
        let disp_rms = (self
            .flow
            .displacements()
            .iter()
            .map(|&d| d * d)
            .sum::<f64>()
            / n)
            .sqrt();
        (var, mean, curv_shift, disp_rms)
    }
}

fn steady_baseline(tool: &SurfaceMorphogenesisTool) -> (f64, f64) {
    match tool.kinetics_kind {
        MorphogenesisKineticsKind::Schnakenberg => {
            let u0 = (tool.schnak_a + tool.schnak_b).max(0.1);
            let v0 = tool.schnak_b / (u0 * u0).max(1e-4);
            (u0, v0)
        }
        MorphogenesisKineticsKind::GrayScott => (1.0, 0.0),
    }
}

fn seed_initial_pattern(
    runner: &mut KineticsRunner,
    pts: &[Point3<f64>],
    tool: &SurfaceMorphogenesisTool,
) {
    seed_pattern_with_phase(runner, pts, tool, 0.0);
}

fn seed_pattern_with_phase(
    runner: &mut KineticsRunner,
    pts: &[Point3<f64>],
    tool: &SurfaceMorphogenesisTool,
    phase: f64,
) {
    let (u0, v0) = steady_baseline(tool);
    for (idx, p) in pts.iter().enumerate() {
        // Intrinsic 3D coordinate seeding guarantees exact continuity across all seams and poles
        let harmonic = (2.3 * p.x + phase).sin() * (2.1 * p.y - 0.7 * phase).cos()
            + 0.65 * (2.7 * p.z + 1.3 * phase).sin();
        match tool.kinetics_kind {
            MorphogenesisKineticsKind::Schnakenberg => {
                runner.u_mut()[idx] = (u0 + 0.14 * harmonic).max(0.02);
                runner.v_mut()[idx] = (v0 - 0.09 * harmonic).max(0.02);
            }
            MorphogenesisKineticsKind::GrayScott => {
                let spots = ((2.2 * (p.x + phase.cos())).sin() * (2.2 * (p.y + phase.sin())).cos()
                    + 0.5 * (2.4 * p.z).cos())
                .max(0.0);
                if spots > 0.55 {
                    runner.u_mut()[idx] = 0.50;
                    runner.v_mut()[idx] = 0.28 + 0.08 * spots;
                } else {
                    runner.u_mut()[idx] = u0;
                    runner.v_mut()[idx] = v0;
                }
            }
        }
    }
}

fn sync_polar_caps(runner: &mut KineticsRunner, nu: usize, nv: usize) {
    if nu == 0 || nv < 2 {
        return;
    }
    for pole_j in [0_usize, nv - 1] {
        let u_mean = (0..nu)
            .map(|i| runner.u()[flatten_2d_index(pole_j, i, nv)])
            .sum::<f64>()
            / nu as f64;
        let v_mean = (0..nu)
            .map(|i| runner.v()[flatten_2d_index(pole_j, i, nv)])
            .sum::<f64>()
            / nu as f64;
        for i in 0..nu {
            let idx = flatten_2d_index(pole_j, i, nv);
            runner.u_mut()[idx] = u_mean;
            runner.v_mut()[idx] = v_mean;
        }
    }
}
