//! Zero-allocation flat-grid `MeanCurvatureFlow`, `HeatEquationSolver`, CFL clamping,
//! `laplace_beltrami` diagnostics, and `FiniteStateGuard` rollback for [`super::ManifoldFlowStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::custom_surface::SurfaceType;
use super::{ManifoldFlowMode, ManifoldFlowStudio};
use crate::framework::recovery::FiniteStateGuard;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::pure_math::differential_geometry::heat_equation::HeatEquationSolver;
use math_explorer::pure_math::differential_geometry::mean_curvature_flow::{
    DiscreteSurface, MeanCurvatureFlow,
};
use math_explorer::pure_math::differential_geometry::operators::laplace_beltrami;
use math_explorer::pure_math::differential_geometry::surface::{
    KleinBottle, ParametricSurface, Sphere, Torus,
};
use nalgebra::Point3;
use std::f64::consts::{PI, TAU};

struct DynSurface<'a>(&'a dyn ParametricSurface);

impl ParametricSurface for DynSurface<'_> {
    fn position(&self, u: f64, v: f64) -> Point3<f64> {
        self.0.position(u, v)
    }

    fn partial_u(&self, u: f64, v: f64) -> nalgebra::Vector3<f64> {
        self.0.partial_u(u, v)
    }

    fn partial_v(&self, u: f64, v: f64) -> nalgebra::Vector3<f64> {
        self.0.partial_v(u, v)
    }
}

fn sample_perturbed_surface_point(
    surface: &dyn ParametricSurface,
    u: f64,
    v: f64,
    pert: f64,
) -> Point3<f64> {
    let base = surface.position(u, v);
    if pert.abs() < 1e-9 {
        return base;
    }
    let n = surface.partial_u(u, v).cross(&surface.partial_v(u, v));
    let norm = n.norm();
    if norm > 1e-10 && norm.is_finite() {
        let wave = (2.0 * u).cos() * (2.0 * v).sin();
        Point3::from(base.coords + (n / norm) * (pert * wave))
    } else {
        base
    }
}

fn initial_thermal_spot(u: f64, v: f64, u_range: (f64, f64), v_range: (f64, f64)) -> f64 {
    let uc = 0.5 * (u_range.0 + u_range.1);
    let vc = 0.5 * (v_range.0 + v_range.1);
    let su = (0.18 * (u_range.1 - u_range.0)).max(0.15);
    let sv = (0.18 * (v_range.1 - v_range.0)).max(0.15);
    let r2 = ((u - uc) / su).powi(2) + ((v - vc) / sv).powi(2);
    100.0 * (-r2).exp()
}

fn flow_uv_domain_and_closure(
    surface_type: SurfaceType,
    sandbox_uv: ((f64, f64), (f64, f64)),
) -> ((f64, f64), (f64, f64), bool, bool) {
    match surface_type {
        SurfaceType::Sphere => ((0.0, TAU), (0.08, PI - 0.08), true, false),
        SurfaceType::Torus => ((0.0, TAU), (0.0, TAU), true, true),
        SurfaceType::KleinBottle => ((0.0, TAU), (0.0, TAU), false, true),
        SurfaceType::CustomHeightfield | SurfaceType::CustomParametric => {
            (sandbox_uv.0, sandbox_uv.1, false, false)
        }
    }
}

fn bilinear_sample(
    grid: &[f64],
    nu: usize,
    nv: usize,
    u_range: (f64, f64),
    v_range: (f64, f64),
    u: f64,
    v: f64,
) -> f64 {
    if grid.len() != nu * nv || nu < 2 || nv < 2 {
        return 0.0;
    }
    let fu =
        ((u - u_range.0) / (u_range.1 - u_range.0).max(1e-9)).clamp(0.0, 1.0) * (nu - 1) as f64;
    let fv =
        ((v - v_range.0) / (v_range.1 - v_range.0).max(1e-9)).clamp(0.0, 1.0) * (nv - 1) as f64;
    let i0 = (fu.floor() as usize).min(nu - 2);
    let j0 = (fv.floor() as usize).min(nv - 2);
    let tu = fu - i0 as f64;
    let tv = fv - j0 as f64;
    let q00 = grid[flatten_2d_index(j0, i0, nv)];
    let q10 = grid[flatten_2d_index(j0, i0 + 1, nv)];
    let q01 = grid[flatten_2d_index(j0 + 1, i0, nv)];
    let q11 = grid[flatten_2d_index(j0 + 1, i0 + 1, nv)];
    let a = q00 * (1.0 - tu) + q10 * tu;
    let b = q01 * (1.0 - tu) + q11 * tu;
    a * (1.0 - tv) + b * tv
}

impl ManifoldFlowStudio {
    /// Re-initializes both `MeanCurvatureFlow` and `HeatEquationSolver` flat buffers from current parameters.
    pub fn reset_simulation(&mut self) {
        self.clamp_model_parameters();
        match self.surface_type {
            SurfaceType::Sphere => self.init_with_surface(&Sphere {
                radius: self.sphere_radius,
            }),
            SurfaceType::Torus => self.init_with_surface(&Torus {
                major_radius: self.torus_major_radius,
                minor_radius: self.torus_minor_radius,
            }),
            SurfaceType::KleinBottle => self.init_with_surface(&KleinBottle {
                radius: self.klein_radius,
            }),
            SurfaceType::CustomHeightfield => {
                let hf = self.custom_sandbox.build_heightfield();
                self.init_with_surface(&hf);
            }
            SurfaceType::CustomParametric => {
                let cp = self.custom_sandbox.build_parametric();
                self.init_with_surface(&cp);
            }
        }
        self.sim_time = 0.0;
        self.step_count = 0;
        self.recovery_note = None;
        self.metric_history.clear();
        self.metric_history.push([0.0, 1.0]);
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    fn init_with_surface(&mut self, surface: &dyn ParametricSurface) {
        let dyn_surf = DynSurface(surface);
        let nu = self.u_res;
        let nv = self.v_res;
        let sandbox_uv = self.custom_sandbox.uv_domain_for(self.surface_type);
        let (u_range, v_range, closed_u, closed_v) =
            flow_uv_domain_and_closure(self.surface_type, sandbox_uv);
        let pert = self.initial_perturbation;

        let discrete = DiscreteSurface::from_parametric(
            u_range,
            v_range,
            nu,
            nv,
            closed_u,
            closed_v,
            |u, v| sample_perturbed_surface_point(surface, u, v, pert),
        );

        let static_surf =
            DiscreteSurface::from_parametric(u_range, v_range, nu, nv, false, false, |u, v| {
                surface.position(u, v)
            });
        self.static_points = static_surf.points;

        let heat_solver =
            HeatEquationSolver::new(&dyn_surf, self.alpha, u_range, v_range, (nu, nv), |u, v| {
                initial_thermal_spot(u, v, u_range, v_range)
            });
        let heat_cfl = heat_solver.recommended_cfl_dt();
        let heat_init = heat_solver.total_heat().max(1e-9);
        let _center_sample = heat_solver.value(nu / 2, nv / 2);
        self.temp_grid = heat_solver.u_grid;
        self.temp_scratch = heat_solver.next_grid;

        self.mcf = MeanCurvatureFlow::new(discrete);
        let mcf_cfl = self.mcf.recommended_cfl_dt();
        let area_init = self.mcf.surface.surface_area().max(1e-9);

        self.cfl_dt = match self.mode {
            ManifoldFlowMode::MeanCurvature => mcf_cfl,
            ManifoldFlowMode::HeatDiffusion => heat_cfl,
        };
        self.effective_dt = self.dt.min(self.cfl_dt);
        self.initial_metric = match self.mode {
            ManifoldFlowMode::MeanCurvature => area_init,
            ManifoldFlowMode::HeatDiffusion => heat_init,
        };
        self.current_metric = self.initial_metric;
        self.update_probe_laplace_beltrami_on(surface, u_range, v_range);
        self.save_guard_backup();
    }

    fn save_guard_backup(&mut self) {
        self.backup_points.clone_from(&self.mcf.surface.points);
        self.backup_temps.clone_from(&self.temp_grid);
    }

    fn rollback_from_backup(&mut self) {
        self.mcf.surface.points.clone_from(&self.backup_points);
        self.mcf.refresh_curvature_norms();
        self.temp_grid.clone_from(&self.backup_temps);
        self.is_playing = false;
        self.recovery_note = Some(
            "FiniteStateGuard rolled back a non-finite step and clamped dt to the CFL ceiling."
                .to_string(),
        );
    }

    /// Advances `count` CFL-clamped steps, rolling back via [`FiniteStateGuard`] if non-finite.
    pub fn advance_substeps(&mut self, count: usize) {
        if !self.is_finite_state() {
            self.rollback_from_backup();
            self.state_rev = self.state_rev.wrapping_add(1);
            return;
        }
        self.save_guard_backup();
        match self.mode {
            ManifoldFlowMode::MeanCurvature => {
                self.cfl_dt = self.mcf.recommended_cfl_dt();
                self.effective_dt = self.dt.min(self.cfl_dt);
                for _ in 0..count {
                    self.mcf.step(self.effective_dt);
                    self.sim_time += self.effective_dt;
                    self.step_count += 1;
                    if !self.is_finite_state() {
                        break;
                    }
                }
                if self.is_finite_state() {
                    self.current_metric = self.mcf.surface.surface_area();
                    self.refresh_probe_only();
                }
            }
            ManifoldFlowMode::HeatDiffusion => self.step_heat_current_surface(count),
        }

        if !self.is_finite_state() {
            self.rollback_from_backup();
        } else {
            let ratio = (self.current_metric / self.initial_metric.max(1e-9)).clamp(0.0, 10.0);
            if self.metric_history.len() >= 512 {
                self.metric_history.remove(0);
            }
            self.metric_history.push([self.sim_time, ratio]);
        }
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    fn step_heat_current_surface(&mut self, count: usize) {
        match self.surface_type {
            SurfaceType::Sphere => self.step_heat_with_surface(
                &Sphere {
                    radius: self.sphere_radius,
                },
                count,
            ),
            SurfaceType::Torus => self.step_heat_with_surface(
                &Torus {
                    major_radius: self.torus_major_radius,
                    minor_radius: self.torus_minor_radius,
                },
                count,
            ),
            SurfaceType::KleinBottle => self.step_heat_with_surface(
                &KleinBottle {
                    radius: self.klein_radius,
                },
                count,
            ),
            SurfaceType::CustomHeightfield => {
                let hf = self.custom_sandbox.build_heightfield();
                self.step_heat_with_surface(&hf, count);
            }
            SurfaceType::CustomParametric => {
                let cp = self.custom_sandbox.build_parametric();
                self.step_heat_with_surface(&cp, count);
            }
        }
    }

    fn step_heat_with_surface(&mut self, surface: &dyn ParametricSurface, count: usize) {
        let dyn_surf = DynSurface(surface);
        let sandbox_uv = self.custom_sandbox.uv_domain_for(self.surface_type);
        let (u_range, v_range, _, _) = flow_uv_domain_and_closure(self.surface_type, sandbox_uv);
        let mut solver = HeatEquationSolver {
            surface: &dyn_surf,
            alpha: self.alpha,
            u_grid: std::mem::take(&mut self.temp_grid),
            next_grid: std::mem::take(&mut self.temp_scratch),
            grid_res: (self.u_res, self.v_res),
            range_u: u_range,
            range_v: v_range,
        };
        self.cfl_dt = solver.recommended_cfl_dt();
        self.effective_dt = self.dt.min(self.cfl_dt);
        for _ in 0..count {
            solver.step(self.effective_dt);
            self.sim_time += self.effective_dt;
            self.step_count += 1;
        }
        let mut heat_now = solver.total_heat();
        if !heat_now.is_finite() {
            solver.set_value(0, 0, f64::NAN);
            heat_now = f64::NAN;
        }
        self.current_metric = heat_now;
        self.temp_grid = solver.u_grid;
        self.temp_scratch = solver.next_grid;
        self.update_probe_laplace_beltrami_on(surface, u_range, v_range);
    }

    pub(super) fn refresh_probe_only(&mut self) {
        let sandbox_uv = self.custom_sandbox.uv_domain_for(self.surface_type);
        let (u_range, v_range, _, _) = flow_uv_domain_and_closure(self.surface_type, sandbox_uv);
        match self.surface_type {
            SurfaceType::Sphere => self.update_probe_laplace_beltrami_on(
                &Sphere {
                    radius: self.sphere_radius,
                },
                u_range,
                v_range,
            ),
            SurfaceType::Torus => self.update_probe_laplace_beltrami_on(
                &Torus {
                    major_radius: self.torus_major_radius,
                    minor_radius: self.torus_minor_radius,
                },
                u_range,
                v_range,
            ),
            SurfaceType::KleinBottle => self.update_probe_laplace_beltrami_on(
                &KleinBottle {
                    radius: self.klein_radius,
                },
                u_range,
                v_range,
            ),
            SurfaceType::CustomHeightfield => {
                let hf = self.custom_sandbox.build_heightfield();
                self.update_probe_laplace_beltrami_on(&hf, u_range, v_range);
            }
            SurfaceType::CustomParametric => {
                let cp = self.custom_sandbox.build_parametric();
                self.update_probe_laplace_beltrami_on(&cp, u_range, v_range);
            }
        }
    }

    fn update_probe_laplace_beltrami_on(
        &mut self,
        surface: &dyn ParametricSurface,
        u_range: (f64, f64),
        v_range: (f64, f64),
    ) {
        let dyn_surf = DynSurface(surface);
        let u = u_range.0 + self.probe_uv_frac[0].clamp(0.05, 0.95) * (u_range.1 - u_range.0);
        let v = v_range.0 + self.probe_uv_frac[1].clamp(0.05, 0.95) * (v_range.1 - v_range.0);
        let (nu, nv) = (self.u_res, self.v_res);
        let field_ref: &[f64] = match self.mode {
            ManifoldFlowMode::MeanCurvature => &self.mcf.curvature_norms,
            ManifoldFlowMode::HeatDiffusion => &self.temp_grid,
        };
        let interp =
            |uu: f64, vv: f64| bilinear_sample(field_ref, nu, nv, u_range, v_range, uu, vv);
        let lb = laplace_beltrami(&dyn_surf, u, v, &interp);
        self.probe_laplace_beltrami = if lb.is_finite() { lb } else { 0.0 };
    }
}
