//! Curvilinear `CompiledExpr` evaluation, local orthonormal basis projection, Gauss/Stokes
//! integral verification, and zero-allocation 3D Helmholtz decomposition.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CoordinateFrameKind, CurvilinearDiagnostics, CurvilinearVectorStudio, QuiverSample3D,
    MAX_HELMHOLTZ_GRID, MIN_HELMHOLTZ_GRID,
};
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use math_commons::ast::{is_universal_constant, CompiledExpr};
use math_explorer::pure_math::vector_calculus::coordinates::{
    Cartesian, Cylindrical, OrthogonalCoordinateSystem, Spherical,
};
use math_explorer::pure_math::vector_calculus::operators::{curl, divergence, gradient, laplacian};
use math_explorer::pure_math::vector_calculus::theorems::{
    helmholtz_decompose, verify_divergence_theorem, verify_stokes_theorem, Domain,
    HelmholtzSummary, HelmholtzWorkspace, StokesPatch,
};
use nalgebra::Vector3;

pub(super) struct DynCoords<'a>(pub(super) &'a dyn OrthogonalCoordinateSystem);

impl OrthogonalCoordinateSystem for DynCoords<'_> {
    fn scale_factors(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.0.scale_factors(point)
    }

    fn local_to_cartesian(&self, point: &Vector3<f64>) -> Vector3<f64> {
        self.0.local_to_cartesian(point)
    }

    fn cartesian_to_local(&self, cart: &Vector3<f64>) -> Vector3<f64> {
        self.0.cartesian_to_local(cart)
    }
}

fn eval_compiled_3d(expr: &CompiledExpr, p: &Vector3<f64>, params: &[f64]) -> f64 {
    let val = expr.eval_3d(p[0], p[1], p[2], params);
    if val.is_finite() {
        val.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn sample_and_decompose_helmholtz(
    coords: &DynCoords<'_>,
    domain: &Domain,
    poisson_iters: usize,
    workspace: &mut HelmholtzWorkspace,
    field_samples: &mut [Vector3<f64>],
    vec_fn: &dyn Fn(&Vector3<f64>) -> Vector3<f64>,
) -> HelmholtzSummary {
    let n = workspace.nx;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = workspace.cell_center(domain, i, j, k);
                let idx = workspace.index(i, j, k);
                field_samples[idx] = vec_fn(&p);
            }
        }
    }
    helmholtz_decompose(coords, field_samples, domain, workspace, poisson_iters)
}

/// Projects a local curvilinear vector `v_local = (v1, v2, v3)` at curvilinear `point`
/// into 3D Cartesian space using the orthonormalized tangent frame of `coords`.
pub(super) fn curvilinear_vector_to_cartesian(
    coords: &dyn OrthogonalCoordinateSystem,
    point: &Vector3<f64>,
    v_local: &Vector3<f64>,
) -> Vector3<f64> {
    let eps = 1e-5;
    let mut cart = Vector3::zeros();
    for axis in 0..3 {
        let mut p_plus = *point;
        let mut p_minus = *point;
        p_plus[axis] += eps;
        p_minus[axis] -= eps;
        let tangent = coords.local_to_cartesian(&p_plus) - coords.local_to_cartesian(&p_minus);
        let norm = tangent.norm();
        if norm > 1e-12 && norm.is_finite() {
            cart += (tangent / norm) * v_local[axis];
        }
    }
    cart
}

pub(super) fn build_field_bar(
    label: &'static str,
    axes: &[&'static str; 3],
    expr: &str,
) -> crate::framework::MathExpressionInputBar {
    let mut bar = crate::framework::MathExpressionInputBar::new(label, "0", vec![])
        .with_axes(axes)
        .with_inline_sliders(false);
    bar.set_expression(expr);
    bar
}

impl CurvilinearVectorStudio {
    /// Switches the active coordinate system and resets domain, patch, probe, and formula bars.
    pub fn switch_frame(&mut self, next_frame: CoordinateFrameKind) {
        if self.frame == next_frame {
            return;
        }
        self.frame = next_frame;
        let vars = next_frame.axis_vars();
        let (f1, f2, f3, phi) = next_frame.default_expressions();
        self.f1_bar = build_field_bar("F_1", &vars, f1);
        self.f2_bar = build_field_bar("F_2", &vars, f2);
        self.f3_bar = build_field_bar("F_3", &vars, f3);
        self.scalar_bar = build_field_bar("Phi", &vars, phi);
        let (d_min, d_max) = next_frame.default_domain();
        self.domain_min = d_min;
        self.domain_max = d_max;
        let (s_axis, s_fixed, s_u, s_v) = next_frame.default_stokes_patch();
        self.stokes_axis = s_axis;
        self.stokes_fixed = s_fixed;
        self.stokes_u = s_u;
        self.stokes_v = s_v;
        self.probe_point = next_frame.default_probe();
        self.recompile_expressions();
        self.mark_dirty();
    }

    pub(super) fn mark_dirty(&mut self) {
        self.dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    /// Recompiles `(F_1, F_2, F_3, Phi)` against the active coordinate variables plus free sliders.
    pub fn recompile_expressions(&mut self) {
        let (Some(e1), Some(e2), Some(e3), Some(es)) = (
            self.f1_bar.expr(),
            self.f2_bar.expr(),
            self.f3_bar.expr(),
            self.scalar_bar.expr(),
        ) else {
            self.compile_error = Some("Syntax error in one or more field expressions".to_string());
            return;
        };

        let axes = self.frame.axis_vars();
        let mut discovered = Vec::new();
        for expr in [e1, e2, e3, es] {
            for v in expr.free_variables() {
                let is_axis = axes.iter().any(|ax| v.eq_ignore_ascii_case(ax));
                if !is_axis
                    && !is_universal_constant(&v)
                    && !discovered.iter().any(|existing| existing == &v)
                {
                    discovered.push(v);
                }
            }
        }
        if discovered.len() > MAX_SYNTHESIZED_PARAMS {
            self.compile_error = Some(format!(
                "Field defines {} free parameters (maximum {MAX_SYNTHESIZED_PARAMS})",
                discovered.len()
            ));
            return;
        }
        let shared_refs: Vec<&str> = discovered.iter().map(String::as_str).collect();

        match (
            CompiledExpr::compile_with_vars_and_params(e1, &axes, &shared_refs),
            CompiledExpr::compile_with_vars_and_params(e2, &axes, &shared_refs),
            CompiledExpr::compile_with_vars_and_params(e3, &axes, &shared_refs),
            CompiledExpr::compile_with_vars_and_params(es, &axes, &shared_refs),
        ) {
            (Ok(c1), Ok(c2), Ok(c3), Ok(cs)) => {
                self.param_sliders.sync_names(&discovered);
                self.compiled_quad = (c1, c2, c3, cs);
                self.compile_error = None;
                self.dirty = true;
            }
            (Err(e), _, _, _) | (_, Err(e), _, _) | (_, _, Err(e), _) | (_, _, _, Err(e)) => {
                self.compile_error = Some(e.to_string());
            }
        }
    }

    pub(super) fn recompute_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        self.clamp_all_bounds();
        match self.frame {
            CoordinateFrameKind::Cartesian => self.evaluate_with_coords(&Cartesian),
            CoordinateFrameKind::Cylindrical => self.evaluate_with_coords(&Cylindrical),
            CoordinateFrameKind::Spherical => self.evaluate_with_coords(&Spherical),
        }
        self.dirty = false;
    }

    fn evaluate_with_coords(&mut self, raw_coords: &dyn OrthogonalCoordinateSystem) {
        let coords = DynCoords(raw_coords);
        let n = self.grid_res.clamp(MIN_HELMHOLTZ_GRID, MAX_HELMHOLTZ_GRID);
        if self.workspace.nx != n || self.workspace.ny != n || self.workspace.nz != n {
            self.workspace = HelmholtzWorkspace::new(n, n, n);
            self.field_samples.resize(n * n * n, Vector3::zeros());
        }

        let (c1, c2, c3, cs) = &self.compiled_quad;
        let params = self.param_sliders.param_slice();
        let vec_fn = |p: &Vector3<f64>| -> Vector3<f64> {
            Vector3::new(
                eval_compiled_3d(c1, p, params),
                eval_compiled_3d(c2, p, params),
                eval_compiled_3d(c3, p, params),
            )
        };
        let scalar_fn = |p: &Vector3<f64>| -> f64 { eval_compiled_3d(cs, p, params) };

        let domain = Domain {
            min: Vector3::from_row_slice(&self.domain_min),
            max: Vector3::from_row_slice(&self.domain_max),
        };
        let patch = self.current_stokes_patch();
        let probe = Vector3::from_row_slice(&self.probe_point);
        let _ = coords.cartesian_to_local(&coords.local_to_cartesian(&probe));

        let steps = self.integration_steps.clamp(6, 48);
        let (gauss_volume, gauss_flux, gauss_residual) =
            verify_divergence_theorem(&coords, &domain, vec_fn, steps);
        let (stokes_surface_curl, stokes_loop_circulation, stokes_residual) =
            verify_stokes_theorem(&coords, &patch, vec_fn, steps);
        let helmholtz = sample_and_decompose_helmholtz(
            &coords,
            &domain,
            self.poisson_iters.clamp(20, 240),
            &mut self.workspace,
            &mut self.field_samples,
            &vec_fn,
        );

        self.diagnostics = CurvilinearDiagnostics {
            probe_gradient: gradient(&coords, scalar_fn, &probe),
            probe_divergence: divergence(&coords, vec_fn, &probe),
            probe_curl: curl(&coords, vec_fn, &probe),
            probe_laplacian: laplacian(&coords, scalar_fn, &probe),
            probe_scale_factors: coords.scale_factors(&probe),
            gauss_volume,
            gauss_flux,
            gauss_residual,
            stokes_surface_curl,
            stokes_loop_circulation,
            stokes_residual,
            helmholtz,
        };
        self.rebuild_quivers(raw_coords, &domain);
    }

    fn current_stokes_patch(&self) -> StokesPatch {
        StokesPatch::new(
            self.stokes_axis.min(2),
            self.stokes_fixed,
            (self.stokes_u[0], self.stokes_u[1]),
            (self.stokes_v[0], self.stokes_v[1]),
        )
    }

    fn rebuild_quivers(&mut self, coords: &dyn OrthogonalCoordinateSystem, domain: &Domain) {
        let n = self.workspace.nx;
        let stride = (n / 5).max(1);
        self.quivers.clear();
        for k in (1..(n - 1)).step_by(stride) {
            for j in (1..(n - 1)).step_by(stride) {
                for i in (1..(n - 1)).step_by(stride) {
                    let idx = self.workspace.index(i, j, k);
                    let p_local = self.workspace.cell_center(domain, i, j, k);
                    let pos = coords.local_to_cartesian(&p_local);
                    let tot =
                        curvilinear_vector_to_cartesian(coords, &p_local, &self.field_samples[idx]);
                    let irr = curvilinear_vector_to_cartesian(
                        coords,
                        &p_local,
                        &self.workspace.irrotational[idx],
                    );
                    let sol = curvilinear_vector_to_cartesian(
                        coords,
                        &p_local,
                        &self.workspace.solenoidal[idx],
                    );
                    self.quivers.push(QuiverSample3D {
                        pos: [pos[0], pos[1], pos[2]],
                        total: [tot[0], tot[1], tot[2]],
                        irrotational: [irr[0], irr[1], irr[2]],
                        solenoidal: [sol[0], sol[1], sol[2]],
                    });
                }
            }
        }
    }
}
