#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Custom AST metric connection evaluation and radial/throat profiles for [`ChristoffelInspectorTool`].

use eframe::egui;
use math_explorer::nalgebra::SVector;
use math_explorer::physics::high_energy::{MorrisThorneWormhole, SchwarzschildBlackHole};
use math_explorer::pure_math::tensor::{static_christoffel_symbols, StaticMetric};

use super::super::playground::sandbox::{
    build_lapse_static_metric, build_wormhole_static_metric, MIN_REGULAR_LAPSE, MIN_REGULAR_THROAT,
};
use super::{ChristoffelEvaluation, ChristoffelInspectorTool, InspectorManifold};

impl ChristoffelInspectorTool {
    pub(super) fn draw_inspector_sandbox_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut quick_changed = false;
        if self.state.manifold == InspectorManifold::CustomRadialLapse4D {
            ui.horizontal_wrapped(|ui| {
                ui.small("Templates:");
                for (label, expr) in [
                    ("Schwarzschild (1 - 2/r)", "1 - 2 / r"),
                    ("Reissner-Nordstrom", "1 - Rs / r + q^2 / r^2"),
                ] {
                    if ui.small_button(label).clicked() {
                        self.set_custom_lapse_expression(expr);
                        quick_changed = true;
                    }
                }
            });
        }
        let changed = match self.state.manifold {
            InspectorManifold::CustomRadialLapse4D => self.sandbox.lapse_bar.ui(ui),
            InspectorManifold::CustomWormhole3D => self.sandbox.wormhole_bar.ui(ui),
            _ => false,
        };
        if changed {
            self.sync_sandbox_to_state();
        }
        changed || quick_changed
    }

    pub(super) fn sync_sandbox_to_state(&mut self) {
        self.state.custom_lapse_expr = self.sandbox.lapse_bar.raw_text.clone();
        self.state.custom_wormhole_expr = self.sandbox.wormhole_bar.raw_text.clone();
        self.state.custom_params = self.sandbox.merged_params_map();
    }

    pub(super) fn evaluate_custom_lapse(&self) -> Option<ChristoffelEvaluation> {
        let s = &self.state;
        let compiled = self.sandbox.lapse_bar.active_compiled()?;
        let params = self.sandbox.lapse_bar.param_slice();
        let f_here = compiled.eval_2d(s.radial_coord, s.scale_radius, params);
        if !f_here.is_finite() || f_here <= MIN_REGULAR_LAPSE {
            return None;
        }
        let custom = build_lapse_static_metric(compiled, s.scale_radius, params);
        let pt = SVector::<f64, 4>::new(0.0, s.radial_coord, s.theta, 0.4);
        let approx = static_christoffel_symbols(&custom, &pt).ok()?;
        let ref_gamma = SchwarzschildBlackHole::from_schwarzschild_radius(s.scale_radius)
            .ok()
            .and_then(|bh| bh.christoffel_at(&pt).ok());

        let mut analytical = [[[0.0; 4]; 4]; 4];
        let mut numerical = [[[0.0; 4]; 4]; 4];
        let mut max_abs_diff = 0.0_f64;
        for mu in 0..4 {
            for a in 0..4 {
                for b in 0..4 {
                    let nv = approx[mu][(a, b)];
                    let ev = ref_gamma.map_or(nv, |g| g[mu][(a, b)]);
                    analytical[mu][a][b] = ev;
                    numerical[mu][a][b] = nv;
                    max_abs_diff = max_abs_diff.max((ev - nv).abs());
                }
            }
        }
        Some(ChristoffelEvaluation {
            dim: 4,
            analytical,
            numerical,
            max_abs_diff,
        })
    }

    pub(super) fn evaluate_custom_wormhole(&self) -> Option<ChristoffelEvaluation> {
        let s = &self.state;
        let compiled = self.sandbox.wormhole_bar.active_compiled()?;
        let params = self.sandbox.wormhole_bar.param_slice();
        let b_here = compiled.eval_2d(s.radial_coord, s.scale_radius, params);
        if !b_here.is_finite() || b_here <= MIN_REGULAR_THROAT {
            return None;
        }
        let custom = build_wormhole_static_metric(compiled, s.scale_radius, params);
        let pt = SVector::<f64, 3>::new(0.0, s.radial_coord, 0.4);
        let approx = static_christoffel_symbols(&custom, &pt).ok()?;
        let ref_gamma = MorrisThorneWormhole::new(s.scale_radius)
            .ok()
            .and_then(|wh| wh.christoffel_at(&pt).ok());

        let mut analytical = [[[0.0; 4]; 4]; 4];
        let mut numerical = [[[0.0; 4]; 4]; 4];
        let mut max_abs_diff = 0.0_f64;
        for mu in 0..3 {
            for a in 0..3 {
                for b in 0..3 {
                    let nv = approx[mu][(a, b)];
                    let ev = ref_gamma.map_or(nv, |g| g[mu][(a, b)]);
                    analytical[mu][a][b] = ev;
                    numerical[mu][a][b] = nv;
                    max_abs_diff = max_abs_diff.max((ev - nv).abs());
                }
            }
        }
        Some(ChristoffelEvaluation {
            dim: 3,
            analytical,
            numerical,
            max_abs_diff,
        })
    }

    pub(super) fn custom_lapse_profiles(&self, rs: f64, n: usize) -> Vec<(String, Vec<[f64; 2]>)> {
        let Some(compiled) = self.sandbox.lapse_bar.active_compiled() else {
            return Vec::new();
        };
        let params = self.sandbox.lapse_bar.param_slice();
        let custom = build_lapse_static_metric(compiled, rs, params);
        let r_start = (1.12 * self.sandbox.estimate_lapse_horizon(rs)).max(0.5);
        let mut g00 = Vec::with_capacity(n);
        let mut g11 = Vec::with_capacity(n);
        for i in 0..n {
            let r = r_start + 14.0 * (i as f64) / ((n - 1) as f64);
            let pt = SVector::<f64, 4>::new(0.0, r, self.state.theta, 0.0);
            if let Ok(g) = static_christoffel_symbols(&custom, &pt) {
                g00.push([r, g[1][(0, 0)]]);
                g11.push([r, g[0][(0, 1)]]);
            }
        }
        vec![
            ("Gamma^r_(ct ct)(r) [Custom f(r)]".to_string(), g00),
            ("Gamma^ct_(ct r)(r) [Custom f(r)]".to_string(), g11),
        ]
    }

    pub(super) fn custom_wormhole_profiles(
        &self,
        b0: f64,
        n: usize,
    ) -> Vec<(String, Vec<[f64; 2]>)> {
        let Some(compiled) = self.sandbox.wormhole_bar.active_compiled() else {
            return Vec::new();
        };
        let params = self.sandbox.wormhole_bar.param_slice();
        let custom = build_wormhole_static_metric(compiled, b0, params);
        let mut g_phi = Vec::with_capacity(n);
        let mut g_l = Vec::with_capacity(n);
        for i in 0..n {
            let l = -8.0 + 16.0 * (i as f64) / ((n - 1) as f64);
            let pt = SVector::<f64, 3>::new(0.0, l, 0.0);
            if let Ok(g) = static_christoffel_symbols(&custom, &pt) {
                g_phi.push([l, g[2][(1, 2)]]);
                g_l.push([l, 0.1 * g[1][(2, 2)]]);
            }
        }
        vec![
            ("Gamma^phi_(l phi)(l) [Custom b(l)]".to_string(), g_phi),
            ("0.1 * Gamma^l_(phi phi)(l) [Custom b(l)]".to_string(), g_l),
        ]
    }
}
