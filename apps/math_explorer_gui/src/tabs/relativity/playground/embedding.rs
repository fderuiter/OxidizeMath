#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D Flamm/wormhole surface wireframes, projected reference rings, and radial effective
//! potential plot curves for [`GeodesicPlaygroundTool`].

use std::f64::consts::TAU;

use crate::app::SemanticPalette;
use egui_plot::{Line, LineStyle, PlotPoints};
use math_explorer::physics::high_energy::{
    effective_potential, flamm_paraboloid_z, kerr_veff, MorrisThorneWormhole,
};

use super::sim::to_geodesic_kind;
use super::{GeodesicPlaygroundTool, SpacetimePreset, TrajectorySample};

impl GeodesicPlaygroundTool {
    pub(super) fn reference_ring_projected(
        &self,
        radius: f64,
        z_height: f64,
        segments: usize,
    ) -> Vec<[f64; 2]> {
        let n = segments.max(12);
        (0..=n)
            .map(|i| {
                let (s, c) = (TAU * (i as f64) / (n as f64)).sin_cos();
                self.state
                    .camera
                    .project(&[radius * c, radius * s, z_height])
            })
            .collect()
    }

    pub(super) fn build_3d_reference_rings(&self, palette: &SemanticPalette) -> Vec<Line<'static>> {
        let (r_plus, r_ergo, _, isco, r_ph) =
            self.relativistic_radii_and_omega(self.state.initial_radius);
        let throat_r = if r_plus > 0.0 {
            r_plus
        } else {
            self.state.scale_radius
        };
        let mut rings = vec![Line::new(
            "Outer Horizon / Throat r+",
            PlotPoints::new(self.reference_ring_projected(throat_r, 0.0, 56)),
        )
        .color(palette.error)];
        if self.state.spacetime == SpacetimePreset::KerrEquatorial
            && (r_ergo - throat_r).abs() > 1e-4
        {
            let z_e = flamm_paraboloid_z(throat_r, r_ergo).unwrap_or(0.0);
            rings.push(
                Line::new(
                    "Static Limit Ergosphere r_ergo",
                    PlotPoints::new(self.reference_ring_projected(r_ergo, z_e, 56)),
                )
                .color(palette.error)
                .style(LineStyle::Dashed { length: 6.0 }),
            );
        }
        if !self.state.spacetime.is_wormhole() && r_ph > throat_r {
            let z_ph = self.surface_z_at(throat_r, r_ph);
            rings.push(
                Line::new(
                    "Photon Sphere",
                    PlotPoints::new(self.reference_ring_projected(r_ph, z_ph, 56)),
                )
                .color(palette.warning),
            );
        }
        if !self.state.spacetime.is_wormhole() && isco > throat_r {
            let z_i = self.surface_z_at(throat_r, isco);
            rings.push(
                Line::new(
                    "ISCO",
                    PlotPoints::new(self.reference_ring_projected(isco, z_i, 56)),
                )
                .color(palette.info)
                .style(LineStyle::Dotted { spacing: 5.0 }),
            );
        }
        rings
    }

    pub(super) fn surface_z_at(&self, throat_r: f64, r: f64) -> f64 {
        if self.state.spacetime == SpacetimePreset::CustomRadialLapse {
            let f = self.sandbox.eval_lapse(r, self.state.scale_radius).max(0.0);
            let r_eff = (r * (1.0 - f)).clamp(0.0, r);
            2.0 * (r_eff * (r - r_eff).max(0.0)).sqrt()
        } else {
            flamm_paraboloid_z(throat_r, r).unwrap_or(0.0)
        }
    }

    pub(super) fn projected_worldline(
        &self,
        buf: &[TrajectorySample],
        active: Option<TrajectorySample>,
    ) -> (Vec<[f64; 2]>, Option<[f64; 2]>) {
        let limit = self.state.scrub_index.saturating_add(1).min(buf.len());
        let traj = buf[..limit]
            .iter()
            .map(|s| self.state.camera.project(&s.embed_xyz))
            .collect();
        (
            traj,
            active.map(|s| self.state.camera.project(&s.embed_xyz)),
        )
    }

    pub(super) fn build_embedding_wireframe(&self) -> Vec<Vec<[f64; 2]>> {
        let mut lines = Vec::with_capacity(14);
        let rs = self.state.scale_radius;
        if !self.state.spacetime.is_wormhole() {
            let (r_plus, _, _, _, _) = self.relativistic_radii_and_omega(self.state.initial_radius);
            let r_base = r_plus.max(0.3);
            for ring_idx in 1..=5 {
                let r = r_base * (1.0 + 1.35 * f64::from(ring_idx));
                lines.push(self.reference_ring_projected(r, self.surface_z_at(r_base, r), 40));
            }
            for spoke in 0..8 {
                let (s, c) = (TAU * f64::from(spoke) / 8.0).sin_cos();
                let meridian = (0..18)
                    .map(|k| {
                        let r = r_base + 7.0 * rs * (k as f64) / 17.0;
                        self.state
                            .camera
                            .project(&[r * c, r * s, self.surface_z_at(r_base, r)])
                    })
                    .collect();
                lines.push(meridian);
            }
        } else {
            self.build_wormhole_wireframe(&mut lines, rs);
        }
        lines
    }

    fn build_wormhole_wireframe(&self, lines: &mut Vec<Vec<[f64; 2]>>, b0: f64) {
        let is_custom = self.state.spacetime == SpacetimePreset::CustomWormholeProfile;
        let wh = MorrisThorneWormhole::new(b0).ok();
        let eval_r = |l: f64| {
            if is_custom {
                self.sandbox.eval_wormhole_b(l, b0).max(0.1)
            } else {
                wh.as_ref()
                    .and_then(|w| w.areal_radius(l).ok())
                    .unwrap_or(b0)
            }
        };
        for level in [-6.0, -3.0, 3.0, 6.0] {
            let z = b0.max(0.1) * (level / b0.max(0.1)).asinh();
            lines.push(self.reference_ring_projected(eval_r(level), z, 40));
        }
        for spoke in 0..8 {
            let (s, c) = (TAU * f64::from(spoke) / 8.0).sin_cos();
            let meridian = (0..=20)
                .map(|k| {
                    let l = -8.0 + 16.0 * (k as f64) / 20.0;
                    let r = eval_r(l);
                    let z = b0.max(0.1) * (l / b0.max(0.1)).asinh();
                    self.state.camera.project(&[r * c, r * s, z])
                })
                .collect();
            lines.push(meridian);
        }
    }

    pub(super) fn effective_potential_curve(&self, points: usize) -> Vec<[f64; 2]> {
        let n = points.max(16);
        let ell = self.state.impact_or_momentum;
        let rs = self.state.scale_radius;
        if self.state.spacetime.is_wormhole() {
            return self.wormhole_potential_curve(n, rs, ell);
        }
        let mut curve = Vec::with_capacity(n);
        let kind = to_geodesic_kind(self.state.mode);
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => {
                for i in 0..n {
                    let r = 1.04 * rs + 16.0 * (i as f64) / ((n - 1) as f64);
                    if let Ok(v) = effective_potential(rs, r, ell, kind) {
                        curve.push([r, v]);
                    }
                }
            }
            SpacetimePreset::KerrEquatorial => {
                let a = 0.5 * rs * self.state.spin_chi.clamp(-0.95, 0.95);
                let (r_plus, _, _, _, _) =
                    self.relativistic_radii_and_omega(self.state.initial_radius);
                let e = self.launch_energy;
                let kappa = kind.kappa();
                for i in 0..n {
                    let r = 1.04 * r_plus + 16.0 * (i as f64) / ((n - 1) as f64);
                    let v = kerr_veff(rs, a, r, e, ell, kappa);
                    if v.is_finite() {
                        curve.push([r, v]);
                    }
                }
            }
            SpacetimePreset::CustomRadialLapse => {
                let r_h = self.sandbox.estimate_lapse_horizon(rs).max(0.25);
                let kappa = kind.kappa();
                for i in 0..n {
                    let r = 1.05 * r_h + 16.0 * (i as f64) / ((n - 1) as f64);
                    let f = self.sandbox.eval_lapse(r, rs);
                    if f.is_finite() {
                        curve.push([r, 0.5 * f * (kappa + (ell * ell) / (r * r))]);
                    }
                }
            }
            SpacetimePreset::MorrisThorneWormhole | SpacetimePreset::CustomWormholeProfile => {}
        }
        curve
    }

    fn wormhole_potential_curve(&self, n: usize, rs: f64, ell: f64) -> Vec<[f64; 2]> {
        let mut curve = Vec::with_capacity(n);
        if self.state.spacetime == SpacetimePreset::MorrisThorneWormhole {
            if let Ok(wh) = MorrisThorneWormhole::new(rs) {
                for i in 0..n {
                    let l = -10.0 + 20.0 * (i as f64) / ((n - 1) as f64);
                    if let Ok(v) = wh.effective_potential(l, ell) {
                        curve.push([l, v]);
                    }
                }
            }
        } else {
            for i in 0..n {
                let l = -10.0 + 20.0 * (i as f64) / ((n - 1) as f64);
                let b_val = self.sandbox.eval_wormhole_b(l, rs);
                if b_val.is_finite() && b_val > 0.1 {
                    curve.push([l, (ell * ell) / (2.0 * b_val * b_val)]);
                }
            }
        }
        curve
    }
}
