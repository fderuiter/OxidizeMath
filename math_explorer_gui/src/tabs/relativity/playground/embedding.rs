#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! 3D Flamm/wormhole surface wireframes, projected reference rings, sample builders,
//! and radial effective potential curves for [`GeodesicPlaygroundTool`].

use std::f64::consts::TAU;

use crate::app::SemanticPalette;
use egui_plot::{Line, LineStyle, PlotPoints};
use math_explorer::physics::high_energy::{
    combined_clock_rate, effective_potential, flamm_paraboloid_z, GeodesicKind,
    KerrEquatorialMetric, MorrisThorneWormhole, C,
};
use math_explorer::pure_math::analysis::ode::ArrayState;

use super::sim::Sample4dMeta;
use super::sim_integrators::planar_velocity_beta;
use super::{
    GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset, TrajectorySample, MAX_TRAJECTORY_SAMPLES,
};

impl GeodesicPlaygroundTool {
    pub(super) fn push_4d_sample(
        buf: &mut Vec<TrajectorySample>,
        meta: Sample4dMeta<'_>,
        proper_tau: f64,
        state: &ArrayState<8>,
        metric_norm: f64,
    ) {
        if buf.len() > MAX_TRAJECTORY_SAMPLES {
            return;
        }
        let r = state.0[1].max(meta.throat_r);
        let phi = state.0[3];
        let z = flamm_paraboloid_z(meta.throat_r, r).unwrap_or(0.0);
        let (sin_p, cos_p) = phi.sin_cos();
        let ut = state.0[4].max(1e-6);
        let dphi_dct = state.0[7] / ut;
        let velocity_beta = planar_velocity_beta(r, phi, state.0[5] / ut, dphi_dct);
        let dtau_dt = match meta.mode {
            GeodesicMode::NullPhoton => 0.0,
            GeodesicMode::TimelikeParticle => (1.0 / ut).clamp(0.0, 1.0),
        };
        let (grav, rate, gamma) = compute_4d_clock_factors(meta, r, dtau_dt, dphi_dct);
        buf.push(TrajectorySample {
            coord_ct: state.0[0],
            proper_tau,
            radial_coord: r,
            phi,
            radial_vel: state.0[5],
            dphi_dct,
            velocity_beta,
            embed_xyz: [r * cos_p, r * sin_p, z],
            metric_norm,
            clock_rate: rate,
            grav_dilation: grav,
            lorentz_gamma: gamma,
        });
    }

    pub(super) fn push_wormhole_sample(
        buf: &mut Vec<TrajectorySample>,
        mode: GeodesicMode,
        wh: &MorrisThorneWormhole,
        proper_tau: f64,
        state: &ArrayState<6>,
        metric_norm: f64,
    ) {
        if buf.len() > MAX_TRAJECTORY_SAMPLES {
            return;
        }
        let l = state.0[1];
        let phi = state.0[2];
        let xyz = wh.embedding_xyz(l, phi).unwrap_or([0.0, 0.0, 0.0]);
        let ut = state.0[3].max(1.0);
        let dphi_dct = state.0[5] / ut;
        let r_areal = wh.areal_radius(l).unwrap_or(wh.throat_radius);
        let velocity_beta = planar_velocity_beta(r_areal, phi, state.0[4] / ut, dphi_dct);
        let clock_rate = match mode {
            GeodesicMode::NullPhoton => 0.0,
            GeodesicMode::TimelikeParticle => (1.0 / ut).clamp(0.0, 1.0),
        };
        buf.push(TrajectorySample {
            coord_ct: state.0[0],
            proper_tau,
            radial_coord: l,
            phi,
            radial_vel: state.0[4],
            dphi_dct,
            velocity_beta,
            embed_xyz: xyz,
            metric_norm,
            clock_rate,
            grav_dilation: 1.0,
            lorentz_gamma: ut,
        });
    }

    pub(super) fn reference_ring_world(
        radius: f64,
        z_height: f64,
        segments: usize,
    ) -> Vec<[f64; 3]> {
        let n = segments.max(12);
        (0..=n)
            .map(|i| {
                let (s, c) = (TAU * (i as f64) / (n as f64)).sin_cos();
                [radius * c, radius * s, z_height]
            })
            .collect()
    }

    pub(super) fn reference_ring_projected(
        &self,
        radius: f64,
        z_height: f64,
        segments: usize,
    ) -> Vec<[f64; 2]> {
        Self::reference_ring_world(radius, z_height, segments)
            .into_iter()
            .map(|p| self.state.camera.project(&p))
            .collect()
    }

    pub(super) fn reference_rings_world(
        &self,
        palette: &SemanticPalette,
    ) -> Vec<(&'static str, Vec<[f64; 3]>, eframe::egui::Color32)> {
        let (r_plus, r_ergo, _, isco, r_ph) =
            self.relativistic_radii_and_omega(self.state.initial_radius);
        let throat_r = if r_plus > 0.0 {
            r_plus
        } else {
            self.state.scale_radius
        };
        let mut rings = vec![(
            "Outer Horizon / Throat r+",
            Self::reference_ring_world(throat_r, 0.0, 56),
            palette.error,
        )];
        if self.state.spacetime == SpacetimePreset::KerrEquatorial
            && (r_ergo - throat_r).abs() > 1e-4
        {
            let z_e = flamm_paraboloid_z(throat_r, r_ergo).unwrap_or(0.0);
            rings.push((
                "Static Limit Ergosphere r_ergo",
                Self::reference_ring_world(r_ergo, z_e, 56),
                palette.error,
            ));
        }
        if !self.state.spacetime.is_wormhole() && r_ph > throat_r {
            let z_ph = self.surface_z_at(throat_r, r_ph);
            rings.push((
                "Photon Sphere",
                Self::reference_ring_world(r_ph, z_ph, 56),
                palette.warning,
            ));
        }
        if !self.state.spacetime.is_wormhole() && isco > throat_r {
            let z_i = self.surface_z_at(throat_r, isco);
            rings.push((
                "ISCO",
                Self::reference_ring_world(isco, z_i, 56),
                palette.info,
            ));
        }
        rings
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

    pub(super) fn build_embedding_wireframe_world(&self) -> Vec<Vec<[f64; 3]>> {
        let mut lines = Vec::with_capacity(14);
        let rs = self.state.scale_radius;
        if !self.state.spacetime.is_wormhole() {
            let (r_plus, _, _, _, _) = self.relativistic_radii_and_omega(self.state.initial_radius);
            let r_base = r_plus.max(0.3);
            for ring_idx in 1..=5 {
                let r = r_base * (1.0 + 1.35 * f64::from(ring_idx));
                lines.push(Self::reference_ring_world(
                    r,
                    self.surface_z_at(r_base, r),
                    40,
                ));
            }
            for spoke in 0..8 {
                let (s, c) = (TAU * f64::from(spoke) / 8.0).sin_cos();
                let meridian = (0..18)
                    .map(|k| {
                        let r = r_base + 7.0 * rs * (k as f64) / 17.0;
                        [r * c, r * s, self.surface_z_at(r_base, r)]
                    })
                    .collect();
                lines.push(meridian);
            }
        } else {
            self.build_wormhole_wireframe_world(&mut lines, rs);
        }
        lines
    }

    pub(super) fn build_embedding_wireframe(&self) -> Vec<Vec<[f64; 2]>> {
        self.build_embedding_wireframe_world()
            .into_iter()
            .map(|ring| {
                ring.into_iter()
                    .map(|p| self.state.camera.project(&p))
                    .collect()
            })
            .collect()
    }

    fn build_wormhole_wireframe_world(&self, lines: &mut Vec<Vec<[f64; 3]>>, b0: f64) {
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
            lines.push(Self::reference_ring_world(eval_r(level), z, 40));
        }
        for spoke in 0..8 {
            let (s, c) = (TAU * f64::from(spoke) / 8.0).sin_cos();
            let meridian = (0..=20)
                .map(|k| {
                    let l = -8.0 + 16.0 * (k as f64) / 20.0;
                    let r = eval_r(l);
                    let z = b0.max(0.1) * (l / b0.max(0.1)).asinh();
                    [r * c, r * s, z]
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
        let kind = match self.state.mode {
            GeodesicMode::NullPhoton => GeodesicKind::Null,
            GeodesicMode::TimelikeParticle => GeodesicKind::Timelike,
        };
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
                    let diff = ell - a * e;
                    let v = 0.5 * (1.0 - rs / r + (a * a) / (r * r)) * kappa
                        + 0.5 * (ell * ell - a * a * e * e) / (r * r)
                        - 0.5 * rs * diff * diff / (r * r * r);
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

fn compute_4d_clock_factors(
    meta: Sample4dMeta<'_>,
    r: f64,
    dtau_dt: f64,
    dphi_dct: f64,
) -> (f64, f64, f64) {
    if let Some(schwarz) = meta.bh {
        let r_safe = r.max(1.001 * meta.throat_r);
        let g = schwarz.gravitational_time_dilation(r_safe).unwrap_or(0.0);
        let beta = if g > 1e-6 && dtau_dt > 0.0 {
            (1.0 - (dtau_dt / g).clamp(0.0, 1.0).powi(2))
                .max(0.0)
                .sqrt()
                .min(0.999)
        } else {
            0.0
        };
        let rt = combined_clock_rate(schwarz, r_safe, beta * C).unwrap_or(dtau_dt);
        let gam = if beta > 0.0 {
            1.0 / (1.0 - beta * beta).sqrt()
        } else {
            1.0
        };
        (g, rt, gam)
    } else {
        let rs =
            (meta.throat_r * meta.throat_r + meta.spin_a * meta.spin_a) / meta.throat_r.max(1e-6);
        let kerr = KerrEquatorialMetric::new(rs, meta.spin_a).ok();
        let zamo_lapse = kerr
            .and_then(|k| k.zamo_clock_dilation(r.max(1.005 * meta.throat_r)).ok())
            .unwrap_or(0.0);
        let rate = if dtau_dt > 0.0 {
            kerr.and_then(|k| {
                k.clock_dilation(r.max(1.005 * meta.throat_r), dphi_dct)
                    .ok()
            })
            .unwrap_or(dtau_dt)
        } else {
            0.0
        };
        let gam = if zamo_lapse > 1e-6 && rate > 0.0 {
            (zamo_lapse / rate).max(1.0)
        } else {
            1.0
        };
        (zamo_lapse, rate, gam)
    }
}
