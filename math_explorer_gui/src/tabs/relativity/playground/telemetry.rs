#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Exact Kerr/ISCO horizons, frame dragging, effective potential curves, observables,
//! custom AST sample builders, and 2D atlas panel wiring.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::async_sim::atlas::AtlasTeleportEvent;
use eframe::egui;
use math_commons::ast::CompiledExpr;
use math_explorer::physics::high_energy::{
    photon_sphere_radius, KerrEquatorialMetric, SchwarzschildBlackHole,
};
use math_explorer::pure_math::analysis::ode::ArrayState;
use scientific_metadata::theory::ParameterConstraint;

use super::{
    kerr_prograde_critical_b, GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset,
    TrajectorySample, MAX_TRAJECTORY_SAMPLES,
};

#[derive(Clone, Copy)]
pub(crate) struct CustomSampleCtx<'a> {
    pub(crate) mode: GeodesicMode,
    pub(crate) compiled: &'a CompiledExpr,
    pub(crate) scale: f64,
    pub(crate) params: &'a [f64],
}

pub(crate) fn push_custom_lapse_sample(
    buf: &mut Vec<TrajectorySample>,
    ctx: CustomSampleCtx<'_>,
    proper_tau: f64,
    state: &ArrayState<8>,
    metric_norm: f64,
) {
    if buf.len() > MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let r = state.0[1].max(0.08);
    let phi = state.0[3];
    let f = ctx.compiled.eval_2d(r, ctx.scale, ctx.params).max(0.0);
    let r_eff = (r * (1.0 - f)).clamp(0.0, r);
    let z = 2.0 * (r_eff * (r - r_eff).max(0.0)).sqrt();
    let (sin_p, cos_p) = phi.sin_cos();
    let grav = f.sqrt();
    let ut = state.0[4].max(1e-6);
    let dtau_dt = match ctx.mode {
        GeodesicMode::NullPhoton => 0.0,
        GeodesicMode::TimelikeParticle => (1.0 / ut).clamp(0.0, 1.0),
    };
    let gamma = if grav > 1e-6 && dtau_dt > 0.0 {
        (grav / dtau_dt).max(1.0)
    } else {
        1.0
    };
    buf.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: r,
        phi,
        radial_vel: state.0[5],
        embed_xyz: [r * cos_p, r * sin_p, z],
        metric_norm,
        clock_rate: dtau_dt,
        grav_dilation: grav,
        lorentz_gamma: gamma,
    });
}

pub(crate) fn push_custom_wormhole_sample(
    buf: &mut Vec<TrajectorySample>,
    ctx: CustomSampleCtx<'_>,
    proper_tau: f64,
    state: &ArrayState<6>,
    metric_norm: f64,
) {
    if buf.len() > MAX_TRAJECTORY_SAMPLES {
        return;
    }
    let l = state.0[1];
    let phi = state.0[2];
    let b0 = ctx.scale.max(0.1);
    let r = ctx.compiled.eval_2d(l, ctx.scale, ctx.params).max(0.1);
    let z = b0 * (l / b0).asinh();
    let (sin_p, cos_p) = phi.sin_cos();
    let ut = state.0[3].max(1.0);
    let clock_rate = match ctx.mode {
        GeodesicMode::NullPhoton => 0.0,
        GeodesicMode::TimelikeParticle => (1.0 / ut).clamp(0.0, 1.0),
    };
    buf.push(TrajectorySample {
        coord_ct: state.0[0],
        proper_tau,
        radial_coord: l,
        phi,
        radial_vel: state.0[4],
        embed_xyz: [r * cos_p, r * sin_p, z],
        metric_norm,
        clock_rate,
        grav_dilation: 1.0,
        lorentz_gamma: ut,
    });
}

impl GeodesicPlaygroundTool {
    /// Teleports parameters from a 2D atlas click/Enter activation and rebuilds both worldlines.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "impact_or_momentum" => self.state.impact_or_momentum = val,
                "initial_radius" => self.state.initial_radius = val,
                "spin_chi" => {
                    self.state.spin_chi = val;
                    if self.state.spacetime == SpacetimePreset::Schwarzschild && val.abs() > 1e-4 {
                        self.state.spacetime = SpacetimePreset::KerrEquatorial;
                    }
                }
                _ => {}
            }
        }
        self.clamp_state_in_place();
        self.rebuild_trajectory(560);
        self.state.scrub_index = self.samples.len().saturating_sub(1);
        self.revision = self.revision.wrapping_add(1);
    }

    pub(super) fn render_relativity_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let prev_axes = self.atlas.axis_keys();
        let get_val = |k: &str, s: &super::GeodesicPlaygroundState| match k {
            "impact_or_momentum" => s.impact_or_momentum,
            "initial_radius" => s.initial_radius,
            "spin_chi" => s.spin_chi,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::mem::discriminant(&self.state.spacetime).hash(&mut h);
        std::mem::discriminant(&self.state.mode).hash(&mut h);
        self.state.scale_radius.to_bits().hash(&mut h);
        self.state.radial_bias.to_bits().hash(&mut h);
        if prev_axes.1 != "spin_chi" {
            self.state.spin_chi.to_bits().hash(&mut h);
        }
        if prev_axes.1 != "initial_radius" {
            self.state.initial_radius.to_bits().hash(&mut h);
        }
        self.sandbox.lapse_bar.state_fingerprint().hash(&mut h);
        self.sandbox.wormhole_bar.state_fingerprint().hash(&mut h);

        let snap = self.build_cartographer(prev_axes);
        let teleport = self.atlas.show_panel(
            ui,
            "relativity_lensing",
            get_val(prev_axes.0, &self.state),
            get_val(prev_axes.1, &self.state),
            h.finish(),
            move |_| Arc::new(snap),
        );
        let cur_axes = self.atlas.axis_keys();
        if cur_axes != prev_axes {
            self.state.atlas_axes = [cur_axes.0.into(), cur_axes.1.into()];
            self.revision = self.revision.wrapping_add(1);
        }
        if let Some(ev) = teleport {
            self.apply_atlas_teleport(ev);
            ui.ctx().request_repaint();
        }
    }

    /// Exact outer horizon $r_+$, static limit $r_{\text{ergo}}$, ZAMO $\omega(r)$, ISCO, and photon sphere.
    #[must_use]
    pub fn relativistic_radii_and_omega(&self, sample_r: f64) -> (f64, f64, f64, f64, f64) {
        let rs = self.state.scale_radius;
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => {
                let isco = SchwarzschildBlackHole::from_schwarzschild_radius(rs)
                    .and_then(|bh| bh.isco())
                    .unwrap_or(3.0 * rs);
                let r_ph = photon_sphere_radius(rs).unwrap_or(1.5 * rs);
                (rs, rs, 0.0, isco, r_ph)
            }
            SpacetimePreset::KerrEquatorial => {
                let chi = self.state.spin_chi.clamp(-0.95, 0.95);
                let a = 0.5 * rs * chi;
                let Ok(kerr) = KerrEquatorialMetric::new(rs, a) else {
                    return (rs, rs, 0.0, 3.0 * rs, 1.5 * rs);
                };
                let r_plus = kerr.outer_horizon_radius();
                let r_ergo = kerr.ergosphere_radius();
                let omega = kerr
                    .frame_dragging_omega(sample_r.max(1.005 * r_plus))
                    .unwrap_or(0.0);
                let base_isco = SchwarzschildBlackHole::from_schwarzschild_radius(rs)
                    .and_then(|bh| bh.isco())
                    .unwrap_or(3.0 * rs);
                let isco = kerr_bpt_isco(base_isco / 6.0, chi, self.state.impact_or_momentum);
                let sign = if chi * self.state.impact_or_momentum >= 0.0 {
                    -1.0
                } else {
                    1.0
                };
                let r_ph = rs * (1.0 + ((2.0 / 3.0) * (sign * chi).clamp(-1.0, 1.0).acos()).cos());
                (r_plus, r_ergo, omega, isco, r_ph)
            }
            SpacetimePreset::CustomRadialLapse => {
                let r_h = self.sandbox.estimate_lapse_horizon(rs);
                let ref_r = r_h.max(0.5 * rs);
                (r_h, r_h, 0.0, 3.0 * ref_r, 1.5 * ref_r)
            }
            SpacetimePreset::MorrisThorneWormhole | SpacetimePreset::CustomWormholeProfile => {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            }
        }
    }

    fn trajectory_summary_stats(&self, r_cur: f64) -> (f64, f64, f64, f64, f64) {
        let r0 = self.state.initial_radius;
        let r_min = self
            .samples
            .iter()
            .map(|s| s.radial_coord)
            .fold(r0, f64::min);
        let r_final = self.samples.last().map_or(r0, |s| s.radial_coord);
        let delta_phi = match (self.samples.first(), self.samples.last()) {
            (Some(f), Some(l)) => (l.phi - f.phi).abs(),
            _ => 0.0,
        };
        let throat_crossed = if self.state.spacetime.is_wormhole() {
            let min_l = self
                .samples
                .iter()
                .map(|s| s.radial_coord)
                .fold(r_cur, f64::min);
            let max_l = self
                .samples
                .iter()
                .map(|s| s.radial_coord)
                .fold(r_cur, f64::max);
            f64::from(u8::from(min_l < -0.05 && max_l > 0.05))
        } else {
            0.0
        };
        let twin_r_min = if self.twin_samples.is_empty() {
            r_min
        } else {
            self.twin_samples
                .iter()
                .map(|s| s.radial_coord)
                .fold(r0, f64::min)
        };
        (r_min, r_final, delta_phi, throat_crossed, twin_r_min)
    }

    pub(super) fn collect_observables(&self) -> Vec<(&'static str, f64)> {
        let active = self.active_sample();
        let r_cur = active.map_or(self.state.initial_radius, |s| s.radial_coord);
        let (r_plus, r_ergo, omega, isco, r_ph) = self.relativistic_radii_and_omega(r_cur);
        let (r_min, r_final, delta_phi, throat, twin_r_min) = self.trajectory_summary_stats(r_cur);
        let (d_ct, d_tau) = match (active, self.active_twin_sample()) {
            (Some(p), Some(t)) => (t.coord_ct - p.coord_ct, t.proper_tau - p.proper_tau),
            _ => (0.0, 0.0),
        };
        let b_crit = if self.state.spacetime.is_wormhole() {
            self.state.scale_radius
        } else if self.state.spacetime == SpacetimePreset::KerrEquatorial {
            kerr_prograde_critical_b(self.state.scale_radius, self.state.spin_chi)
        } else {
            kerr_prograde_critical_b(self.state.scale_radius, 0.0)
        };
        let r0 = self.state.initial_radius;
        let b_cur = self.state.impact_or_momentum;
        vec![
            ("launch_energy", clean_finite(self.launch_energy, 1.0)),
            (
                "effective_energy",
                clean_finite(0.5 * self.launch_energy.powi(2), 0.5),
            ),
            ("critical_impact_b", clean_finite(b_crit, 0.0)),
            ("impact_or_momentum", clean_finite(b_cur, 0.0)),
            ("spin_chi", clean_finite(self.state.spin_chi, 0.0)),
            ("r_min", clean_finite(r_min, r0)),
            ("r_final", clean_finite(r_final, r0)),
            ("delta_phi", clean_finite(delta_phi, 0.0)),
            (
                "captured",
                f64::from(u8::from(self.halted_near_singularity)),
            ),
            ("throat_crossed", throat),
            ("outer_horizon_radius", clean_finite(r_plus, 0.0)),
            ("ergosphere_radius", clean_finite(r_ergo, 0.0)),
            ("frame_dragging_omega", clean_finite(omega, 0.0)),
            ("isco", clean_finite(isco, 0.0)),
            ("photon_sphere_radius", clean_finite(r_ph, 0.0)),
            (
                "clock_rate",
                clean_finite(active.map_or(0.0, |s| s.clock_rate), 0.0),
            ),
            (
                "proper_tau",
                clean_finite(active.map_or(0.0, |s| s.proper_tau), 0.0),
            ),
            (
                "coord_ct",
                clean_finite(active.map_or(0.0, |s| s.coord_ct), 0.0),
            ),
            ("twin_r_min", clean_finite(twin_r_min, r_min)),
            (
                "twin_captured",
                f64::from(u8::from(self.twin_halted_near_singularity)),
            ),
            ("twin_delta_ct", clean_finite(d_ct, 0.0)),
            ("twin_delta_tau", clean_finite(d_tau, 0.0)),
        ]
    }

    pub(super) fn build_theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let min_r = self.min_launch_radius();
        let max_r = if self.state.spacetime.is_wormhole() {
            12.0
        } else {
            18.0
        };
        HashMap::from([
            ("scale_radius".into(), p_bounds(0.5, 5.0, 0.05, 2.0)),
            ("spin_chi".into(), p_bounds(-0.95, 0.95, 0.02, 0.65)),
            (
                "initial_radius".into(),
                p_bounds(min_r, max_r, 0.1, 10.0_f64.clamp(min_r, max_r)),
            ),
            (
                "impact_or_momentum".into(),
                p_bounds(-14.0, 14.0, 0.05, 5.35),
            ),
            ("twin_delta_b".into(), p_bounds(-2.0, 2.0, 0.01, 0.0)),
            ("radial_bias".into(), p_bounds(-0.98, 0.98, 0.02, -0.82)),
            ("step_size".into(), p_bounds(0.005, 0.12, 0.005, 0.04)),
        ])
    }

    pub(super) fn collect_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::with_capacity(5);
        let limit = self
            .state
            .scrub_index
            .saturating_add(1)
            .min(self.samples.len());
        let traj = self.samples[..limit]
            .iter()
            .map(|s| self.state.camera.project(&s.embed_xyz))
            .collect();
        datasets.push(("Geodesic Trajectory (Projected)".to_string(), traj));
        if !self.twin_samples.is_empty() {
            let t_lim = self
                .state
                .scrub_index
                .saturating_add(1)
                .min(self.twin_samples.len());
            let t_traj = self.twin_samples[..t_lim]
                .iter()
                .map(|s| self.state.camera.project(&s.embed_xyz))
                .collect();
            datasets.push(("Twin Geodesic Trajectory (Projected)".to_string(), t_traj));
        }
        let (r_plus, _, _, _, _) = self.relativistic_radii_and_omega(self.state.initial_radius);
        let ring_r = if r_plus > 0.0 {
            r_plus
        } else {
            self.state.scale_radius
        };
        datasets.push((
            "Horizon or Throat Ring".to_string(),
            self.reference_ring_projected(ring_r, 0.0, 48),
        ));
        let veff = self.effective_potential_curve(64);
        if let (Some(f), Some(l)) = (veff.first(), veff.last()) {
            let e_lvl = 0.5 * self.launch_energy * self.launch_energy;
            datasets.push(("0.5 * e^2".to_string(), vec![[f[0], e_lvl], [l[0], e_lvl]]));
        }
        datasets.push(("Effective Potential V_eff(r)".to_string(), veff));
        datasets
    }
}

const fn p_bounds(min: f64, max: f64, step: f64, default: f64) -> ParameterConstraint {
    ParameterConstraint {
        min,
        max,
        step,
        default: Some(default),
    }
}

#[inline]
fn kerr_bpt_isco(m_geom: f64, chi: f64, impact_or_momentum: f64) -> f64 {
    if chi.abs() <= 1e-9 {
        return 6.0 * m_geom;
    }
    let c = chi.clamp(-0.99, 0.99);
    let factor = (1.0 - c * c).max(0.0).cbrt();
    let z1 = 1.0 + factor * ((1.0 + c).max(0.0).cbrt() + (1.0 - c).max(0.0).cbrt());
    let z2 = (3.0 * c * c + z1 * z1).sqrt();
    let root = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).max(0.0).sqrt();
    if c * impact_or_momentum >= 0.0 {
        m_geom * (3.0 + z2 - root)
    } else {
        m_geom * (3.0 + z2 + root)
    }
}

#[inline]
fn clean_finite(val: f64, fallback: f64) -> f64 {
    if val.is_finite() {
        val
    } else {
        fallback
    }
}
