#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::flow::{AttractorMode, CompiledFlowJob};
use super::AttractorPlotter;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
use math_explorer::physics::chaos::metrics::lorenz_lyapunov;
use math_explorer::pure_math::analysis::ode::{RungeKutta4, SolverExt};
use nalgebra::Vector3;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Supported `(x, y)` parameter axis pairs for [`AttractorCartographer`].
pub const ATTRACTOR_ATLAS_AXIS_PAIRS: &[(&str, &str)] =
    &[("sigma", "rho"), ("rho", "beta"), ("param_0", "param_1")];

const ESCAPE_NORM: f64 = 1e5;

/// Discrete 3D dynamical flow stability regimes mapped by [`AttractorCartographer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AttractorRegime {
    /// Origin sink or stable equilibrium point ($\lambda_1 < -0.02$).
    EquilibriumSink = 0,
    /// Stable periodic limit cycle or Hopf boundary ($-0.02 \le \lambda_1 \le 0.04$).
    PeriodicOrHopf = 1,
    /// Chaotic strange attractor ($\lambda_1 > 0.04$).
    StrangeAttractor = 2,
    /// Trajectory diverges outside bounded phase space.
    UnboundedFlow = 3,
}

impl AttractorRegime {
    /// Converts a raw `u8` cell regime code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::EquilibriumSink,
            1 => Self::PeriodicOrHopf,
            2 => Self::StrangeAttractor,
            _ => Self::UnboundedFlow,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::EquilibriumSink => "Equilibrium Sink",
            Self::PeriodicOrHopf => "Periodic / Hopf Window",
            Self::StrangeAttractor => "Strange Attractor",
            Self::UnboundedFlow => "Unbounded Flow",
        }
    }
}

/// Analytical subcritical Hopf threshold $\rho_H(\sigma, \beta)$ for the Lorenz system.
#[inline]
#[must_use]
pub fn lorenz_hopf_threshold(sigma: f64, beta: f64) -> Option<f64> {
    let denom = sigma - beta - 1.0;
    if sigma > 0.0 && beta > 0.0 && denom > 1e-6 {
        let rho_h = sigma * (sigma + beta + 3.0) / denom;
        rho_h.is_finite().then_some(rho_h)
    } else {
        None
    }
}

/// Default serialized atlas axis keys for [`super::preset`].
#[must_use]
pub fn default_attractor_atlas_axes() -> [String; 2] {
    ["sigma".to_string(), "rho".to_string()]
}

/// Predicate skipping default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_attractor_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "sigma" && axes[1] == "rho"
}

fn attractor_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (resolved, label, min, max, step) = match key {
        "sigma" => ("sigma", "Prandtl (σ)", 0.0, 50.0, 0.1),
        "beta" => ("beta", "Geometric (β)", 0.0, 10.0, 0.1),
        "param_0" => ("param_0", "Custom Param 0", 0.0, 10.0, 0.05),
        "param_1" => ("param_1", "Custom Param 1", 0.0, 10.0, 0.05),
        _ => ("rho", "Rayleigh (ρ)", 0.0, 100.0, 0.5),
    };
    AtlasAxisSpec::from_theory(
        theory_params,
        resolved,
        label,
        ParameterConstraint {
            min,
            max,
            step,
            default: None,
        },
    )
}

/// Zero-allocation 2D Lyapunov and Hopf boundary cartographer for 3D continuous flows.
#[derive(Clone, Debug)]
pub struct AttractorCartographer {
    axes: AtlasAxisPair,
    mode: AttractorMode,
    custom_job: CompiledFlowJob,
    base_sigma: f64,
    base_rho: f64,
    base_beta: f64,
    base_p0: f64,
    base_p1: f64,
}

impl AttractorCartographer {
    /// Builds an [`AttractorCartographer`] bounded by `theory_params`.
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        mode: AttractorMode,
        custom_job: CompiledFlowJob,
        base_lorenz: (f64, f64, f64),
        base_custom: (f64, f64),
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: attractor_axis_spec(theory_params, axis_keys.0),
                y: attractor_axis_spec(theory_params, axis_keys.1),
            },
            mode,
            custom_job,
            base_sigma: base_lorenz.0,
            base_rho: base_lorenz.1,
            base_beta: base_lorenz.2,
            base_p0: base_custom.0,
            base_p1: base_custom.1,
        }
    }

    fn unbounded_cell() -> AtlasCell {
        AtlasCell {
            regime: AttractorRegime::UnboundedFlow as u8,
            primary: 2.0,
            secondary: 0.0,
        }
    }

    fn eval_lorenz_cell(&self, sigma: f64, rho: f64, beta: f64) -> AtlasCell {
        if sigma <= 0.0 || beta <= 0.0 {
            return Self::unbounded_cell();
        }
        let dt = 0.015;
        let probe = LorenzBuilder::new()
            .sigma(sigma)
            .rho(rho)
            .beta(beta)
            .build(LorenzState::new(1.0, 1.0, 1.0));
        let mut pos = Vector3::new(1.0, 1.0, 1.0);
        let mut rk4 = RungeKutta4::new(&pos);
        for _ in 0..80 {
            pos = rk4.solve(&probe, 0.0, &pos, dt);
            if !(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite())
                || pos.norm() > ESCAPE_NORM
            {
                return Self::unbounded_cell();
            }
        }
        let Ok(lam) = lorenz_lyapunov(&probe, &mut rk4, pos, dt, 20, 0.24) else {
            return Self::unbounded_cell();
        };
        let rho_h = lorenz_hopf_threshold(sigma, beta).unwrap_or(f64::INFINITY);
        let regime = if rho <= 1.0 || (lam <= 0.04 && rho < rho_h) {
            AttractorRegime::EquilibriumSink
        } else if lam > 0.04 {
            AttractorRegime::StrangeAttractor
        } else {
            AttractorRegime::PeriodicOrHopf
        };
        AtlasCell {
            regime: regime as u8,
            primary: lam,
            secondary: rho_h,
        }
    }

    fn eval_custom_cell(&self, p0: f64, p1: f64) -> AtlasCell {
        let overrides = [(0usize, p0), (1usize, p1)];
        let view = self.custom_job.bind_params(&overrides);
        let dt = 0.015;
        let mut pos = Vector3::new(1.0, 1.0, 1.0);
        for _ in 0..80 {
            pos = view.rk4_step(pos, dt);
            if !(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite())
                || pos.norm() > ESCAPE_NORM
            {
                return Self::unbounded_cell();
            }
        }
        let mut rk4 = RungeKutta4::new(&pos);
        let Ok(lam) = lorenz_lyapunov(&view, &mut rk4, pos, dt, 20, 0.24) else {
            return Self::unbounded_cell();
        };
        let regime = if lam > 0.04 {
            AttractorRegime::StrangeAttractor
        } else if lam >= -0.02 {
            AttractorRegime::PeriodicOrHopf
        } else {
            AttractorRegime::EquilibriumSink
        };
        AtlasCell {
            regime: regime as u8,
            primary: lam,
            secondary: pos.norm(),
        }
    }
}

impl RegimeCartographer for AttractorCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let mut sigma = self.base_sigma;
        let mut rho = self.base_rho;
        let mut beta = self.base_beta;
        let mut c0 = self.base_p0;
        let mut c1 = self.base_p1;
        let mut custom_axes = false;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "sigma" => sigma = v,
                "rho" => rho = v,
                "beta" => beta = v,
                "param_0" => {
                    c0 = v;
                    custom_axes = true;
                }
                "param_1" => {
                    c1 = v;
                    custom_axes = true;
                }
                _ => {}
            }
        }
        if custom_axes || self.mode == AttractorMode::Custom {
            self.eval_custom_cell(c0, c1)
        } else {
            self.eval_lorenz_cell(sigma, rho, beta)
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match AttractorRegime::from_u8(cell.regime) {
            AttractorRegime::EquilibriumSink => {
                (1.0 / (1.0 - cell.primary).max(1.0)).clamp(0.2, 1.0)
            }
            AttractorRegime::PeriodicOrHopf => 0.80,
            AttractorRegime::StrangeAttractor => (cell.primary * 0.9).clamp(0.3, 1.0),
            AttractorRegime::UnboundedFlow => 0.75,
        };
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        AttractorRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Largest Lyapunov λ₁"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = AttractorRegime::from_u8(cell.regime);
        if reg == AttractorRegime::UnboundedFlow {
            format!(
                "{} = {:.3}, {} = {:.3}\nRegime: {}",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label()
            )
        } else if cell.secondary.is_finite() && self.mode == AttractorMode::Lorenz {
            format!(
                "{} = {:.3}, {} = {:.3}\nRegime: {}\nλ₁ = {:+.3}  (ρ_H = {:.2})",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.primary,
                cell.secondary
            )
        } else {
            format!(
                "{} = {:.3}, {} = {:.3}\nRegime: {}\nλ₁ = {:+.3}",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.primary
            )
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            AttractorRegime::EquilibriumSink,
            AttractorRegime::PeriodicOrHopf,
            AttractorRegime::StrangeAttractor,
            AttractorRegime::UnboundedFlow,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 4, 0.7, high_contrast),
            )
        })
        .collect()
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        let mut pts = Vec::new();
        match (self.axes.x.key, self.axes.y.key) {
            ("sigma", "rho") => {
                let s_min = (self.base_beta + 1.15).max(self.axes.x.constraint.min);
                let s_max = self.axes.x.constraint.max;
                if s_max <= s_min {
                    return None;
                }
                for i in 0..=96 {
                    let s = s_min + (i as f64 / 96.0) * (s_max - s_min);
                    if let Some(rh) = lorenz_hopf_threshold(s, self.base_beta) {
                        if (self.axes.y.constraint.min..=self.axes.y.constraint.max).contains(&rh) {
                            pts.push([s, rh]);
                        }
                    }
                }
            }
            ("rho", "beta") => {
                let b_min = self.axes.y.constraint.min.max(0.05);
                let b_max = (self.base_sigma - 1.15).min(self.axes.y.constraint.max);
                if b_max <= b_min {
                    return None;
                }
                for i in 0..=96 {
                    let b = b_min + (i as f64 / 96.0) * (b_max - b_min);
                    if let Some(rh) = lorenz_hopf_threshold(self.base_sigma, b) {
                        if (self.axes.x.constraint.min..=self.axes.x.constraint.max).contains(&rh) {
                            pts.push([rh, b]);
                        }
                    }
                }
            }
            _ => return None,
        }
        (pts.len() >= 2).then_some(("Hopf Curve ρ_H(σ, β)", pts))
    }
}

impl AttractorPlotter {
    fn custom_param_pair(&self) -> (f64, f64) {
        let s = self.flow_bank.param_slice();
        (
            s.first().copied().unwrap_or(0.2),
            s.get(1).copied().unwrap_or(0.2),
        )
    }

    fn set_custom_param_slot(&mut self, slot: usize, val: f64) {
        let mut pmap = self.flow_bank.params_map();
        if let Some(target) = pmap.values_mut().nth(slot) {
            *target = val.clamp(0.0, 10.0);
            self.flow_bank.set_params_map(&pmap);
        }
    }

    /// Applies an [`AtlasTeleportEvent`] to Lorenz or custom 3D flow parameters and live trajectories.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let mut switched_mode = false;
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "sigma" => {
                    switched_mode |= self.mode != AttractorMode::Lorenz;
                    self.mode = AttractorMode::Lorenz;
                    self.system.sigma = val.clamp(0.0, 50.0);
                }
                "rho" => {
                    switched_mode |= self.mode != AttractorMode::Lorenz;
                    self.mode = AttractorMode::Lorenz;
                    self.system.rho = val.clamp(0.0, 100.0);
                }
                "beta" => {
                    switched_mode |= self.mode != AttractorMode::Lorenz;
                    self.mode = AttractorMode::Lorenz;
                    self.system.beta = val.clamp(0.0, 10.0);
                }
                "param_0" => {
                    switched_mode |= self.mode != AttractorMode::Custom;
                    self.mode = AttractorMode::Custom;
                    self.set_custom_param_slot(0, val);
                }
                "param_1" => {
                    switched_mode |= self.mode != AttractorMode::Custom;
                    self.mode = AttractorMode::Custom;
                    self.set_custom_param_slot(1, val);
                }
                _ => {}
            }
        }
        if switched_mode || self.diverged() || self.system.state.vec.norm() < 1e-3 {
            self.reset();
        } else {
            self.on_slider_changed();
        }
    }

    pub(super) fn render_attractor_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let (p0, p1) = self.custom_param_pair();
        let get_val = |k: &str| match k {
            "sigma" => self.system.sigma,
            "rho" => self.system.rho,
            "beta" => self.system.beta,
            "param_0" => p0,
            "param_1" => p1,
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.mode as u8).hash(&mut h);
        for bar in self.flow_bank.bars() {
            bar.raw_text.hash(&mut h);
        }
        for (idx, &pv) in self.flow_bank.param_slice().iter().enumerate() {
            let skip = (idx == 0 && (xk == "param_0" || yk == "param_0"))
                || (idx == 1 && (xk == "param_1" || yk == "param_1"));
            if !skip {
                pv.to_bits().hash(&mut h);
            }
        }
        for (name, val) in [
            ("sigma", self.system.sigma),
            ("rho", self.system.rho),
            ("beta", self.system.beta),
        ] {
            if name != xk && name != yk {
                val.to_bits().hash(&mut h);
            }
        }

        let theory_params = self.theory_parameters();
        let mode = self.mode;
        let job = CompiledFlowJob::from_bank(&self.flow_bank);
        let (sigma, rho, beta) = (self.system.sigma, self.system.rho, self.system.beta);
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "attractor",
            get_val(xk),
            get_val(yk),
            h.finish(),
            |keys| {
                Arc::new(AttractorCartographer::new(
                    &theory_params,
                    keys,
                    mode,
                    job,
                    (sigma, rho, beta),
                    (p0, p1),
                ))
            },
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}
