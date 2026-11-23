//! Zero-allocation steady-state Bateman regimen cartographer and `TheoryDescribable` metadata
//! for [`super::PharmacokineticsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::challenge::CHALLENGE_PK_THERAPEUTIC_WINDOW;
use super::{
    AbsorptionMode, MAX_DOSE_MG, MAX_INTERVAL_H, MAX_KA, MAX_KE, MAX_MEC, MAX_MTC, MAX_VOLUME_L,
    MIN_INTERVAL_H, MIN_RATE, MIN_VOLUME_L, PharmacokineticsTool,
};
use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent, RegimeCartographer,
    atlas_regime_color,
};
use eframe::egui;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Supported 2D parameter atlas axis pairs for [`PharmacokineticsTool`].
pub const PK_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("dose_mg", "interval_hours"), ("ka", "ke")];

/// Closed-form steady-state `(C_min_ss, C_max_ss)` for periodic oral Bateman dosing.
#[must_use]
#[inline]
pub fn steady_state_bateman_extrema(
    dose_mg: f64,
    tau_h: f64,
    bioavailability: f64,
    ka: f64,
    ke: f64,
    volume_l: f64,
) -> (f64, f64) {
    let d = dose_mg.max(0.0);
    let tau = tau_h.max(0.1);
    let f = bioavailability.clamp(0.0, 1.0);
    let v_d = volume_l.max(0.1);
    let k_a = ka.max(1e-4);
    let k_e = ke.max(1e-4);
    if d <= 0.0 || f <= 0.0 {
        return (0.0, 0.0);
    }

    let r_e = (-k_e * tau).exp();
    let r_a = (-k_a * tau).exp();
    let acc_e = 1.0 / (1.0 - r_e).max(1e-12);
    let acc_a = 1.0 / (1.0 - r_a).max(1e-12);

    if (k_a - k_e).abs() > 1e-6 {
        let scale = (f * d * k_a) / (v_d * (k_a - k_e));
        let eval =
            |t: f64| (scale * (acc_e * (-k_e * t).exp() - acc_a * (-k_a * t).exp())).max(0.0);
        let ratio = (k_a * acc_a) / (k_e * acc_e).max(1e-18);
        let t_peak = if ratio > 0.0 {
            (ratio.ln() / (k_a - k_e)).clamp(0.0, tau)
        } else {
            0.0
        };
        let c_trough = eval(0.0).min(eval(tau));
        let c_peak = eval(t_peak).max(c_trough);
        (c_trough, c_peak)
    } else {
        let k = k_e;
        let denom = (1.0 - r_e).max(1e-12);
        let shift = (tau * r_e) / (denom * denom);
        let eval = |t: f64| ((f * d * k / v_d) * (-k * t).exp() * (t / denom + shift)).max(0.0);
        let t_peak = (1.0 / k - (tau * r_e) / denom).clamp(0.0, tau);
        let c_trough = eval(0.0).min(eval(tau));
        let c_peak = eval(t_peak).max(c_trough);
        (c_trough, c_peak)
    }
}

/// Stack-resident zero-allocation steady-state ADME regimen cartographer.
pub struct PharmacokineticsCartographer {
    axes: AtlasAxisPair,
    dose_mg: f64,
    interval_hours: f64,
    bioavailability: f64,
    ka: f64,
    ke: f64,
    volume_l: f64,
    mec: f64,
    mtc: f64,
}

fn build_pk_axes(
    tool: &PharmacokineticsTool,
    axis_keys: (&'static str, &'static str),
) -> Option<AtlasAxisPair> {
    let theory = tool.theory_parameters();
    let c = |min, max, step| ParameterConstraint { min, max, step };
    match axis_keys {
        ("dose_mg", "interval_hours") => {
            let mut x = AtlasAxisSpec::from_theory(
                &theory,
                "dose_mg",
                "Maintenance Dose D (mg)",
                c(10.0, MAX_DOSE_MG, 10.0),
            );
            x.constraint.min = x.constraint.min.max(10.0);
            x.constraint.max = 800.0_f64.max(tool.dose_mg * 1.4).min(x.constraint.max);
            let mut y = AtlasAxisSpec::from_theory(
                &theory,
                "interval_hours",
                "Dosing Interval τ (h)",
                c(MIN_INTERVAL_H, MAX_INTERVAL_H, 0.5),
            );
            y.constraint.max = 36.0_f64
                .max(tool.interval_hours * 1.4)
                .min(y.constraint.max);
            Some(AtlasAxisPair { x, y })
        }
        ("ka", "ke") => {
            let mut x = AtlasAxisSpec::from_theory(
                &theory,
                "ka",
                "Absorption Rate k_a (1/h)",
                c(0.05, MAX_KA, 0.05),
            );
            x.constraint.min = x.constraint.min.max(0.05);
            x.constraint.max = 4.0_f64.max(tool.effective_ka * 1.25).min(x.constraint.max);
            let mut y = AtlasAxisSpec::from_theory(
                &theory,
                "ke",
                "Elimination Rate k_e (1/h)",
                c(MIN_RATE, MAX_KE, 0.01),
            );
            y.constraint.max = 0.8_f64.max(tool.ke * 1.4).min(y.constraint.max);
            Some(AtlasAxisPair { x, y })
        }
        _ => None,
    }
}

impl PharmacokineticsCartographer {
    /// Constructs a [`PharmacokineticsCartographer`] for `("dose_mg", "interval_hours")` or `("ka", "ke")`.
    #[must_use]
    pub fn new(
        tool: &PharmacokineticsTool,
        axis_keys: (&'static str, &'static str),
    ) -> Option<Self> {
        let axes = build_pk_axes(tool, axis_keys)?;
        Some(Self {
            axes,
            dose_mg: tool.dose_mg,
            interval_hours: tool.interval_hours,
            bioavailability: tool.bioavailability,
            ka: tool.effective_ka.max(MIN_RATE),
            ke: tool.ke.max(MIN_RATE),
            volume_l: tool.volume_l,
            mec: tool.mec.max(0.05),
            mtc: tool.mtc.max(tool.mec + 0.1),
        })
    }
}

impl RegimeCartographer for PharmacokineticsCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (d, tau, ka, ke) = match (self.axes.x.key, self.axes.y.key) {
            ("dose_mg", "interval_hours") => (p1, p2, self.ka, self.ke),
            _ => (self.dose_mg, self.interval_hours, p1, p2),
        };
        let (c_min, c_max) =
            steady_state_bateman_extrema(d, tau, self.bioavailability, ka, ke, self.volume_l);
        let regime = if c_max > self.mtc {
            3
        } else if c_max < self.mec {
            0
        } else if c_min >= self.mec {
            1
        } else {
            2
        };
        AtlasCell {
            regime,
            primary: c_max,
            secondary: c_min,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (cell.primary / (self.mtc * 1.5).max(1e-6)).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        match regime {
            0 => "Subtherapeutic Underdose (C_max < MEC)",
            1 => "Therapeutic Window (MEC ≤ C_ss ≤ MTC)",
            2 => "Trough Breakthrough (C_min < MEC ≤ C_max)",
            _ => "Supratherapeutic Toxic Peak (C_max > MTC)",
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Steady Peak C_max,ss (mg/L)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {p1:.2}, {} = {p2:.2}\nRegime: {}\nC_max,ss = {:.2} mg/L | C_min,ss = {:.2} mg/L",
            self.axes.x.label,
            self.axes.y.label,
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if (self.axes.x.key, self.axes.y.key) != ("dose_mg", "interval_hours") {
            return None;
        }
        let mut mtc_ceil = Vec::with_capacity(49);
        let steps = 48;
        for i in 0..=steps {
            let frac = (i as f64) / (steps as f64);
            let tau = self.axes.y.constraint.min + frac * self.axes.y.span();
            let (_, unit_max) = steady_state_bateman_extrema(
                1.0,
                tau,
                self.bioavailability,
                self.ka,
                self.ke,
                self.volume_l,
            );
            if unit_max > 1e-9 {
                let d_mtc = self.mtc / unit_max;
                if (self.axes.x.constraint.min..=self.axes.x.constraint.max).contains(&d_mtc) {
                    mtc_ceil.push([d_mtc, tau]);
                }
            }
        }
        (mtc_ceil.len() >= 2).then_some(("Toxic Ceiling (C_max,ss = MTC)", mtc_ceil))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        (0..4_u8)
            .map(|r| {
                (
                    self.regime_label(r),
                    atlas_regime_color(r, 4, 0.75, high_contrast),
                )
            })
            .collect()
    }
}

impl PharmacokineticsTool {
    pub(crate) fn pk_atlas_coords(&self, axis_keys: (&str, &str)) -> (f64, f64) {
        match axis_keys {
            ("dose_mg", "interval_hours") => (self.dose_mg, self.interval_hours),
            ("ka", "ke") => (self.effective_ka, self.ke),
            _ => (self.dose_mg, self.interval_hours),
        }
    }

    pub(crate) fn pk_atlas_signature(&self, axis_keys: (&str, &str)) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.bioavailability.to_bits().hash(&mut h);
        self.volume_l.to_bits().hash(&mut h);
        self.mec.to_bits().hash(&mut h);
        self.mtc.to_bits().hash(&mut h);
        match axis_keys {
            ("dose_mg", "interval_hours") => {
                self.effective_ka.to_bits().hash(&mut h);
                self.ke.to_bits().hash(&mut h);
            }
            _ => {
                self.dose_mg.to_bits().hash(&mut h);
                self.interval_hours.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }

    pub(crate) fn apply_pk_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let locked_challenge = self.active_challenge == Some(CHALLENGE_PK_THERAPEUTIC_WINDOW);
        for (k, v) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            if !v.is_finite() {
                continue;
            }
            match k {
                "dose_mg" => self.dose_mg = v.clamp(0.0, MAX_DOSE_MG),
                "interval_hours" => self.interval_hours = v.clamp(MIN_INTERVAL_H, MAX_INTERVAL_H),
                "ka" if !locked_challenge => {
                    self.absorption_mode = AbsorptionMode::DirectKa;
                    self.ka = v.clamp(MIN_RATE, MAX_KA);
                }
                "ke" if !locked_challenge => self.ke = v.clamp(MIN_RATE, MAX_KE),
                _ => {}
            }
        }
        self.recalculate();
    }
}

impl TheoryDescribable for PharmacokineticsTool {
    fn theory_description(&self) -> String {
        format!(
            "One-compartment oral ADME Bateman kinetics $C(t) = \\frac{{F D k_a}}{{V_d (k_a - k_e)}}(e^{{-k_e t}} - e^{{-k_a t}})$ with multi-dose linear superposition $C_{{\\Sigma}}(t) = \\sum_{{i}} C(t - t_i)$, biphasic two-pulse release $f_1 C(t) + f_2 C(t - t_{{\\text{{lag}}}})$, and chiral isomer mixtures ($F = {:.2}$, $D = {:.1}$ mg, $k_a = {:.3}\\text{{ h}}^{{-1}}$, $k_e = {:.3}\\text{{ h}}^{{-1}}$, $V_d = {:.1}$ L).",
            self.bioavailability, self.dose_mg, self.effective_ka, self.ke, self.volume_l
        )
    }

    fn phonetic_description(&self) -> String {
        "One-compartment Bateman pharmacokinetics and multi-dose ADME superposition theory".into()
    }

    fn theory_citation(&self) -> String {
        "Clinical Pharmacokinetics and Pharmacodynamics: Concepts and Applications (Rowland & Tozer, 2011); Über eine Erweiterung der Theorie des radioaktiven Zerfalls (Bateman, 1910)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "Bateman Function".into(),
            "First-order extravascular absorption and mono-exponential elimination model".into(),
        );
        map.insert(
            "Linear Superposition".into(),
            "Multi-dose accumulation across maintenance, loading, and missed-dose schedules".into(),
        );
        map.insert(
            "Two-Pulse & Chiral ADME".into(),
            "Biphasic extended-release pulses and stereoselective d/l-enantiomer plasma curves"
                .into(),
        );
        map
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let c = |min, max, step| ParameterConstraint { min, max, step };
        HashMap::from([
            ("dose_mg".to_string(), c(0.0, MAX_DOSE_MG, 10.0)),
            ("bioavailability".to_string(), c(0.05, 1.0, 0.01)),
            ("ka".to_string(), c(MIN_RATE, MAX_KA, 0.05)),
            ("ke".to_string(), c(MIN_RATE, MAX_KE, 0.01)),
            ("volume_l".to_string(), c(MIN_VOLUME_L, MAX_VOLUME_L, 1.0)),
            (
                "interval_hours".to_string(),
                c(MIN_INTERVAL_H, MAX_INTERVAL_H, 0.5),
            ),
            ("mec".to_string(), c(0.05, MAX_MEC, 0.1)),
            ("mtc".to_string(), c(0.1, MAX_MTC, 0.5)),
        ])
    }
}
