//! Zero-allocation 2D confounding regime atlas for [`super::simpson_paradox::SimpsonParadoxLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::simpson_paradox::{
    SimpsonParadoxLab, TableCounts, MILD_CTRL_RATE, MILD_TREAT_RATE, SEVERE_CTRL_RATE,
    SEVERE_TREAT_RATE,
};
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Confounding regime across the `(mild_treatment_share, severe_treatment_share)` allocation plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SimpsonRegime {
    /// Non-reversed harmful pooled estimate ($\text{RR}_{\text{pool}} > 1.03$).
    ConsistentHarmful = 0,
    /// Unconfounded / balanced protective pooled estimate ($\text{RR}_{\text{pool}} < 0.97$).
    BalancedProtective = 1,
    /// Confounding washes out the protective signal ($\text{RR}_{\text{pool}} \approx 1.0$).
    ConfoundedNullWashout = 2,
    /// Full Simpson's Paradox reversal ($\text{RR}_1 < 1, \text{RR}_2 < 1$, yet $\text{RR}_{\text{pool}} > 1$).
    SimpsonReversal = 3,
}

impl SimpsonRegime {
    /// Converts a raw `u8` regime index into [`SimpsonRegime`].
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::ConsistentHarmful,
            1 => Self::BalancedProtective,
            2 => Self::ConfoundedNullWashout,
            _ => Self::SimpsonReversal,
        }
    }

    /// Human-readable regime title.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ConsistentHarmful => "Non-Reversed Harm (RR_pool > 1)",
            Self::BalancedProtective => "Balanced Protective (RR_pool < 0.97)",
            Self::ConfoundedNullWashout => "Confounded Null Washout (RR_pool ≈ 1)",
            Self::SimpsonReversal => "Simpson's Reversal (RR₁,₂ < 1, RR_pool > 1)",
        }
    }
}

/// Pure-stack 2×2 contingency relative-risk cartographer (`zero` heap allocations per cell).
#[derive(Clone, Debug)]
pub struct SimpsonCartographer {
    axes: AtlasAxisPair,
    stratum_size: u32,
}

impl SimpsonCartographer {
    /// Constructs a [`SimpsonCartographer`] for `(x_key, y_key)` at fixed `stratum_size`.
    #[must_use]
    pub fn new(
        theory: &HashMap<String, ParameterConstraint>,
        keys: (&'static str, &'static str),
        stratum_size: u32,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: simpson_axis_spec(theory, keys.0),
                y: simpson_axis_spec(theory, keys.1),
            },
            stratum_size: stratum_size.clamp(20, 800),
        }
    }
}

fn simpson_axis_spec(
    theory: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    match key {
        "severe_treatment_share" => AtlasAxisSpec::from_theory(
            theory,
            "severe_treatment_share",
            "Severe Stratum Treatment Share",
            ParameterConstraint {
                min: 0.10,
                max: 0.90,
                step: 0.01,
            },
        ),
        _ => AtlasAxisSpec::from_theory(
            theory,
            "mild_treatment_share",
            "Mild Stratum Treatment Share",
            ParameterConstraint {
                min: 0.10,
                max: 0.90,
                step: 0.01,
            },
        ),
    }
}

#[inline]
fn exact_relative_risk(c: TableCounts) -> f64 {
    let treat_tot = (c.treat_event + c.treat_no_event).max(1) as f64;
    let ctrl_tot = (c.ctrl_event + c.ctrl_no_event).max(1) as f64;
    let p_treat = c.treat_event as f64 / treat_tot;
    let p_ctrl = (c.ctrl_event as f64 / ctrl_tot).max(1e-9);
    p_treat / p_ctrl
}

impl RegimeCartographer for SimpsonCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (mut mild_share, mut sev_share) = (0.20, 0.80);
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "mild_treatment_share" => mild_share = v.clamp(0.10, 0.90),
                "severe_treatment_share" => sev_share = v.clamp(0.10, 0.90),
                _ => {}
            }
        }
        let mild = TableCounts::from_share_and_rates(
            self.stratum_size,
            mild_share,
            MILD_TREAT_RATE,
            MILD_CTRL_RATE,
        );
        let sev = TableCounts::from_share_and_rates(
            self.stratum_size,
            sev_share,
            SEVERE_TREAT_RATE,
            SEVERE_CTRL_RATE,
        );
        let pool = mild.pooled_with(sev);
        let rr_mild = exact_relative_risk(mild);
        let rr_sev = exact_relative_risk(sev);
        let rr_pool = exact_relative_risk(pool);
        let reversal = (rr_mild < 1.0 && rr_sev < 1.0 && rr_pool > 1.0)
            || (rr_mild > 1.0 && rr_sev > 1.0 && rr_pool < 1.0);
        let regime = if reversal {
            SimpsonRegime::SimpsonReversal
        } else if (rr_pool - 1.0).abs() <= 0.03 {
            SimpsonRegime::ConfoundedNullWashout
        } else if rr_pool < 0.97 {
            SimpsonRegime::BalancedProtective
        } else {
            SimpsonRegime::ConsistentHarmful
        };
        AtlasCell {
            regime: regime as u8,
            primary: rr_pool,
            secondary: rr_pool - 0.5 * (rr_mild + rr_sev),
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let strength = ((cell.primary - 1.0).abs() / 0.35).clamp(0.25, 1.0);
        atlas_regime_color(cell.regime, 4, strength, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        SimpsonRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Pooled Relative Risk (RR_pool)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {:.2}, {} = {:.2}\nRegime: {}\nPooled RR = {:.3}  |  Confounding Shift = {:+.3}",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            SimpsonRegime::SimpsonReversal,
            SimpsonRegime::ConfoundedNullWashout,
            SimpsonRegime::BalancedProtective,
            SimpsonRegime::ConsistentHarmful,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 4, 0.75, high_contrast),
            )
        })
        .collect()
    }
}

/// Canonical `ParameterConstraint` table for Simpson's Paradox allocation axes.
#[must_use]
pub fn simpson_theory_parameters() -> HashMap<String, ParameterConstraint> {
    HashMap::from([
        (
            "mild_treatment_share".to_string(),
            ParameterConstraint {
                min: 0.10,
                max: 0.90,
                step: 0.01,
            },
        ),
        (
            "severe_treatment_share".to_string(),
            ParameterConstraint {
                min: 0.10,
                max: 0.90,
                step: 0.01,
            },
        ),
        (
            "stratum_size".to_string(),
            ParameterConstraint {
                min: 20.0,
                max: 800.0,
                step: 10.0,
            },
        ),
    ])
}

impl SimpsonParadoxLab {
    /// Applies a 2D atlas teleport across `100%` of allocation slider bounds and pins `CalloutManager`.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "mild_treatment_share" => self.config.mild_treatment_share = val.clamp(0.10, 0.90),
                "severe_treatment_share" => {
                    self.config.severe_treatment_share = val.clamp(0.10, 0.90);
                }
                _ => {}
            }
        }
        self.apply_allocation_sliders();
        if let Some(ref pool) = self.pooled_eval {
            self.callout_manager.set_auto_callout(
                "atlas_teleport",
                [pool.risk.relative_risk, 1.0],
                format!(
                    "Atlas: {} (RR = {:.2})",
                    ev.regime_label, pool.risk.relative_risk
                ),
                [22.0, 18.0],
            );
        }
    }

    pub(super) fn render_simpson_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let val_of = |k: &str| match k {
            "mild_treatment_share" => self.config.mild_treatment_share,
            "severe_treatment_share" => self.config.severe_treatment_share,
            _ => 0.50,
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.config.stratum_size.hash(&mut hasher);
        let theory = simpson_theory_parameters();
        let n_stratum = self.config.stratum_size;
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "simpson_paradox_atlas",
            val_of(xk),
            val_of(yk),
            hasher.finish(),
            move |keys| Arc::new(SimpsonCartographer::new(&theory, keys, n_stratum)),
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}

// [cite:clinical_trials_statistics]
