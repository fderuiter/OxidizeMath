//! Zero-allocation 2D parameter regime atlas for [`super::parrondo::ParrondoJumpRatchetTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::parrondo::{ParrondoParadoxTool, ParrondoVariant};
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
pub use math_explorer::pure_math::statistics::markov::parrondo::evaluate_jump_ratchet_stack;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Discrete dynamical drift regime for `ParrondoJumpRatchetTool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ParrondoJumpRegime {
    /// Combined mixture or SSA jump-ratchet drift is strictly negative.
    LosingDrift = 0,
    /// Paradox active: both pure games lose ($\mu_A < 0, \mu_B < 0$) while coupled ratchet wins.
    ParadoxRatchet = 1,
    /// Near-zero fair drift boundary ($|\mu| \le 10^{-4}$).
    NeutralBoundary = 2,
    /// Positive drift where at least one pure game already has non-negative expectation.
    DirectPositiveDrift = 3,
}

impl ParrondoJumpRegime {
    /// Converts a raw `u8` regime index into [`ParrondoJumpRegime`].
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::LosingDrift,
            1 => Self::ParadoxRatchet,
            2 => Self::NeutralBoundary,
            _ => Self::DirectPositiveDrift,
        }
    }

    /// Human-readable regime label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LosingDrift => "Losing Drift (μ < 0)",
            Self::ParadoxRatchet => "Paradox Ratchet (μ_A<0, μ_B<0, μ>0)",
            Self::NeutralBoundary => "Critical Fair Boundary (μ ≈ 0)",
            Self::DirectPositiveDrift => "Unbiased / Direct Positive Drift",
        }
    }
}

/// Stack-allocated 2D cartographer for discrete convex mixtures and continuous-time jump ratchets.
#[derive(Clone, Debug)]
pub struct ParrondoJumpCartographer {
    axes: AtlasAxisPair,
    variant: ParrondoVariant,
    base_eps: f64,
    base_gamma: f64,
    base_lambda: f64,
}

impl ParrondoJumpCartographer {
    /// Constructs a [`ParrondoJumpCartographer`] for `(x_key, y_key)`.
    #[must_use]
    pub fn new(
        theory: &HashMap<String, ParameterConstraint>,
        keys: (&'static str, &'static str),
        variant: ParrondoVariant,
        base_eps: f64,
        base_gamma: f64,
        base_lambda: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: jump_axis_spec(theory, keys.0),
                y: jump_axis_spec(theory, keys.1),
            },
            variant,
            base_eps,
            base_gamma,
            base_lambda,
        }
    }
}

fn jump_axis_spec(
    theory: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    match key {
        "mix_gamma" => AtlasAxisSpec::from_theory(
            theory,
            "mix_gamma",
            "Mixture Weight γ (Game A)",
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        ),
        "jump_switch_rate" => AtlasAxisSpec::from_theory(
            theory,
            "jump_switch_rate",
            "SSA Jump Rate λ",
            ParameterConstraint {
                min: 0.1,
                max: 5.0,
                step: 0.05,
            },
        ),
        _ => AtlasAxisSpec::from_theory(
            theory,
            "bias_epsilon",
            "Coin Bias Disadvantage (ε)",
            ParameterConstraint {
                min: 0.0,
                max: 0.05,
                step: 0.0005,
            },
        ),
    }
}

impl RegimeCartographer for ParrondoJumpCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let mut eps = self.base_eps;
        let mut gamma = self.base_gamma;
        let mut lambda = self.base_lambda;
        let mut uses_ssa = false;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "bias_epsilon" => eps = v,
                "mix_gamma" => gamma = v,
                "jump_switch_rate" => {
                    lambda = v;
                    uses_ssa = true;
                }
                _ => {}
            }
        }
        let (da, db, dmix, dabb, dssa) =
            evaluate_jump_ratchet_stack(self.variant, eps, gamma, lambda);
        let primary = if uses_ssa { dssa } else { dmix };
        let regime = if primary.abs() <= 1e-4 {
            ParrondoJumpRegime::NeutralBoundary
        } else if primary < 0.0 {
            ParrondoJumpRegime::LosingDrift
        } else if da < -1e-6 && db < -1e-6 {
            ParrondoJumpRegime::ParadoxRatchet
        } else {
            ParrondoJumpRegime::DirectPositiveDrift
        };
        AtlasCell {
            regime: regime as u8,
            primary,
            secondary: dabb,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (cell.primary.abs() / 0.04).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        ParrondoJumpRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        if self.axes.y.key == "jump_switch_rate" || self.axes.x.key == "jump_switch_rate" {
            "SSA Jump Drift μ(SSA)"
        } else {
            "Mixture Drift μ(Mix)"
        }
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {:.4}, {} = {:.4}\nRegime: {}\n{} = {:+.5}  |  Periodic μ(ABB) = {:+.5}",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            self.primary_metric_label(),
            cell.primary,
            cell.secondary
        )
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            ParrondoJumpRegime::ParadoxRatchet,
            ParrondoJumpRegime::DirectPositiveDrift,
            ParrondoJumpRegime::NeutralBoundary,
            ParrondoJumpRegime::LosingDrift,
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

impl ParrondoParadoxTool {
    /// Applies a 2D atlas teleport across `100%` of slider bounds and synchronizes `CalloutManager`.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "bias_epsilon" => self.state.bias_epsilon = val.clamp(0.0, 0.05),
                "mix_gamma" => self.state.mix_gamma = val.clamp(0.0, 1.0),
                "jump_switch_rate" => self.state.jump_switch_rate = val.clamp(0.1, 5.0),
                _ => {}
            }
        }
        self.rebuild_full_simulation();
        let x = self.state.steps as f64;
        let y = x * self.metrics.drift_mix;
        self.callouts.set_auto_callout(
            "atlas_teleport",
            [x, y],
            format!("Atlas: {}", ev.regime_label),
            [-110.0, -26.0],
        );
    }

    pub(super) fn render_jump_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let val_of = |k: &str| match k {
            "bias_epsilon" => self.state.bias_epsilon,
            "mix_gamma" => self.state.mix_gamma,
            "jump_switch_rate" => self.state.jump_switch_rate,
            _ => 0.0,
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.state.variant as u8).hash(&mut hasher);
        for (k, v) in [
            ("bias_epsilon", self.state.bias_epsilon),
            ("mix_gamma", self.state.mix_gamma),
            ("jump_switch_rate", self.state.jump_switch_rate),
        ] {
            if k != xk && k != yk {
                v.to_bits().hash(&mut hasher);
            }
        }
        let theory = self.theory_parameters();
        let variant = self.state.variant;
        let (eps, gamma, lambda) = (
            self.state.bias_epsilon,
            self.state.mix_gamma,
            self.state.jump_switch_rate,
        );
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "parrondo_jump_atlas",
            val_of(xk),
            val_of(yk),
            hasher.finish(),
            move |keys| {
                Arc::new(ParrondoJumpCartographer::new(
                    &theory, keys, variant, eps, gamma, lambda,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}

// [cite:paradox_lab]
