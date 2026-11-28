//! 2D [`RegimeCartographer`] for [`super::ParrondoParadoxTool`] (`Parrondo's Kelly Ratchet`)
//! backed by [`math_explorer::pure_math::statistics::markov::parrondo::solve_parrondo_ratchet_stack`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ParrondoParadoxTool;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
pub use math_explorer::pure_math::statistics::markov::parrondo::{
    solve_parrondo_ratchet_stack, ParrondoGame, MAX_SCHEDULE_LEN,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

/// Supported 2D parameter atlas axis pairs for [`ParrondoParadoxTool`].
pub const PARRONDO_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[
    ("p_a", "epsilon"),
    ("p_a_base", "epsilon"),
    ("p_b0_base", "p_b1_base"),
];

pub(super) const WEDGE_PA_MIN: f64 = 0.42;
pub(super) const WEDGE_PA_MAX: f64 = 0.51;
pub(super) const WEDGE_EPS_MIN: f64 = -0.02;
pub(super) const WEDGE_EPS_MAX: f64 = 0.05;

/// Zero-allocation 2D parameter cartographer for [`ParrondoParadoxTool`].
#[derive(Clone, Debug)]
pub struct ParrondoCartographer {
    axes: AtlasAxisPair,
    modulus: usize,
    p_a_base: f64,
    p_b0_base: f64,
    p_b1_base: f64,
    epsilon: f64,
    schedule: [ParrondoGame; MAX_SCHEDULE_LEN],
    schedule_len: usize,
}

fn parrondo_axis_pair(
    theory: &std::collections::HashMap<String, ParameterConstraint>,
    axis_keys: (&'static str, &'static str),
) -> Option<AtlasAxisPair> {
    let c = |min, max, step| ParameterConstraint { min, max, step };
    match axis_keys {
        ("p_a", "epsilon") => Some(AtlasAxisPair {
            x: AtlasAxisSpec::from_theory(
                theory,
                "p_a",
                "Effective Game A Win Prob p_A",
                c(WEDGE_PA_MIN, WEDGE_PA_MAX, 0.002),
            ),
            y: AtlasAxisSpec::from_theory(
                theory,
                "epsilon",
                "Bias Perturbation ε",
                c(WEDGE_EPS_MIN, 0.045, 0.0005),
            ),
        }),
        ("p_a_base", "epsilon") => Some(AtlasAxisPair {
            x: AtlasAxisSpec::from_theory(
                theory,
                "p_a_base",
                "Game A Base Win Prob p_A,0",
                c(0.35, 0.65, 0.005),
            ),
            y: AtlasAxisSpec::from_theory(
                theory,
                "epsilon",
                "Bias Perturbation ε",
                c(WEDGE_EPS_MIN, 0.045, 0.0005),
            ),
        }),
        ("p_b0_base", "p_b1_base") => Some(AtlasAxisPair {
            x: AtlasAxisSpec::from_theory(
                theory,
                "p_b0_base",
                "Trap Win Prob p_B0,0 (r = 0)",
                c(0.03, 0.50, 0.005),
            ),
            y: AtlasAxisSpec::from_theory(
                theory,
                "p_b1_base",
                "Non-Trap Win Prob p_B1,0 (r ≠ 0)",
                c(0.50, 0.95, 0.005),
            ),
        }),
        _ => None,
    }
}

impl ParrondoCartographer {
    /// Constructs a stack-resident [`ParrondoCartographer`] for any supported [`PARRONDO_ATLAS_AXIS_PAIRS`] pair.
    #[must_use]
    pub fn new(
        tool: &ParrondoParadoxTool,
        axis_keys: (&'static str, &'static str),
    ) -> Option<Self> {
        let theory = tool.theory_parameters();
        let axes = parrondo_axis_pair(&theory, axis_keys)?;
        let mut schedule = [ParrondoGame::GameA; MAX_SCHEDULE_LEN];
        let steps = tool.active_schedule.steps();
        let schedule_len = steps.len().min(MAX_SCHEDULE_LEN);
        schedule[..schedule_len].copy_from_slice(&steps[..schedule_len]);
        Some(Self {
            axes,
            modulus: tool.modulus,
            p_a_base: tool.p_a_base,
            p_b0_base: tool.p_b0_base,
            p_b1_base: tool.p_b1_base,
            epsilon: tool.epsilon,
            schedule,
            schedule_len,
        })
    }
}

impl RegimeCartographer for ParrondoCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (p_a_base, p_b0_base, p_b1_base, eps) = match (self.axes.x.key, self.axes.y.key) {
            ("p_a", "epsilon") => (p1 + p2, self.p_b0_base, self.p_b1_base, p2),
            ("p_a_base", "epsilon") => (p1, self.p_b0_base, self.p_b1_base, p2),
            _ => (self.p_a_base, p1, p2, self.epsilon),
        };
        let Some(summary) = solve_parrondo_ratchet_stack(
            self.modulus,
            p_a_base,
            p_b0_base,
            p_b1_base,
            eps,
            &self.schedule[..self.schedule_len],
        ) else {
            return AtlasCell {
                regime: 3,
                primary: -1.0,
                secondary: 0.0,
            };
        };
        let regime = if summary.is_paradox {
            1
        } else if summary.gain_schedule.abs() <= 1e-4 {
            2
        } else if summary.gain_schedule > 0.0 {
            0
        } else {
            3
        };
        AtlasCell {
            regime,
            primary: summary.gain_schedule,
            secondary: summary.pi_sched_trap,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (0.5 + 15.0 * cell.primary).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        match regime {
            0 => "Positive Drift (Non-Paradox)",
            1 => "Paradox Wedge (A<0, B<0, Sched>0)",
            2 => "Fair Ratchet Boundary (|g| ≤ 1e-4)",
            _ => "Negative Ratchet Drift (E[ΔW] < 0)",
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Schedule Drift E[ΔW]"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {p1:.4}, {} = {p2:.4}\nRegime: {}\nE[ΔW_sched] = {:+.5} | Trap π(0) = {:.4}",
            self.axes.x.label,
            self.axes.y.label,
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        let (y_min, y_max) = (self.axes.y.constraint.min, self.axes.y.constraint.max);
        match (self.axes.x.key, self.axes.y.key) {
            ("p_a", "epsilon") => {
                Some(("Fair Game A (p_A = 0.5)", vec![[0.5, y_min], [0.5, y_max]]))
            }
            ("p_a_base", "epsilon") => Some((
                "Fair Game A (p_A,0 - ε = 0.5)",
                vec![[0.5 + y_min, y_min], [0.5 + y_max, y_max]],
            )),
            _ => None,
        }
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

// [cite:clinical_trials_statistics]
