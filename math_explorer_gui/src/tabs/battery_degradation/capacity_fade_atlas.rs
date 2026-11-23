//! Zero-allocation lithium-ion state-of-health (SOH) parameter cartographer
//! for [`super::capacity_fade::CapacityFadeTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::capacity_fade::CapacityFadeTool;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use math_explorer::applied::battery_degradation::{Cycles, DepthOfDischarge, PowerLawModel};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

/// Supported 2D parameter atlas axis pairs for [`CapacityFadeTool`].
pub const CAPACITY_FADE_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("dod", "cycles_to_simulate")];

/// Stack-resident zero-allocation battery state-of-health regime cartographer.
pub struct CapacityFadeCartographer {
    axes: AtlasAxisPair,
    model: PowerLawModel,
}

impl CapacityFadeCartographer {
    /// Builds a [`CapacityFadeCartographer`] for `("dod", "cycles_to_simulate")`.
    #[must_use]
    pub fn new(tool: &CapacityFadeTool, axis_keys: (&'static str, &'static str)) -> Option<Self> {
        if axis_keys != ("dod", "cycles_to_simulate") {
            return None;
        }
        let theory = tool.theory_parameters();
        let mut x = AtlasAxisSpec::from_theory(
            &theory,
            "dod",
            "Depth of Discharge DoD (%)",
            ParameterConstraint {
                min: 5.0,
                max: 100.0,
                step: 1.0,
                default: Some(80.0),
            },
        );
        x.constraint.min = x.constraint.min.max(5.0);
        let y = AtlasAxisSpec::from_theory(
            &theory,
            "cycles_to_simulate",
            "Cycle Horizon N (cycles)",
            ParameterConstraint {
                min: 100.0,
                max: 10_000.0,
                step: 50.0,
                default: Some(2000.0),
            },
        );
        Some(Self {
            axes: AtlasAxisPair { x, y },
            model: PowerLawModel::standard(),
        })
    }
}

impl RegimeCartographer for CapacityFadeCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let soh = match (
            Cycles::new(p2.max(0.0)),
            DepthOfDischarge::new(p1.clamp(0.0, 100.0)),
        ) {
            (Ok(cycles), Ok(dod)) => self.model.capacity(cycles, dod).as_f64(),
            _ => 0.0,
        };
        let fade_pct = (1.0 - soh).max(0.0) * 100.0;
        let regime = if soh >= 0.80 {
            1
        } else if soh >= 0.70 {
            2
        } else {
            3
        };
        AtlasCell {
            regime,
            primary: soh * 100.0,
            secondary: fade_pct,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (cell.primary / 100.0).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        match regime {
            1 => "First-Life Healthy (SOH ≥ 80%)",
            2 => "Second-Life Window (70% ≤ SOH < 80%)",
            _ => "End-of-Life Depleted (SOH < 70%)",
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Retained Capacity SOH (%)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "DoD = {p1:.1}%, Cycles = {p2:.0}\nRegime: {}\nSOH = {:.1}% | Fade = {:.1}%",
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        let mut pts_80 = Vec::with_capacity(48);
        let steps = 47;
        for i in 0..=steps {
            let frac = (i as f64) / (steps as f64);
            let dod = self.axes.x.constraint.min + frac * self.axes.x.span();
            let (n80, _) = CapacityFadeTool::projected_life_cycles(dod);
            if (self.axes.y.constraint.min..=self.axes.y.constraint.max).contains(&n80) {
                pts_80.push([dod, n80]);
            }
        }
        (pts_80.len() >= 2).then_some(("First-Life Knee (80% SOH)", pts_80))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [1_u8, 2, 3]
            .into_iter()
            .map(|r| {
                (
                    self.regime_label(r),
                    atlas_regime_color(r, 4, 0.75, high_contrast),
                )
            })
            .collect()
    }
}

impl CapacityFadeTool {
    pub(crate) fn apply_capacity_fade_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (k, v) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            if !v.is_finite() {
                continue;
            }
            match k {
                "dod" => self.dod = v.clamp(1.0, 100.0),
                "cycles_to_simulate" => self.cycles_to_simulate = v.clamp(100.0, 10_000.0),
                _ => {}
            }
        }
    }
}
