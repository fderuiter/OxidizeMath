//! Zero-allocation 2D parameter regime atlas for [`super::braess::BraessNetworkParadoxTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::braess::{
    BraessNetworkParadoxTool, BraessState, avg_physical_latency, solve_flows_for_demand,
};
use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent, RegimeCartographer,
    atlas_regime_color,
};
use eframe::egui;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Congestion efficiency regime across the Wheatstone `(demand, shortcut_cost / shortcut_toll)` plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BraessRegime {
    /// Shortcut carries zero selfish flow; network operates as parallel highways.
    ShortcutBypassed = 0,
    /// Selfish Wardrop equilibrium matches Pigou social optimum ($\text{PoA} \approx 1$).
    SociallyOptimal = 1,
    /// Shortcut used with excess congestion ($1 < \text{PoA}$), yet still faster than closing it.
    InefficientEquilibrium = 2,
    /// Braess Paradox slowdown: selfish latency strictly exceeds closed-shortcut latency!
    BraessParadoxSlowdown = 3,
}

impl BraessRegime {
    /// Converts a raw `u8` regime index into [`BraessRegime`].
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::ShortcutBypassed,
            1 => Self::SociallyOptimal,
            2 => Self::InefficientEquilibrium,
            _ => Self::BraessParadoxSlowdown,
        }
    }

    /// Human-readable regime title.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ShortcutBypassed => "Shortcut Unused (Parallel Split)",
            Self::SociallyOptimal => "Pigou Social Optimum (PoA ≈ 1)",
            Self::InefficientEquilibrium => "Congested Equilibrium (PoA > 1)",
            Self::BraessParadoxSlowdown => "Braess Paradox Slowdown (L_W > L_closed)",
        }
    }
}

/// Stack-allocated `[f64; 3]` Wardrop vs. Pigou congestion cartographer (`zero` heap allocations).
#[derive(Clone, Debug)]
pub struct BraessCartographer {
    axes: AtlasAxisPair,
    base_state: BraessState,
}

impl BraessCartographer {
    /// Constructs a [`BraessCartographer`] for `(x_key, y_key)`.
    #[must_use]
    pub fn new(
        theory: &HashMap<String, ParameterConstraint>,
        keys: (&'static str, &'static str),
        base_state: BraessState,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: braess_axis_spec(theory, keys.0),
                y: braess_axis_spec(theory, keys.1),
            },
            base_state,
        }
    }
}

fn braess_axis_spec(
    theory: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    match key {
        "shortcut_cost" => AtlasAxisSpec::from_theory(
            theory,
            "shortcut_cost",
            "Shortcut Base Cost (c_AB)",
            ParameterConstraint {
                min: 0.0,
                max: 1.5,
                step: 0.02,
            },
        ),
        "shortcut_toll" => AtlasAxisSpec::from_theory(
            theory,
            "shortcut_toll",
            "Pigouvian Shortcut Toll (τ)",
            ParameterConstraint {
                min: 0.0,
                max: 1.5,
                step: 0.02,
            },
        ),
        _ => AtlasAxisSpec::from_theory(
            theory,
            "demand",
            "Commuter Demand (D)",
            ParameterConstraint {
                min: 0.2,
                max: 2.0,
                step: 0.02,
            },
        ),
    }
}

impl RegimeCartographer for BraessCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let mut probe = BraessState {
            shortcut_open: true,
            ..self.base_state.clone()
        };
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "demand" => probe.demand = v.clamp(0.2, 2.0),
                "shortcut_cost" => probe.shortcut_cost = v.clamp(0.0, 1.5),
                "shortcut_toll" => probe.shortcut_toll = v.clamp(0.0, 1.5),
                _ => {}
            }
        }
        let (wardrop, pigou, closed_lat) = solve_flows_for_demand(&probe, probe.demand);
        let wardrop_lat = avg_physical_latency(wardrop, &probe);
        let pigou_lat = avg_physical_latency(pigou, &probe);
        let poa = wardrop_lat / pigou_lat.max(1e-9);

        let regime = if wardrop_lat > closed_lat + 1e-4 {
            BraessRegime::BraessParadoxSlowdown
        } else if poa > 1.002 {
            BraessRegime::InefficientEquilibrium
        } else if wardrop[2] <= 1e-5 {
            BraessRegime::ShortcutBypassed
        } else {
            BraessRegime::SociallyOptimal
        };
        AtlasCell {
            regime: regime as u8,
            primary: poa,
            secondary: wardrop_lat - closed_lat,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let excess = ((cell.primary - 1.0) / 0.35).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, excess.max(0.25), high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        BraessRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Price of Anarchy (PoA)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {:.3}, {} = {:.3}\nRegime: {}\nPrice of Anarchy = {:.4}  |  L_Wardrop - L_closed = {:+.4}",
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
            BraessRegime::BraessParadoxSlowdown,
            BraessRegime::InefficientEquilibrium,
            BraessRegime::SociallyOptimal,
            BraessRegime::ShortcutBypassed,
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

impl BraessNetworkParadoxTool {
    /// Applies a 2D atlas teleport across `100%` of slider bounds and synchronizes `CalloutManager`.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "demand" => self.state.demand = val.clamp(0.2, 2.0),
                "shortcut_cost" => self.state.shortcut_cost = val.clamp(0.0, 1.5),
                "shortcut_toll" => self.state.shortcut_toll = val.clamp(0.0, 1.5),
                _ => {}
            }
        }
        self.state.shortcut_open = true;
        self.recompute_all();
        let target = if self.state.continuum_mfg_mode {
            [
                0.0,
                self.mfg_m_final
                    .get(self.mfg_m_final.len() / 2)
                    .map_or(0.5, |p| p[1]),
            ]
        } else {
            [self.state.demand, self.metrics.wardrop_latency]
        };
        self.callouts.set_auto_callout(
            "atlas_teleport",
            target,
            format!(
                "Atlas: {} (PoA {:.3})",
                ev.regime_label, self.metrics.price_of_anarchy
            ),
            [18.0, -26.0],
        );
    }

    pub(super) fn render_braess_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let val_of = |k: &str| match k {
            "demand" => self.state.demand,
            "shortcut_cost" => self.state.shortcut_cost,
            "shortcut_toll" => self.state.shortcut_toll,
            _ => 0.0,
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.state.congestion_slope.to_bits().hash(&mut hasher);
        self.state.bypass_latency.to_bits().hash(&mut hasher);
        for (k, v) in [
            ("demand", self.state.demand),
            ("shortcut_cost", self.state.shortcut_cost),
            ("shortcut_toll", self.state.shortcut_toll),
        ] {
            if k != xk && k != yk {
                v.to_bits().hash(&mut hasher);
            }
        }
        let theory = self.theory_parameters();
        let base_state = self.state.clone();
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "braess_network_atlas",
            val_of(xk),
            val_of(yk),
            hasher.finish(),
            move |keys| Arc::new(BraessCartographer::new(&theory, keys, base_state)),
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}

// [cite:paradox_lab]
