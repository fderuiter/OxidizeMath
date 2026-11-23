#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    solve_wardrop_in_place, AffineLatency, BraessCongestionLab, ShortcutPrediction, WardropError,
    DEMAND_CURVE_SAMPLES,
};
use crate::accessibility::AccessibleHoverText;
use eframe::egui::{self, Pos2};
use egui_plot::{Line, Plot, PlotPoint, Points};
use std::collections::BTreeMap;

/// Actions requested by the Braess sidebar controls for `GraphEditorTool` to execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BraessUiAction {
    /// No topology mutation requested.
    None,
    /// Load the 4-node Braess diamond preset (`s=0, u=1, v=2, t=3`).
    LoadPreset,
    /// Toggle the zero-cost `u -> v` (`1 -> 2`) shortcut edge.
    ToggleShortcut(bool),
    /// Recompute equilibrium after affine parameter edits.
    Recompute,
}

impl BraessCongestionLab {
    /// Synchronizes `edge_latencies` length with `edges`, clamping negative coefficients.
    pub fn sync_with_edges(&mut self, edges: &[(usize, usize, f64)]) {
        while self.edge_latencies.len() < edges.len() {
            let idx = self.edge_latencies.len();
            let w = edges[idx].2;
            self.edge_latencies.push(AffineLatency::clamped(0.0, w));
        }
        self.edge_latencies.truncate(edges.len());
        for lat in &mut self.edge_latencies {
            *lat = AffineLatency::clamped(lat.a, lat.b);
        }
    }

    /// Returns true if any non-default congestion state needs JSON persistence.
    pub fn has_custom_state(&self, edges: &[(usize, usize, f64)]) -> bool {
        if !self.config.is_default() {
            return true;
        }
        self.edge_latencies.iter().enumerate().any(|(i, lat)| {
            let w = edges.get(i).map_or(1.0, |e| e.2);
            lat.a.abs() > 1e-12 || (lat.b - w).abs() > 1e-12
        })
    }

    /// Recomputes Wardrop equilibrium, counterfactual shortcut delta, and `CalloutManager`.
    pub fn recompute(
        &mut self,
        node_positions: &BTreeMap<usize, Pos2>,
        edges: &[(usize, usize, f64)],
    ) {
        self.sync_with_edges(edges);
        if !self.config.total_demand.is_finite() {
            self.config.total_demand = 1.0;
        } else {
            self.config.total_demand = self.config.total_demand.clamp(0.1, 10.0);
        }
        let m = edges.len();
        self.equilibrium_flows.resize(m, 0.0);
        self.equilibrium_latencies.resize(m, 0.0);
        self.scratch
            .prepare_topology(node_positions, edges, &self.edge_latencies);

        let shortcut_idx = edges.iter().position(|&(u, v, _)| u == 1 && v == 2);
        self.config.shortcut_enabled = shortcut_idx.is_some();

        self.status = solve_wardrop_in_place(
            &mut self.scratch,
            self.config.source_node,
            self.config.sink_node,
            self.config.total_demand,
            None,
            &mut self.equilibrium_flows,
            &mut self.equilibrium_latencies,
        );

        self.without_shortcut_latency = shortcut_idx.and_then(|skip| {
            let mut flows = std::mem::take(&mut self.scratch.baseline_flows);
            let mut lats = std::mem::take(&mut self.scratch.baseline_latencies);
            let res = solve_wardrop_in_place(
                &mut self.scratch,
                self.config.source_node,
                self.config.sink_node,
                self.config.total_demand,
                Some(skip),
                &mut flows,
                &mut lats,
            )
            .ok()
            .map(|s| s.commuter_latency);
            self.scratch.baseline_flows = flows;
            self.scratch.baseline_latencies = lats;
            res
        });

        self.rebuild_demand_curves(shortcut_idx);
        self.sync_callouts();
    }

    fn rebuild_demand_curves(&mut self, shortcut_idx: Option<usize>) {
        self.active_curve.clear();
        self.no_shortcut_curve.clear();
        if self.status.is_err() {
            return;
        }
        let mut flows = std::mem::take(&mut self.scratch.sweep_flows);
        let mut lats = std::mem::take(&mut self.scratch.sweep_latencies);
        for i in 1..=DEMAND_CURVE_SAMPLES {
            let d = (i as f64) * 0.10;
            if let Ok(act) = solve_wardrop_in_place(
                &mut self.scratch,
                self.config.source_node,
                self.config.sink_node,
                d,
                None,
                &mut flows,
                &mut lats,
            ) {
                self.active_curve
                    .push(PlotPoint::new(d, act.commuter_latency));
            }
            if let Some(skip) = shortcut_idx {
                if let Ok(base) = solve_wardrop_in_place(
                    &mut self.scratch,
                    self.config.source_node,
                    self.config.sink_node,
                    d,
                    Some(skip),
                    &mut flows,
                    &mut lats,
                ) {
                    self.no_shortcut_curve
                        .push(PlotPoint::new(d, base.commuter_latency));
                }
            }
        }
        self.scratch.sweep_flows = flows;
        self.scratch.sweep_latencies = lats;
    }

    fn sync_callouts(&mut self) {
        let Ok(summary) = self.status else {
            self.callout_manager.clear_auto();
            return;
        };
        let d = self.config.total_demand;
        if let Some(base) = self
            .without_shortcut_latency
            .filter(|&b| summary.commuter_latency > b + 1e-6)
        {
            self.callout_manager.set_auto_callout(
                "nash_trap",
                [d, summary.commuter_latency],
                format!(
                    "Nash Trap: {:.2} -> {:.2} (+{:.2})",
                    base,
                    summary.commuter_latency,
                    summary.commuter_latency - base
                ),
                [-35.0, -24.0],
            );
        } else {
            self.callout_manager.set_auto_callout(
                "nash_trap",
                [d, summary.commuter_latency],
                format!("Wardrop Latency: {:.2}", summary.commuter_latency),
                [20.0, -20.0],
            );
        }
    }

    /// Renders sidebar controls for Braess's paradox and affine edge latencies.
    pub fn show_sidebar(
        &mut self,
        ui: &mut egui::Ui,
        edges: &mut [(usize, usize, f64)],
    ) -> BraessUiAction {
        let mut action = BraessUiAction::None;
        ui.heading("Braess Congestion Lab");
        if ui
            .checkbox(
                &mut self.config.enabled,
                "Enable Wardrop Congestion Routing",
            )
            .changed()
        {
            action = BraessUiAction::Recompute;
        }
        if ui
            .button("🚗 Load Braess Diamond Preset (s -> u, v -> t)")
            .accessible_hover_text(
                "Populate 4-node Braess diamond network with affine latencies l(f)=f and l(f)=1",
            )
            .clicked()
        {
            return BraessUiAction::LoadPreset;
        }
        if !self.config.enabled {
            return action;
        }

        self.render_hypothesis_box(ui);
        if let Some(next_act) = self.render_routing_controls(ui, edges) {
            action = next_act;
        }
        self.render_equilibrium_readout(ui);
        self.render_demand_plot(ui);
        let _ = self.callout_manager.render_ui(ui);
        action
    }

    fn render_hypothesis_box(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.group(|ui| {
            ui.label(
                egui::RichText::new(
                    "Predict Before Toggling: What happens to commuter travel time when opening the zero-cost u -> v shortcut?",
                )
                .small()
                .strong(),
            );
            ui.radio_value(
                &mut self.config.prediction,
                Some(ShortcutPrediction::SpeedsUpCommuters),
                "Commuters Arrive Faster",
            );
            ui.radio_value(
                &mut self.config.prediction,
                Some(ShortcutPrediction::NoChange),
                "Travel Time Unchanged (1.50)",
            );
            ui.radio_value(
                &mut self.config.prediction,
                Some(ShortcutPrediction::SlowsAllCommutersNashTrap),
                "Everyone Slows Down (1.50 -> 2.00 Nash Trap!)",
            );
            if let Some(pred) = self.config.prediction {
                let ok = pred == ShortcutPrediction::SlowsAllCommutersNashTrap;
                let col = if ok { palette.success } else { palette.warning };
                ui.colored_label(
                    col,
                    "Selfish routing abandons the 50/50 split (1.50) for s->u->v->t, congesting both bottlenecks to 2.00!",
                );
            }
        });
    }

    fn render_routing_controls(
        &mut self,
        ui: &mut egui::Ui,
        edges: &mut [(usize, usize, f64)],
    ) -> Option<BraessUiAction> {
        let mut out = None;
        let mut shortcut = self.config.shortcut_enabled;
        if ui
            .checkbox(
                &mut shortcut,
                "⚡ Enable Zero-Cost u -> v Shortcut (1 -> 2)",
            )
            .accessible_hover_text(
                "Toggle zero-latency shortcut edge 1 -> 2 to test Braess's paradox",
            )
            .changed()
        {
            return Some(BraessUiAction::ToggleShortcut(shortcut));
        }

        ui.horizontal(|ui| {
            ui.label("Source s:");
            if ui
                .add(egui::DragValue::new(&mut self.config.source_node))
                .changed()
            {
                out = Some(BraessUiAction::Recompute);
            }
            ui.label("Sink t:");
            if ui
                .add(egui::DragValue::new(&mut self.config.sink_node))
                .changed()
            {
                out = Some(BraessUiAction::Recompute);
            }
        });
        if ui
            .add(egui::Slider::new(&mut self.config.total_demand, 0.1..=3.0).text("Demand D"))
            .changed()
        {
            out = Some(BraessUiAction::Recompute);
        }

        egui::CollapsingHeader::new("Affine Edge Latencies l_e(f) = a·f + b").show(ui, |ui| {
            if self.render_edge_latency_editors(ui, edges) {
                out = Some(BraessUiAction::Recompute);
            }
        });
        out
    }

    fn render_edge_latency_editors(
        &mut self,
        ui: &mut egui::Ui,
        edges: &mut [(usize, usize, f64)],
    ) -> bool {
        let mut changed = false;
        self.sync_with_edges(edges);
        for (i, edge) in edges.iter_mut().enumerate() {
            let lat = &mut self.edge_latencies[i];
            ui.horizontal(|ui| {
                ui.label(format!("{}->{}:", edge.0, edge.1));
                let a_ch = ui
                    .add(
                        egui::DragValue::new(&mut lat.a)
                            .speed(0.05)
                            .range(0.0..=20.0)
                            .prefix("a="),
                    )
                    .changed();
                let b_ch = ui
                    .add(
                        egui::DragValue::new(&mut lat.b)
                            .speed(0.05)
                            .range(0.0..=20.0)
                            .prefix("b="),
                    )
                    .changed();
                if a_ch || b_ch {
                    *lat = AffineLatency::clamped(lat.a, lat.b);
                    edge.2 = lat.b;
                    changed = true;
                }
            });
        }
        changed
    }

    fn render_equilibrium_readout(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        match self.status {
            Ok(summary) => {
                let trap = self
                    .without_shortcut_latency
                    .filter(|&b| summary.commuter_latency > b + 1e-6);
                let col = if trap.is_some() {
                    palette.error
                } else {
                    palette.success
                };
                ui.colored_label(
                    col,
                    format!("Commuter Latency: {:.2}", summary.commuter_latency),
                );
                if let Some(base) = trap {
                    ui.colored_label(
                        palette.error,
                        format!(
                            "Braess Slowdown: {:.2} -> {:.2} (+{:.1}%)",
                            base,
                            summary.commuter_latency,
                            100.0 * (summary.commuter_latency - base) / base.max(1e-9)
                        ),
                    );
                }
                ui.label(format!(
                    "Beckmann Φ(f*): {:.3} ({} FW iters)",
                    summary.beckmann_potential, summary.iterations
                ));
            }
            Err(WardropError::MissingTerminal) => {
                ui.colored_label(
                    palette.warning,
                    "Source s or Sink t node ID does not exist in the current graph.",
                );
            }
            Err(WardropError::Unreachable) => {
                ui.colored_label(
                    palette.error,
                    format!(
                        "Sink node {} is unreachable from source node {} — connect a directed path.",
                        self.config.sink_node, self.config.source_node
                    ),
                );
            }
        }
    }

    fn render_demand_plot(&mut self, ui: &mut egui::Ui) {
        if self.active_curve.is_empty() {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let current_pt = self
            .status
            .ok()
            .map(|s| [self.config.total_demand, s.commuter_latency]);
        Plot::new("braess_wardrop_latency_plot")
            .height(155.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                if !self.no_shortcut_curve.is_empty() {
                    plot_ui.line(
                        Line::new("Without u->v", &self.no_shortcut_curve[..])
                            .color(palette.success),
                    );
                }
                plot_ui.line(
                    Line::new("Active Network", &self.active_curve[..])
                        .color(palette.series_color(3)),
                );
                if let Some(pt) = current_pt {
                    plot_ui.points(Points::new("Equilibrium", vec![pt]).radius(5.0_f32));
                }
                self.callout_manager.render_on_plot(plot_ui);
            });
    }
}

// [cite:dijkstra]
