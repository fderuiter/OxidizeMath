#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::render_tool_header;
use crate::accessibility::{
    announce_live_status, paint_canvas_focus_ring, register_canvas_accessibility,
    AccessibleHoverText,
};
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Plot};
use math_explorer::engine_graph::{
    max_product_bp_into, sum_product_bp_into, BpConfig, FactorGraphView, FactorSpec,
};
use serde::{Deserialize, Serialize};

/// Canonical factor-graph topologies in `FactorBeliefPropagationTool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BpTopology {
    /// 4-node Markov chain $X_0 - X_1 - X_2 - X_3$ (exact tree BP).
    Chain4,
    /// 5-node star tree centered at hub $X_0$ (exact tree BP + Bethe $\ln Z$).
    StarTree5,
    /// 4-node loopy $2 \times 2$ cycle $X_0 - X_1 - X_3 - X_2 - X_0$ (damped loopy BP).
    LoopyCycle4,
}

impl BpTopology {
    const ALL: [Self; 3] = [Self::Chain4, Self::StarTree5, Self::LoopyCycle4];

    fn label(self) -> &'static str {
        match self {
            Self::Chain4 => "4-Node Chain (Exact Tree BP)",
            Self::StarTree5 => "5-Node Star Tree (Exact Bethe ln Z)",
            Self::LoopyCycle4 => "2x2 Loopy Potts Cycle (Damped BP)",
        }
    }

    fn num_vars(self) -> usize {
        match self {
            Self::Chain4 | Self::LoopyCycle4 => 4,
            Self::StarTree5 => 5,
        }
    }

    fn edges(self) -> &'static [(usize, usize)] {
        match self {
            Self::Chain4 => &[(0, 1), (1, 2), (2, 3)],
            Self::StarTree5 => &[(0, 1), (0, 2), (0, 3), (0, 4)],
            Self::LoopyCycle4 => &[(0, 1), (1, 3), (3, 2), (2, 0)],
        }
    }
}

/// Serializable bounded state for `FactorBeliefPropagationTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactorBpState {
    /// Active factor-graph topology.
    pub topology: BpTopology,
    /// Discrete state cardinality $K \in \{2, 3\}$.
    pub num_states: usize,
    /// Pairwise Potts/Ising log-coupling $J \in [-2.0, 2.5]$.
    pub coupling: f64,
    /// External log-field bias on root $X_0$ toward state $0$.
    pub root_bias: f64,
    /// Pinned variable index receiving secondary bias toward state $K-1$.
    pub pinned_node: usize,
    /// Log-message damping factor $\lambda \in [0.0, 0.90]$.
    pub damping: f64,
    /// Maximum BP sweep iterations.
    pub max_iters: usize,
}

impl Default for FactorBpState {
    fn default() -> Self {
        Self {
            topology: BpTopology::StarTree5,
            num_states: 3,
            coupling: 1.15,
            root_bias: 0.95,
            pinned_node: 3,
            damping: 0.35,
            max_iters: 60,
        }
    }
}

impl FactorBpState {
    fn validate(&self) -> Result<(), String> {
        let ok = (2..=3).contains(&self.num_states)
            && self.coupling.is_finite()
            && (-2.0..=2.5).contains(&self.coupling)
            && self.root_bias.is_finite()
            && (-2.5..=2.5).contains(&self.root_bias)
            && self.pinned_node < 5
            && self.damping.is_finite()
            && (0.0..=0.90).contains(&self.damping)
            && (1..=200).contains(&self.max_iters);
        if !ok {
            return Err("FactorBpState parameter out of bounds".into());
        }
        Ok(())
    }
}

/// Summary sum-product and max-product diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct BpMetrics {
    /// Whether sum-product BP converged within tolerance.
    pub converged: bool,
    /// Sweeps executed by sum-product BP.
    pub iterations: usize,
    /// Final max-norm message residual.
    pub max_residual: f64,
    /// Bethe variational free energy $F_{\text{Bethe}}$.
    pub bethe_free_energy: f64,
    /// Log-partition estimate $\ln Z = -F_{\text{Bethe}}$.
    pub log_partition: f64,
    /// Unnormalized log-score of decoded MAP configuration.
    pub map_log_score: f64,
}

/// Interactive factor-graph sum-product and max-product BP studio (`engine_graph`).
pub struct FactorBeliefPropagationTool {
    state: FactorBpState,
    metrics: BpMetrics,
    revision: u64,
    cardinalities: Vec<usize>,
    factors: Vec<FactorSpec>,
    potentials: Vec<f64>,
    beliefs: Vec<f64>,
    max_marginals: Vec<f64>,
    map_states: Vec<usize>,
    sp_scratch: Vec<f64>,
    mp_scratch: Vec<f64>,
}

impl Default for FactorBeliefPropagationTool {
    fn default() -> Self {
        let mut tool = Self {
            state: FactorBpState::default(),
            metrics: BpMetrics::default(),
            revision: 1,
            cardinalities: Vec::new(),
            factors: Vec::new(),
            potentials: Vec::new(),
            beliefs: Vec::new(),
            max_marginals: Vec::new(),
            map_states: Vec::new(),
            sp_scratch: Vec::new(),
            mp_scratch: Vec::new(),
        };
        tool.rebuild_and_solve();
        tool
    }
}

impl FactorBeliefPropagationTool {
    #[cfg(test)]
    pub(super) fn metrics(&self) -> BpMetrics {
        self.metrics
    }

    #[cfg(test)]
    pub(super) fn set_topology_for_test(&mut self, topology: BpTopology) {
        self.state.topology = topology;
        self.rebuild_and_solve();
    }

    /// Executes sum-product and max-product BP over preallocated buffers with zero heap allocations.
    pub fn step_bp_zero_alloc(&mut self) {
        let Ok(fg) = FactorGraphView::new(&self.cardinalities, &self.factors, &self.potentials)
        else {
            return;
        };
        let cfg = BpConfig {
            max_iters: self.state.max_iters.clamp(1, 200),
            tolerance: 1.0e-10,
            damping: self.state.damping.clamp(0.0, 0.90),
        };
        if let Ok(sp) = sum_product_bp_into(&fg, &cfg, &mut self.beliefs, &mut self.sp_scratch) {
            let map_score = max_product_bp_into(
                &fg,
                &cfg,
                &mut self.max_marginals,
                &mut self.map_states,
                &mut self.mp_scratch,
            )
            .map(|r| r.log_map_score)
            .unwrap_or(f64::NAN);
            self.metrics = BpMetrics {
                converged: sp.converged,
                iterations: sp.iterations,
                max_residual: sp.max_residual,
                bethe_free_energy: -sp.log_partition,
                log_partition: sp.log_partition,
                map_log_score: map_score,
            };
        }
    }

    fn rebuild_and_solve(&mut self) {
        let n_vars = self.state.topology.num_vars();
        let k = self.state.num_states.clamp(2, 3);
        let pin = self.state.pinned_node % n_vars;
        self.cardinalities.clear();
        self.cardinalities.resize(n_vars, k);
        self.factors.clear();
        self.potentials.clear();

        for v in 0..n_vars {
            let start = self.potentials.len();
            for s in 0..k {
                let mut log_u = 0.0;
                if v == 0 && s == 0 {
                    log_u += self.state.root_bias;
                }
                if v == pin && s == k - 1 {
                    log_u += 0.65 * self.state.root_bias.abs();
                }
                self.potentials.push(log_u.exp());
            }
            self.factors.push(FactorSpec::unary(v, start, k));
        }

        for &(u, v) in self.state.topology.edges() {
            let start = self.potentials.len();
            for su in 0..k {
                for sv in 0..k {
                    let log_psi = if su == sv { self.state.coupling } else { 0.0 };
                    self.potentials.push(log_psi.exp());
                }
            }
            self.factors.push(FactorSpec::pairwise(u, v, start, k, k));
        }

        if let Ok(fg) = FactorGraphView::new(&self.cardinalities, &self.factors, &self.potentials) {
            self.beliefs.resize(fg.beliefs_len(), 0.0);
            self.max_marginals.resize(fg.beliefs_len(), 0.0);
            self.map_states.resize(n_vars, 0);
            self.sp_scratch.resize(fg.bp_scratch_len(), 0.0);
            self.mp_scratch.resize(fg.bp_scratch_len(), 0.0);
            self.step_bp_zero_alloc();
        }
        self.revision = self.revision.wrapping_add(1);
    }

    fn node_positions(topology: BpTopology, rect: egui::Rect) -> Vec<egui::Pos2> {
        let (cx, cy) = (rect.center().x, rect.center().y);
        let s = (rect.width() * 0.36).min(210.0);
        match topology {
            BpTopology::Chain4 => (0..4)
                .map(|i| egui::pos2(cx - s + (2.0 * s) * (i as f32) / 3.0, cy))
                .collect(),
            BpTopology::StarTree5 => [(0.0, 0.0), (-s, -40.0), (s, -40.0), (-s, 40.0), (s, 40.0)]
                .into_iter()
                .map(|(dx, dy)| egui::pos2(cx + dx, cy + dy))
                .collect(),
            BpTopology::LoopyCycle4 => [(-0.65, -38.0), (0.65, -38.0), (-0.65, 38.0), (0.65, 38.0)]
                .into_iter()
                .map(|(fx, dy)| egui::pos2(cx + s * fx, cy + dy))
                .collect(),
        }
    }

    fn render_factor_canvas(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let width = ui.available_width().max(240.0);
        let (response, painter) =
            ui.allocate_painter(egui::Vec2::new(width, 136.0), egui::Sense::click());
        let n_vars = self.state.topology.num_vars();
        if response.clicked()
            || (response.has_focus()
                && ui.input(|i| i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter)))
        {
            self.state.pinned_node = (self.state.pinned_node + 1) % n_vars;
            self.rebuild_and_solve();
            announce_live_status(
                ui.ctx(),
                format!(
                    "Pinned evidence moved to X{}, ln Z {:.3}",
                    self.state.pinned_node, self.metrics.log_partition
                ),
            );
        }
        let summary = format!(
            "{} with K={}, ln Z={:.3}, MAP={:?}",
            self.state.topology.label(),
            self.state.num_states,
            self.metrics.log_partition,
            self.map_states
        );
        register_canvas_accessibility(
            ui,
            &response,
            egui::accesskit::Role::GraphicsDocument,
            "Factor graph diagram (Click or press Space to cycle pinned evidence node)",
            Some(&summary),
        );
        painter.rect_filled(response.rect, 6.0, palette.surface_fill);
        self.paint_factor_graph(&painter, response.rect, &palette);
        paint_canvas_focus_ring(ui, &response);
    }

    fn paint_factor_graph(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &SemanticPalette,
    ) {
        let pos = Self::node_positions(self.state.topology, rect);
        let font = egui::FontId::proportional(11.5_f32);
        for &(u, v) in self.state.topology.edges() {
            let mid = egui::pos2(0.5 * (pos[u].x + pos[v].x), 0.5 * (pos[u].y + pos[v].y));
            painter.line_segment(
                [pos[u], pos[v]],
                egui::Stroke::new(2.0_f32, palette.graph_edge),
            );
            let sq = egui::Rect::from_center_size(mid, egui::Vec2::splat(11.0));
            painter.rect_filled(sq, 2.0, palette.accent);
        }
        let pin = self.state.pinned_node % pos.len();
        for (i, &p) in pos.iter().enumerate() {
            let map_s = self.map_states.get(i).copied().unwrap_or(0);
            let ring = palette.series[map_s % palette.series.len()];
            painter.circle_filled(p, 15.0_f32, palette.surface_fill);
            painter.circle_stroke(
                p,
                15.0_f32,
                egui::Stroke::new(if i == pin { 3.0_f32 } else { 2.0_f32 }, ring),
            );
            painter.text(
                p,
                egui::Align2::CENTER_CENTER,
                format!("X{i}:{map_s}"),
                font.clone(),
                palette.node_stroke,
            );
        }
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        egui::ComboBox::from_label("Topology")
            .selected_text(self.state.topology.label())
            .show_ui(ui, |ui| {
                for top in BpTopology::ALL {
                    changed |= ui
                        .selectable_value(&mut self.state.topology, top, top.label())
                        .changed();
                }
            });
        ui.separator();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.num_states, 2..=3).text("States K"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.coupling, -2.0..=2.5).text("Coupling J"))
            .accessible_hover_text("Pairwise Potts log-coupling (>0 attractive, <0 repulsive)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.root_bias, -2.5..=2.5).text("Root Bias h₀"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.damping, 0.0..=0.90).text("Damping λ"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.max_iters, 5..=200).text("Max Sweeps"))
            .changed();
        if changed {
            self.rebuild_and_solve();
        }
    }

    fn render_belief_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let k = self.state.num_states.clamp(2, 3);
        let n_vars = self.state.topology.num_vars();
        Plot::new(ui.id().with("prob_factor_bp_plot"))
            .legend(Legend::default())
            .x_axis_label("Variable Index i")
            .y_axis_label("Marginal Belief bᵢ(s)")
            .include_y(0.0)
            .include_y(1.05)
            .show(ui, |plot_ui| {
                let width = 0.72 / (k as f64);
                for s in 0..k {
                    let offset = ((s as f64) - 0.5 * ((k - 1) as f64)) * width;
                    let mut bars = Vec::with_capacity(n_vars);
                    for v in 0..n_vars {
                        let prob = self.beliefs.get(v * k + s).copied().unwrap_or(0.0);
                        bars.push(Bar::new((v as f64) + offset, prob).width(width * 0.90));
                    }
                    plot_ui.bar_chart(
                        BarChart::new(format!("State s = {s}"), bars)
                            .color(palette.series[s % palette.series.len()]),
                    );
                }
            });
    }
}

impl InteractiveTool for FactorBeliefPropagationTool {
    fn name(&self) -> &'static str {
        "Factor Graph & Belief Propagation"
    }
    fn domain(&self) -> &'static str {
        "probability"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "prob_bp_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            render_tool_header(ui, "Factor Graph & BP");
            self.render_controls(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let m = self.metrics;
            let status_col = if m.converged {
                palette.success
            } else {
                palette.warning
            };
            ui.horizontal_wrapped(|ui| {
                let status = format!(
                    "Converged: {} ({} sweeps, res {:.2e})",
                    m.converged, m.iterations, m.max_residual
                );
                ui.colored_label(status_col, status);
                ui.separator();
                ui.colored_label(palette.series[0], format!("ln Z: {:.4}", m.log_partition));
                ui.separator();
                let bethe = format!("F_Bethe: {:.4}", m.bethe_free_energy);
                ui.colored_label(palette.accent, bethe);
                ui.separator();
                let map_txt = format!("MAP {:?} (score {:.3})", self.map_states, m.map_log_score);
                ui.colored_label(palette.info, map_txt);
            });
            ui.add_space(4.0);
            self.render_factor_canvas(ui);
            ui.add_space(4.0);
            self.render_belief_plot(ui);
        });
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn has_finite_parameters(&self) -> bool {
        self.state.validate().is_ok()
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }
    fn try_load_state(&mut self, json: &str) -> Result<(), String> {
        self.state_bounds().validate_json(json)?;
        let parsed: FactorBpState =
            serde_json::from_str(json).map_err(|e| format!("Invalid FactorBpState: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.rebuild_and_solve();
        Ok(())
    }
    fn clamp_parameters(&mut self) -> bool {
        self.state.num_states = self.state.num_states.clamp(2, 3);
        self.state.coupling = self.state.coupling.clamp(-2.0, 2.5);
        self.state.root_bias = self.state.root_bias.clamp(-2.5, 2.5);
        self.state.pinned_node = self.state.pinned_node.min(4);
        self.state.damping = self.state.damping.clamp(0.0, 0.90);
        self.state.max_iters = self.state.max_iters.clamp(1, 200);
        self.rebuild_and_solve();
        true
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let (k, n_vars) = (
            self.state.num_states.clamp(2, 3),
            self.state.topology.num_vars(),
        );
        (0..k)
            .map(|s| {
                let pts = (0..n_vars)
                    .map(|v| [v as f64, *self.beliefs.get(v * k + s).unwrap_or(&0.0)])
                    .collect();
                (format!("Belief b_i({s})"), pts)
            })
            .collect()
    }
    fn step_simulation(&mut self) {
        self.state.pinned_node = (self.state.pinned_node + 1) % self.state.topology.num_vars();
        self.rebuild_and_solve();
    }
}

// [cite:probability]
