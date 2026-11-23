//! Coupled Kuramoto phase-oscillator network tool in `ChaosTab`.
//!
//! Integrates $\dot{\theta}_i = \omega_i + \frac{K}{N}\sum_j A_{ij}\sin(\theta_j - \theta_i)$
//! using preallocated RK4 scratch buffers, subscribes to [`GraphTopologyChannel`] when linked,
//! and publishes live coherence $R(t) \in [0.0, 1.0]$ onto
//! [`crate::app::workspace_bus::ScalarSignalChannel`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod view;

use crate::app::workspace_bus::{
    GraphTopologyChannel, SharedWorkspaceBus, KURAMOTO_COHERENCE_SERIES, MAX_SCALAR_SAMPLES,
};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::physics::chaos::kuramoto::{default_small_world_ring_edges, KuramotoSystem};
use std::collections::VecDeque;

const fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct KuramotoToolState {
    standalone_nodes: usize,
    coupling_k: f64,
    frequency_spread: f64,
    dt: f64,
    #[serde(default, skip_serializing_if = "is_false")]
    is_running: bool,
}

/// Interactive Kuramoto oscillator network synchronization laboratory.
pub struct KuramotoOscillatorTool {
    pub(super) system: KuramotoSystem,
    pub(super) standalone_nodes: usize,
    pub(super) coupling_k: f64,
    pub(super) frequency_spread: f64,
    pub(super) dt: f64,
    pub(super) is_running: bool,
    pub(super) positions: Vec<[f32; 2]>,
    pub(super) edges: Vec<(usize, usize, f64)>,
    pub(super) coherence_trace: VecDeque<[f64; 2]>,
    pub(super) synced_bus_revision: Option<u64>,
    pub(super) using_bus_topology: bool,
    last_published_sample_time: Option<f64>,
    state_rev: u64,
}

impl Default for KuramotoOscillatorTool {
    fn default() -> Self {
        let standalone_nodes = 12;
        let coupling_k = 3.5;
        let frequency_spread = 0.8;
        let dt = 0.03;
        let edges = default_small_world_ring_edges(standalone_nodes);
        let system = KuramotoSystem::builder()
            .num_oscillators(standalone_nodes)
            .coupling_k(coupling_k)
            .frequency_spread(frequency_spread)
            .edges(edges.clone())
            .build();
        let mut tool = Self {
            system,
            standalone_nodes,
            coupling_k,
            frequency_spread,
            dt,
            is_running: false,
            positions: circular_normalized_positions(standalone_nodes),
            edges,
            coherence_trace: VecDeque::with_capacity(MAX_SCALAR_SAMPLES),
            synced_bus_revision: None,
            using_bus_topology: false,
            last_published_sample_time: None,
            state_rev: 1,
        };
        tool.record_current_coherence_sample();
        tool
    }
}

fn circular_normalized_positions(n: usize) -> Vec<[f32; 2]> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![[0.0, 0.0]];
    }
    (0..n)
        .map(|i| {
            let angle =
                std::f32::consts::TAU * (i as f32) / (n as f32) - std::f32::consts::FRAC_PI_2;
            [0.82 * angle.cos(), 0.82 * angle.sin()]
        })
        .collect()
}

impl KuramotoOscillatorTool {
    fn bump_rev(&mut self) {
        self.state_rev = self.state_rev.saturating_add(1).max(1);
    }

    pub(super) fn record_current_coherence_sample(&mut self) -> [f64; 2] {
        let r = self.current_coherence();
        let sample = [self.system.time(), r];
        if self.coherence_trace.len() >= MAX_SCALAR_SAMPLES {
            self.coherence_trace.pop_front();
        }
        self.coherence_trace.push_back(sample);
        sample
    }

    /// Advances the oscillator network by one RK4 step of `self.dt`.
    pub fn step_once(&mut self) -> [f64; 2] {
        self.system.step_rk4(self.dt);
        self.record_current_coherence_sample()
    }

    /// Current coherence order parameter $R(t) \in [0.0, 1.0]$.
    #[must_use]
    pub fn current_coherence(&self) -> f64 {
        self.system.order_parameter().0
    }

    /// Resets oscillator phases and clears the coherence trace.
    pub fn reset_simulation(&mut self) {
        self.is_running = false;
        if !self.using_bus_topology {
            self.edges = default_small_world_ring_edges(self.standalone_nodes);
            self.positions = circular_normalized_positions(self.standalone_nodes);
            self.system.set_topology(self.standalone_nodes, &self.edges);
        }
        self.system.set_coupling_k(self.coupling_k);
        self.system.set_frequency_spread(self.frequency_spread);
        self.system.reset_phases();
        self.coherence_trace.clear();
        self.last_published_sample_time = None;
        self.record_current_coherence_sample();
        self.bump_rev();
    }

    pub(crate) fn ingest_topology_channel(&mut self, channel: &GraphTopologyChannel) {
        let rev = channel.revision();
        if rev == 0 || self.synced_bus_revision == Some(rev) {
            return;
        }
        self.synced_bus_revision = Some(rev);
        self.using_bus_topology = true;
        self.positions = channel.positions().to_vec();
        self.edges = channel.edges().to_vec();
        self.system.set_topology(self.positions.len(), &self.edges);
        self.record_current_coherence_sample();
        self.bump_rev();
    }

    fn publish_latest_coherence(&mut self, bus: &mut SharedWorkspaceBus) {
        if let Some(&[t, r]) = self.coherence_trace.back() {
            if self.last_published_sample_time != Some(t) {
                bus.signals.push_sample(KURAMOTO_COHERENCE_SERIES, t, r);
                self.last_published_sample_time = Some(t);
            }
        }
    }

    pub(super) fn on_bus_unlinked(&mut self) {
        self.last_published_sample_time = None;
        if self.using_bus_topology {
            self.using_bus_topology = false;
            self.synced_bus_revision = None;
            self.reset_simulation();
        }
    }

    pub(super) fn render_controls(&mut self, ui: &mut egui::Ui, reset_bus_series: &mut bool) {
        if self.using_bus_topology {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.label(
                egui::RichText::new(format!(
                    "🔗 Linked Topology (rev {})",
                    self.synced_bus_revision.unwrap_or(0)
                ))
                .small()
                .color(palette.success),
            );
        }

        if let Some(action) = PlaybackTransport::in_thread(
            "kuramoto_oscillator_transport",
            "Kuramoto Oscillators",
            self.is_running,
        )
        .reset_label("↻ Reset Phases")
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.is_running = true,
                PlaybackAction::Pause => self.is_running = false,
                PlaybackAction::Step => {
                    self.is_running = false;
                    let _ = self.step_once();
                }
                PlaybackAction::Reset => {
                    self.reset_simulation();
                    *reset_bus_series = true;
                }
            }
            self.bump_rev();
        }

        ui.separator();
        self.render_parameter_sliders(ui, reset_bus_series);
        self.render_coherence_readouts(ui);
    }

    fn render_parameter_sliders(&mut self, ui: &mut egui::Ui, reset_bus_series: &mut bool) {
        if !self.using_bus_topology
            && ui
                .add(egui::Slider::new(&mut self.standalone_nodes, 3..=128).text("Oscillators (N)"))
                .changed()
        {
            self.reset_simulation();
            *reset_bus_series = true;
        }
        if ui
            .add(egui::Slider::new(&mut self.coupling_k, 0.0..=20.0).text("Coupling (K)"))
            .changed()
        {
            self.system.set_coupling_k(self.coupling_k);
            self.bump_rev();
        }
        if ui
            .add(egui::Slider::new(&mut self.frequency_spread, 0.0..=3.0).text("Detuning (Δω)"))
            .changed()
        {
            self.system.set_frequency_spread(self.frequency_spread);
            self.bump_rev();
        }
        if ui
            .add(egui::Slider::new(&mut self.dt, 0.005..=0.10).text("Step (dt)"))
            .changed()
        {
            self.bump_rev();
        }
    }

    fn render_coherence_readouts(&self, ui: &mut egui::Ui) {
        ui.separator();
        let (r, psi) = self.system.order_parameter();
        let kc = self.system.critical_coupling_estimate();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let r_color = if r >= 0.70 {
            palette.success
        } else {
            palette.axis_x
        };
        ui.label(
            egui::RichText::new(format!("Coherence R(t): {r:.3}"))
                .strong()
                .color(r_color),
        );
        ui.label(format!("Mean Phase ψ(t): {psi:.2} rad"));
        if kc.is_finite() {
            ui.label(format!("Est. Critical K_c: {kc:.2}"));
        } else {
            ui.label("Est. Critical K_c: ∞ (disconnected)");
        }
        ui.label(format!(
            "Active Nodes: {} | Edges: {}",
            self.system.num_oscillators(),
            self.edges.len()
        ));
    }
}

impl InteractiveTool for KuramotoOscillatorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        &self.system
    }

    fn name(&self) -> &'static str {
        "Kuramoto Oscillators"
    }

    fn domain(&self) -> &'static str {
        "chaos"
    }

    fn sync_workspace_bus(&mut self, bus: &mut SharedWorkspaceBus) {
        self.ingest_topology_channel(&bus.topology);
        self.publish_latest_coherence(bus);
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn has_finite_parameters(&self) -> bool {
        self.coupling_k.is_finite() && self.frequency_spread.is_finite() && self.dt.is_finite()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&KuramotoToolState {
            standalone_nodes: self.standalone_nodes,
            coupling_k: self.coupling_k,
            frequency_spread: self.frequency_spread,
            dt: self.dt,
            is_running: self.is_running,
        })
        .ok()
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: KuramotoToolState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Kuramoto oscillator state: {e}"))?;
        if !(3..=128).contains(&s.standalone_nodes)
            || !s.coupling_k.is_finite()
            || !(0.0..=20.0).contains(&s.coupling_k)
            || !s.frequency_spread.is_finite()
            || !(0.0..=3.0).contains(&s.frequency_spread)
            || !s.dt.is_finite()
            || !(0.005..=0.10).contains(&s.dt)
        {
            return Err("Kuramoto oscillator parameters out of bounds".to_string());
        }
        self.standalone_nodes = s.standalone_nodes;
        self.coupling_k = s.coupling_k;
        self.frequency_spread = s.frequency_spread;
        self.dt = s.dt;
        if self.using_bus_topology {
            self.system.set_coupling_k(self.coupling_k);
            self.system.set_frequency_spread(self.frequency_spread);
        } else {
            self.reset_simulation();
        }
        self.is_running = s.is_running;
        self.bump_rev();
        Ok(())
    }

    fn step_simulation(&mut self) {
        let _ = self.step_once();
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.render_tool_inside(parent_ui);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Coupled Kuramoto Sync",
    "domain": "chaos",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "standalone_nodes": 16,
    "coupling_k": 5.2,
    "frequency_spread": 0.45,
    "dt": 0.04
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "KuramotoOscillatorTool",
        display_name: "Kuramoto Oscillators",
        domain: "chaos",
        tags: &["synchronization", "network", "oscillators", "coherence"],
        build: || Box::new(KuramotoOscillatorTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "kuramoto_oscillator_starter",
        title: "Coupled Kuramoto Sync",
        description: "Phase synchronization transition and order-parameter coherence R(t) across coupled oscillator networks.",
        category: "Nonlinear Dynamics",
        icon: "🔄",
        domain: "chaos",
        tool_name: Some("KuramotoOscillatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kuramoto_tool_publishes_bounded_coherence_to_scalar_signal_channel() {
        let mut tool = KuramotoOscillatorTool::default();
        let mut bus = SharedWorkspaceBus::default();
        bus.topology.publish(
            &[[-0.8, 0.0], [0.0, 0.8], [0.8, 0.0], [0.0, -0.8]],
            &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
        );
        tool.sync_workspace_bus(&mut bus);
        assert_eq!(tool.system.num_oscillators(), 4);

        for _ in 0..1_100 {
            let [t, r] = tool.step_once();
            assert!((0.0..=1.0).contains(&r));
            bus.signals.push_sample(KURAMOTO_COHERENCE_SERIES, t, r);
        }
        assert_eq!(tool.coherence_trace.len(), MAX_SCALAR_SAMPLES);
        let series = bus.signals.series(KURAMOTO_COHERENCE_SERIES).unwrap();
        assert_eq!(series.samples().len(), MAX_SCALAR_SAMPLES);
    }
}
