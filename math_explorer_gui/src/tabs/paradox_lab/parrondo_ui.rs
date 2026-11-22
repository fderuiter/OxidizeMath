#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::parrondo::{ParrondoParadoxTool, ParrondoSchedule, ParrondoState, ParrondoVariant};
use super::render_tool_header;
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{ExportCapabilities, InteractiveTool, StarterPresetMetadata, ToolMetadata};
use eframe::egui;
use egui_plot::{HLine, Legend, Line, LineStyle, Plot, PlotPoints};

impl ParrondoParadoxTool {
    pub(super) fn render_parrondo_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "parrondo_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            render_tool_header(ui, "Parrondo's Paradox");
            self.render_controls(ui);
        });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_drift_banner(ui);
            ui.add_space(4.0);
            self.render_trajectory_plot(ui);
        });
    }

    fn render_controls(&mut self, ui: &mut egui::Ui) {
        let mut changed = self.render_variant_selector(ui);
        ui.separator();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.bias_epsilon, 0.0..=0.05)
                    .text("Bias ε")
                    .step_by(0.0005),
            )
            .accessible_hover_text("Coin disadvantage epsilon subtracted from fair win probabilities")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.mix_gamma, 0.0..=1.0).text("Mix γ (Game A)"))
            .accessible_hover_text("Probability gamma of choosing Game A in Random Mixture")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.jump_switch_rate, 0.1..=5.0).text("SSA Jump λ"))
            .accessible_hover_text("Continuous-time game switching rate for GillespieSolver")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.steps, 20..=400).text("Rounds"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.walkers, 32..=2048).text("Walkers"))
            .changed();
        ui.horizontal(|ui| {
            ui.label("RNG Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.state.seed).speed(1.0))
                .changed();
        });

        ui.separator();
        ui.label("Stationary Slope Overlay:");
        changed |= self.render_schedule_selector(ui);
        ui.separator();
        self.render_step_buttons(ui, changed);
        ui.separator();
        self.render_tensor_summary(ui);
    }

    fn render_variant_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.label("Markov Formulation:");
        ui.horizontal(|ui| {
            changed |= ui
                .selectable_value(
                    &mut self.state.variant,
                    ParrondoVariant::CapitalDependent,
                    "Mod-3 Capital",
                )
                .accessible_hover_text("3-state capital-dependent ratchet (X_t mod 3)")
                .changed();
            changed |= ui
                .selectable_value(
                    &mut self.state.variant,
                    ParrondoVariant::HistoryDependent,
                    "4-State History",
                )
                .accessible_hover_text("4-state history-dependent ratchet (LL, LW, WL, WW)")
                .changed();
        });
        changed
    }

    fn render_schedule_selector(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        egui::ComboBox::from_id_salt("parrondo_focus_schedule")
            .selected_text(match self.state.focus_schedule {
                ParrondoSchedule::PureA => "Pure Game A",
                ParrondoSchedule::PureB => "Pure Game B",
                ParrondoSchedule::PeriodicABB => "Periodic ABB",
                ParrondoSchedule::RandomMix => "Random Mixture",
            })
            .show_ui(ui, |ui| {
                for (sched, label) in [
                    (ParrondoSchedule::PeriodicABB, "Periodic ABB"),
                    (ParrondoSchedule::RandomMix, "Random Mixture"),
                    (ParrondoSchedule::PureA, "Pure Game A"),
                    (ParrondoSchedule::PureB, "Pure Game B"),
                ] {
                    changed |= ui
                        .selectable_value(&mut self.state.focus_schedule, sched, label)
                        .changed();
                }
            });
        changed
    }

    fn render_step_buttons(&mut self, ui: &mut egui::Ui, changed: bool) {
        if changed {
            self.rebuild_full_simulation();
        }
        ui.horizontal(|ui| {
            if ui
                .button("⟲ Reset to t=0")
                .accessible_hover_text("Reset trajectories to round 0 for manual stepping")
                .clicked()
            {
                self.prepare_stepping();
                self.revision = self.revision.wrapping_add(1);
            }
            if ui
                .button("⏭ +10 Rounds")
                .accessible_hover_text("Advance zero-allocation stepper by 10 rounds")
                .clicked()
            {
                for _ in 0..10 {
                    self.step_parrondo_zero_alloc();
                }
                self.revision = self.revision.wrapping_add(1);
            }
            if ui
                .button("▶ Full Horizon")
                .accessible_hover_text("Simulate all rounds across ensemble and GillespieSolver")
                .clicked()
            {
                self.rebuild_full_simulation();
            }
        });
    }

    fn render_tensor_summary(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let m = self.metrics();
        ui.colored_label(
            palette.muted_text,
            format!(
                "TransitionTensor [0..2]: P_avg(0→1) = {:.4}, P_t=0.5(0→1) = {:.4}",
                m.tensor_avg_p01, m.tensor_mid_p01
            ),
        );
    }

    fn render_drift_banner(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let m = self.metrics();
        let color_for = |d: f64| if d >= 0.0 { palette.success } else { palette.error };
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(color_for(m.drift_a), format!("μ(A): {:+.5}", m.drift_a));
            ui.separator();
            ui.colored_label(color_for(m.drift_b), format!("μ(B): {:+.5}", m.drift_b));
            ui.separator();
            ui.colored_label(color_for(m.drift_abb), format!("μ(ABB): {:+.5}", m.drift_abb));
            ui.separator();
            ui.colored_label(color_for(m.drift_mix), format!("μ(Mix): {:+.5}", m.drift_mix));
        });
    }

    fn render_trajectory_plot(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let end = (self.current_step + 1).min(self.traj_a.len()).max(1);

        Plot::new(ui.id().with("parrondo_bankroll_plot"))
            .legend(Legend::default())
            .x_axis_label("Round t")
            .y_axis_label("Expected / Ensemble Capital E[X_t]")
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    HLine::new("Break-even", 0.0)
                        .color(palette.graph_edge)
                        .style(LineStyle::Dashed { length: 6.0_f32 }),
                );
                for (name, slice, color, width) in [
                    ("Game A (Pure)", &self.traj_a[..end], palette.error, 2.0_f32),
                    ("Game B (Pure)", &self.traj_b[..end], palette.series[1], 2.0_f32),
                    ("Periodic ABB", &self.traj_abb[..end], palette.success, 2.5_f32),
                    ("Random Mixture", &self.traj_mix[..end], palette.series[0], 2.0_f32),
                    ("Gillespie SSA Ratchet", &self.traj_ssa[..end], palette.series[4], 1.4_f32),
                ] {
                    plot_ui.line(
                        Line::new(name, PlotPoints::from(slice.to_vec()))
                            .color(color)
                            .width(width),
                    );
                }
                plot_ui.line(
                    Line::new(
                        "Markov Stationary Drift",
                        PlotPoints::from(self.traj_exact[..end].to_vec()),
                    )
                    .color(palette.accent)
                    .style(LineStyle::Dashed { length: 8.0_f32 })
                    .width(2.2_f32),
                );
            });
    }
}

impl InteractiveTool for ParrondoParadoxTool {
    fn name(&self) -> &'static str {
        "Parrondo's Paradox"
    }
    fn domain(&self) -> &'static str {
        "paradox_lab"
    }
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.render_parrondo_inside(parent_ui);
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
        let parsed: ParrondoState =
            serde_json::from_str(json).map_err(|e| format!("Invalid Parrondo state: {e}"))?;
        parsed.validate()?;
        self.state = parsed;
        self.rebuild_full_simulation();
        Ok(())
    }
    fn clamp_parameters(&mut self) -> bool {
        self.state.bias_epsilon = self.state.bias_epsilon.clamp(0.0, 0.20);
        self.state.mix_gamma = self.state.mix_gamma.clamp(0.0, 1.0);
        self.state.jump_switch_rate = self.state.jump_switch_rate.clamp(0.05, 10.0);
        self.state.steps = self.state.steps.clamp(10, 600);
        self.state.walkers = self.state.walkers.clamp(16, 4096);
        self.rebuild_full_simulation();
        true
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let end = (self.current_step + 1).min(self.traj_a.len());
        vec![
            ("Game A (Pure)".into(), self.traj_a[..end].to_vec()),
            ("Game B (Pure)".into(), self.traj_b[..end].to_vec()),
            ("Periodic ABB".into(), self.traj_abb[..end].to_vec()),
            ("Random Mixture".into(), self.traj_mix[..end].to_vec()),
            ("Markov Stationary Drift".into(), self.traj_exact[..end].to_vec()),
            ("Gillespie SSA Ratchet".into(), self.traj_ssa[..end].to_vec()),
        ]
    }
    fn step_simulation(&mut self) {
        self.state.seed = self.state.seed.wrapping_add(1);
        self.rebuild_full_simulation();
    }
}

impl scientific_metadata::theory::TheoryDescribable for ParrondoParadoxTool {
    fn theory_description(&self) -> String {
        "Parrondo's Brownian ratchet: two individually losing games ($\\mu_A < 0, \\mu_B < 0$) yield strictly positive expected bankroll drift ($\\mu_{ABB} > 0, \\mu_{\\text{mix}} > 0$) when alternated periodically or mixed stochastically across stationary distributions $\\pi P = \\pi$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Parrondo ratchet combining two losing Markov games into a winning expectation".into()
    }
    fn theory_citation(&self) -> String {
        "Harmer & Abbott (1999), Losing strategies can win by Parrondo's paradox, Nature 402, 864; Parrondo, Harmer & Abbott (2000), PRL 85, 5226".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

pub const PRESET_PARRONDO_RATCHET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Parrondo's Paradox",
    "domain": "paradox_lab",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "bias_epsilon": 0.003,
    "mix_gamma": 0.5,
    "jump_switch_rate": 1.5,
    "variant": "CapitalDependent",
    "focus_schedule": "PeriodicABB",
    "steps": 150,
    "walkers": 512,
    "seed": 7
  }
}"#;

inventory::submit! {
    ToolMetadata {
        id: "ParrondoParadoxTool",
        display_name: "Parrondo's Paradox",
        domain: "paradox_lab",
        tags: &["markov", "stochastic", "ratchet", "game_theory"],
        build: || Box::new(ParrondoParadoxTool::default()),
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "parrondo_ratchet_starter",
        title: "Parrondo's Brownian Ratchet (ABB)",
        description: "Alternating two losing coins (epsilon = 0.003) produces positive expected capital growth.",
        category: "Paradox Lab",
        icon: "🎲",
        domain: "paradox_lab",
        tool_name: Some("ParrondoParadoxTool"),
        preset_generator: || PRESET_PARRONDO_RATCHET,
    }
}

// [cite:paradox_lab]
