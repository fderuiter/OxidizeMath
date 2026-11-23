//! [`InteractiveTool`], `TheoryDescribable`, and `inventory` registrations for [`super::Glicko2TournamentTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::Glicko2TournamentTool;
use crate::framework::InteractiveTool;
use crate::presets::{Glicko2TournamentPresetConfig, MAX_TOURNAMENT_PERIODS};
use eframe::egui;

impl InteractiveTool for Glicko2TournamentTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Glicko-2 Tournament"
    }
    fn domain(&self) -> &'static str {
        "game_theory"
    }
    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn has_finite_parameters(&self) -> bool {
        self.tau.is_finite()
            && self.draw_probability.is_finite()
            && self.competitors.iter().all(|c| {
                c.initial.rating.value().is_finite()
                    && c.initial.rating_deviation.value().is_finite()
                    && c.initial.volatility.value().is_finite()
            })
    }

    fn clamp_parameters(&mut self) -> bool {
        let tau = if self.tau.is_finite() {
            self.tau.clamp(0.3, 1.2)
        } else {
            0.5
        };
        let draw = if self.draw_probability.is_finite() {
            self.draw_probability.clamp(0.0, 1.0)
        } else {
            0.15
        };
        let changed = (tau - self.tau).abs() > f64::EPSILON
            || (draw - self.draw_probability).abs() > f64::EPSILON;
        (self.tau, self.draw_probability) = (tau, draw);
        self.target_rounds = self.target_rounds.clamp(1, MAX_TOURNAMENT_PERIODS);
        if changed {
            let _ = self.rebuild_trajectory();
        }
        changed
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.export_config()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: Glicko2TournamentPresetConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Glicko2TournamentTool state: {e}"))?;
        self.apply_config(cfg)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        self.poll_io(&ctx);
        let side_id = crate::framework::scoped_panel_id(parent_ui, "glicko2_control_panel");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_side_controls(ui);
        });
        self.show_bottom_dock(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_central_plot(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for Glicko2TournamentTool {
    fn theory_description(&self) -> String {
        "Multi-competitor Bayesian skill estimation under Mark Glickman's Glicko-2 system: each rating period freezes pre-period ratings $(\\mu_j, \\phi_j)$, updates volatility $\\sigma'$ via Illinois root-finding, inflates deviation $\\phi^* = \\sqrt{\\phi^2 + \\sigma'^2}$, and computes posterior $(\\mu', \\phi')$ with 95% confidence interval $r \\pm 2\\,\\text{RD}$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Glicko two Bayesian tournament rating simulator with rating deviation confidence bands and volatility".into()
    }
    fn theory_citation(&self) -> String {
        "Parameter estimation in large dynamic paired comparison experiments (Glickman, Applied Statistics, 1999); Example of the Glicko-2 system (Glickman, 2012)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::from([(
            "tau".to_string(),
            "System constant constraining log-volatility change per period (0.3 to 1.2)."
                .to_string(),
        )])
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "Glicko2TournamentTool",
        display_name: "Glicko-2 Tournament",
        domain: "game_theory",
        tags: &["glicko2", "ratings", "swiss", "tournament", "bayesian"],
        build: || Box::new(Glicko2TournamentTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "glicko2_swiss_upset",
        title: "Swiss Upset & Volatility Spike",
        description: "r ± 2RD & σ': Five-player Swiss tournament where an underdog prodigy triggers a volatility surge.",
        category: "Game Theory",
        icon: "♟️",
        domain: "game_theory",
        tool_name: Some("Glicko2TournamentTool"),
        preset_generator: || crate::starter_gallery::PRESET_GLICKO2_SWISS_UPSET,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "glicko2_round_robin_drift",
        title: "Round-Robin Frozen vs. Sequential Drift",
        description: "Δr = r_seq - r_frozen: Quantifies order-dependent rating drift vs. canonical frozen periods.",
        category: "Game Theory",
        icon: "🎯",
        domain: "game_theory",
        tool_name: Some("Glicko2TournamentTool"),
        preset_generator: || crate::starter_gallery::PRESET_GLICKO2_ROUND_ROBIN_DRIFT,
    }
}
