#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::PlotPoint;
use std::sync::mpsc::Receiver;

#[path = "glicko2_tournament/io.rs"]
pub(crate) mod io;
#[path = "glicko2_tournament/model.rs"]
pub(crate) mod model;
#[path = "glicko2_tournament/ui.rs"]
mod ui;

use model::{Glicko2TournamentState, RosterPlayer, ScheduledMatch, StandingRow};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TournamentSubView {
    Leaderboard,
    MatchupMatrix,
}

pub struct Glicko2TournamentTool {
    pub(crate) tau: f64,
    pub(crate) period: usize,
    pub(crate) roster: Vec<RosterPlayer>,
    pub(crate) matches: Vec<ScheduledMatch>,

    // Draft UI editors for adding players / matches
    pub(crate) draft_player_name: String,
    pub(crate) draft_player_rating: f64,
    pub(crate) draft_player_rd: f64,
    pub(crate) draft_match_a: usize,
    pub(crate) draft_match_b: usize,
    pub(crate) draft_score_a: f64,
    pub(crate) active_subview: TournamentSubView,

    // Cached evaluation results & zero-allocation plot series
    pub(crate) standings: Vec<StandingRow>,
    pub(crate) win_prob_matrix: Vec<Vec<f64>>,
    pub(crate) error_bar_lines: Vec<(String, [PlotPoint; 6])>,
    pub(crate) post_rating_points: Vec<PlotPoint>,
    pub(crate) pre_rating_points: Vec<PlotPoint>,
    pub(crate) eval_error: Option<String>,

    // Non-blocking file open/save channels
    pub(crate) open_csv_rx: Option<Receiver<Result<(String, String), String>>>,
    pub(crate) save_file_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) io_status: Option<(String, bool)>,
}

impl Default for Glicko2TournamentTool {
    fn default() -> Self {
        let (roster, matches) = model::glickman_canonical_roster_and_matches();
        let mut tool = Self {
            tau: 0.5,
            period: 1,
            roster,
            matches,
            draft_player_name: String::from("Challenger"),
            draft_player_rating: 1500.0,
            draft_player_rd: 350.0,
            draft_match_a: 0,
            draft_match_b: 1,
            draft_score_a: 1.0,
            active_subview: TournamentSubView::Leaderboard,
            standings: Vec::new(),
            win_prob_matrix: Vec::new(),
            error_bar_lines: Vec::new(),
            post_rating_points: Vec::new(),
            pre_rating_points: Vec::new(),
            eval_error: None,
            open_csv_rx: None,
            save_file_rx: None,
            io_status: None,
        };
        tool.recalculate();
        tool
    }
}

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
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn has_finite_parameters(&self) -> bool {
        self.tau.is_finite()
            && self
                .roster
                .iter()
                .all(|p| p.rating.is_finite() && p.rd.is_finite() && p.volatility.is_finite())
            && self.matches.iter().all(|m| m.score_a.is_finite())
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&Glicko2TournamentState {
            tau: self.tau,
            period: self.period,
            roster: self.roster.clone(),
            matches: self.matches.clone(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: Glicko2TournamentState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Glicko2TournamentTool state: {e}"))?;
        parsed.validate()?;
        self.tau = parsed.tau;
        self.period = parsed.period;
        self.roster = parsed.roster;
        self.matches = parsed.matches;
        self.draft_match_a = 0;
        self.draft_match_b = (1usize).min(self.roster.len().saturating_sub(1));
        self.recalculate();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.poll_io_channels(parent_ui.ctx());
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "glicko2_tournament_side_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            self.show_controls_panel(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_main_workbench(ui);
        });
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "Glicko2TournamentTool",
        display_name: "Glicko-2 Tournament",
        domain: "game_theory",
        tags: &["glicko2", "ratings", "tournament", "ranking"],
        build: || Box::new(Glicko2TournamentTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for Glicko2TournamentTool {
    fn theory_description(&self) -> String {
        format!(
            "Simultaneous Glicko-2 tournament rating system ($\\tau = {:.2}$, period $t = {}$): maps $(r, \\text{{RD}})$ to $(\\mu, \\phi)$ via $173.7178$, updates volatility $\\sigma'$ by Illinois root-finding, inflates inactive uncertainty $\\phi^* = \\sqrt{{\\phi^2 + \\sigma^2}}$, and computes head-to-head win probabilities $E(\\mu_i, \\mu_j, \\phi_j) = (1 + e^{{-g(\\phi_j)(\\mu_i - \\mu_j)}})^{{-1}}$.",
            self.tau, self.period
        )
    }
    fn phonetic_description(&self) -> String {
        "Glicko two competitive tournament rating volatility and interval estimation theory".into()
    }
    fn theory_citation(&self) -> String {
        "Example of the Glicko-2 System (Glickman, 2012); Parameter Estimation in Large Dynamic Paired Comparison Experiments (Glickman, 1999)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "95% Skill Interval".into(),
            "Approximate 95% credible skill band r +/- 2 * RD on the Glicko scale".into(),
        );
        map.insert(
            "Inactivity RD Growth".into(),
            "Rating deviation expands via sqrt(phi^2 + sigma^2) when zero games are played".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Round-Robin Masters League",
    "domain": "game_theory",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "tau": 0.6,
    "period": 1,
    "roster": [
      { "name": "Apex", "rating": 1620.0, "rd": 85.0, "volatility": 0.06 },
      { "name": "Bastion", "rating": 1540.0, "rd": 110.0, "volatility": 0.06 },
      { "name": "Cipher", "rating": 1490.0, "rd": 140.0, "volatility": 0.07 },
      { "name": "Dynamo", "rating": 1450.0, "rd": 210.0, "volatility": 0.08 },
      { "name": "Sabbatical", "rating": 1580.0, "rd": 95.0, "volatility": 0.06 }
    ],
    "matches": [
      { "player_a": 3, "player_b": 0, "score_a": 1.0 },
      { "player_a": 0, "player_b": 1, "score_a": 0.5 },
      { "player_a": 1, "player_b": 2, "score_a": 1.0 },
      { "player_a": 2, "player_b": 3, "score_a": 0.5 },
      { "player_a": 0, "player_b": 2, "score_a": 1.0 }
    ]
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "glicko2_tournament_starter",
        title: "Round-Robin Masters League",
        description: "Simultaneous Glicko-2 rating updates with upset volatility shifts, inactivity RD inflation, and pairwise win matrix.",
        category: "Game Theory",
        icon: "🏆",
        domain: "game_theory",
        tool_name: Some("Glicko2TournamentTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "glicko2_tournament/tests.rs"]
mod tests;
