//! Interactive Bimatrix Nash Equilibrium & Coalitional Shapley Value Workbench (`NashShapleyTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::applied::game_theory::cooperative::{
    is_convex, is_in_core, is_superadditive, max_core_excess, shapley_values,
};
use math_explorer::applied::game_theory::normal_form::{
    evaluate_bimatrix_profile, iterated_dominance, lemke_howson_equilibrium,
    solve_zero_sum_minimax, support_enumeration, BimatrixGameView, DominanceMode,
    DominanceWorkspace, LemkeHowsonWorkspace, MinimaxWorkspace, SupportEquilibrium,
    MAX_SUPPORT_ENUM_DIM,
};
use serde::{Deserialize, Serialize};

pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

pub const MAX_UI_STRATEGIES: usize = 6;
pub const MAX_UI_COALITION_PLAYERS: usize = 5;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum NashShapleyViewMode {
    BimatrixNash,
    CoalitionalShapley,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NashShapleyConfig {
    /// Payoff scale display reference (first scalar field for perturbation tests).
    pub payoff_probe: f64,
    pub mode: NashShapleyViewMode,
    pub rows: usize,
    pub cols: usize,
    pub row_payoffs: Vec<f64>,
    pub col_payoffs: Vec<f64>,
    pub zero_sum_lock: bool,
    pub weak_dominance: bool,
    pub initial_label: usize,
    pub coalition_players: usize,
    pub characteristic_values: Vec<f64>,
}

impl NashShapleyConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.payoff_probe.is_finite() || !(-1e4..=1e4).contains(&self.payoff_probe) {
            return Err("payoff_probe must be finite and in [-10000, 10000]".into());
        }
        if !(2..=MAX_SUPPORT_ENUM_DIM).contains(&self.rows)
            || !(2..=MAX_SUPPORT_ENUM_DIM).contains(&self.cols)
        {
            return Err("rows and cols must be in 2..=8".into());
        }
        let expected_cells = self.rows * self.cols;
        if self.row_payoffs.len() != expected_cells || self.col_payoffs.len() != expected_cells {
            return Err("bimatrix payoff lengths must equal rows * cols".into());
        }
        if self
            .row_payoffs
            .iter()
            .chain(self.col_payoffs.iter())
            .any(|v| !v.is_finite() || !(-1e4..=1e4).contains(v))
        {
            return Err("bimatrix payoffs must be finite and in [-10000, 10000]".into());
        }
        if self.initial_label >= self.rows + self.cols {
            return Err("initial_label must be < rows + cols".into());
        }
        if !(2..=6).contains(&self.coalition_players) {
            return Err("coalition_players must be in 2..=6".into());
        }
        let expected_masks = 1usize << self.coalition_players;
        if self.characteristic_values.len() != expected_masks {
            return Err("characteristic_values length must equal 2^coalition_players".into());
        }
        if self
            .characteristic_values
            .iter()
            .any(|v| !v.is_finite() || !(-1e4..=1e4).contains(v))
        {
            return Err("characteristic_values must be finite and in [-10000, 10000]".into());
        }
        if self.characteristic_values[0].abs() > 1e-6 {
            return Err("empty coalition worth v(0) must be 0.0".into());
        }
        Ok(())
    }
}

pub struct NashShapleyTool {
    pub(crate) cfg: NashShapleyConfig,
    pub(crate) row_strategy: Vec<f64>,
    pub(crate) col_strategy: Vec<f64>,
    pub(crate) row_alive: Vec<bool>,
    pub(crate) col_alive: Vec<bool>,
    pub(crate) dom_rounds: usize,
    pub(crate) row_payoff: f64,
    pub(crate) col_payoff: f64,
    pub(crate) max_regret: f64,
    pub(crate) pivots: usize,
    pub(crate) enumerated_count: usize,
    pub(crate) enumerated_eqs: [SupportEquilibrium; 8],
    pub(crate) shapley: Vec<f64>,
    pub(crate) superadditive: bool,
    pub(crate) convex: bool,
    pub(crate) in_core: bool,
    pub(crate) core_excess: f64,
    dom_ws: DominanceWorkspace,
    mm_ws: MinimaxWorkspace,
    lh_ws: LemkeHowsonWorkspace,
    state_rev: u64,
}

impl Default for NashShapleyTool {
    fn default() -> Self {
        let cfg = presets::default_config();
        let mut tool = Self {
            cfg,
            row_strategy: vec![0.0; 2],
            col_strategy: vec![0.0; 2],
            row_alive: vec![true; 2],
            col_alive: vec![true; 2],
            dom_rounds: 0,
            row_payoff: 0.0,
            col_payoff: 0.0,
            max_regret: 0.0,
            pivots: 0,
            enumerated_count: 0,
            enumerated_eqs: [SupportEquilibrium {
                rows: 0,
                cols: 0,
                row_strategy: [0.0; MAX_SUPPORT_ENUM_DIM],
                col_strategy: [0.0; MAX_SUPPORT_ENUM_DIM],
                row_payoff: 0.0,
                col_payoff: 0.0,
            }; 8],
            shapley: vec![0.0; 3],
            superadditive: true,
            convex: true,
            in_core: true,
            core_excess: 0.0,
            dom_ws: DominanceWorkspace::new(MAX_SUPPORT_ENUM_DIM, MAX_SUPPORT_ENUM_DIM),
            mm_ws: MinimaxWorkspace::new(MAX_SUPPORT_ENUM_DIM, MAX_SUPPORT_ENUM_DIM),
            lh_ws: LemkeHowsonWorkspace::new(MAX_SUPPORT_ENUM_DIM, MAX_SUPPORT_ENUM_DIM),
            state_rev: 1,
        };
        tool.recompute();
        tool
    }
}

impl NashShapleyTool {
    pub(crate) fn recompute(&mut self) {
        if self.cfg.zero_sum_lock {
            for (b, &a) in self
                .cfg
                .col_payoffs
                .iter_mut()
                .zip(self.cfg.row_payoffs.iter())
            {
                *b = -a;
            }
        }
        self.cfg.initial_label = self
            .cfg
            .initial_label
            .min(self.cfg.rows + self.cfg.cols - 1);
        self.recompute_bimatrix();
        self.recompute_coalition();
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    fn recompute_bimatrix(&mut self) {
        let rows = self.cfg.rows;
        let cols = self.cfg.cols;
        self.row_strategy.resize(rows, 0.0);
        self.col_strategy.resize(cols, 0.0);
        self.row_alive.resize(rows, true);
        self.col_alive.resize(cols, true);
        self.dom_ws.ensure_capacity(rows, cols);
        self.mm_ws.ensure_capacity(rows, cols);
        self.lh_ws.ensure_capacity(rows, cols);

        let Ok(game) =
            BimatrixGameView::new(&self.cfg.row_payoffs, &self.cfg.col_payoffs, rows, cols)
        else {
            return;
        };
        let mode = if self.cfg.weak_dominance {
            DominanceMode::Weak
        } else {
            DominanceMode::Strict
        };
        if let Ok(dom) = iterated_dominance(
            game,
            mode,
            &mut self.dom_ws,
            &mut self.row_alive,
            &mut self.col_alive,
        ) {
            self.dom_rounds = dom.rounds;
        }
        if self.cfg.zero_sum_lock {
            if let Ok(mm) = solve_zero_sum_minimax(
                &self.cfg.row_payoffs,
                rows,
                cols,
                &mut self.mm_ws,
                &mut self.row_strategy,
                &mut self.col_strategy,
            ) {
                self.pivots = mm.pivots;
            }
        } else if let Ok(lh) = lemke_howson_equilibrium(
            game,
            self.cfg.initial_label,
            &mut self.lh_ws,
            &mut self.row_strategy,
            &mut self.col_strategy,
        ) {
            self.pivots = lh.pivots;
        }
        if let Ok((rp, cp, reg)) =
            evaluate_bimatrix_profile(game, &self.row_strategy, &self.col_strategy)
        {
            (self.row_payoff, self.col_payoff, self.max_regret) = (rp, cp, reg);
        }
        self.enumerated_count = support_enumeration(game, &mut self.enumerated_eqs).unwrap_or(0);
    }

    fn recompute_coalition(&mut self) {
        let n = self.cfg.coalition_players;
        self.cfg.characteristic_values[0] = 0.0;
        self.shapley.resize(n, 0.0);
        if shapley_values(n, &self.cfg.characteristic_values, &mut self.shapley).is_ok() {
            self.superadditive =
                is_superadditive(n, &self.cfg.characteristic_values).unwrap_or(false);
            self.convex = is_convex(n, &self.cfg.characteristic_values).unwrap_or(false);
            self.in_core =
                is_in_core(n, &self.cfg.characteristic_values, &self.shapley).unwrap_or(false);
            self.core_excess = max_core_excess(n, &self.cfg.characteristic_values, &self.shapley)
                .map_or(0.0, |(ex, _)| ex);
        }
    }

    pub fn resize_bimatrix(&mut self, new_rows: usize, new_cols: usize) {
        let nr = new_rows.clamp(2, MAX_UI_STRATEGIES);
        let nc = new_cols.clamp(2, MAX_UI_STRATEGIES);
        if nr == self.cfg.rows && nc == self.cfg.cols {
            return;
        }
        let mut next_a = vec![0.0; nr * nc];
        let mut next_b = vec![0.0; nr * nc];
        for r in 0..self.cfg.rows.min(nr) {
            for c in 0..self.cfg.cols.min(nc) {
                let old_idx = flatten_2d_index(c, r, self.cfg.cols);
                let new_idx = flatten_2d_index(c, r, nc);
                next_a[new_idx] = self.cfg.row_payoffs[old_idx];
                next_b[new_idx] = self.cfg.col_payoffs[old_idx];
            }
        }
        self.cfg.rows = nr;
        self.cfg.cols = nc;
        self.cfg.row_payoffs = next_a;
        self.cfg.col_payoffs = next_b;
        self.recompute();
    }

    pub fn resize_coalition(&mut self, players: usize) {
        let np = players.clamp(2, MAX_UI_COALITION_PLAYERS);
        if np == self.cfg.coalition_players {
            return;
        }
        let new_len = 1usize << np;
        let mut next_v = vec![0.0; new_len];
        for (m, slot) in next_v
            .iter_mut()
            .enumerate()
            .take(self.cfg.characteristic_values.len().min(new_len))
        {
            *slot = self.cfg.characteristic_values[m];
        }
        next_v[0] = 0.0;
        self.cfg.coalition_players = np;
        self.cfg.characteristic_values = next_v;
        self.recompute();
    }
}

impl InteractiveTool for NashShapleyTool {
    fn name(&self) -> &'static str {
        "Nash & Shapley Equilibria"
    }

    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.state_rev
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::new(4096, 6, 128)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("row_payoff", self.row_payoff),
            ("col_payoff", self.col_payoff),
            ("max_regret", self.max_regret),
            ("enumerated_equilibria", self.enumerated_count as f64),
            ("core_excess", self.core_excess),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: NashShapleyConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid NashShapleyTool state: {e}"))?;
        cfg.validate()?;
        self.cfg = cfg;
        self.recompute();
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.cfg.mode {
            NashShapleyViewMode::BimatrixNash => vec![
                (
                    "Row Mixed Strategy p*".to_string(),
                    self.row_strategy
                        .iter()
                        .enumerate()
                        .map(|(i, &p)| [i as f64, p])
                        .collect(),
                ),
                (
                    "Column Mixed Strategy q*".to_string(),
                    self.col_strategy
                        .iter()
                        .enumerate()
                        .map(|(j, &q)| [j as f64, q])
                        .collect(),
                ),
            ],
            NashShapleyViewMode::CoalitionalShapley => vec![(
                "Shapley Value Allocation".to_string(),
                self.shapley
                    .iter()
                    .enumerate()
                    .map(|(i, &phi)| [i as f64, phi])
                    .collect(),
            )],
        }
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "nash_shapley_side");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_controls(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_main_view(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for NashShapleyTool {
    fn theory_description(&self) -> String {
        "Computes Nash equilibria $(p^*, q^*)$ of finite bimatrix games via iterated dominance, zero-sum primal-dual Simplex, $8\\times 8$ support enumeration, and lexicographically perturbed Lemke-Howson pivoting, alongside exact coalitional Shapley values $\\phi_i(v) = \\sum_{S \\subseteq N \\setminus \\{i\\}} \\frac{|S|!(n-|S|-1)!}{n!}[v(S \\cup \\{i\\}) - v(S)]$ and Core excess diagnostics.".into()
    }

    fn phonetic_description(&self) -> String {
        "Normal form bimatrix Nash equilibrium Lemke Howson and cooperative Shapley value workbench"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Non-Cooperative Games (Nash, Annals of Mathematics, 1951); Equilibrium Points in Bimatrix Games (Lemke & Howson, 1964); A Value for n-Person Games (Shapley, 1953)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NashShapleyTool",
        display_name: "Nash & Shapley Equilibria",
        domain: "game_theory",
        tags: &["nash", "lemke-howson", "minimax", "shapley", "core", "bimatrix"],
        build: || Box::new(NashShapleyTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "nash_matching_pennies_rps",
        title: "Zero-Sum RPS Minimax & Glove Market",
        description: "v* = 0, p* = (1/3, 1/3, 1/3): Symmetric zero-sum Rock-Paper-Scissors paired with a 3-player glove market.",
        category: "Game Theory",
        icon: "⚖️",
        domain: "game_theory",
        tool_name: Some("NashShapleyTool"),
        preset_generator: || presets::PRESET_RPS_ZERO_SUM,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "shapley_airport_cost_sharing",
        title: "Airport Runway Shapley & Core Allocation",
        description: "φ_i(v) ∈ Core(v): Convex 4-player coalition synergy game verifying Shapley Core membership.",
        category: "Game Theory",
        icon: "🤝",
        domain: "game_theory",
        tool_name: Some("NashShapleyTool"),
        preset_generator: || presets::PRESET_CONVEX_SYNERGY_SHAPLEY,
    }
}
