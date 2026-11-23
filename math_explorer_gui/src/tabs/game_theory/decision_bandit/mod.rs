//! Interactive Sequential Decision Trees, EVPI/EVSI, and Multi-Armed Bandits (`DecisionBanditTool`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::applied::decision_theory::{
    backward_induction, evpi, evsi, simulate_bernoulli_thompson, simulate_bernoulli_ucb1,
    DecisionEdge, DecisionNode, DecisionTreeView, DecisionTreeWorkspace, RiskAttitude,
    ThompsonBandit, Ucb1Bandit, VoiWorkspace,
};
use oxidize_core::rng::OxidizeRng;
use serde::{Deserialize, Serialize};

pub mod presets;
mod ui;

#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionWorkbenchMode {
    DecisionTreeVoi,
    BanditRegret,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiskMode {
    Neutral,
    Cara,
    Crra,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DecisionBanditConfig {
    /// Exploration constant c >= 0 (first scalar field for perturbation tests).
    pub ucb_c: f64,
    pub mode: DecisionWorkbenchMode,
    pub risk_mode: RiskMode,
    pub risk_param: f64,
    pub safe_payoff: f64,
    pub success_payoff: f64,
    pub failure_payoff: f64,
    pub prior_good: f64,
    pub signal_sensitivity: f64,
    pub signal_specificity: f64,
    pub arm_probs: Vec<f64>,
    pub horizon: usize,
    pub rng_seed: u64,
}

impl DecisionBanditConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.ucb_c.is_finite() || !(0.0..=10.0).contains(&self.ucb_c) {
            return Err("ucb_c must be finite and in [0, 10]".into());
        }
        if !self.risk_param.is_finite() || !(0.0005..=5.0).contains(&self.risk_param) {
            return Err("risk_param must be finite and in [0.0005, 5.0]".into());
        }
        for v in [self.safe_payoff, self.success_payoff, self.failure_payoff] {
            if !v.is_finite() || !(-5000.0..=5000.0).contains(&v) {
                return Err("payoffs must be finite and in [-5000, 5000]".into());
            }
            if self.risk_mode == RiskMode::Crra && v <= 0.0 {
                return Err("CRRA requires strictly positive wealth outcomes".into());
            }
        }
        for p in [
            self.prior_good,
            self.signal_sensitivity,
            self.signal_specificity,
        ] {
            if !p.is_finite() || !(0.01..=0.999).contains(&p) {
                return Err("probabilities must be in [0.01, 0.999]".into());
            }
        }
        if !(2..=6).contains(&self.arm_probs.len()) {
            return Err("arm_probs length must be in 2..=6".into());
        }
        if self
            .arm_probs
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err("arm_probs entries must be in [0, 1]".into());
        }
        if !(20..=1000).contains(&self.horizon) {
            return Err("horizon must be in 20..=1000".into());
        }
        Ok(())
    }
}

pub struct DecisionBanditTool {
    pub(crate) cfg: DecisionBanditConfig,
    pub(crate) root_ce: f64,
    pub(crate) risky_ce: f64,
    pub(crate) optimal_branch: usize,
    pub(crate) evpi_val: f64,
    pub(crate) evsi_val: f64,
    pub(crate) evsi_curve: Vec<[f64; 2]>,
    pub(crate) ucb_regret: Vec<[f64; 2]>,
    pub(crate) thompson_regret: Vec<[f64; 2]>,
    pub(crate) ucb_pulls: Vec<u64>,
    pub(crate) thompson_pulls: Vec<u64>,
    tree_ws: DecisionTreeWorkspace,
    voi_ws: VoiWorkspace,
    state_rev: u64,
}

impl Default for DecisionBanditTool {
    fn default() -> Self {
        let mut tool = Self {
            cfg: presets::default_config(),
            root_ce: 0.0,
            risky_ce: 0.0,
            optimal_branch: 0,
            evpi_val: 0.0,
            evsi_val: 0.0,
            evsi_curve: Vec::new(),
            ucb_regret: Vec::new(),
            thompson_regret: Vec::new(),
            ucb_pulls: Vec::new(),
            thompson_pulls: Vec::new(),
            tree_ws: DecisionTreeWorkspace::new(8, 8),
            voi_ws: VoiWorkspace::new(4),
            state_rev: 1,
        };
        tool.recompute();
        tool
    }
}

impl DecisionBanditTool {
    fn current_risk(&self) -> RiskAttitude {
        match self.cfg.risk_mode {
            RiskMode::Neutral => RiskAttitude::Neutral,
            RiskMode::Cara => RiskAttitude::Cara {
                alpha: self.cfg.risk_param.clamp(0.0005, 5.0),
            },
            RiskMode::Crra => RiskAttitude::Crra {
                gamma: self.cfg.risk_param.clamp(0.0005, 5.0),
            },
        }
    }

    pub(crate) fn recompute(&mut self) {
        if self.cfg.risk_mode == RiskMode::Crra {
            self.cfg.safe_payoff = self.cfg.safe_payoff.max(1.0);
            self.cfg.success_payoff = self.cfg.success_payoff.max(1.0);
            self.cfg.failure_payoff = self.cfg.failure_payoff.max(1.0);
        }
        self.recompute_decision_tree_and_voi();
        self.recompute_bandits();
        self.state_rev = self.state_rev.wrapping_add(1).max(1);
    }

    fn recompute_decision_tree_and_voi(&mut self) {
        let p = self.cfg.prior_good.clamp(0.01, 0.99);
        let nodes = [
            DecisionNode::Decision {
                edge_start: 0,
                edge_len: 2,
            },
            DecisionNode::Terminal {
                payoff: self.cfg.safe_payoff,
            },
            DecisionNode::Chance {
                edge_start: 2,
                edge_len: 2,
            },
            DecisionNode::Terminal {
                payoff: self.cfg.success_payoff,
            },
            DecisionNode::Terminal {
                payoff: self.cfg.failure_payoff,
            },
        ];
        let edges = [
            DecisionEdge {
                target: 1,
                probability: 1.0,
            },
            DecisionEdge {
                target: 2,
                probability: 1.0,
            },
            DecisionEdge {
                target: 3,
                probability: p,
            },
            DecisionEdge {
                target: 4,
                probability: 1.0 - p,
            },
        ];
        if let Ok(tree) = DecisionTreeView::new(&nodes, &edges, 0) {
            if let Ok(res) = backward_induction(tree, self.current_risk(), &mut self.tree_ws) {
                self.root_ce = res.root_certainty_equivalent;
                self.risky_ce = self.tree_ws.node_certainty_equivalent(2).unwrap_or(0.0);
                self.optimal_branch = self.tree_ws.optimal_branch(0).unwrap_or(0);
            }
        }
        self.recompute_voi_metrics(p);
    }

    fn recompute_voi_metrics(&mut self, p: f64) {
        let payoffs = [
            self.cfg.success_payoff,
            self.cfg.failure_payoff,
            self.cfg.safe_payoff,
            self.cfg.safe_payoff,
        ];
        let prior = [p, 1.0 - p];
        if let Ok(pi) = evpi(&payoffs, 2, 2, &prior) {
            self.evpi_val = pi.evpi;
        }
        let sens = self.cfg.signal_sensitivity.clamp(0.5, 0.999);
        let spec = self.cfg.signal_specificity.clamp(0.5, 0.999);
        let lik = [sens, 1.0 - sens, 1.0 - spec, spec];
        if let Ok(si) = evsi(&payoffs, 2, 2, &prior, &lik, 2, &mut self.voi_ws) {
            self.evsi_val = si.evsi;
        }

        self.evsi_curve.clear();
        for step in 0..=25 {
            let acc = 0.5 + 0.499 * (step as f64 / 25.0);
            let sweep_lik = [acc, 1.0 - acc, 1.0 - acc, acc];
            if let Ok(si) = evsi(&payoffs, 2, 2, &prior, &sweep_lik, 2, &mut self.voi_ws) {
                self.evsi_curve.push([acc, si.evsi]);
            }
        }
    }

    fn recompute_bandits(&mut self) {
        let n_arms = self.cfg.arm_probs.len();
        let horizon = self.cfg.horizon.clamp(20, 1000);
        let mut raw_ucb = vec![0.0; horizon];
        let mut raw_ts = vec![0.0; horizon];

        if let Ok(mut ucb) = Ucb1Bandit::new(n_arms, self.cfg.ucb_c.max(0.0)) {
            let mut rng = OxidizeRng::from_seed(self.cfg.rng_seed);
            let _ = simulate_bernoulli_ucb1(&mut ucb, &self.cfg.arm_probs, &mut rng, &mut raw_ucb);
            self.ucb_pulls = ucb.pull_counts().to_vec();
        }
        if let Ok(mut ts) = ThompsonBandit::new(n_arms) {
            let mut rng = OxidizeRng::from_seed(self.cfg.rng_seed ^ 0x9E37_79B9_7F4A_7C15);
            let _ =
                simulate_bernoulli_thompson(&mut ts, &self.cfg.arm_probs, &mut rng, &mut raw_ts);
            self.thompson_pulls = ts.pull_counts().to_vec();
        }
        self.ucb_regret = raw_ucb
            .into_iter()
            .enumerate()
            .map(|(t, r)| [(t + 1) as f64, r])
            .collect();
        self.thompson_regret = raw_ts
            .into_iter()
            .enumerate()
            .map(|(t, r)| [(t + 1) as f64, r])
            .collect();
    }
}

impl InteractiveTool for DecisionBanditTool {
    fn name(&self) -> &'static str {
        "Decision Trees & Bandits"
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
        ToolStateBounds::new(4096, 6, 64)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("root_certainty_equivalent", self.root_ce),
            ("evpi", self.evpi_val),
            ("evsi", self.evsi_val),
            (
                "ucb1_final_regret",
                self.ucb_regret.last().map_or(0.0, |p| p[1]),
            ),
            (
                "thompson_final_regret",
                self.thompson_regret.last().map_or(0.0, |p| p[1]),
            ),
        ]
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.cfg).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let cfg: DecisionBanditConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid DecisionBanditTool state: {e}"))?;
        cfg.validate()?;
        self.cfg = cfg;
        self.recompute();
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        match self.cfg.mode {
            DecisionWorkbenchMode::DecisionTreeVoi => vec![
                (
                    "EVSI vs Signal Accuracy".to_string(),
                    self.evsi_curve.clone(),
                ),
                (
                    "EVPI Ceiling".to_string(),
                    vec![[0.5, self.evpi_val], [1.0, self.evpi_val]],
                ),
            ],
            DecisionWorkbenchMode::BanditRegret => vec![
                (
                    "UCB1 Cumulative Regret".to_string(),
                    self.ucb_regret.clone(),
                ),
                (
                    "Thompson Cumulative Regret".to_string(),
                    self.thompson_regret.clone(),
                ),
            ],
        }
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "decision_bandit_side");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            self.show_controls(ui);
        });
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_main_view(ui);
        });
    }
}

impl scientific_metadata::theory::TheoryDescribable for DecisionBanditTool {
    fn theory_description(&self) -> String {
        "Evaluates sequential decision trees by reverse-topological certainty-equivalent folding under CARA and CRRA utilities, quantifies Blackwell's information ordering $0 \\le \\mathrm{EVSI} \\le \\mathrm{EVPI}$, and compares seeded $\\mathrm{UCB1}_i(t) = \\hat{\\mu}_i + c\\sqrt{\\ln t / n_i}$ against Beta-Bernoulli Thompson Sampling.".into()
    }

    fn phonetic_description(&self) -> String {
        "Sequential decision tree backward induction expected value of sample information and multi armed bandits".into()
    }

    fn theory_citation(&self) -> String {
        "Applied Statistical Decision Theory (Raiffa & Schlaifer, 1961); Finite-time Analysis of the Multiarmed Bandit Problem (Auer, Cesa-Bianchi & Fischer, 2002)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "DecisionBanditTool",
        display_name: "Decision Trees & Bandits",
        domain: "game_theory",
        tags: &["decision-tree", "cara", "crra", "evpi", "evsi", "ucb1", "thompson"],
        build: || Box::new(DecisionBanditTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "decision_cara_seismic_voi",
        title: "CARA Risk-Averse Drilling & EVSI",
        description: "0 ≤ EVSI ≤ EVPI: Exponential CARA certainty equivalent and Bayesian signal value curve.",
        category: "Game Theory",
        icon: "🌳",
        domain: "game_theory",
        tool_name: Some("DecisionBanditTool"),
        preset_generator: || presets::PRESET_CARA_SEISMIC_VOI,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "bandit_thompson_vs_ucb1",
        title: "Thompson vs. UCB1 Regret Race",
        description: "R_T = O(log T): Four-arm stationary Bernoulli bandit comparing Thompson Sampling and UCB1.",
        category: "Game Theory",
        icon: "🎰",
        domain: "game_theory",
        tool_name: Some("DecisionBanditTool"),
        preset_generator: || presets::PRESET_THOMPSON_VS_UCB1,
    }
}
