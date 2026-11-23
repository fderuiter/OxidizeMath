#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{build_static_symbolic_grid, ReplicatorDynamicsTool};
use crate::framework::math_input::{MathExpressionInputBar, ParameterSliderBank};
use math_commons::ast::Expr;
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

pub(super) const HAWK_DOVE_FORMULAS: &[&[&str]] = &[&["(v - c) / 2", "v"], &["0", "v / 2"]];
const RPS_FORMULAS: &[&[&str]] = &[&["0", "-s", "s"], &["s", "0", "-s"], &["-s", "s", "0"]];

const PRESET_SYMBOLIC_HAWK_DOVE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Symbolic Hawk-Dove Cost-Benefit",
    "domain": "game_theory",
    "timestamp": "2026-05-01T00:00:00Z"
  },
  "config": {
    "payoff_matrix": [[-1.0, 2.0], [0.0, 1.0]],
    "initial_population": [0.1, 0.9],
    "time_horizon": 20.0,
    "dt": 0.05,
    "strategy_names": ["Hawk", "Dove"],
    "symbolic_payoffs": [["(v - c) / 2", "v"], ["0", "v / 2"]],
    "symbolic_params": {"c": 4.0, "v": 2.0}
  }
}"#;

pub(super) fn default_hawk_dove_params() -> BTreeMap<String, f64> {
    let mut map = BTreeMap::new();
    map.insert("c".to_string(), 4.0);
    map.insert("v".to_string(), 2.0);
    map
}

impl ReplicatorDynamicsTool {
    pub(super) fn reset_symbolic_from_numeric_matrix(&mut self) {
        let rows = self.payoff_matrix.nrows();
        let cols = self.payoff_matrix.ncols();
        let mut bars = Vec::with_capacity(rows);
        let mut exprs = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut bar_row = Vec::with_capacity(cols);
            let mut expr_row = Vec::with_capacity(cols);
            for j in 0..cols {
                let val = self.payoff_matrix[(i, j)];
                let text = format!("{val}");
                let bar = MathExpressionInputBar::new("", &text, Vec::new())
                    .with_axes(&[])
                    .with_inline_sliders(false);
                let expr = bar.expr().cloned().unwrap_or(Expr::Number(val));
                bar_row.push(bar);
                expr_row.push(expr);
            }
            bars.push(bar_row);
            exprs.push(expr_row);
        }
        self.symbolic_mode = false;
        self.symbolic_bars = bars;
        self.last_valid_exprs = exprs;
        self.symbolic_compiled = vec![vec![None; cols]; rows];
        self.symbolic_sliders = ParameterSliderBank::new().with_range(-5.0..=10.0, 2.0);
        self.symbolic_error = None;
    }

    pub(crate) fn load_preset_hawk_dove(&mut self) {
        self.spatial.seeded_graph = None;
        self.payoff_matrix = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, 1.0]);
        self.initial_population = DVector::from_vec(vec![0.1, 0.9]);
        self.strategy_names = vec!["Hawk".to_string(), "Dove".to_string()];
        self.time_horizon = 20.0;
        self.symbolic_mode = false;
        let (bars, exprs) = build_static_symbolic_grid(HAWK_DOVE_FORMULAS);
        self.symbolic_bars = bars;
        self.last_valid_exprs = exprs;
        self.symbolic_compiled = vec![vec![None; 2]; 2];
        self.symbolic_sliders
            .set_params_map(&default_hawk_dove_params());
        self.symbolic_error = None;
        self.run_simulation();
    }

    pub(crate) fn load_preset_symbolic_hawk_dove(&mut self) {
        self.load_preset_hawk_dove();
        self.symbolic_mode = true;
        self.sync_and_eval_symbolic_payoffs();
        self.run_simulation();
    }

    pub(crate) fn load_preset_prisoners_dilemma(&mut self) {
        self.spatial.seeded_graph = None;
        self.payoff_matrix = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 5.0, 1.0]);
        self.initial_population = DVector::from_vec(vec![0.9, 0.1]);
        self.strategy_names = vec!["Cooperate".to_string(), "Defect".to_string()];
        self.time_horizon = 10.0;
        self.reset_symbolic_from_numeric_matrix();
        self.run_simulation();
    }

    pub(crate) fn load_preset_stag_hunt(&mut self) {
        self.spatial.seeded_graph = None;
        self.payoff_matrix = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 3.0, 2.0]);
        self.initial_population = DVector::from_vec(vec![0.55, 0.45]);
        self.strategy_names = vec!["Stag".to_string(), "Hare".to_string()];
        self.time_horizon = 12.0;
        self.reset_symbolic_from_numeric_matrix();
        self.run_simulation();
    }

    pub(crate) fn load_preset_rps(&mut self) {
        self.spatial.seeded_graph = None;
        self.payoff_matrix =
            DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0]);
        self.initial_population = DVector::from_vec(vec![0.50, 0.30, 0.20]);
        self.strategy_names = vec![
            "Rock".to_string(),
            "Paper".to_string(),
            "Scissors".to_string(),
        ];
        self.time_horizon = 50.0;
        self.symbolic_mode = false;
        let (bars, exprs) = build_static_symbolic_grid(RPS_FORMULAS);
        self.symbolic_bars = bars;
        self.last_valid_exprs = exprs;
        self.symbolic_compiled = vec![vec![None; 3]; 3];
        let mut rps_params = BTreeMap::new();
        rps_params.insert("s".to_string(), 1.0);
        self.symbolic_sliders.set_params_map(&rps_params);
        self.symbolic_error = None;
        self.run_simulation();
    }

    pub(crate) fn load_preset_spiral_rps(&mut self) {
        self.spatial.seeded_graph = None;
        self.payoff_matrix =
            DMatrix::from_row_slice(3, 3, &[0.0, -0.7, 1.0, 1.0, 0.0, -0.7, -0.7, 1.0, 0.0]);
        self.initial_population = DVector::from_vec(vec![0.55, 0.25, 0.20]);
        self.strategy_names = vec![
            "Rock".to_string(),
            "Paper".to_string(),
            "Scissors".to_string(),
        ];
        self.time_horizon = 40.0;
        self.reset_symbolic_from_numeric_matrix();
        self.run_simulation();
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hawk_dove",
        title: "Hawk-Dove Dynamics",
        description: "ẋ_i = x_i((Ax)_i - xᵀAx): Evolutionary game model of conflict and resource competition.",
        category: "Game Theory",
        icon: "🦅",
        domain: "game_theory",
        tool_name: Some("ReplicatorDynamicsTool"),
        preset_generator: || crate::starter_gallery::PRESET_HAWK_DOVE,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "replicator_symbolic_hawk_dove",
        title: "Symbolic Hawk-Dove Cost-Benefit",
        description: "A = [[(v-c)/2, v], [0, v/2]]: Symbolic evolutionary payoff matrix with live resource (v) and injury cost (c) sliders.",
        category: "Game Theory",
        icon: "🧬",
        domain: "game_theory",
        tool_name: Some("ReplicatorDynamicsTool"),
        preset_generator: || PRESET_SYMBOLIC_HAWK_DOVE,
    }
}
