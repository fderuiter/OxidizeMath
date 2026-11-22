use crate::framework::math_input::{
    MathExpressionInputBar, ParameterSliderBank, MAX_SYNTHESIZED_PARAMS,
};
use crate::framework::InteractiveTool;
use crate::presets::ReplicatorPresetConfig;
use eframe::egui;
use math_commons::ast::{CompiledExpr, Expr};
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;
use nalgebra::{DMatrix, DVector};
use std::collections::{BTreeMap, BTreeSet};

#[path = "replicator/ui.rs"]
mod ui;

#[cfg(test)]
#[path = "replicator/tests.rs"]
mod tests;

const HAWK_DOVE_FORMULAS: &[&[&str]] = &[&["(v - c) / 2", "v"], &["0", "v / 2"]];
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

pub struct ReplicatorDynamicsTool {
    payoff_matrix: DMatrix<f64>,
    initial_population: DVector<f64>,
    trajectory: Vec<(f64, DVector<f64>)>,
    time_horizon: f64,
    dt: f64,
    strategy_names: Vec<String>,
    playback_time: f64,
    is_playing: bool,
    playback_speed: f64,
    symbolic_mode: bool,
    pub(crate) symbolic_bars: Vec<Vec<MathExpressionInputBar>>,
    last_valid_exprs: Vec<Vec<Expr>>,
    symbolic_compiled: Vec<Vec<Option<CompiledExpr>>>,
    pub(crate) symbolic_sliders: ParameterSliderBank,
    symbolic_error: Option<String>,
}

impl Default for ReplicatorDynamicsTool {
    fn default() -> Self {
        let payoff = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, 0.0, 1.0]);
        let init = DVector::from_vec(vec![0.1, 0.9]);
        let (bars, exprs) = build_static_symbolic_grid(HAWK_DOVE_FORMULAS);
        let rows = bars.len();
        let mut sliders = ParameterSliderBank::new().with_range(-5.0..=10.0, 2.0);
        sliders.set_params_map(&default_hawk_dove_params());
        let mut tool = Self {
            payoff_matrix: payoff,
            initial_population: init,
            trajectory: Vec::new(),
            time_horizon: 20.0,
            dt: 0.05,
            strategy_names: vec!["Hawk".to_string(), "Dove".to_string()],
            playback_time: 0.0,
            is_playing: false,
            playback_speed: 1.0,
            symbolic_mode: false,
            symbolic_bars: bars,
            last_valid_exprs: exprs,
            symbolic_compiled: vec![vec![None; rows]; rows],
            symbolic_sliders: sliders,
            symbolic_error: None,
        };
        tool.run_simulation();
        tool
    }
}

fn default_hawk_dove_params() -> BTreeMap<String, f64> {
    let mut map = BTreeMap::new();
    map.insert("c".to_string(), 4.0);
    map.insert("v".to_string(), 2.0);
    map
}

type SymbolicGrid = (Vec<Vec<MathExpressionInputBar>>, Vec<Vec<Expr>>);

fn build_static_symbolic_grid(formulas: &[&[&str]]) -> SymbolicGrid {
    let mut bars = Vec::with_capacity(formulas.len());
    let mut exprs = Vec::with_capacity(formulas.len());
    for row in formulas {
        let mut bar_row = Vec::with_capacity(row.len());
        let mut expr_row = Vec::with_capacity(row.len());
        for &cell in *row {
            let bar = MathExpressionInputBar::new("", cell, Vec::new())
                .with_axes(&[])
                .with_inline_sliders(false);
            let expr = bar.expr().cloned().unwrap_or(Expr::Number(0.0));
            bar_row.push(bar);
            expr_row.push(expr);
        }
        bars.push(bar_row);
        exprs.push(expr_row);
    }
    (bars, exprs)
}

fn build_dynamic_symbolic_grid(formulas: &[Vec<String>]) -> Result<SymbolicGrid, String> {
    let mut bars = Vec::with_capacity(formulas.len());
    let mut exprs = Vec::with_capacity(formulas.len());
    for (i, row) in formulas.iter().enumerate() {
        let mut bar_row = Vec::with_capacity(row.len());
        let mut expr_row = Vec::with_capacity(row.len());
        for (j, cell) in row.iter().enumerate() {
            let bar = MathExpressionInputBar::new("", cell, Vec::new())
                .with_axes(&[])
                .with_inline_sliders(false);
            let Some(expr) = bar.expr().cloned() else {
                let reason = bar
                    .error_message()
                    .unwrap_or_else(|| "invalid expression".to_string());
                return Err(format!("Invalid symbolic payoff at ({i}, {j}): {reason}"));
            };
            bar_row.push(bar);
            expr_row.push(expr);
        }
        bars.push(bar_row);
        exprs.push(expr_row);
    }
    Ok((bars, exprs))
}

fn compile_symbolic_exprs(
    exprs: &[Vec<Expr>],
    sliders: &mut ParameterSliderBank,
) -> Result<Vec<Vec<Option<CompiledExpr>>>, String> {
    let mut union_vars = BTreeSet::new();
    for row in exprs {
        for expr in row {
            for v in expr.free_variables() {
                union_vars.insert(v);
            }
        }
    }
    let discovered: Vec<String> = union_vars.into_iter().collect();
    if discovered.len() > MAX_SYNTHESIZED_PARAMS {
        return Err(format!(
            "Payoff matrix defines {} free parameters (maximum {MAX_SYNTHESIZED_PARAMS})",
            discovered.len()
        ));
    }
    sliders.sync_names(&discovered);
    let shared_refs: Vec<&str> = sliders.active_names().iter().map(String::as_str).collect();
    let mut compiled_grid = Vec::with_capacity(exprs.len());
    for row in exprs {
        let mut compiled_row = Vec::with_capacity(row.len());
        for expr in row {
            let compiled = CompiledExpr::compile_with_vars_and_params(expr, &[], &shared_refs)
                .map_err(|e| e.to_string())?;
            compiled_row.push(Some(compiled));
        }
        compiled_grid.push(compiled_row);
    }
    Ok(compiled_grid)
}

impl ReplicatorDynamicsTool {
    pub(crate) fn recompute_trajectory_preserving_playback(&mut self) {
        let sum = self.initial_population.sum();
        if sum > 0.0 {
            self.initial_population /= sum;
        } else {
            let n = self.initial_population.len();
            self.initial_population = DVector::from_element(n, 1.0 / n as f64);
        }

        if let Ok(system) = ReplicatorDynamics::new(self.payoff_matrix.clone()) {
            self.trajectory =
                system.simulate(self.initial_population.clone(), self.time_horizon, self.dt);
            self.playback_time = self.playback_time.clamp(0.0, self.time_horizon);
        }
    }

    pub(crate) fn run_simulation(&mut self) {
        self.recompute_trajectory_preserving_playback();
        self.playback_time = 0.0;
        self.is_playing = false;
    }

    pub(crate) fn sync_and_eval_symbolic_payoffs(&mut self) {
        let rows = self.symbolic_bars.len();
        for i in 0..rows {
            for j in 0..self.symbolic_bars[i].len() {
                if let Some(expr) = self.symbolic_bars[i][j].expr() {
                    self.last_valid_exprs[i][j] = expr.clone();
                }
            }
        }

        match compile_symbolic_exprs(&self.last_valid_exprs, &mut self.symbolic_sliders) {
            Ok(grid) => {
                self.symbolic_compiled = grid;
                self.symbolic_error = None;
                self.evaluate_symbolic_matrix();
            }
            Err(err) => {
                self.symbolic_error = Some(err);
            }
        }
    }

    pub(crate) fn evaluate_symbolic_matrix(&mut self) {
        let params = self.symbolic_sliders.param_slice();
        for (i, row) in self.symbolic_compiled.iter().enumerate() {
            for (j, cell) in row.iter().enumerate() {
                if let Some(compiled) = cell {
                    let val = compiled.eval_slice(params);
                    if val.is_finite() {
                        self.payoff_matrix[(i, j)] = val.clamp(-1e4, 1e4);
                    }
                }
            }
        }
    }

    pub(crate) fn step_once(&mut self) {
        self.is_playing = false;
        if self.trajectory.is_empty() {
            return;
        }
        self.playback_time += self.dt;
        if self.playback_time > self.time_horizon {
            self.playback_time = 0.0;
        }
    }

    pub(crate) fn reset_playback(&mut self) {
        self.is_playing = false;
        self.playback_time = 0.0;
    }

    pub(crate) fn load_preset_hawk_dove(&mut self) {
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

    pub(crate) fn load_preset_rps(&mut self) {
        self.payoff_matrix =
            DMatrix::from_row_slice(3, 3, &[0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0]);
        self.initial_population = DVector::from_vec(vec![0.33, 0.33, 0.34]);
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
}

impl InteractiveTool for ReplicatorDynamicsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Replicator Dynamics"
    }
    fn domain(&self) -> &'static str {
        "game_theory"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.export_config()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let config: ReplicatorPresetConfig = serde_json::from_str(state)
            .map_err(|e| format!("Invalid ReplicatorDynamicsTool state: {e}"))?;
        self.apply_config(config)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if self.is_playing && !self.trajectory.is_empty() {
            let dt = f64::from(parent_ui.ctx().input(|i| i.stable_dt));
            self.playback_time += dt * self.playback_speed;
            if self.playback_time > self.time_horizon {
                self.playback_time = 0.0;
            }
            parent_ui.ctx().request_repaint();
        }

        let panel_id = crate::framework::scoped_panel_id(parent_ui, "replicator_control_panel");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            self.show_settings_panel(ui);
        });

        self.show_bottom_panel(parent_ui);

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.show_trajectory_plot(ui);
        });
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ReplicatorDynamicsTool",
        display_name: "Replicator Dynamics",
        domain: "game_theory",
        tags: &[],
        build: || Box::new(ReplicatorDynamicsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for ReplicatorDynamicsTool {
    fn theory_description(&self) -> String {
        "Multi-strategy evolutionary replicator dynamics: $\\dot{x}_i = x_i \\left( (A x)_i - x^T A x \\right)$, where strategy frequency $x_i$ increases if its expected payoff $(A x)_i$ exceeds the population average fitness $x^T A x$.".into()
    }
    fn phonetic_description(&self) -> String {
        "Evolutionary game theory replicator dynamics".into()
    }
    fn theory_citation(&self) -> String {
        "Evolution and the Theory of Games (Maynard Smith, 1982); Evolutionary Games and Population Dynamics (Hofbauer & Sigmund, 1998)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

impl ReplicatorDynamicsTool {
    fn export_config(&self) -> ReplicatorPresetConfig {
        let rows = self.payoff_matrix.nrows();
        let cols = self.payoff_matrix.ncols();
        let mut matrix = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(self.payoff_matrix[(i, j)]);
            }
            matrix.push(row);
        }
        let initial_pop: Vec<f64> = self.initial_population.iter().copied().collect();
        let (symbolic_payoffs, symbolic_params) = if self.symbolic_mode {
            let sym_rows = self
                .symbolic_bars
                .iter()
                .map(|row| row.iter().map(|b| b.raw_text.trim().to_string()).collect())
                .collect();
            (Some(sym_rows), self.symbolic_sliders.export_active_map())
        } else {
            (None, BTreeMap::new())
        };

        ReplicatorPresetConfig {
            payoff_matrix: matrix,
            initial_population: initial_pop,
            time_horizon: self.time_horizon,
            dt: self.dt,
            strategy_names: self.strategy_names.clone(),
            symbolic_payoffs,
            symbolic_params,
        }
    }

    fn apply_config(&mut self, config: ReplicatorPresetConfig) -> Result<(), String> {
        config.validate()?;

        let rows = config.payoff_matrix.len();
        let mut sliders = self.symbolic_sliders.clone();
        let (sym_mode, bars, exprs, compiled) = if let Some(ref sym_rows) = config.symbolic_payoffs
        {
            let (bars, exprs) = build_dynamic_symbolic_grid(sym_rows)?;
            sliders.set_params_map(&config.symbolic_params);
            let compiled = compile_symbolic_exprs(&exprs, &mut sliders)?;
            (true, bars, exprs, compiled)
        } else {
            let formatted: Vec<Vec<String>> = config
                .payoff_matrix
                .iter()
                .map(|r| r.iter().map(|v| format!("{v}")).collect())
                .collect();
            let (bars, exprs) = build_dynamic_symbolic_grid(&formatted)?;
            (false, bars, exprs, vec![vec![None; rows]; rows])
        };

        let mut flat = Vec::with_capacity(rows * rows);
        for row in &config.payoff_matrix {
            flat.extend_from_slice(row);
        }

        self.payoff_matrix = DMatrix::from_row_slice(rows, rows, &flat);
        self.initial_population = DVector::from_vec(config.initial_population);
        self.time_horizon = config.time_horizon;
        self.dt = config.dt;
        self.strategy_names = config.strategy_names;
        self.symbolic_mode = sym_mode;
        self.symbolic_bars = bars;
        self.last_valid_exprs = exprs;
        self.symbolic_compiled = compiled;
        self.symbolic_sliders = sliders;
        self.symbolic_error = None;
        if self.symbolic_mode {
            self.evaluate_symbolic_matrix();
        }
        self.run_simulation();
        Ok(())
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
