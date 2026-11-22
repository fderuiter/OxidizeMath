//! Symbolic AST bytecode payoff matrix evaluation and starter card for [`super::ReplicatorDynamicsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ReplicatorDynamicsTool;
use crate::framework::math_input::{
    MathExpressionInputBar, ParameterSliderBank, MAX_SYNTHESIZED_PARAMS,
};
use math_commons::ast::{CompiledExpr, Expr};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const HAWK_DOVE_FORMULAS: &[&[&str]] = &[&["(v - c) / 2", "v"], &["0", "v / 2"]];
pub(super) const RPS_FORMULAS: &[&[&str]] =
    &[&["0", "-s", "s"], &["s", "0", "-s"], &["-s", "s", "0"]];

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

pub(super) type SymbolicGrid = (Vec<Vec<MathExpressionInputBar>>, Vec<Vec<Expr>>);

pub(super) fn default_hawk_dove_params() -> BTreeMap<String, f64> {
    let mut map = BTreeMap::new();
    map.insert("c".to_string(), 4.0);
    map.insert("v".to_string(), 2.0);
    map
}

pub(super) fn build_static_symbolic_grid(formulas: &[&[&str]]) -> SymbolicGrid {
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

pub(super) fn build_dynamic_symbolic_grid(formulas: &[Vec<String>]) -> Result<SymbolicGrid, String> {
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

pub(super) fn compile_symbolic_exprs(
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
        self.recompute_spatial_trajectory();
    }

    pub(crate) fn load_preset_symbolic_hawk_dove(&mut self) {
        self.load_preset_hawk_dove();
        self.symbolic_mode = true;
        self.sync_and_eval_symbolic_payoffs();
        self.run_simulation();
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
