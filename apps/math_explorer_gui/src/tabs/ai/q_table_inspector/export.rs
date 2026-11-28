#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::ExportPrimitive;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, Move};
use math_explorer::ai::reinforcement_learning::{
    move_to_index, DenseRlWorkspace, GridCellRole, MarkovDecisionProcess,
};

use crate::app::{contrasting_text_color, ThemeMode};
use crate::async_sim::SimCommand;
use crate::framework::{ExportCell, ExportTable};
use crate::tabs::ai::grid_world::export::{move_label, move_plot_delta, role_label};
use crate::tabs::ai::rl_sculpt::{role_base_fill, RlSolverMode};

use super::{q_value_cell_color, HeatmapView, QTableInspectorTool, SculptedQTableRunner};

impl QTableInspectorTool {
    fn cell_export_scalar(&self, state: GridState) -> f64 {
        if self.episodes_trained == 0 && matches!(self.view, HeatmapView::MaxQ) {
            self.latest_snap.bellman.v_star_at(state)
        } else {
            self.get_q_value_for_view(&state)
        }
    }

    fn export_scalar_bounds(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for y in 0..self.mdp.height {
            for x in 0..self.mdp.width {
                let st = GridState { x, y };
                let role = self.mdp.cell_role(st);
                if self.mdp.is_terminal(&st) || role == GridCellRole::SolidWall {
                    continue;
                }
                let val = self.cell_export_scalar(st);
                if val.is_finite() {
                    lo = lo.min(val);
                    hi = hi.max(val);
                }
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            return (0.0, 1.0);
        }
        if (hi - lo).abs() < 1e-6 {
            hi = lo + 1.0;
        }
        (lo, hi)
    }

    pub(super) fn build_q_table_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = ThemeMode::Dark.palette();
        let (lo, hi) = self.export_scalar_bounds();
        let w = self.mdp.width.max(1);
        let h = self.mdp.height.max(1);
        let mut prims = Vec::with_capacity((w * h * 2) as usize);

        for y in 0..h {
            let py = f64::from(h - 1 - y);
            for x in 0..w {
                let px = f64::from(x);
                let st = GridState { x, y };
                let role = self.mdp.cell_role(st);
                let fill = if matches!(
                    role,
                    GridCellRole::Goal | GridCellRole::Trap | GridCellRole::SolidWall
                ) {
                    role_base_fill(role, &palette)
                } else {
                    let v = self.cell_export_scalar(st);
                    let norm = ((v - lo) / (hi - lo)).clamp(0.0, 1.0);
                    q_value_cell_color(norm, &palette)
                };
                prims.push(ExportPrimitive::Bar {
                    name: format!("State ({x},{y})"),
                    min: [px, py],
                    max: [px + 0.94, py + 0.94],
                    fill,
                    stroke_color: palette.graph_edge,
                    stroke_width: 1.0,
                });

                if !self.mdp.is_terminal(&st) && role != GridCellRole::SolidWall {
                    let idx = flatten_2d_index(x as usize, y as usize, w as usize);
                    let action = self
                        .latest_snap
                        .greedy_actions
                        .get(idx)
                        .copied()
                        .flatten()
                        .filter(|_| self.episodes_trained > 0)
                        .or_else(|| self.latest_snap.bellman.optimal_action_at(st));
                    if let Some(mv) = action {
                        let (dx, dy) = move_plot_delta(mv);
                        let cx = px + 0.47;
                        let cy = py + 0.47;
                        prims.push(ExportPrimitive::Arrow {
                            name: "Optimal Policy".to_string(),
                            origin: [cx - dx * 0.6, cy - dy * 0.6],
                            tip: [cx + dx, cy + dy],
                            tip_length: Some(7.0),
                            color: contrasting_text_color(fill),
                        });
                    }
                }
            }
        }
        prims
    }

    fn build_q_table_row(&self, st: GridState, w: i32) -> Vec<ExportCell> {
        let idx = flatten_2d_index(st.x as usize, st.y as usize, w as usize);
        let q_row = self
            .latest_snap
            .q_table
            .get(idx)
            .copied()
            .unwrap_or([0.0; 4]);
        let max_q = match self.solver_mode {
            RlSolverMode::TabularQ => q_row.iter().copied().fold(q_row[0], f64::max),
            RlSolverMode::PolicyGradient => self
                .latest_snap
                .state_values
                .get(idx)
                .copied()
                .unwrap_or(0.0),
        };
        let v_star = self.latest_snap.bellman.v_star_at(st);
        let v_pi = self.latest_snap.bellman.v_pi_at(st);
        let action = self
            .latest_snap
            .greedy_actions
            .get(idx)
            .copied()
            .flatten()
            .filter(|_| self.episodes_trained > 0)
            .or_else(|| self.latest_snap.bellman.optimal_action_at(st));
        let role = self.mdp.cell_role(st);
        vec![
            ExportCell::Int(i64::from(st.x)),
            ExportCell::Int(i64::from(st.y)),
            ExportCell::Text(role_label(role, st == self.mdp.start).to_string()),
            ExportCell::Float(q_row[move_to_index(Move::Up)]),
            ExportCell::Float(q_row[move_to_index(Move::Down)]),
            ExportCell::Float(q_row[move_to_index(Move::Left)]),
            ExportCell::Float(q_row[move_to_index(Move::Right)]),
            ExportCell::Float(max_q),
            ExportCell::Float(v_star),
            ExportCell::Float(v_pi),
            ExportCell::Float((max_q - v_star).abs()),
            ExportCell::Text(move_label(action).to_string()),
        ]
    }

    pub(super) fn build_q_table_export_tables(&self) -> Vec<ExportTable> {
        let (w, h) = (self.mdp.width.max(1), self.mdp.height.max(1));
        let mut rows = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                rows.push(self.build_q_table_row(GridState { x, y }, w));
            }
        }
        let headers = [
            "x",
            "y",
            "role",
            "q_up",
            "q_down",
            "q_left",
            "q_right",
            "max_q",
            "v_star",
            "v_pi",
            "bellman_residual",
            "greedy_action",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        vec![ExportTable::new("Q-Table & Bellman Values", headers, rows)]
    }

    pub(super) fn step_q_table_episode(&mut self) {
        let seed = 0x517A_B1E0_u64.wrapping_add(u64::from(self.episodes_trained));
        let step_budget = super::episode_step_budget(&self.mdp);
        let mut runner = SculptedQTableRunner::from_mdp(self.mdp.clone(), self.solver_mode);
        runner.workspace = DenseRlWorkspace::new(
            self.mdp.width as usize,
            self.mdp.height as usize,
            step_budget,
            Some(seed),
        );
        if self.latest_snap.q_table.len() == runner.workspace.q_table().len()
            && self.latest_snap.state_values.len() == runner.workspace.state_values().len()
        {
            runner
                .workspace
                .q_table_mut()
                .copy_from_slice(&self.latest_snap.q_table);
            if self.latest_snap.theta_logits.len() == runner.workspace.theta_logits().len() {
                runner
                    .workspace
                    .theta_logits_mut()
                    .copy_from_slice(&self.latest_snap.theta_logits);
            }
            runner
                .workspace
                .state_values_mut()
                .copy_from_slice(&self.latest_snap.state_values);
        }
        runner.episodes_trained = self.episodes_trained;
        runner.run_one_episode();
        self.latest_audio = runner.latest_audio;
        self.episodes_trained = runner.episodes_trained;
        self.latest_snap = runner.build_q_snapshot();
        let _ = self.worker.send_command(SimCommand::Step(1));
        self.revision = self.revision.wrapping_add(1).max(1);
    }
}

impl scientific_metadata::theory::TheoryDescribable for QTableInspectorTool {
    fn theory_description(&self) -> String {
        "Bellman optimality $V^*(s) = \\max_a \\sum_{s'} P(s'|s,a)[R + \\gamma V^*(s')]$ compared against empirical temporal-difference $Q(s,a)$ and policy-gradient $V_\\pi(s)$ value tables.".into()
    }
    fn phonetic_description(&self) -> String {
        "Q-learning and Bellman optimality equation theory".into()
    }
    fn theory_citation(&self) -> String {
        "Reinforcement Learning: An Introduction (Sutton & Barto, 2018)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
