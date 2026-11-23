#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui;
use egui_plot::export_primitives::{build_grid_primitives, ExportPrimitive, GridCellSpec};
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, Move};

const ALL_MOVES: [Move; 4] = [Move::Up, Move::Down, Move::Left, Move::Right];

fn greedy_delta(action: Move) -> (f64, f64) {
    match action {
        Move::Up => (0.0, 0.32),
        Move::Down => (0.0, -0.32),
        Move::Left => (-0.32, 0.0),
        Move::Right => (0.32, 0.0),
    }
}

impl GridWorldTool {
    fn cell_role(&self, state: GridState) -> &'static str {
        if state == self.env.goal {
            "goal"
        } else if self.env.traps.contains(&state) {
            "trap"
        } else if state == self.env.start {
            "start"
        } else {
            "empty"
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let total_rows = self.env.height as usize;
        let mut cells = Vec::with_capacity((self.env.width * self.env.height) as usize);
        let mut overlays = Vec::new();

        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let state = GridState { x, y };
                let role = self.cell_role(state);
                let label = if state == self.current_state {
                    "Agent".into()
                } else {
                    format!("{role} ({x},{y})")
                };
                cells.push(GridCellSpec {
                    row: y as usize,
                    col: x as usize,
                    total_rows,
                    fill: self.cell_fill(state, &palette),
                    stroke_color: palette.node_stroke,
                    stroke_width: 1.2,
                    label,
                });

                let mut best_move = Move::Right;
                let mut best_q = f64::NEG_INFINITY;
                for m in ALL_MOVES {
                    let q = self.agent.get_q_value(&state, &m);
                    if q > best_q {
                        best_q = q;
                        best_move = m;
                    }
                }
                if role != "goal" && role != "trap" && best_q.abs() > 1e-9 {
                    let cx = f64::from(x) + 0.47;
                    let cy = f64::from(self.env.height - 1 - y) + 0.47;
                    let (dx, dy) = greedy_delta(best_move);
                    overlays.push(ExportPrimitive::Arrow {
                        name: format!("Policy ({x},{y})"),
                        origin: [cx - dx * 0.4, cy - dy * 0.4],
                        tip: [cx + dx * 0.5, cy + dy * 0.5],
                        tip_length: Some(6.0),
                        color: egui::Color32::WHITE,
                    });
                }
            }
        }

        let agent_cx = f64::from(self.current_state.x) + 0.47;
        let agent_cy = f64::from(self.env.height - 1 - self.current_state.y) + 0.47;
        overlays.push(ExportPrimitive::Points {
            name: "Agent".into(),
            points: vec![[agent_cx, agent_cy]],
            color: palette.warning,
            radius: 7.5,
        });

        let mut out = build_grid_primitives(&cells);
        out.extend(overlays);
        out
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let mut state_rows = Vec::with_capacity((self.env.width * self.env.height) as usize);
        for y in 0..self.env.height {
            for x in 0..self.env.width {
                let state = GridState { x, y };
                state_rows.push(vec![
                    ExportCell::Int(i64::from(x)),
                    ExportCell::Int(i64::from(y)),
                    ExportCell::Text(self.cell_role(state).into()),
                    ExportCell::Bool(state == self.current_state),
                    ExportCell::Float(self.agent.get_q_value(&state, &Move::Up)),
                    ExportCell::Float(self.agent.get_q_value(&state, &Move::Down)),
                    ExportCell::Float(self.agent.get_q_value(&state, &Move::Left)),
                    ExportCell::Float(self.agent.get_q_value(&state, &Move::Right)),
                ]);
            }
        }
        let summary_row = vec![vec![
            ExportCell::UInt(u64::from(self.episodes)),
            ExportCell::UInt(u64::from(self.steps)),
            ExportCell::Float(self.total_reward),
            ExportCell::Float(self.learning_rate),
            ExportCell::Float(self.discount_factor),
            ExportCell::Float(self.exploration_rate),
        ]];
        vec![
            ExportTable::new(
                "Grid States",
                vec![
                    "x".into(),
                    "y".into(),
                    "role".into(),
                    "is_agent".into(),
                    "q_up".into(),
                    "q_down".into(),
                    "q_left".into(),
                    "q_right".into(),
                ],
                state_rows,
            ),
            ExportTable::new(
                "Episode Summary",
                vec![
                    "episodes".into(),
                    "steps".into(),
                    "total_reward".into(),
                    "learning_rate".into(),
                    "discount_factor".into(),
                    "exploration_rate".into(),
                ],
                summary_row,
            ),
        ]
    }
}
