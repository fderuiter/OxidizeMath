#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{q_value_cell_color, QTableInspectorTool, QTableRunner, QTableSnapshot};
use crate::app::SemanticPalette;
use crate::async_sim::SimulationRunner;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui;
use egui_plot::export_primitives::{build_grid_primitives, ExportPrimitive, GridCellSpec};
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, Move};

const ALL_MOVES: [Move; 4] = [Move::Up, Move::Down, Move::Left, Move::Right];

fn move_name(action: Move) -> &'static str {
    match action {
        Move::Up => "Up",
        Move::Down => "Down",
        Move::Left => "Left",
        Move::Right => "Right",
    }
}

fn move_delta(action: Move) -> (f64, f64) {
    match action {
        Move::Up => (0.0, 0.34),
        Move::Down => (0.0, -0.34),
        Move::Left => (-0.34, 0.0),
        Move::Right => (0.34, 0.0),
    }
}

impl QTableInspectorTool {
    pub(super) fn step_export_simulation(&mut self) {
        let target = self.episodes_trained.saturating_add(5).min(250);
        let mut runner = QTableRunner::new();
        for _ in 0..target {
            runner.step_once();
        }
        let snapshot = runner.get_snapshot();
        if let Some(data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<QTableSnapshot>())
        {
            self.cached_q_values = data.q_values.clone();
            self.episodes_trained = data.episodes_trained;
        }
    }

    fn cell_role_and_best(&self, state: GridState) -> (&'static str, f64, Move) {
        let role = if state == self.env.goal {
            "goal"
        } else if self.env.traps.contains(&state) {
            "trap"
        } else if state == self.env.start {
            "start"
        } else {
            "empty"
        };
        let mut best_move = Move::Right;
        let mut best_q = f64::NEG_INFINITY;
        for action in ALL_MOVES {
            let q = self.get_q(&state, &action);
            if q > best_q {
                best_q = q;
                best_move = action;
            }
        }
        (role, best_q, best_move)
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let (min_q, max_q) = self.compute_q_bounds();
        let span = (max_q - min_q).abs();
        let total_rows = self.env.height as usize;
        let mut cells = Vec::with_capacity((self.env.width * self.env.height) as usize);
        let mut arrows = Vec::new();

        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let state = GridState { x, y };
                let val = self.get_q_value_for_view(&state);
                let norm = if span > 1e-9 {
                    (val - min_q) / (max_q - min_q)
                } else {
                    0.35
                };
                let (role, best_q, best_move) = self.cell_role_and_best(state);
                let (fill, label) = match role {
                    "goal" => (palette.success, format!("G ({val:.2})")),
                    "trap" => (palette.error, format!("T ({val:.2})")),
                    "start" => (q_value_cell_color(norm, &palette), format!("S ({val:.2})")),
                    _ => (q_value_cell_color(norm, &palette), format!("{val:.2}")),
                };
                cells.push(GridCellSpec {
                    row: y as usize,
                    col: x as usize,
                    total_rows,
                    fill,
                    stroke_color: palette.node_stroke,
                    stroke_width: 1.2,
                    label,
                });
                if role != "goal" && role != "trap" && best_q.abs() > 1e-9 {
                    let cx = f64::from(x) + 0.47;
                    let cy = f64::from(self.env.height - 1 - y) + 0.47;
                    let (dx, dy) = move_delta(best_move);
                    arrows.push(ExportPrimitive::Arrow {
                        name: format!("Greedy ({x},{y})"),
                        origin: [cx - dx * 0.45, cy - dy * 0.45],
                        tip: [cx + dx * 0.55, cy + dy * 0.55],
                        tip_length: Some(6.0),
                        color: egui::Color32::WHITE,
                    });
                }
            }
        }

        let mut out = build_grid_primitives(&cells);
        out.extend(arrows);
        out
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let mut rows = Vec::with_capacity((self.env.width * self.env.height) as usize);
        for y in 0..self.env.height {
            for x in 0..self.env.width {
                let state = GridState { x, y };
                let (role, best_q, best_move) = self.cell_role_and_best(state);
                rows.push(vec![
                    ExportCell::Int(i64::from(x)),
                    ExportCell::Int(i64::from(y)),
                    ExportCell::Text(role.into()),
                    ExportCell::Float(self.get_q(&state, &Move::Up)),
                    ExportCell::Float(self.get_q(&state, &Move::Down)),
                    ExportCell::Float(self.get_q(&state, &Move::Left)),
                    ExportCell::Float(self.get_q(&state, &Move::Right)),
                    ExportCell::Float(best_q),
                    ExportCell::Text(move_name(best_move).into()),
                ]);
            }
        }
        vec![ExportTable::new(
            "Q-Table",
            vec![
                "x".into(),
                "y".into(),
                "role".into(),
                "q_up".into(),
                "q_down".into(),
                "q_left".into(),
                "q_right".into(),
                "max_q".into(),
                "best_action".into(),
            ],
            rows,
        )]
    }
}
