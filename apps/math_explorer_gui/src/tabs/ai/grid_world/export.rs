#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::{ExportPrimitive, SharedPoints};
use math_explorer::ai::reinforcement_learning::{move_to_index, GridCellRole, GridState, Move};

use crate::app::{contrasting_text_color, ThemeMode};
use crate::framework::{ExportCell, ExportTable};
use crate::tabs::ai::rl_sculpt::solve_bellman_with_policy;

use super::GridWorldTool;

pub(in crate::tabs::ai) const fn move_label(action: Option<Move>) -> &'static str {
    match action {
        Some(Move::Up) => "Up",
        Some(Move::Down) => "Down",
        Some(Move::Left) => "Left",
        Some(Move::Right) => "Right",
        None => "None",
    }
}

pub(in crate::tabs::ai) const fn role_label(role: GridCellRole, is_start: bool) -> &'static str {
    match role {
        GridCellRole::Goal => "Goal",
        GridCellRole::Trap => "Trap",
        GridCellRole::SolidWall => "Wall",
        GridCellRole::WindUp => "WindUp",
        GridCellRole::WindDown => "WindDown",
        GridCellRole::WindLeft => "WindLeft",
        GridCellRole::WindRight => "WindRight",
        GridCellRole::Free if is_start => "Start",
        GridCellRole::Free => "Free",
    }
}

pub(in crate::tabs::ai) const fn move_plot_delta(action: Move) -> (f64, f64) {
    match action {
        Move::Up => (0.0, 0.24),
        Move::Down => (0.0, -0.24),
        Move::Left => (-0.24, 0.0),
        Move::Right => (0.24, 0.0),
    }
}

impl GridWorldTool {
    pub(super) fn build_grid_world_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = ThemeMode::Dark.palette();
        let bounds = self.overlay_bounds();
        let owned_bellman;
        let bellman = if self.bellman_dirty {
            owned_bellman = solve_bellman_with_policy(&self.mdp, &self.workspace, self.solver_mode);
            &owned_bellman
        } else {
            &self.bellman
        };
        let w = self.mdp.width.max(1);
        let h = self.mdp.height.max(1);
        let mut prims = Vec::with_capacity((w * h * 2 + 2) as usize);

        for y in 0..h {
            let py = f64::from(h - 1 - y);
            for x in 0..w {
                let px = f64::from(x);
                let state = GridState { x, y };
                let fill = self.cell_fill(state, bounds, &palette);
                prims.push(ExportPrimitive::Bar {
                    name: format!("Cell ({x},{y})"),
                    min: [px, py],
                    max: [px + 0.94, py + 0.94],
                    fill,
                    stroke_color: palette.graph_edge,
                    stroke_width: 1.0,
                });
                let action = self
                    .greedy_action_for_display(state)
                    .or_else(|| bellman.optimal_action_at(state));
                if let Some(mv) = action {
                    let (dx, dy) = move_plot_delta(mv);
                    let cx = px + 0.47;
                    let cy = py + 0.47;
                    prims.push(ExportPrimitive::Arrow {
                        name: "Policy".to_string(),
                        origin: [cx - dx * 0.6, cy - dy * 0.6],
                        tip: [cx + dx, cy + dy],
                        tip_length: Some(7.0),
                        color: contrasting_text_color(fill),
                    });
                }
            }
        }
        let ax = f64::from(self.current_state.x) + 0.47;
        let ay = f64::from(h - 1 - self.current_state.y) + 0.47;
        prims.push(ExportPrimitive::Points {
            name: "Agent".to_string(),
            points: SharedPoints::from(vec![[ax, ay]]),
            color: palette.warning,
            radius: 6.5,
        });
        prims
    }

    pub(super) fn build_grid_world_tables(&self) -> Vec<ExportTable> {
        let owned_bellman;
        let bellman = if self.bellman_dirty {
            owned_bellman = solve_bellman_with_policy(&self.mdp, &self.workspace, self.solver_mode);
            &owned_bellman
        } else {
            &self.bellman
        };
        let w = self.mdp.width.max(1);
        let h = self.mdp.height.max(1);
        let mut rows = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let state = GridState { x, y };
                let s_idx = self.mdp.cell_index(state);
                let q_row = self
                    .workspace
                    .q_table()
                    .get(s_idx)
                    .copied()
                    .unwrap_or([0.0; 4]);
                let role = self.mdp.cell_role(state);
                let action = self
                    .greedy_action_for_display(state)
                    .or_else(|| bellman.optimal_action_at(state));
                rows.push(vec![
                    ExportCell::Int(i64::from(x)),
                    ExportCell::Int(i64::from(y)),
                    ExportCell::Text(role_label(role, state == self.mdp.start).to_string()),
                    ExportCell::Float(q_row[move_to_index(Move::Up)]),
                    ExportCell::Float(q_row[move_to_index(Move::Down)]),
                    ExportCell::Float(q_row[move_to_index(Move::Left)]),
                    ExportCell::Float(q_row[move_to_index(Move::Right)]),
                    ExportCell::Float(bellman.v_star_at(state)),
                    ExportCell::Float(bellman.v_pi_at(state)),
                    ExportCell::Text(move_label(action).to_string()),
                    ExportCell::Bool(state == self.current_state),
                ]);
            }
        }
        vec![ExportTable::new(
            "Grid World Policy & Values",
            vec![
                "x".into(),
                "y".into(),
                "role".into(),
                "q_up".into(),
                "q_down".into(),
                "q_left".into(),
                "q_right".into(),
                "v_star".into(),
                "v_pi".into(),
                "greedy_action".into(),
                "is_agent".into(),
            ],
            rows,
        )]
    }
}
