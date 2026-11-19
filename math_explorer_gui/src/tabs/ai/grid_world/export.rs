#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use egui_plot::export_primitives::ExportPrimitive;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, Move};

impl GridWorldTool {
    pub(super) fn cold_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut primitives =
            Vec::with_capacity((self.env.width * self.env.height).max(0) as usize + 1);

        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let state = GridState { x, y };
                let fill = if state == self.env.goal {
                    palette.success
                } else if self.env.traps.contains(&state) {
                    palette.error
                } else if state == self.current_state {
                    palette.accent
                } else if state == self.env.start {
                    palette.info
                } else {
                    palette.surface_fill
                };
                let cx = f64::from(x);
                let cy = f64::from(self.env.height - 1 - y);
                primitives.push(ExportPrimitive::Bar {
                    name: format!("cell_{x}_{y}"),
                    min: [cx - 0.46, cy - 0.46],
                    max: [cx + 0.46, cy + 0.46],
                    fill,
                    stroke_color: palette.graph_edge,
                    stroke_width: 1.0,
                });
            }
        }

        primitives.push(ExportPrimitive::Points {
            name: "agent_position".to_string(),
            points: vec![[
                f64::from(self.current_state.x),
                f64::from(self.env.height - 1 - self.current_state.y),
            ]],
            color: palette.accent,
            radius: 6.5,
        });

        primitives
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut max_q_series = Vec::new();
        let mut coords = Vec::new();
        for y in 0..self.env.height {
            for x in 0..self.env.width {
                let state = GridState { x, y };
                let max_q = [Move::Up, Move::Down, Move::Left, Move::Right]
                    .iter()
                    .map(|a| self.agent.get_q_value(&state, a))
                    .fold(f64::NEG_INFINITY, f64::max);
                let val = if max_q.is_finite() { max_q } else { 0.0 };
                max_q_series.push([f64::from(y * self.env.width + x), val]);
                coords.push([f64::from(x), f64::from(y)]);
            }
        }

        vec![
            (
                "agent_position".to_string(),
                vec![[
                    f64::from(self.current_state.x),
                    f64::from(self.current_state.y),
                ]],
            ),
            ("state_max_q_values".to_string(), max_q_series),
            ("grid_coordinates".to_string(), coords),
        ]
    }
}
