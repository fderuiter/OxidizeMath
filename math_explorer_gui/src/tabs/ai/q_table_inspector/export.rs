#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{q_value_cell_color, QTableInspectorTool};
use egui_plot::export_primitives::ExportPrimitive;
use math_explorer::ai::reinforcement_learning::{
    grid_world::{GridState, Move},
    MarkovDecisionProcess,
};

impl QTableInspectorTool {
    pub(super) fn step_export_episode(&mut self) {
        let alpha = 0.25_f64;
        let gamma = self.env.gamma;
        let all_moves = [Move::Up, Move::Down, Move::Left, Move::Right];
        let mut state = self.env.start;

        for step_idx in 0..64usize {
            if self.env.is_terminal(&state) {
                break;
            }
            let actions = self.env.actions(&state);
            if actions.is_empty() {
                break;
            }

            let explore_hash = (self.episodes_trained as usize)
                .wrapping_mul(31)
                .wrapping_add(step_idx);
            let chosen = if explore_hash % 4 == 0 {
                actions[explore_hash % actions.len()]
            } else {
                *actions
                    .iter()
                    .max_by(|a, b| {
                        self.get_q(&state, a)
                            .partial_cmp(&self.get_q(&state, b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(&actions[0])
            };

            let next_state = self.env.step(&state, &chosen);
            let reward = self.env.reward(&state, &chosen, &next_state);
            let max_next_q = if self.env.is_terminal(&next_state) {
                0.0
            } else {
                all_moves
                    .iter()
                    .map(|m| self.get_q(&next_state, m))
                    .fold(0.0_f64, f64::max)
            };
            let old_q = self.get_q(&state, &chosen);
            let updated_q = old_q + alpha * (reward + gamma * max_next_q - old_q);
            self.cached_q_values.insert((state, chosen), updated_q);
            state = next_state;
        }
        self.episodes_trained = self.episodes_trained.saturating_add(1);
    }

    pub(super) fn cold_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let (min_q, max_q) = self.compute_q_bounds();
        let mut primitives =
            Vec::with_capacity((self.env.width * self.env.height).max(0) as usize);

        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let state = GridState { x, y };
                let q_val = self.get_q_value_for_view(&state);
                let fill = if state == self.env.goal {
                    palette.success
                } else if self.env.traps.contains(&state) {
                    palette.error
                } else {
                    let norm = ((q_val - min_q) / (max_q - min_q)).clamp(0.0, 1.0);
                    q_value_cell_color(norm, &palette)
                };
                let cx = f64::from(x);
                let cy = f64::from(self.env.height - 1 - y);
                primitives.push(ExportPrimitive::Bar {
                    name: format!("q_cell_{x}_{y}"),
                    min: [cx - 0.46, cy - 0.46],
                    max: [cx + 0.46, cy + 0.46],
                    fill,
                    stroke_color: palette.graph_edge,
                    stroke_width: 1.0,
                });
            }
        }

        primitives
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut q_series = Vec::new();
        let mut coords = Vec::new();
        for y in 0..self.env.height {
            for x in 0..self.env.width {
                let state = GridState { x, y };
                q_series.push([
                    f64::from(y * self.env.width + x),
                    self.get_q_value_for_view(&state),
                ]);
                coords.push([f64::from(x), f64::from(y)]);
            }
        }

        vec![
            ("q_heatmap_values".to_string(), q_series),
            ("grid_coordinates".to_string(), coords),
        ]
    }
}
