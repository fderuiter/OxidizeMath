#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkMetricsTool;
use eframe::egui;
use egui_plot::export_primitives::ExportPrimitive;

impl NetworkMetricsTool {
    pub(super) fn cold_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut primitives = Vec::with_capacity(self.edges.len() + self.node_positions.len());

        for (idx, &(u, v, _)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                primitives.push(ExportPrimitive::Polyline {
                    name: format!("edge_{idx}"),
                    points: vec![
                        [f64::from(pos_u.x), -f64::from(pos_u.y)],
                        [f64::from(pos_v.x), -f64::from(pos_v.y)],
                    ],
                    color: palette.graph_edge,
                    stroke_width: 2.0,
                    closed: false,
                    fill: None,
                });
            }
        }

        let mut max_val = 0.0_f64;
        let mut min_val = f64::MAX;
        for &val in self.metric_values.values() {
            max_val = max_val.max(val);
            min_val = min_val.min(val);
        }
        let range = max_val - min_val;

        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let pos = self.node_positions[&id];
            let val = self.metric_values.get(&id).copied().unwrap_or(0.0);
            let t = if range > 0.0 {
                ((val - min_val) / range).clamp(0.0, 1.0) as f32
            } else {
                0.5
            };
            let low = palette.axis_z;
            let high = palette.axis_x;
            let lerp =
                |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t).round() as u8;
            let color = egui::Color32::from_rgb(
                lerp(low.r(), high.r()),
                lerp(low.g(), high.g()),
                lerp(low.b(), high.b()),
            );
            primitives.push(ExportPrimitive::Points {
                name: format!("node_{id}"),
                points: vec![[f64::from(pos.x), -f64::from(pos.y)]],
                color,
                radius: 9.0,
            });
        }

        primitives
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();

        let centrality = sorted_ids
            .iter()
            .map(|id| [*id as f64, self.metric_values.get(id).copied().unwrap_or(0.0)])
            .collect();
        let positions = sorted_ids
            .iter()
            .map(|id| {
                let p = self.node_positions[id];
                [f64::from(p.x), f64::from(p.y)]
            })
            .collect();
        let edges = self
            .edges
            .iter()
            .map(|&(u, v, _)| [u as f64, v as f64])
            .collect();

        vec![
            ("node_centrality".to_string(), centrality),
            ("node_positions".to_string(), positions),
            ("graph_edges".to_string(), edges),
        ]
    }
}
