#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NeuralNetworkVizTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui::{self, Color32};
use egui_plot::export_primitives::{
    build_graph_primitives, ExportPrimitive, GraphEdgeSpec, GraphNodeSpec,
};

fn neuron_voltage_color(v_mv: f64, palette: &SemanticPalette) -> Color32 {
    let t = (((v_mv as f32) + 65.0) / 105.0).clamp(0.0, 1.0);
    let low = palette.axis_z;
    let high = palette.axis_x;
    let lerp = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t).round() as u8;
    Color32::from_rgb(
        lerp(low.r(), high.r()),
        lerp(low.g(), high.g()),
        lerp(low.b(), high.b()),
    )
}

impl NeuralNetworkVizTool {
    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let n = self.neurons.len();
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            let v = self.neurons[i].v();
            let pos = self.positions.get(i).copied().unwrap_or([0.0, 0.0]);
            nodes.push(GraphNodeSpec {
                id: i,
                pos: [f64::from(pos[0]), f64::from(pos[1])],
                fill: neuron_voltage_color(v, &palette),
                stroke_color: palette.node_stroke,
                radius: 12.0,
                label: format!("N{i} ({v:.0}mV)"),
            });
        }

        let mut edges = Vec::new();
        for post in 0..n {
            for pre in 0..n {
                let w = self
                    .weights
                    .get(post)
                    .and_then(|r| r.get(pre))
                    .copied()
                    .unwrap_or(0.0);
                if w.abs() > 1e-12 {
                    let base = if w > 0.0 {
                        palette.success
                    } else {
                        palette.error
                    };
                    edges.push(GraphEdgeSpec {
                        from_id: pre,
                        to_id: post,
                        color: Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 200),
                        width: (1.0 + w.abs() as f32).min(5.0),
                        directed: true,
                        label: format!("N{pre}->N{post}"),
                    });
                }
            }
        }
        build_graph_primitives(&nodes, &edges, true)
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let n = self.neurons.len();
        let mut neuron_rows = Vec::with_capacity(n);
        for i in 0..n {
            let pos = self.positions.get(i).copied().unwrap_or([0.0, 0.0]);
            let ext = self.external_input.get(i).copied().unwrap_or(0.0);
            neuron_rows.push(vec![
                ExportCell::UInt(i as u64),
                ExportCell::Float(f64::from(pos[0])),
                ExportCell::Float(f64::from(pos[1])),
                ExportCell::Float(self.neurons[i].v()),
                ExportCell::Float(ext),
            ]);
        }

        let mut synapse_rows = Vec::new();
        for post in 0..n {
            for pre in 0..n {
                let w = self
                    .weights
                    .get(post)
                    .and_then(|r| r.get(pre))
                    .copied()
                    .unwrap_or(0.0);
                if w.abs() > 1e-12 {
                    synapse_rows.push(vec![
                        ExportCell::UInt(pre as u64),
                        ExportCell::UInt(post as u64),
                        ExportCell::Float(w),
                    ]);
                }
            }
        }

        let mut tables = vec![ExportTable::new(
            "Neurons",
            vec![
                "neuron_id".into(),
                "pos_x".into(),
                "pos_y".into(),
                "membrane_mv".into(),
                "external_current".into(),
            ],
            neuron_rows,
        )];
        if !synapse_rows.is_empty() {
            tables.push(ExportTable::new(
                "Synapses",
                vec![
                    "presynaptic_id".into(),
                    "postsynaptic_id".into(),
                    "weight".into(),
                ],
                synapse_rows,
            ));
        }
        tables
    }
}
