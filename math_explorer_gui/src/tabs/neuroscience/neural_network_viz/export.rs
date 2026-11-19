#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NeuralNetworkVizTool;
use eframe::egui::Color32;
use egui_plot::export_primitives::ExportPrimitive;

impl NeuralNetworkVizTool {
    pub(super) fn step_export_simulation(&mut self) {
        self.is_running = false;
        for _ in 0..10 {
            self.step();
        }
    }

    pub(super) fn cold_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut primitives = Vec::new();
        let n = self.neurons.len();

        for i in 0..n {
            for j in 0..n {
                let weight = self.weights[i][j];
                if weight != 0.0 {
                    let origin = [f64::from(self.positions[j][0]), -f64::from(self.positions[j][1])];
                    let tip = [f64::from(self.positions[i][0]), -f64::from(self.positions[i][1])];
                    let color = if weight > 0.0 {
                        palette.success
                    } else {
                        palette.error
                    };
                    primitives.push(ExportPrimitive::Arrow {
                        name: format!("synapse_{j}_to_{i}"),
                        origin,
                        tip,
                        tip_length: Some(8.0),
                        color,
                    });
                }
            }
        }

        for i in 0..n {
            let v = self.neurons[i].v() as f32;
            let norm = ((v + 65.0) / 105.0).clamp(0.0, 1.0);
            let low = palette.axis_z;
            let high = palette.axis_x;
            let lerp =
                |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * norm).round() as u8;
            let color = Color32::from_rgb(
                lerp(low.r(), high.r()),
                lerp(low.g(), high.g()),
                lerp(low.b(), high.b()),
            );
            primitives.push(ExportPrimitive::Points {
                name: format!("neuron_{i}"),
                points: vec![[f64::from(self.positions[i][0]), -f64::from(self.positions[i][1])]],
                color,
                radius: 9.0,
            });
        }

        primitives
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let n = self.neurons.len();
        let voltages = (0..n).map(|i| [i as f64, self.neurons[i].v()]).collect();
        let positions = (0..n)
            .map(|i| [f64::from(self.positions[i][0]), f64::from(self.positions[i][1])])
            .collect();

        let mut datasets = vec![
            ("neuron_voltages_mv".to_string(), voltages),
            ("neuron_ring_positions".to_string(), positions),
        ];

        let mut synapses = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let w = self.weights[i][j];
                if w != 0.0 {
                    synapses.push([(j * n + i) as f64, w]);
                }
            }
        }
        if !synapses.is_empty() {
            datasets.push(("synaptic_weights".to_string(), synapses));
        }

        datasets
    }
}
