#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::f64::consts::TAU;

use eframe::egui::Color32;
use egui_plot::{ExportPrimitive, SharedPoints};

use crate::app::{SemanticPalette, ThemeMode};
use crate::framework::{ExportCell, ExportTable};

use super::kuramoto_ui::phase_to_color;
use super::{NeuralDynamicsMode, NeuralNetworkVizTool};

fn neuron_voltage_color(v_mv: f64, palette: &SemanticPalette) -> Color32 {
    let t = (((v_mv as f32) + 65.0) / 105.0).clamp(0.0, 1.0);
    let (low, high) = (palette.axis_z, palette.axis_x);
    let lerp = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t).round() as u8;
    Color32::from_rgb(
        lerp(low.r(), high.r()),
        lerp(low.g(), high.g()),
        lerp(low.b(), high.b()),
    )
}

impl NeuralNetworkVizTool {
    fn append_synapse_primitives(
        &self,
        palette: &SemanticPalette,
        prims: &mut Vec<ExportPrimitive>,
    ) {
        let n = self.neurons.len().min(self.positions.len());
        for pre in 0..n {
            let start = self.csr_row_ptr.get(pre).copied().unwrap_or(0);
            let end = self.csr_row_ptr.get(pre + 1).copied().unwrap_or(0);
            for idx in start..end {
                let post = self.csr_col_idx[idx];
                let weight = self.csr_weights[idx];
                if post >= n || weight == 0.0 {
                    continue;
                }
                let origin = [
                    f64::from(self.positions[pre][0]),
                    -f64::from(self.positions[pre][1]),
                ];
                let tip = [
                    f64::from(self.positions[post][0]),
                    -f64::from(self.positions[post][1]),
                ];
                let color = if weight > 0.0 {
                    palette.success
                } else {
                    palette.error
                };
                prims.push(ExportPrimitive::Polyline {
                    name: format!("Synapse {pre}->{post}"),
                    points: SharedPoints::from(vec![origin, tip]),
                    color,
                    stroke_width: (1.0 + weight.abs() as f32).min(4.0),
                    closed: false,
                    fill: None,
                });
                prims.push(ExportPrimitive::Arrow {
                    name: format!("Synapse {pre}->{post}"),
                    origin,
                    tip,
                    tip_length: Some(8.0),
                    color,
                });
            }
        }
    }

    fn build_kuramoto_primitives(&self, palette: &SemanticPalette) -> Vec<ExportPrimitive> {
        let mut prims = Vec::new();
        let ring: Vec<[f64; 2]> = (0..=48)
            .map(|k| {
                let ang = (k as f64 / 48.0) * TAU;
                [ang.cos() * 48.0, ang.sin() * 48.0]
            })
            .collect();
        prims.push(ExportPrimitive::Polyline {
            name: "Phase Circle".to_string(),
            points: SharedPoints::from(ring),
            color: palette.muted_text,
            stroke_width: 1.2,
            closed: true,
            fill: None,
        });
        self.append_synapse_primitives(palette, &mut prims);
        let phases = self.kuramoto.phases();
        let n = phases.len().min(self.positions.len());
        for (i, &theta) in phases.iter().enumerate().take(n) {
            let cx = f64::from(self.positions[i][0]);
            let cy = -f64::from(self.positions[i][1]);
            let color = phase_to_color(theta, palette);
            prims.push(ExportPrimitive::Points {
                name: format!("Oscillator {i}"),
                points: SharedPoints::from(vec![[cx, cy]]),
                color,
                radius: 9.0,
            });
            let tip = [cx + theta.cos() * 14.0, cy - theta.sin() * 14.0];
            prims.push(ExportPrimitive::Polyline {
                name: format!("Phase Pointer {i}"),
                points: SharedPoints::from(vec![[cx, cy], tip]),
                color: palette.node_stroke,
                stroke_width: 2.0,
                closed: false,
                fill: None,
            });
        }
        let (r, psi) = self.kuramoto.order_parameter();
        prims.push(ExportPrimitive::Arrow {
            name: "Order Parameter r(t)".to_string(),
            origin: [0.0, 0.0],
            tip: [psi.cos() * r * 48.0, -psi.sin() * r * 48.0],
            tip_length: Some(9.0),
            color: palette.accent,
        });
        prims
    }

    pub(super) fn build_nn_plot_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = ThemeMode::Dark.palette();
        if self.mode == NeuralDynamicsMode::Kuramoto {
            return self.build_kuramoto_primitives(&palette);
        }
        let mut prims = Vec::new();
        self.append_synapse_primitives(&palette, &mut prims);
        let n = self.neurons.len().min(self.positions.len());
        for i in 0..n {
            let v = self.neurons[i].v();
            let color = neuron_voltage_color(v, &palette);
            let pt = [
                f64::from(self.positions[i][0]),
                -f64::from(self.positions[i][1]),
            ];
            prims.push(ExportPrimitive::Points {
                name: format!("Neuron {i}"),
                points: SharedPoints::from(vec![pt]),
                color,
                radius: 9.5,
            });
        }
        prims
    }

    fn build_neurons_table(&self) -> ExportTable {
        let n = self.neurons.len();
        let phases = self.kuramoto.phases();
        let mut neuron_rows = Vec::with_capacity(n);
        for i in 0..n {
            let pos = self.positions.get(i).copied().unwrap_or([0.0, 0.0]);
            let i_ext = self.external_input.get(i).copied().unwrap_or(0.0);
            let phase = phases.get(i).copied().unwrap_or(0.0);
            neuron_rows.push(vec![
                ExportCell::Int(i as i64),
                ExportCell::Float(f64::from(pos[0])),
                ExportCell::Float(f64::from(pos[1])),
                ExportCell::Float(self.neurons[i].v()),
                ExportCell::Float(i_ext),
                ExportCell::Float(phase),
            ]);
        }
        ExportTable::new(
            "Neurons",
            vec![
                "neuron_id".into(),
                "x".into(),
                "y".into(),
                "voltage_mv".into(),
                "external_current_ua".into(),
                "kuramoto_phase_rad".into(),
            ],
            neuron_rows,
        )
    }

    pub(super) fn build_nn_export_tables(&self) -> Vec<ExportTable> {
        let n = self.neurons.len();
        let mut tables = vec![self.build_neurons_table()];
        let mut synapse_rows = Vec::new();
        for pre in 0..n {
            let start = self.csr_row_ptr.get(pre).copied().unwrap_or(0);
            let end = self.csr_row_ptr.get(pre + 1).copied().unwrap_or(0);
            for idx in start..end {
                synapse_rows.push(vec![
                    ExportCell::Int(pre as i64),
                    ExportCell::Int(self.csr_col_idx[idx] as i64),
                    ExportCell::Float(self.csr_weights[idx]),
                ]);
            }
        }
        if !synapse_rows.is_empty() {
            tables.push(ExportTable::new(
                "Synapses",
                vec!["pre_neuron".into(), "post_neuron".into(), "weight".into()],
                synapse_rows,
            ));
        }
        if self.mode == NeuralDynamicsMode::Kuramoto && !self.order_history.is_empty() {
            let order_rows = self
                .order_history
                .iter()
                .map(|&[t, r]| vec![ExportCell::Float(t), ExportCell::Float(r)])
                .collect();
            tables.push(ExportTable::new(
                "Kuramoto Order Parameter",
                vec!["time".into(), "order_r".into()],
                order_rows,
            ));
        }
        tables
    }
}
