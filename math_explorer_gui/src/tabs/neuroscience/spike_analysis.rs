use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoint, Points};
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::collections::VecDeque;

pub struct SpikeAnalysisTool {
    // Simulation State
    neuron: HodgkinHuxleyNeuron,
    time: f64,
    is_running: bool,
    history_points: Vec<PlotPoint>, // [time, voltage]

    // Analysis State
    spike_times: Vec<f64>,
    spike_plot_points: Vec<PlotPoint>,
    isis: VecDeque<f64>, // Inter-Spike Intervals (ms)
    last_voltage: f64,

    // UI Controls
    input_current: f64,
    spike_threshold: f64,
    simulation_speed: usize,
}

impl Default for SpikeAnalysisTool {
    fn default() -> Self {
        let neuron = HodgkinHuxleyNeuron::new(-65.0);

        Self {
            neuron,
            time: 0.0,
            is_running: false,
            history_points: Vec::new(),
            spike_times: Vec::new(),
            spike_plot_points: Vec::new(),
            isis: VecDeque::new(),
            last_voltage: -65.0,
            input_current: 10.0,
            spike_threshold: 0.0,
            simulation_speed: 10,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SpikeAnalysisState {
    input_current: f64,
    spike_threshold: f64,
    simulation_speed: usize,
}

impl InteractiveTool for SpikeAnalysisTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Spike Train Analysis"
    }
    fn domain(&self) -> &'static str {
        "neuroscience"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SpikeAnalysisState {
            input_current: self.input_current,
            spike_threshold: self.spike_threshold,
            simulation_speed: self.simulation_speed,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: SpikeAnalysisState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid SpikeAnalysisTool state: {e}"))?;
        if !s.input_current.is_finite() || !s.spike_threshold.is_finite() || s.simulation_speed == 0 {
            return Err("Invalid parameters in SpikeAnalysisTool".to_string());
        }
        self.input_current = s.input_current;
        self.spike_threshold = s.spike_threshold;
        self.simulation_speed = s.simulation_speed;
        self.reset();
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        // --- Controls Panel ---
        egui::SidePanel::left("spike_analysis_controls").show(ctx, |ui| {
            ui.heading("Configuration");
            ui.separator();

            ui.add(egui::Slider::new(&mut self.input_current, 0.0..=50.0).text("Input Current (µA/cm²) - I_ext"));

            let threshold_changed = ui
                .add(egui::Slider::new(&mut self.spike_threshold, -50.0..=20.0).text("Spike Threshold (mV) - Threshold"))
                .changed();
            if threshold_changed {
                for pt in &mut self.spike_plot_points {
                    pt.y = self.spike_threshold;
                }
            }

            ui.separator();
            ui.add(egui::Slider::new(&mut self.simulation_speed, 1..=100).text("Simulation Speed (steps/frame)"));

            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button(if self.is_running {
                        "⏸ Pause"
                    } else {
                        "▶ Run"
                    })
                    .accessible_hover_text(if self.is_running {
                        "Pause the spike analysis simulation"
                    } else {
                        "Start the spike analysis simulation"
                    })
                    .clicked()
                {
                    self.is_running = !self.is_running;
                }
                if ui
                    .button("↻ Reset")
                    .accessible_hover_text("Reset the simulation to its initial state")
                    .clicked()
                {
                    self.reset();
                }
            });

            ui.separator();
            ui.label(format!("Time: {:.2} ms", self.time));
            ui.label(format!("Spikes Detected: {}", self.spike_times.len()));
            if let Some(last_isi) = self.isis.back() {
                ui.label(format!("Last ISI: {:.2} ms", last_isi));
            }
            if !self.isis.is_empty() {
                let mean_isi: f64 = self.isis.iter().sum::<f64>() / self.isis.len() as f64;
                ui.label(format!("Mean ISI: {:.2} ms", mean_isi));
                ui.label(format!("Firing Rate: {:.2} Hz", 1000.0 / mean_isi));
            }
        });

        // --- Visualization Panel ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // Update Simulation
            if self.is_running {
                self.step_simulation();
                ctx.request_repaint();
            }

            // Top Plot: Voltage Trace & Raster
            let height = ui.available_height() / 2.0;
            ui.push_id("voltage_trace", |ui| {
                Plot::new("voltage_plot")
                    .height(height)
                    .x_axis_label("Time (ms)")
                    .y_axis_label("Membrane Potential (mV)")
                    .show(ui, |plot_ui| {
                        // Voltage Line
                        plot_ui.line(Line::new(
                            "Voltage",
                            &self.history_points[..],
                        ));

                        // Threshold Line
                        plot_ui.hline(
                            egui_plot::HLine::new("Threshold", self.spike_threshold)
                                .color(egui::Color32::RED),
                        );

                        // Spike Markers (Raster)
                        plot_ui.points(
                            Points::new("Spikes", &self.spike_plot_points[..])
                                .radius(3.0_f32)
                                .color(egui::Color32::GREEN),
                        );
                    });
            });

            ui.separator();

            // Bottom Plot: ISI Histogram
            ui.push_id("isi_histogram", |ui| {
                Plot::new("isi_plot")
                    .x_axis_label("Inter-Spike Interval (ms)")
                    .y_axis_label("Count")
                    .show(ui, |plot_ui| {
                        if !self.isis.is_empty() {
                            let bars = self.compute_histogram(20); // 20 bins
                            plot_ui.bar_chart(
                                BarChart::new("ISI Distribution", bars).color(egui::Color32::GOLD),
                            );
                        }
                    });
            });
        });
    }
}

impl SpikeAnalysisTool {
    fn step_simulation(&mut self) {
        let dt = 0.01; // Time step (ms)

        for _ in 0..self.simulation_speed {
            // Update Neuron
            self.neuron.update(dt, self.input_current);
            let v = self.neuron.v();
            self.time += dt;

            // Spike Detection (Rising Edge)
            if v >= self.spike_threshold && self.last_voltage < self.spike_threshold {
                self.record_spike(self.time);
            }

            // Record History (subsample to avoid memory explosion)
            // Recording every 10th step = 0.1ms resolution
            // Or just check time % 0.1 < dt
            if (self.time * 10.0).round() % 1.0 == 0.0 {
                self.history_points.push(PlotPoint::new(self.time, v));
            }

            self.last_voltage = v;
        }

        // Manage History Size (keep last 2000ms. 2000ms / 0.1ms = 20,000 points.)
        if self.history_points.len() > 20_000 {
            let excess = self.history_points.len() - 20_000;
            self.history_points.drain(0..excess);
        }
    }

    fn record_spike(&mut self, time: f64) {
        if let Some(&last_spike_time) = self.spike_times.last() {
            let isi = time - last_spike_time;
            self.isis.push_back(isi);
        }
        self.spike_times.push(time);
        self.spike_plot_points.push(PlotPoint::new(time, self.spike_threshold));

        // Keep ISI history manageable? Maybe last 1000 intervals.
        if self.isis.len() > 1000 {
            self.isis.pop_front();
        }
        // Keep spike times aligned with view?
        // If we clear history, we might lose context.
        // For now, simple vector is fine.
    }

    fn reset(&mut self) {
        self.neuron = HodgkinHuxleyNeuron::new(-65.0);
        self.time = 0.0;
        self.history_points.clear();
        self.spike_times.clear();
        self.spike_plot_points.clear();
        self.isis.clear();
        self.last_voltage = -65.0;
        self.is_running = false;
    }

    fn compute_histogram(&self, num_bins: usize) -> Vec<Bar> {
        if self.isis.is_empty() {
            return Vec::new();
        }

        let (min_isi, max_isi) = self
            .isis
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        if (max_isi - min_isi).abs() < 1e-6 {
            // All ISIs are the same (perfect periodicity)
            return vec![Bar::new(min_isi, self.isis.len() as f64).width(0.5)];
        }

        let bin_width = (max_isi - min_isi) / num_bins as f64;
        let mut counts = vec![0; num_bins];

        for &isi in &self.isis {
            let mut bin_idx = ((isi - min_isi) / bin_width).floor() as usize;
            if bin_idx >= num_bins {
                bin_idx = num_bins - 1;
            }
            counts[bin_idx] += 1;
        }

        counts
            .into_iter()
            .enumerate()
            .map(|(i, count)| {
                let center = min_isi + (i as f64 + 0.5) * bin_width;
                Bar::new(center, count as f64).width(bin_width * 0.9)
            })
            .collect()
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "SpikeAnalysisTool",
        domain: "neuroscience",
        tags: &[],
        build: || Box::new(SpikeAnalysisTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SpikeAnalysisTool {
    fn theory_description(&self) -> String {
        format!(
            "Hodgkin-Huxley action potential kinetics and inter-spike interval (ISI) statistical distribution under injected current $I_{{ext}} = {:.1}\\,\\mu\\text{{A/cm}}^2$: $C_m \\frac{{dV}}{{dt}} = I_{{ext}} - \\bar{{g}}_{{Na}} m^3 h (V - E_{{Na}}) - \\bar{{g}}_K n^4 (V - E_K) - g_L (V - E_L)$ with detection threshold {:.1}\\,\\text{{mV}}.",
            self.input_current, self.spike_threshold
        )
    }
    fn phonetic_description(&self) -> String { "Hodgkin-Huxley action potential spike train and inter-spike interval theory".into() }
    fn theory_citation(&self) -> String { "A Quantitative Description of Membrane Current and Its Application to Conduction and Excitation in Nerve (Hodgkin & Huxley, 1952); Neuronal Dynamics (Gerstner et al., 2014)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_analysis_simulation_and_borrowed_slices() {
        let mut tool = SpikeAnalysisTool::default();
        assert!(tool.history_points.is_empty());
        assert!(tool.spike_plot_points.is_empty());

        // Step simulation to generate history points
        tool.simulation_speed = 50;
        tool.input_current = 20.0;
        for _ in 0..100 {
            tool.step_simulation();
        }

        assert!(!tool.history_points.is_empty());

        // Verify borrowed slice conversions for history_points and spike_plot_points
        {
            let history_slice: &[PlotPoint] = &tool.history_points[..];
            let history_points: egui_plot::PlotPoints<'_> = history_slice.into();
            assert!(matches!(history_points, egui_plot::PlotPoints::Borrowed(_)));
            assert_eq!(history_points.points().len(), tool.history_points.len());

            let spike_slice: &[PlotPoint] = &tool.spike_plot_points[..];
            let spike_plot_points: egui_plot::PlotPoints<'_> = spike_slice.into();
            assert!(matches!(spike_plot_points, egui_plot::PlotPoints::Borrowed(_)));
            assert_eq!(spike_plot_points.points().len(), tool.spike_plot_points.len());
        }

        // Test threshold update adjusts existing spike points
        let old_threshold = tool.spike_threshold;
        tool.record_spike(12.34);
        assert!(!tool.spike_plot_points.is_empty());
        let last_pt = tool.spike_plot_points.last().unwrap();
        assert_eq!(last_pt.y, old_threshold);

        let new_threshold = 15.0;
        tool.spike_threshold = new_threshold;
        for pt in &mut tool.spike_plot_points {
            pt.y = tool.spike_threshold;
        }
        assert_eq!(tool.spike_plot_points.last().unwrap().y, new_threshold);

        // Reset clears all buffers
        tool.reset();
        assert!(tool.history_points.is_empty());
        assert!(tool.spike_plot_points.is_empty());
        assert!(tool.spike_times.is_empty());
        assert_eq!(tool.time, 0.0);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spike Train Raster & ISI",
    "domain": "neuroscience",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "input_current": 15.0,
    "spike_threshold": 0.0,
    "simulation_speed": 20
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spike_analysis_starter",
        title: "Spike Train Raster & ISI",
        description: "Inter-spike interval (ISI) histogram and Fano factor variance analysis of neural spike trains.",
        category: "Electrophysiology",
        icon: "📈",
        domain: "neuroscience",
        tool_name: Some("SpikeAnalysisTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
