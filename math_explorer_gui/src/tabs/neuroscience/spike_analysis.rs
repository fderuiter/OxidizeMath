use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoint, Points};
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::collections::VecDeque;

pub struct SpikeAnalysisTool {
    neuron: HodgkinHuxleyNeuron,
    time: f64,
    is_running: bool,
    history_points: Vec<PlotPoint>,
    spike_times: Vec<f64>,
    spike_plot_points: Vec<PlotPoint>,
    isis: VecDeque<f64>,
    last_voltage: f64,
    input_current: f64,
    spike_threshold: f64,
    simulation_speed: usize,
    spiked_in_frame: bool,
    sonification: crate::audio::SonificationFrame,
}

impl Default for SpikeAnalysisTool {
    fn default() -> Self {
        Self {
            neuron: HodgkinHuxleyNeuron::new(-65.0),
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
            spiked_in_frame: false,
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SpikeAnalysisState {
    input_current: f64,
    spike_threshold: f64,
    simulation_speed: usize,
    #[serde(default)]
    is_running: bool,
}

impl InteractiveTool for SpikeAnalysisTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Spike Train Analysis"
    }
    fn step_simulation(&mut self) {
        Self::step_simulation(self);
    }
    fn domain(&self) -> &'static str {
        "neuroscience"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&SpikeAnalysisState {
            input_current: self.input_current,
            spike_threshold: self.spike_threshold,
            simulation_speed: self.simulation_speed,
            is_running: self.is_running,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: SpikeAnalysisState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid spike analysis state: {e}"))?;
        if !parsed.input_current.is_finite()
            || !(0.0..=50.0).contains(&parsed.input_current)
            || !parsed.spike_threshold.is_finite()
            || !(-50.0..=20.0).contains(&parsed.spike_threshold)
            || !(1..=100).contains(&parsed.simulation_speed)
        {
            return Err("Spike analysis parameters out of bounds".into());
        }
        self.reset();
        self.input_current = parsed.input_current;
        self.spike_threshold = parsed.spike_threshold;
        self.simulation_speed = parsed.simulation_speed;
        self.is_running = parsed.is_running;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_controls(ui);
        self.draw_plots(ui);
    }
}

impl SpikeAnalysisTool {
    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "spike_analysis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Configuration");
            ui.separator();

            ui.add(
                egui::Slider::new(&mut self.input_current, 0.0..=50.0)
                    .text("Input Current (µA/cm²) - I_ext"),
            );

            if ui
                .add(
                    egui::Slider::new(&mut self.spike_threshold, -50.0..=20.0)
                        .text("Spike Threshold (mV) - Threshold"),
                )
                .changed()
            {
                for pt in &mut self.spike_plot_points {
                    pt.y = self.spike_threshold;
                }
            }

            ui.separator();
            ui.add(
                egui::Slider::new(&mut self.simulation_speed, 1..=100)
                    .text("Simulation Speed (steps/frame)"),
            );

            ui.separator();
            if let Some(action) = PlaybackTransport::in_thread(
                "spike_analysis_transport",
                "Spike Train Analysis",
                self.is_running,
            )
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => self.is_running = true,
                    PlaybackAction::Pause => self.is_running = false,
                    PlaybackAction::Step => self.step_once(),
                    PlaybackAction::Reset => self.reset(),
                }
            }

            ui.separator();
            crate::audio::render_sonification_panel(
                ui,
                "spike_analysis",
                &self.sonification,
                !self.is_running && self.time <= 0.0,
            );
            ui.separator();
            self.draw_telemetry_labels(ui);
        });
    }

    fn draw_telemetry_labels(&self, ui: &mut egui::Ui) {
        ui.label(format!("Time: {:.2} ms", self.time));
        ui.label(format!("Spikes Detected: {}", self.spike_times.len()));
        if let Some(last_isi) = self.isis.back() {
            ui.label(format!("Last ISI: {last_isi:.2} ms"));
        }
        let rate_hz = self.mean_firing_rate_hz();
        if rate_hz > 0.0 {
            let mean_isi = 1000.0 / rate_hz;
            ui.label(format!("Mean ISI: {mean_isi:.2} ms"));
            ui.label(format!("Firing Rate: {rate_hz:.2} Hz"));
        }
    }

    fn mean_firing_rate_hz(&self) -> f64 {
        if self.isis.is_empty() {
            return 0.0;
        }
        let mean_isi = self.isis.iter().sum::<f64>() / (self.isis.len() as f64);
        if mean_isi > 1e-6 {
            1000.0 / mean_isi
        } else {
            0.0
        }
    }

    fn draw_plots(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.is_running {
                self.step_simulation();
                ctx.request_repaint();
            }

            let height = ui.available_height() / 2.0;
            ui.push_id("voltage_trace", |ui| {
                Plot::new("voltage_plot")
                    .height(height)
                    .x_axis_label("Time (ms)")
                    .y_axis_label("Membrane Potential (mV)")
                    .show(ui, |plot_ui| {
                        let palette = crate::app::SemanticPalette::from_visuals(
                            &plot_ui.ctx().style().visuals,
                        );
                        if self.history_points.is_empty() {
                            plot_ui.line(
                                Line::new("Voltage", vec![[self.time, self.neuron.v()]])
                                    .color(palette.series_color(0)),
                            );
                        } else {
                            plot_ui.line(
                                Line::new("Voltage", &self.history_points[..])
                                    .color(palette.series_color(0)),
                            );
                        }
                        plot_ui.hline(
                            egui_plot::HLine::new("Threshold", self.spike_threshold)
                                .color(palette.error),
                        );
                        plot_ui.points(
                            Points::new("Spikes", &self.spike_plot_points[..])
                                .radius(3.0_f32)
                                .color(palette.success),
                        );
                    });
            });

            ui.separator();

            ui.push_id("isi_histogram", |ui| {
                let bar_color =
                    crate::app::SemanticPalette::from_visuals(ui.visuals()).series_color(5);
                Plot::new("isi_plot")
                    .x_axis_label("Inter-Spike Interval (ms)")
                    .y_axis_label("Count")
                    .show(ui, |plot_ui| {
                        if !self.isis.is_empty() {
                            let bars = self.compute_histogram(20);
                            plot_ui.bar_chart(
                                BarChart::new("ISI Distribution", bars).color(bar_color),
                            );
                        }
                    });
            });
        });
    }

    fn advance_dt_tick(&mut self) {
        let dt = 0.01;
        self.neuron.update(dt, self.input_current);
        let v = self.neuron.v();
        self.time += dt;

        if v >= self.spike_threshold && self.last_voltage < self.spike_threshold {
            self.spiked_in_frame = true;
            self.record_spike(self.time);
        }

        if (self.time * 10.0).round() % 1.0 == 0.0 {
            self.history_points.push(PlotPoint::new(self.time, v));
        }

        self.last_voltage = v;
        if self.history_points.len() > 20_000 {
            let excess = self.history_points.len() - 20_000;
            self.history_points.drain(0..excess);
        }
    }

    pub(crate) fn step_once(&mut self) {
        self.is_running = false;
        self.spiked_in_frame = false;
        let prev_v = self.last_voltage;
        self.advance_dt_tick();
        self.refresh_sonification(prev_v);
    }

    fn step_simulation(&mut self) {
        self.spiked_in_frame = false;
        let prev_v = self.last_voltage;
        for _ in 0..self.simulation_speed {
            self.advance_dt_tick();
        }
        self.refresh_sonification(prev_v);
    }

    fn refresh_sonification(&mut self, frame_start_v: f64) {
        let cur_v = self.neuron.v();
        let ref_prev = if self.spiked_in_frame {
            self.spike_threshold - 5.0
        } else {
            frame_start_v
        };
        let ref_cur = if self.spiked_in_frame {
            cur_v.max(self.spike_threshold + 5.0)
        } else {
            cur_v
        };
        self.sonification = crate::audio::mappings::spike_analysis_frame(
            ref_cur,
            ref_prev,
            self.spike_threshold,
            self.mean_firing_rate_hz(),
        );
    }

    fn record_spike(&mut self, time: f64) {
        if let Some(&last_spike_time) = self.spike_times.last() {
            let isi = time - last_spike_time;
            self.isis.push_back(isi);
        }
        self.spike_times.push(time);
        self.spike_plot_points
            .push(PlotPoint::new(time, self.spike_threshold));

        if self.isis.len() > 1000 {
            self.isis.pop_front();
        }
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
        self.spiked_in_frame = false;
        self.sonification = crate::audio::SonificationFrame::silent();
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
                Bar::new(center, f64::from(count)).width(bin_width * 0.9)
            })
            .collect()
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SpikeAnalysisTool",
        display_name: "Spike Train Analysis",
        domain: "neuroscience",
        tags: &["isi", "inter-spike interval", "firing rate", "electrophysiology"],
        build: || Box::new(SpikeAnalysisTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for SpikeAnalysisTool {
    fn theory_description(&self) -> String {
        format!(
            "Hodgkin-Huxley action potential kinetics and inter-spike interval (ISI) statistical distribution under injected current $I_{{ext}} = {:.1}\\,\\mu\\text{{A/cm}}^2$: $C_m \\frac{{dV}}{{dt}} = I_{{ext}} - \\bar{{g}}_{{Na}} m^3 h (V - E_{{Na}}) - \\bar{{g}}_K n^4 (V - E_K) - g_L (V - E_L)$ with detection threshold ${:.1}\\,\\text{{mV}}$.",
            self.input_current, self.spike_threshold
        )
    }
    fn phonetic_description(&self) -> String {
        "Hodgkin-Huxley action potential spike train and inter-spike interval theory".into()
    }
    fn theory_citation(&self) -> String {
        "A Quantitative Description of Membrane Current and Its Application to Conduction and Excitation in Nerve (Hodgkin & Huxley, 1952); Neuronal Dynamics (Gerstner et al., 2014)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_analysis_simulation_and_borrowed_slices() {
        let mut tool = SpikeAnalysisTool::default();
        assert!(tool.history_points.is_empty());
        assert!(tool.spike_plot_points.is_empty());

        tool.step_once();
        assert!(!tool.is_running);
        assert_eq!(tool.history_points.len(), 1);
        assert!((tool.time - 0.01).abs() < 1e-9);

        tool.simulation_speed = 50;
        tool.input_current = 20.0;
        for _ in 0..100 {
            tool.step_simulation();
        }

        assert!(!tool.history_points.is_empty());

        {
            let history_slice: &[PlotPoint] = &tool.history_points[..];
            let history_points: egui_plot::PlotPoints<'_> = history_slice.into();
            assert!(matches!(history_points, egui_plot::PlotPoints::Borrowed(_)));
            assert_eq!(history_points.points().len(), tool.history_points.len());

            let spike_slice: &[PlotPoint] = &tool.spike_plot_points[..];
            let spike_plot_points: egui_plot::PlotPoints<'_> = spike_slice.into();
            assert!(matches!(
                spike_plot_points,
                egui_plot::PlotPoints::Borrowed(_)
            ));
            assert_eq!(
                spike_plot_points.points().len(),
                tool.spike_plot_points.len()
            );
        }

        let old_threshold = tool.spike_threshold;
        tool.record_spike(12.34);
        assert!(!tool.spike_plot_points.is_empty());
        assert_eq!(tool.spike_plot_points.last().unwrap().y, old_threshold);

        let new_threshold = 15.0;
        tool.spike_threshold = new_threshold;
        for pt in &mut tool.spike_plot_points {
            pt.y = tool.spike_threshold;
        }
        assert_eq!(tool.spike_plot_points.last().unwrap().y, new_threshold);

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
    "simulation_speed": 20,
    "is_running": false
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "spike_analysis_starter",
        title: "Spike Train Raster & ISI",
        description: "Hodgkin-Huxley membrane voltage spike raster and inter-spike interval (ISI) histogram analysis.",
        category: "Electrophysiology",
        icon: "📈",
        domain: "neuroscience",
        tool_name: Some("SpikeAnalysisTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
