#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SpikeAnalysisTool;
use eframe::egui;
use egui_plot::export_primitives::ExportPrimitive;
use egui_plot::ExportSubplotPanel;

type PanelPayload = (Vec<(String, Vec<[f64; 2]>)>, Vec<ExportPrimitive>);

impl SpikeAnalysisTool {
    pub(super) fn cold_export_subplot_panels(&self) -> Vec<ExportSubplotPanel> {
        let captured = egui_plot::get_last_plot_subplots();
        let find_panel = |id: &str| captured.iter().find(|p| p.id == id);
        let palette = crate::app::palette::DARK_PALETTE;
        let (v_datasets, v_primitives) = self.build_voltage_panel_payload(&palette);
        let (isi_datasets, isi_primitives) = self.build_isi_panel_payload(&palette);

        let v_meta = find_panel("voltage_trace");
        let isi_meta = find_panel("isi_histogram");
        vec![
            ExportSubplotPanel {
                id: "voltage_trace".to_string(),
                screen_rect: v_meta
                    .map(|p| p.screen_rect)
                    .filter(|r| r.width() > 1.0 && r.height() > 1.0)
                    .unwrap_or_else(|| {
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 296.0))
                    }),
                log_x: v_meta.is_some_and(|p| p.log_x),
                log_y: v_meta.is_some_and(|p| p.log_y),
                datasets: v_datasets,
                primitives: v_primitives,
            },
            ExportSubplotPanel {
                id: "isi_histogram".to_string(),
                screen_rect: isi_meta
                    .map(|p| p.screen_rect)
                    .filter(|r| r.width() > 1.0 && r.height() > 1.0)
                    .unwrap_or_else(|| {
                        egui::Rect::from_min_max(egui::pos2(0.0, 304.0), egui::pos2(800.0, 600.0))
                    }),
                log_x: isi_meta.is_some_and(|p| p.log_x),
                log_y: isi_meta.is_some_and(|p| p.log_y),
                datasets: isi_datasets,
                primitives: isi_primitives,
            },
        ]
    }

    fn build_voltage_panel_payload(&self, palette: &crate::app::SemanticPalette) -> PanelPayload {
        let voltage_pts: Vec<[f64; 2]> = if self.history_points.is_empty() {
            vec![[self.time, self.neuron.v()]]
        } else {
            self.history_points.iter().map(|p| [p.x, p.y]).collect()
        };
        let x_min = voltage_pts.first().map_or(0.0, |p| p[0]);
        let x_max = voltage_pts.last().map_or(1.0, |p| p[0]).max(x_min + 1.0);

        let mut primitives = vec![
            ExportPrimitive::Polyline {
                name: "voltage_trace: Voltage".to_string(),
                points: voltage_pts.clone(),
                color: palette.series_color(0),
                stroke_width: 1.75,
                closed: false,
                fill: None,
            },
            ExportPrimitive::Polyline {
                name: "voltage_trace: Threshold".to_string(),
                points: vec![[x_min, self.spike_threshold], [x_max, self.spike_threshold]],
                color: palette.error,
                stroke_width: 1.25,
                closed: false,
                fill: None,
            },
        ];
        let mut datasets = vec![("voltage_trace: Voltage".to_string(), voltage_pts)];

        if !self.spike_plot_points.is_empty() {
            let spikes: Vec<[f64; 2]> = self.spike_plot_points.iter().map(|p| [p.x, p.y]).collect();
            primitives.push(ExportPrimitive::Points {
                name: "voltage_trace: Spikes".to_string(),
                points: spikes.clone(),
                color: palette.success,
                radius: 3.0,
            });
            datasets.push(("voltage_trace: Spikes".to_string(), spikes));
        }

        (datasets, primitives)
    }

    fn build_isi_panel_payload(&self, palette: &crate::app::SemanticPalette) -> PanelPayload {
        let bar_color = palette.series_color(5);
        if self.isis.is_empty() {
            return (
                Vec::new(),
                vec![ExportPrimitive::Polyline {
                    name: "isi_histogram: Baseline".to_string(),
                    points: vec![[0.0, 0.0], [20.0, 0.0]],
                    color: bar_color,
                    stroke_width: 1.25,
                    closed: false,
                    fill: None,
                }],
            );
        }

        let bars = self.compute_histogram(20);
        let mut datasets = Vec::new();
        let mut primitives = Vec::with_capacity(bars.len());
        let mut pts = Vec::with_capacity(bars.len());
        for (idx, bar) in bars.into_iter().enumerate() {
            pts.push([bar.argument, bar.value]);
            let half_w = (bar.bar_width * 0.5).max(0.05);
            let base = bar.base_offset.unwrap_or(0.0);
            primitives.push(ExportPrimitive::Bar {
                name: format!("isi_histogram: bin_{idx}"),
                min: [bar.argument - half_w, base.min(base + bar.value)],
                max: [bar.argument + half_w, base.max(base + bar.value)],
                fill: bar_color,
                stroke_color: bar_color,
                stroke_width: 1.0,
            });
        }
        datasets.push(("isi_histogram: ISI Distribution".to_string(), pts));
        (datasets, primitives)
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut out = Vec::new();
        for panel in self.cold_export_subplot_panels() {
            out.extend(panel.datasets);
        }
        out
    }
}
