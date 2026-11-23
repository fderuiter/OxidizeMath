//! Side panel controls and plots for [`super::ShannonChannelTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{presets, ShannonChannelKind, ShannonChannelTool, MAX_UI_ALPHABET};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, PlotPoints, VLine};

impl ShannonChannelTool {
    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Shannon & Channels");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🔀 Binary Symmetric")
                .accessible_hover_text("Load BSC(0.11) channel and dyadic 4-symbol distribution")
                .clicked()
            {
                self.cfg = presets::default_config();
                self.recompute();
            }
            if ui
                .button("📡 Z-Channel BA")
                .accessible_hover_text("Load asymmetric Z-channel solved via Blahut-Arimoto")
                .clicked()
            {
                self.cfg = presets::z_channel_config();
                self.recompute();
            }
            if ui
                .button("📶 AWGN Hartley")
                .accessible_hover_text("Load 20 kHz Shannon-Hartley AWGN channel at SNR = 31")
                .clicked()
            {
                self.cfg = presets::awgn_hartley_config();
                self.recompute();
            }
        });
        ui.separator();

        let mut changed = self.show_channel_controls(ui);
        ui.separator();
        changed |= self.show_distribution_controls(ui);
        if changed {
            self.recompute();
        }
    }

    fn show_channel_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal_wrapped(|ui| {
            for (kind, label) in [
                (ShannonChannelKind::Bsc, "BSC"),
                (ShannonChannelKind::Bec, "BEC"),
                (ShannonChannelKind::ZChannel, "Z-Chan"),
                (ShannonChannelKind::Awgn, "AWGN"),
            ] {
                if ui
                    .selectable_label(self.cfg.channel_kind == kind, label)
                    .clicked()
                {
                    self.cfg.channel_kind = kind;
                    changed = true;
                }
            }
        });
        match self.cfg.channel_kind {
            ShannonChannelKind::Awgn => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.cfg.snr_linear, 0.1..=100.0).text("Linear SNR"),
                    )
                    .changed();
            }
            _ => {
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.cfg.noise_param, 0.001..=0.95)
                            .text("Noise / Erasure p"),
                    )
                    .changed();
            }
        }
        changed |= ui
            .add(
                egui::Slider::new(&mut self.cfg.bandwidth_hz, 100.0..=100_000.0)
                    .logarithmic(true)
                    .text("Bandwidth B (Hz)"),
            )
            .changed();
        changed
    }

    fn show_distribution_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut n = self.cfg.weights_p.len();
        if ui
            .add(egui::Slider::new(&mut n, 2..=MAX_UI_ALPHABET).text("Alphabet |X|"))
            .changed()
        {
            self.resize_alphabet(n);
            return false;
        }

        let mut changed = ui
            .add(egui::Slider::new(&mut self.cfg.renyi_alpha, 0.0..=5.0).text("Rényi Order α"))
            .changed();
        ui.label("Unnormalized Weights P / Q:");
        egui::Grid::new("shannon_pmf_weights_grid")
            .striped(true)
            .show(ui, |ui| {
                for i in 0..self.cfg.weights_p.len() {
                    ui.label(format!("x_{i}:"));
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.cfg.weights_p[i])
                                .speed(0.05)
                                .range(0.01..=100.0),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.cfg.weights_q[i])
                                .speed(0.05)
                                .range(0.01..=100.0),
                        )
                        .changed();
                    ui.end_row();
                }
            });
        changed
    }

    pub(super) fn show_main_view(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("H(P): {:.4} bits", self.entropy_h));
            ui.separator();
            ui.label(format!("H_α(P): {:.4} bits", self.renyi_h));
            ui.separator();
            ui.label(format!("D_KL(P||Q): {:.4} bits", self.kl_bits));
            ui.separator();
            ui.label(format!("JSD(P||Q): {:.4} bits", self.jsd_bits));
            ui.separator();
            ui.strong(format!("Capacity C: {:.4} bits/use", self.ba_capacity));
            ui.separator();
            ui.colored_label(
                palette.success,
                format!(
                    "I(X*;Y): {:.4} bits ({:.1} bps)",
                    self.induced_mi, self.bandwidth_bps
                ),
            );
        });

        let active_x = match self.cfg.channel_kind {
            ShannonChannelKind::Awgn => self.cfg.snr_linear,
            _ => self.cfg.noise_param,
        };

        Plot::new("shannon_capacity_and_pmf_plot")
            .legend(Legend::default())
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                let cap_pts: PlotPoints = self.capacity_curve.iter().copied().collect();
                plot_ui
                    .line(Line::new("Channel Capacity C", cap_pts).color(palette.series_color(0)));
                let p_bars: Vec<Bar> = self
                    .opt_input
                    .iter()
                    .enumerate()
                    .map(|(i, &prob)| Bar::new(i as f64 * 0.25, prob).width(0.08))
                    .collect();
                plot_ui.bar_chart(
                    BarChart::new("Optimal Input p*(x)", p_bars).color(palette.series_color(1)),
                );
                plot_ui
                    .vline(VLine::new("Operating Point", active_x).color(palette.series_color(2)));
            });
    }
}
