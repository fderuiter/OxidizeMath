#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Live Minsky prime-register bar chart, rule highlighter, prime output tape,
//! and trajectory compressibility plots for [`super::state::FractranMachineTool`].

use super::state::{FractranMachineTool, FractranPreset};
use eframe::egui;
use egui_plot::{Bar, BarChart, ExportPrimitive, Legend, Line, Plot, PlotPoints};
use math_explorer::pure_math::algorithmic_information::fractran::FIRST_64_PRIMES;

/// Minimum number of prime registers displayed on the bar chart ($p_0 = 2 \dots p_9 = 29$).
const MIN_VISIBLE_REGISTERS: usize = 10;

impl FractranMachineTool {
    fn visible_register_count(&self) -> usize {
        let highest_active = self
            .live
            .exponents
            .iter()
            .rposition(|&e| e > 0)
            .map_or(0, |i| i + 1);
        let highest_rule = self
            .program
            .rules()
            .iter()
            .map(|r| r.active_len)
            .max()
            .unwrap_or(0);
        highest_active
            .max(highest_rule)
            .clamp(MIN_VISIBLE_REGISTERS, FIRST_64_PRIMES.len())
    }

    fn format_factored_state(&self) -> String {
        let terms: Vec<String> = self
            .live
            .exponents
            .iter()
            .enumerate()
            .filter(|(_, exp)| **exp > 0)
            .map(|(idx, &exp)| format!("{}^{}", FIRST_64_PRIMES[idx], exp))
            .collect();
        if terms.is_empty() {
            "1 (all prime exponents zero)".to_string()
        } else {
            terms.join(" × ")
        }
    }

    fn render_rule_strip(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(
            egui::RichText::new(format!("State N = {}", self.format_factored_state())).strong(),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Rules:");
            for (idx, rule) in self.program.rules().iter().enumerate() {
                let text = format!("{}/{}", rule.numer, rule.denom);
                let is_next = self.next_rule_index == Some(idx);
                let is_last = self.last_rule_index == Some(idx);
                let rich = if is_next {
                    egui::RichText::new(format!("▶ [{text}]"))
                        .strong()
                        .color(palette.success)
                } else if is_last {
                    egui::RichText::new(format!("✓ {text}")).color(palette.info)
                } else {
                    egui::RichText::new(text)
                };
                ui.label(rich);
            }
        });
    }

    fn render_prime_tape(&self, ui: &mut egui::Ui) {
        if self.config.preset != FractranPreset::Primegame && self.emitted_primes.is_empty() {
            return;
        }
        let tape_str = if self.emitted_primes.is_empty() {
            "Stepping toward 2^2 (step 19)...".to_string()
        } else {
            self.emitted_primes
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("PRIMEGAME Output Tape (2^p):").strong());
            ui.monospace(tape_str);
        });
    }

    pub(crate) fn show_registers_and_rules(&self, ui: &mut egui::Ui) {
        self.render_rule_strip(ui);
        self.render_prime_tape(ui);
        ui.separator();
        ui.label(egui::RichText::new("Prime Register Exponents (p_i -> r_i)").strong());

        let count = self.visible_register_count();
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let bars: Vec<Bar> = (0..count)
            .map(|idx| {
                let prime = FIRST_64_PRIMES[idx];
                let exp = f64::from(self.live.exponents[idx]);
                let fill = if idx < 4 {
                    palette.info
                } else {
                    palette.accent
                };
                Bar::new(idx as f64, exp)
                    .name(format!("p={prime} (r={})", self.live.exponents[idx]))
                    .width(0.68)
                    .fill(fill)
            })
            .collect();

        let plot_id = crate::framework::scoped_panel_id(ui, "fractran_registers_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Prime Register Index i (2, 3, 5, 7, 11, 13, ...)")
            .y_axis_label("Exponent r_i")
            .include_y(0.0)
            .include_y(4.0)
            .show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new("Prime Register Exponents", bars));
            });
    }

    pub(crate) fn show_complexity_panel(&self, parent_ui: &mut egui::Ui) {
        let bottom_id = crate::framework::scoped_panel_id(parent_ui, "fractran_complexity_panel");
        egui::TopBottomPanel::bottom(bottom_id)
            .resizable(true)
            .default_height(220.0)
            .min_height(150.0)
            .show_inside(parent_ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Trajectory Compressibility: Shannon Entropy vs BDM, LZ76 & Prefix K(N)",
                    )
                    .strong(),
                );
                let plot_id = crate::framework::scoped_panel_id(ui, "fractran_complexity_plot");
                Plot::new(plot_id)
                    .legend(Legend::default())
                    .x_axis_label("Execution Step t")
                    .y_axis_label("Information / Upper Bound (bits)")
                    .include_y(0.0)
                    .show(ui, |plot_ui| {
                        for (name, series) in self.trajectory_series() {
                            plot_ui.line(Line::new(name, PlotPoints::from(series)));
                        }
                    });
            });
    }

    fn trajectory_series(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut bdm = Vec::with_capacity(self.trajectory.len());
        let mut lz76 = Vec::with_capacity(self.trajectory.len());
        let mut prefix_k = Vec::with_capacity(self.trajectory.len());
        let mut shannon = Vec::with_capacity(self.trajectory.len());
        for pt in &self.trajectory {
            let x = pt.step as f64;
            bdm.push([x, pt.bdm_bits]);
            lz76.push([x, pt.lz76_bits]);
            prefix_k.push([x, pt.prefix_k_bits]);
            shannon.push([x, pt.shannon_bits]);
        }
        vec![
            ("BDM Upper Bound (bits)".to_string(), bdm),
            ("LZ76 Upper Bound (bits)".to_string(), lz76),
            ("Prefix K(N) Bound (bits)".to_string(), prefix_k),
            ("Shannon Entropy H0 (bits)".to_string(), shannon),
        ]
    }

    pub(crate) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = self.trajectory_series();
        let count = self.visible_register_count();
        let regs: Vec<[f64; 2]> = (0..count)
            .map(|idx| {
                [
                    FIRST_64_PRIMES[idx] as f64,
                    f64::from(self.live.exponents[idx]),
                ]
            })
            .collect();
        datasets.push(("Register Exponents (prime, r_p)".to_string(), regs));
        datasets
    }

    pub(crate) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let mut prims: Vec<ExportPrimitive> = self
            .trajectory_series()
            .into_iter()
            .filter(|(_, pts)| pts.len() >= 2)
            .map(|(name, points)| ExportPrimitive::Polyline {
                name,
                points,
                color: egui::Color32::TRANSPARENT,
                stroke_width: 2.0,
                closed: false,
                fill: None,
            })
            .collect();

        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::dark());
        let count = self.visible_register_count();
        for (idx, (&prime, &exp)) in FIRST_64_PRIMES
            .iter()
            .zip(&self.live.exponents)
            .take(count)
            .enumerate()
        {
            let x = idx as f64;
            let y = f64::from(exp).max(0.15);
            prims.push(ExportPrimitive::Bar {
                name: format!("p={prime}"),
                min: [x - 0.32, 0.0],
                max: [x + 0.32, y],
                fill: palette.info,
                stroke_color: palette.accent,
                stroke_width: 1.0,
            });
        }
        prims
    }
}

// [cite:algorithmic_information_rust]
