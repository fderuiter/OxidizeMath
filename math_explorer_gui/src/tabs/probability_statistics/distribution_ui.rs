#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, density/CDF plots, and plot exports for [`DistributionExplorerTool`].

use super::distribution_tool::{
    DistributionExplorerTool, DistributionFamily, MAX_DIST_SAMPLE_SIZE, MAX_RNG_SEED,
};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Bar, BarChart, ExportPrimitive, Legend, Line, Plot, PlotPoints, VLine};

impl DistributionExplorerTool {
    fn select_family(&mut self, family: DistributionFamily) {
        if self.config.family != family {
            let (a, b) = family.default_params();
            self.config.family = family;
            self.config.param_a = a;
            self.config.param_b = b;
            self.recompute();
            self.bump_revision();
        }
    }

    fn show_family_selectors(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Continuous Families").strong());
        ui.horizontal_wrapped(|ui| {
            for fam in DistributionFamily::CONTINUOUS {
                if ui
                    .selectable_label(self.config.family == fam, fam.label())
                    .accessible_hover_text(format!("Select {} continuous distribution", fam.label()))
                    .clicked()
                {
                    self.select_family(fam);
                }
            }
        });
        ui.add_space(4.0);
        ui.label(egui::RichText::new("Discrete Families").strong());
        ui.horizontal_wrapped(|ui| {
            for fam in DistributionFamily::DISCRETE {
                if ui
                    .selectable_label(self.config.family == fam, fam.label())
                    .accessible_hover_text(format!("Select {} discrete distribution", fam.label()))
                    .clicked()
                {
                    self.select_family(fam);
                }
            }
        });
    }

    fn show_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        let (label_a, label_b) = self.snapshot.param_labels;
        let (a_range, b_range) = parameter_slider_ranges(self.config.family);
        let mut changed = false;

        let mut pa = self.config.param_a.clamp(*a_range.start(), *a_range.end());
        if ui
            .add(egui::Slider::new(&mut pa, a_range).text(label_a))
            .accessible_hover_text(format!("Adjust primary distribution parameter ({label_a})"))
            .changed()
        {
            self.config.param_a = pa;
            changed = true;
        }

        if let Some(lbl_b) = label_b {
            let mut pb = self.config.param_b.clamp(*b_range.start(), *b_range.end());
            if ui
                .add(egui::Slider::new(&mut pb, b_range).text(lbl_b))
                .accessible_hover_text(format!("Adjust secondary distribution parameter ({lbl_b})"))
                .changed()
            {
                self.config.param_b = pb;
                changed = true;
            }
        }

        let mut qp = self.config.quantile_p;
        if ui
            .add(egui::Slider::new(&mut qp, 0.01..=0.99).text("Quantile Level p"))
            .accessible_hover_text("Target cumulative probability p for inverse CDF evaluation")
            .changed()
        {
            self.config.quantile_p = qp.clamp(0.01, 0.99);
            changed = true;
        }

        let mut n = self.config.sample_size;
        if ui
            .add(egui::Slider::new(&mut n, 16..=MAX_DIST_SAMPLE_SIZE).text("Sample Size N"))
            .accessible_hover_text("Number of seeded synthetic draws for MLE and MoM fitting")
            .changed()
        {
            self.config.sample_size = n.clamp(16, MAX_DIST_SAMPLE_SIZE);
            changed = true;
        }

        let mut seed = self.config.seed;
        if ui
            .add(egui::Slider::new(&mut seed, 1..=1_000).text("RNG Seed"))
            .accessible_hover_text("Deterministic OxidizeRng seed for synthetic sampling")
            .changed()
        {
            self.config.seed = seed.clamp(1, MAX_RNG_SEED);
            changed = true;
        }

        if changed {
            self.recompute();
            self.bump_revision();
        }
    }

    fn show_diagnostics_cards(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.group(|ui| {
            ui.label(egui::RichText::new("Exact Moments & Quantile").strong());
            let mean_str = self
                .snapshot
                .mean
                .map_or_else(|| "Undefined".to_string(), |m| format!("{m:.4}"));
            let var_str = self
                .snapshot
                .variance
                .map_or_else(|| "Undefined".to_string(), |v| format!("{v:.4}"));
            ui.label(format!("Mean E[X]: {mean_str}"));
            ui.label(format!("Variance Var[X]: {var_str}"));
            ui.colored_label(
                palette.accent,
                format!(
                    "Quantile Q({:.2}) = {:.4}",
                    self.config.quantile_p, self.snapshot.quantile_x
                ),
            );
        });
        ui.add_space(4.0);
        ui.group(|ui| {
            ui.label(egui::RichText::new("Seeded Parameter Recovery").strong());
            ui.colored_label(palette.success, format!("MLE: {}", self.snapshot.mle_summary));
            ui.colored_label(palette.info, format!("MoM: {}", self.snapshot.mom_summary));
        });
    }

    pub(crate) fn render_workbench(&mut self, parent_ui: &mut egui::Ui) {
        let side_id = crate::framework::scoped_panel_id(parent_ui, "dist_explorer_controls");
        crate::framework::PaneSidePanel::left(side_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Distribution Explorer");
                ui.separator();
                self.show_family_selectors(ui);
                ui.separator();
                self.show_parameter_sliders(ui);
                ui.separator();
                self.show_diagnostics_cards(ui);
            });
        });

        let bottom_id = crate::framework::scoped_panel_id(parent_ui, "dist_explorer_cdf_panel");
        egui::TopBottomPanel::bottom(bottom_id)
            .resizable(true)
            .default_height(210.0)
            .min_height(140.0)
            .show_inside(parent_ui, |ui| {
                self.render_cdf_plot(ui);
            });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            self.render_density_plot(ui);
        });
    }

    fn render_density_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let title = if self.config.family.is_discrete() {
            format!("{} — Exact PMF & Seeded Empirical Frequencies", self.config.family.label())
        } else {
            format!("{} — Exact PDF & Seeded Empirical Histogram", self.config.family.label())
        };
        ui.label(egui::RichText::new(title).strong());

        let plot_id = crate::framework::scoped_panel_id(ui, "dist_explorer_pdf_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Outcome x")
            .y_axis_label("Density / Mass")
            .include_y(0.0)
            .show(ui, |plot_ui| {
                let hist: Vec<Bar> = self
                    .snapshot
                    .histogram_bars
                    .iter()
                    .map(|&[cx, h, w]| Bar::new(cx, h).width(w).fill(palette.info.gamma_multiply(0.45)))
                    .collect();
                if !hist.is_empty() {
                    plot_ui.bar_chart(BarChart::new("Seeded Sample", hist));
                }
                if self.config.family.is_discrete() {
                    let exact_bars: Vec<Bar> = self
                        .snapshot
                        .discrete_bars
                        .iter()
                        .map(|&(k, pmf)| Bar::new(k as f64 - 0.14, pmf).width(0.34).fill(palette.accent))
                        .collect();
                    plot_ui.bar_chart(BarChart::new("Exact PMF", exact_bars));
                } else {
                    plot_ui.line(
                        Line::new("Exact PDF", PlotPoints::from(self.snapshot.density_curve.clone()))
                            .color(palette.accent)
                            .width(2.4_f32),
                    );
                }
                plot_ui.vline(
                    VLine::new("Quantile Q(p)", self.snapshot.quantile_x).color(palette.warning),
                );
            });
    }

    fn render_cdf_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(egui::RichText::new("Cumulative Distribution F(x) & Inverse Quantile Q(p)").strong());
        let plot_id = crate::framework::scoped_panel_id(ui, "dist_explorer_cdf_plot");
        Plot::new(plot_id)
            .legend(Legend::default())
            .x_axis_label("Outcome x")
            .y_axis_label("Cumulative Probability F(x)")
            .include_y(0.0)
            .include_y(1.0)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Exact CDF F(x)", PlotPoints::from(self.snapshot.cdf_curve.clone()))
                        .color(palette.success)
                        .width(2.2_f32),
                );
                plot_ui.vline(
                    VLine::new("Q(p)", self.snapshot.quantile_x).color(palette.warning),
                );
            });
    }

    pub(crate) fn build_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![
            (
                format!("{} Density/PMF", self.config.family.label()),
                self.snapshot.density_curve.clone(),
            ),
            (
                format!("{} CDF", self.config.family.label()),
                self.snapshot.cdf_curve.clone(),
            ),
        ]
    }

    pub(crate) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::SemanticPalette::from_visuals(&egui::Visuals::dark());
        let mut prims = Vec::new();
        if self.snapshot.density_curve.len() >= 2 {
            prims.push(ExportPrimitive::Polyline {
                name: format!("{} Density", self.config.family.label()),
                points: self.snapshot.density_curve.clone(),
                color: palette.accent,
                stroke_width: 2.2,
                closed: false,
                fill: None,
            });
        }
        if self.snapshot.cdf_curve.len() >= 2 {
            prims.push(ExportPrimitive::Polyline {
                name: format!("{} CDF", self.config.family.label()),
                points: self.snapshot.cdf_curve.clone(),
                color: palette.success,
                stroke_width: 2.0,
                closed: false,
                fill: None,
            });
        }
        prims
    }
}

fn parameter_slider_ranges(
    family: DistributionFamily,
) -> (std::ops::RangeInclusive<f64>, std::ops::RangeInclusive<f64>) {
    match family {
        DistributionFamily::Normal => (-5.0..=5.0, 0.15..=4.0),
        DistributionFamily::Gamma => (0.3..=15.0, 0.2..=8.0),
        DistributionFamily::Beta => (0.3..=12.0, 0.3..=12.0),
        DistributionFamily::Weibull => (0.4..=8.0, 0.2..=8.0),
        DistributionFamily::StudentT => (1.5..=25.0, 0.2..=4.0),
        DistributionFamily::LogNormal => (-1.5..=1.5, 0.15..=1.2),
        DistributionFamily::Exponential => (0.2..=8.0, 0.5..=2.0),
        DistributionFamily::ChiSquared => (1.0..=24.0, 0.5..=2.0),
        DistributionFamily::Uniform => (-6.0..=8.0, 0.5..=12.0),
        DistributionFamily::Poisson => (0.4..=20.0, 0.5..=2.0),
        DistributionFamily::Binomial => (2.0..=30.0, 0.05..=0.95),
        DistributionFamily::NegativeBinomial => (0.5..=15.0, 0.10..=0.90),
        DistributionFamily::Geometric | DistributionFamily::Bernoulli => (0.05..=0.95, 0.5..=2.0),
        DistributionFamily::ZeroInflatedPoisson => (0.05..=0.85, 0.5..=14.0),
    }
}

// [cite:clinical_trials_statistics]
