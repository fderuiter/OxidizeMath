//! Side-panel parameter controls, Kong `ESS` diagnostics, and estimator comparison table.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    BootstrapStatKind, MonteCarloWorkbenchTool, WorkbenchBenchmark, WorkbenchViewMode,
    MAX_REPLICATES, MAX_SAMPLES, MIN_REPLICATES, MIN_SAMPLES,
};
use crate::accessibility::AccessibleHoverText;
use crate::app::SemanticPalette;
use crate::framework::{scoped_panel_id, PaneSidePanel};
use eframe::egui;

impl MonteCarloWorkbenchTool {
    pub(super) fn render_workspace(&mut self, ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(ui, "monte_carlo_workbench_controls");
        PaneSidePanel::left(panel_id)
            .default_width(315.0)
            .show_inside(ui, |panel_ui| {
                self.render_controls_panel(panel_ui);
            });

        egui::CentralPanel::default().show_inside(ui, |central_ui| {
            self.render_central_views(central_ui);
        });
    }

    pub(super) fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Monte Carlo & Bootstrap");
        self.render_view_and_benchmark_selectors(ui);
        ui.separator();

        let changed = self.render_budget_and_seed_sliders(ui);
        if changed {
            self.recompute();
        }

        ui.separator();
        self.render_ess_and_estimator_readouts(ui, &palette);
    }

    fn render_view_and_benchmark_selectors(&mut self, ui: &mut egui::Ui) {
        ui.label("Workspace & Export View");
        ui.horizontal_wrapped(|ui| {
            for mode in WorkbenchViewMode::ALL {
                if ui
                    .selectable_label(self.view_mode == mode, mode.label())
                    .accessible_hover_text("Switch chart layout and figure/dataset export focus.")
                    .clicked()
                {
                    self.view_mode = mode;
                }
            }
        });

        ui.add_space(4.0);
        ui.label("Analytical Benchmark");
        let mut bench_changed = false;
        for bench in WorkbenchBenchmark::ALL {
            if ui
                .radio_value(&mut self.benchmark, bench, bench.label())
                .accessible_hover_text("Select integrand with closed-form reference expectation.")
                .changed()
            {
                bench_changed = true;
            }
        }
        if bench_changed {
            self.recompute();
        }
    }

    fn render_budget_and_seed_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.num_samples, MIN_SAMPLES..=MAX_SAMPLES)
                    .text("Sample Budget (N)"),
            )
            .accessible_hover_text("Monte Carlo sample count bounded by MAX_ITERATIONS (5000).")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.bootstrap_replicates,
                    MIN_REPLICATES..=MAX_REPLICATES,
                )
                .text("Bootstrap Replicates (B)"),
            )
            .accessible_hover_text("Nonparametric bootstrap resamples bounded by MAX_ITERATIONS.")
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.confidence_level, 0.80..=0.99)
                    .fixed_decimals(2)
                    .text("Confidence Level"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.importance_tilt, 0.0..=3.5)
                    .fixed_decimals(2)
                    .text("Importance Tilt (θ)"),
            )
            .accessible_hover_text(
                "Proposal drift/exponential tilt controlling SNIS weight concentration and Kong ESS.",
            )
            .changed();

        ui.horizontal(|ui| {
            ui.label("Bootstrap Statistic:");
            for stat in [BootstrapStatKind::Mean, BootstrapStatKind::Median] {
                changed |= ui
                    .radio_value(&mut self.bootstrap_statistic, stat, stat.label())
                    .changed();
            }
        });
        changed |= self.render_seed_and_action_row(ui);
        changed
    }

    fn render_seed_and_action_row(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("OxidizeRng Seed:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.seed).range(0..=1_000_000))
                .accessible_hover_text("Deterministic 64-bit OxidizeRng seed.")
                .changed();
        });

        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🎲 Advance Seed")
                .accessible_hover_text("Increment OxidizeRng seed and rerun preallocated workspaces.")
                .clicked()
            {
                self.seed = self.seed.wrapping_add(1);
                changed = true;
            }
            if ui
                .button("↻ Reset Defaults")
                .accessible_hover_text("Restore canonical 4D Geometric Basket Call workbench settings.")
                .clicked()
            {
                self.reset_defaults();
                changed = false;
            }
        });
        changed
    }

    fn render_ess_and_estimator_readouts(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.heading("Kong ESS & Estimator Readouts");
        ui.label(format!("Exact Reference I* = {:.5}", self.exact_reference));

        match self.est_snis {
            Ok(snis) => {
                let ess_ratio = snis.effective_sample_size / (snis.samples as f64).max(1.0);
                let ess_color = if ess_ratio >= 0.50 {
                    palette.success
                } else if ess_ratio >= 0.15 {
                    palette.warning
                } else {
                    palette.error
                };
                ui.colored_label(
                    ess_color,
                    egui::RichText::new(format!(
                        "SNIS Kong ESS = {:.1} / {} ({:.1}%)",
                        snis.effective_sample_size,
                        snis.samples,
                        ess_ratio * 100.0
                    ))
                    .strong(),
                );
                ui.small(format!(
                    "SNIS Mean = {:.5} ± {:.2e}",
                    snis.mean, snis.standard_error
                ));
            }
            Err(err) => {
                ui.colored_label(
                    palette.error,
                    format!("SNIS Weight Guard Triggered: {err}"),
                );
            }
        }

        ui.add_space(4.0);
        for (label, est_opt) in [
            ("IID MC", self.est_iid),
            ("Antithetic", self.est_antithetic),
            ("Control Variate", self.est_control),
            ("Latin Hypercube", self.est_lhs),
            ("Halton RQMC", self.est_halton),
            ("Sobol RQMC", self.est_sobol),
        ] {
            if let Some(est) = est_opt {
                ui.small(format!(
                    "{label}: {:.5} (SE {:.2e}, CI [{:.4}, {:.4}])",
                    est.mean, est.standard_error, est.ci_lower, est.ci_upper
                ));
            }
        }

        if let Some(comp) = self.boot_comparison {
            ui.add_space(4.0);
            ui.small(format!(
                "Bootstrap SE = {:.4} | BCa z0 = {:+.3}, accel a = {:+.4} | {:.2} ms",
                comp.bca.standard_error,
                comp.bca.bias_correction_z0,
                comp.bca.acceleration,
                self.last_compute_ms
            ));
        }
    }
}

// [cite:clinical_trials_statistics]
