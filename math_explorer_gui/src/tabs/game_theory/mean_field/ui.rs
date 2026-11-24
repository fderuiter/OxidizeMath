//! Side panel controls and plots for [`super::MeanFieldGameTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{presets, MeanFieldGameTool};
use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints, VLine};

impl MeanFieldGameTool {
    fn show_preset_buttons(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("🎯 Balanced MFG")
                .accessible_hover_text("Load baseline Gaussian crowd-congestion equilibrium")
                .clicked()
            {
                self.cfg = presets::default_config();
                self.recompute();
            }
            if ui
                .button("🌊 High Repulsion")
                .accessible_hover_text("Load strong crowd-aversion dispersion regime")
                .clicked()
            {
                self.cfg = presets::high_congestion_config();
                self.recompute();
            }
            if ui
                .button("👥 Bimodal Merge")
                .accessible_hover_text("Load bimodal initial density steering toward x* = 0.25")
                .clicked()
            {
                self.cfg = presets::bimodal_coalescence_config();
                self.recompute();
            }
        });
    }

    pub(super) fn show_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Mean-Field Games");
        self.show_preset_buttons(ui);
        ui.separator();

        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.cfg.congestion_weight, 0.0..=12.0).text("Congestion κ"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.viscosity, 0.02..=0.20).text("Viscosity ν"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.time_horizon, 0.15..=0.60).text("Horizon T"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.cfg.terminal_attraction, 0.2..=5.0)
                    .text("Terminal Pull α"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.target_center, -0.6..=0.6).text("Target x*"))
            .changed();
        changed |= ui
            .checkbox(&mut self.cfg.bimodal_initial, "Bimodal Initial Crowd m₀(x)")
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.grid_points, 21..=41).text("Spatial Grid Nx"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.iterations, 2..=12).text("Fixed-Point Sweeps"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.slice_fraction, 0.0..=1.0).text("Slice t / T"))
            .changed();

        if changed {
            self.recompute();
        }
    }

    pub(super) fn show_main_view(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal_wrapped(|ui| {
            ui.strong(format!("Peak m(x, T): {:.3}", self.peak_terminal_density));
            ui.separator();
            ui.strong(format!("u(0, 0): {:.3}", self.origin_initial_value));
            ui.separator();
            ui.label(format!("Terminal Var[X_T]: {:.4}", self.terminal_variance));
            ui.separator();
            ui.label(format!("CFL Steps Nt: {}", self.cfl_time_steps));
        });

        Plot::new("mfg_coupled_pde_plot")
            .legend(Legend::default())
            .include_y(0.0)
            .show(ui, |plot_ui| {
                let m0: PlotPoints = self.density_initial.iter().copied().collect();
                let ms: PlotPoints = self.density_slice.iter().copied().collect();
                let mt: PlotPoints = self.density_terminal.iter().copied().collect();
                let us: PlotPoints = self.value_slice.iter().copied().collect();
                plot_ui.line(Line::new("Initial m(x, 0)", m0).color(palette.series_color(0)));
                plot_ui.line(Line::new("Slice m(x, t)", ms).color(palette.series_color(1)));
                plot_ui.line(Line::new("Terminal m(x, T)", mt).color(palette.series_color(2)));
                plot_ui.line(Line::new("HJB Value u(x, t)", us).color(palette.series_color(3)));
                plot_ui
                    .vline(VLine::new("Target x*", self.cfg.target_center).color(palette.warning));
            });
    }
}
