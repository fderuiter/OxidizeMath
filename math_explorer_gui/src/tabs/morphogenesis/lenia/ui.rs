//! Side-panel `AnimatedSlider` controls, recovery banner, and paired `egui_plot` inspector
//! for [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{LeniaBrushMode, LeniaSpeciesPreset, LeniaTool};
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::FiniteStateGuard;
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_commons::math_kernel::colormap::ensure_dark_canvas_contrast;
use math_explorer::biology::lenia::{
    GrowthParams, KernelShells, MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MIN_KERNEL_RADIUS,
};

impl LeniaTool {
    pub(crate) fn render_studio(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let hc = crate::app::ThemeMode::from_visuals(ui.visuals())
            == crate::app::ThemeMode::HighContrast;
        if self.high_contrast != hc {
            self.high_contrast = hc;
            self.refresh_pixels_in_place();
        }

        if self.playing && !FiniteStateGuard::diverged(self) {
            self.step_frame_cooperative();
            if self.playing {
                ctx.request_repaint();
            }
        }

        self.draw_side_controls(ui);
        self.draw_central_workspace(ui);
    }

    fn draw_side_controls(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "lenia_studio_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Lenia Studio");
                ui.separator();
                self.draw_recovery_banner(ui);
                self.draw_playback_and_species(ui);
                ui.separator();
                self.draw_kernel_and_growth_sliders(ui);
                ui.separator();
                self.draw_brush_and_grid_controls(ui);
            });
        });
    }

    fn draw_recovery_banner(&mut self, ui: &mut egui::Ui) {
        if !FiniteStateGuard::diverged(self) {
            return;
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        egui::Frame::group(ui.style())
            .stroke(egui::Stroke::new(1.5_f32, palette.error))
            .show(ui, |ui| {
                ui.colored_label(palette.error, "⚠ Instability Guard Triggered");
                if let Some(diag) = &self.diagnostic {
                    ui.small(&diag.message);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Auto-Scale dt").clicked() {
                        FiniteStateGuard::auto_scale_step_size(self);
                    }
                    if ui.button("Clamp Params").clicked() {
                        FiniteStateGuard::clamp_parameters(self);
                    }
                    if ui.button("Reset Orbium").clicked() {
                        FiniteStateGuard::reset_parameters(self);
                    }
                });
            });
        ui.separator();
    }

    fn draw_playback_and_species(&mut self, ui: &mut egui::Ui) {
        let transport_state = if FiniteStateGuard::diverged(self) {
            SimWorkerState::Faulted
        } else if self.playing {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        if let Some(action) =
            PlaybackTransport::from_worker_state("lenia_transport", "Lenia Studio", transport_state)
                .can_reset(true)
                .show(ui)
        {
            match action {
                PlaybackAction::Play => self.playing = true,
                PlaybackAction::Pause => self.playing = false,
                PlaybackAction::Step => {
                    self.playing = false;
                    self.step_once();
                }
                PlaybackAction::Reset => self.reseed_current_genome(),
                PlaybackAction::StepBack | PlaybackAction::Scrub(_) => {}
            }
        }
        ui.small(format!(
            "Step: {:.2} ms | Mass: {:.1} | Speed: {:.2}",
            self.last_step_ms,
            self.engine.telemetry().total_mass,
            self.engine.telemetry().centroid_speed,
        ));
        ui.add_space(4.0);
        ui.label("Species Genomes:");
        ui.horizontal_wrapped(|ui| {
            for sp in LeniaSpeciesPreset::ALL {
                if ui
                    .selectable_label(self.species == sp, sp.label())
                    .clicked()
                {
                    self.apply_species_preset(sp);
                }
            }
        });
    }

    fn draw_kernel_and_growth_sliders(&mut self, ui: &mut egui::Ui) {
        ui.label("Kernel & Growth (Sweepable):");
        let prev = (
            self.radius,
            self.mu,
            self.sigma,
            self.dt,
            self.ring_weights.clone(),
        );
        ui.add(
            AnimatedSlider::new(&mut self.radius, MIN_KERNEL_RADIUS..=MAX_KERNEL_RADIUS)
                .text("Radius R")
                .id_salt("lenia_radius"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.mu, 0.01..=0.60)
                .text("Growth μ")
                .id_salt("lenia_mu"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.sigma, 0.002..=0.15)
                .text("Width σ")
                .id_salt("lenia_sigma"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.dt, 0.005..=0.50)
                .text("Step dt")
                .id_salt("lenia_dt"),
        );
        self.draw_ring_controls(ui);

        if (
            self.radius,
            self.mu,
            self.sigma,
            self.dt,
            &self.ring_weights,
        ) != (prev.0, prev.1, prev.2, prev.3, &prev.4)
        {
            self.sync_engine_parameters();
        }
    }

    fn draw_ring_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Shells:");
            for count in 1..=MAX_KERNEL_RINGS {
                if ui
                    .selectable_label(self.ring_weights.len() == count, format!("{count}"))
                    .clicked()
                {
                    self.ring_weights.resize(count, 0.5);
                }
            }
        });
        for idx in 0..self.ring_weights.len() {
            let label = format!("Ring {} β", idx + 1);
            ui.add(
                AnimatedSlider::new(&mut self.ring_weights[idx], 0.0..=1.0)
                    .text(&label)
                    .id_salt(("lenia_ring_beta", idx)),
            );
        }
    }

    fn draw_brush_and_grid_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("Brush & Perturbation:");
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.brush_mode, LeniaBrushMode::Deposit, "🖌 Deposit");
            ui.selectable_value(&mut self.brush_mode, LeniaBrushMode::Slice, "✂ Slice Tail");
            if ui.button("Clear Grid").clicked() {
                self.engine.clear();
                self.history.clear();
                self.record_telemetry_sample();
                self.refresh_pixels_in_place();
            }
        });
        ui.add(
            AnimatedSlider::new(&mut self.brush_radius, 1.0..=24.0)
                .text("Brush r")
                .id_salt("lenia_brush_r"),
        );
        ui.add(
            AnimatedSlider::new(&mut self.perturbation, 0.0..=0.35)
                .text("Noise")
                .id_salt("lenia_perturb"),
        );
        ui.horizontal_wrapped(|ui| {
            ui.label("Grid:");
            for dim in [64usize, 128, 256] {
                if ui
                    .selectable_label(self.grid_size == dim, format!("{dim}²"))
                    .clicked()
                    && self.grid_size != dim
                {
                    self.grid_size = dim;
                    self.reseed_current_genome();
                }
            }
            if ui.button("🎲 Reseed").clicked() {
                self.rng_seed = self.rng_seed.wrapping_add(1);
                self.reseed_current_genome();
            }
        });
    }

    fn draw_central_workspace(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let avail = ui.available_size();
            if avail.x >= 560.0 {
                ui.columns(2, |cols| {
                    let canvas_side = cols[0].available_width().min(avail.y - 16.0);
                    cols[0].vertical_centered(|ui| {
                        self.draw_interactive_canvas(ui, canvas_side);
                        ui.small("Drag to paint (Shift/Right-drag to slice); Tab + Arrows + Space to stamp.");
                    });
                    self.draw_inspector_plots(&mut cols[1], avail.y);
                });
            } else {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let side = avail.x.min(320.0);
                    self.draw_interactive_canvas(ui, side);
                    self.draw_inspector_plots(ui, 360.0);
                });
            }
        });
    }

    fn draw_inspector_plots(&self, ui: &mut egui::Ui, total_height: f32) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let dark = ui.visuals().dark_mode;
        let col_k = Self::contrast_line_color(palette.accent, dark);
        let col_g = Self::contrast_line_color(palette.success, dark);
        let col_v = Self::contrast_line_color(palette.warning, dark);
        let plot_h = ((total_height - 52.0) * 0.48).clamp(110.0, 320.0);

        let shells = KernelShells::new(self.radius, &self.ring_weights);
        let growth = GrowthParams::new(self.mu, self.sigma);
        let k_pts = shells.sample_radial_profile(96);
        let g_pts = growth.sample_curve(96);

        ui.label("Radial Kernel K(r) & Growth Curve G(u)");
        Plot::new(crate::framework::scoped_panel_id(
            ui,
            "lenia_kernel_growth_plot",
        ))
        .height(plot_h)
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Kernel K(r)", PlotPoints::from(k_pts))
                    .color(col_k)
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new("Growth G(u)", PlotPoints::from(g_pts))
                    .color(col_g)
                    .width(2.0_f32),
            );
        });

        ui.label("Live Soliton Telemetry");
        let (mass_pts, area_pts, speed_pts) = self.build_telemetry_series();
        Plot::new(crate::framework::scoped_panel_id(
            ui,
            "lenia_telemetry_plot",
        ))
        .height(plot_h)
        .legend(Legend::default())
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Total Mass", PlotPoints::from(mass_pts))
                    .color(col_k)
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new("Active Area", PlotPoints::from(area_pts))
                    .color(col_g)
                    .width(1.6_f32),
            );
            plot_ui.line(
                Line::new("Centroid Speed (×100)", PlotPoints::from(speed_pts))
                    .color(col_v)
                    .width(2.0_f32),
            );
        });
    }
}

type TelemetrySeriesTriple = (Vec<[f64; 2]>, Vec<[f64; 2]>, Vec<[f64; 2]>);

impl LeniaTool {
    fn contrast_line_color(base: egui::Color32, dark_mode: bool) -> egui::Color32 {
        if dark_mode {
            let rgb = ensure_dark_canvas_contrast([base.r(), base.g(), base.b()]);
            egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
        } else {
            base
        }
    }

    fn build_telemetry_series(&self) -> TelemetrySeriesTriple {
        let mut mass = Vec::with_capacity(self.history.len().max(1));
        let mut area = Vec::with_capacity(self.history.len().max(1));
        let mut speed = Vec::with_capacity(self.history.len().max(1));
        for sample in &self.history {
            mass.push([sample.time, sample.total_mass]);
            area.push([sample.time, sample.active_area]);
            speed.push([sample.time, sample.centroid_speed * 100.0]);
        }
        (mass, area, speed)
    }

    pub(crate) fn collect_inspector_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let shells = KernelShells::new(self.radius, &self.ring_weights);
        let growth = GrowthParams::new(self.mu, self.sigma);
        let mut mass = Vec::with_capacity(self.history.len().max(1));
        let mut area = Vec::with_capacity(self.history.len().max(1));
        let mut speed = Vec::with_capacity(self.history.len().max(1));
        for sample in &self.history {
            mass.push([sample.time, sample.total_mass]);
            area.push([sample.time, sample.active_area]);
            speed.push([sample.time, sample.centroid_speed]);
        }
        vec![
            ("Kernel K(r)".to_string(), shells.sample_radial_profile(96)),
            ("Growth G(u)".to_string(), growth.sample_curve(96)),
            ("Total Mass".to_string(), mass),
            ("Active Area".to_string(), area),
            ("Centroid Speed".to_string(), speed),
        ]
    }
}
