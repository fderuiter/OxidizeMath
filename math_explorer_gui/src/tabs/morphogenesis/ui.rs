//! Side-panel controls, 2D stability atlas integration, and brushable plot canvas
//! for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::draw_regime_atlas;
use super::cartographer::MorphogenesisCartographer;
use super::runner::MAX_CFL_SUBSTEPS;
#[cfg(test)]
use super::state::initialize_schnakenberg_system;
use super::state::{
    plot_concentration_with_contrast, CanvasInteractionMode, MorphogenesisKinetics,
};
use super::MorphogenesisTool;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{PlaybackAction, PlaybackTransport};
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
#[cfg(test)]
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::FusedEulerSolver;
#[cfg(test)]
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use scientific_metadata::theory::TheoryDescribable;

impl MorphogenesisTool {
    fn sync_texture_from_snapshot(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let Ok(snapshot_pixels) = snapshot.pixels.try_read() else {
            return;
        };
        if snapshot_pixels.is_empty() {
            return;
        }

        let num_pixels = snapshot.width * snapshot.height;
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let high_contrast = self.last_high_contrast.unwrap_or_else(|| {
            crate::app::ThemeMode::from_visuals(&ctx.style().visuals)
                == crate::app::ThemeMode::HighContrast
        });
        if let Some(u_data) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<f64>>())
            .filter(|u| u.len() == num_pixels)
        {
            let recolored = plot_concentration_with_contrast(
                u_data,
                snapshot.width,
                snapshot.height,
                high_contrast,
            );
            self.pixels.copy_from_slice(&recolored.pixels);
        } else if snapshot_pixels.len() == num_pixels {
            self.pixels.copy_from_slice(&snapshot_pixels);
        }

        let image = ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("morphogenesis_tex", image, texture_opts));
        }
        self.dirty = false;
    }

    fn draw_kinetics_and_atlas(&mut self, ui: &mut egui::Ui) {
        let (kinetics, x_val, y_val, d_u, d_v, high_contrast) = {
            let p = self.params.read().unwrap_or_else(|e| e.into_inner());
            let (xv, yv) = match p.kinetics {
                MorphogenesisKinetics::Schnakenberg => (p.a, p.b),
                MorphogenesisKinetics::GrayScott => (p.f, p.k),
            };
            (p.kinetics, xv, yv, p.d_u, p.d_v, p.high_contrast)
        };

        ui.horizontal(|ui| {
            ui.label("Kinetics:");
            if ui
                .selectable_label(
                    kinetics == MorphogenesisKinetics::Schnakenberg,
                    "Schnakenberg",
                )
                .clicked()
                && kinetics != MorphogenesisKinetics::Schnakenberg
            {
                self.switch_kinetics(MorphogenesisKinetics::Schnakenberg);
            }
            if ui
                .selectable_label(kinetics == MorphogenesisKinetics::GrayScott, "Gray-Scott")
                .clicked()
                && kinetics != MorphogenesisKinetics::GrayScott
            {
                self.switch_kinetics(MorphogenesisKinetics::GrayScott);
            }
        });

        let active_kinetics = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .kinetics;
        if active_kinetics == kinetics {
            if let Some((nx, ny)) = draw_regime_atlas(
                ui,
                &mut self.atlas_cache,
                kinetics,
                x_val,
                y_val,
                d_u,
                d_v,
                high_contrast,
            ) {
                self.apply_atlas_coordinates(kinetics, nx, ny);
                ui.ctx().request_repaint();
            }
        }
    }

    fn stabilize_cfl_step(&mut self, ctx: &egui::Context) {
        let safe_dt = {
            let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
            let dt = FusedEulerSolver::cfl_dt_2d(1.0, params.d_u, params.d_v).clamp(1e-4, 1.5);
            params.dt = dt;
            dt
        };
        self.reset_system();
        crate::accessibility::announce_live_status(
            ctx,
            format!("Time step stabilized to CFL bound dt = {safe_dt:.4}"),
        );
    }

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        let mut changed = false;
        let mut cfl_exceeded = false;
        {
            let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
            match params.kinetics {
                MorphogenesisKinetics::Schnakenberg => {
                    changed |= ui
                        .add(egui::Slider::new(&mut params.a, 0.0..=1.0).text("a"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.b, 0.0..=2.0).text("b"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.d_u, 0.1..=5.0).text("D_u"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.d_v, 10.0..=200.0).text("D_v"))
                        .changed();
                }
                MorphogenesisKinetics::GrayScott => {
                    changed |= ui
                        .add(egui::Slider::new(&mut params.f, 0.005..=0.085).text("F"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.k, 0.030..=0.070).text("k"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.d_u, 0.05..=1.0).text("D_u"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.d_v, 0.02..=0.5).text("D_v"))
                        .changed();
                }
            }
            ui.checkbox(&mut params.auto_dt, "Auto CFL Time Step");
            let (substeps, sub_dt) = FusedEulerSolver::cfl_substeps_2d(
                params.dt,
                1.0,
                params.d_u,
                params.d_v,
                MAX_CFL_SUBSTEPS,
            );
            if params.auto_dt {
                ui.small(format!("CFL sub_dt = {sub_dt:.4} ({substeps}x/step)"));
            } else {
                let max_dt = FusedEulerSolver::cfl_dt_2d(1.0, params.d_u, params.d_v);
                changed |= ui
                    .add(egui::Slider::new(&mut params.dt, 1e-4..=1.5).text("Manual dt"))
                    .changed();
                ui.small(format!("2D CFL bound: {max_dt:.4}"));
                cfl_exceeded = 4.0 * params.d_u.max(params.d_v) * params.dt >= 1.0;
            }
        }
        if changed {
            let _ = self.reseed_if_washed_out_entering_pattern();
        }
        if cfl_exceeded && ui.button("⚡ Stabilize Step (CFL)").clicked() {
            self.stabilize_cfl_step(ui.ctx());
        }
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Canvas:");
            ui.selectable_value(&mut self.canvas_mode, CanvasInteractionMode::Brush, "Brush");
            ui.selectable_value(
                &mut self.canvas_mode,
                CanvasInteractionMode::Navigate,
                "Navigate",
            );
        });
        if self.canvas_mode == CanvasInteractionMode::Brush {
            ui.add(egui::Slider::new(&mut self.brush_radius, 1..=12).text("Brush r"));
            ui.small("Primary drag: inject | Secondary: erase");
        }
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Spots").clicked() {
                self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
            }
            if ui.button("Stripes").clicked() {
                self.apply_named_preset(0.14, 0.86, 1.0, 50.0, 0.02);
            }
            if ui.button("Mitosis").clicked() {
                self.apply_gray_scott_preset(0.0367, 0.0649, 0.2, 0.1, 1.0);
            }
            if ui.button("Coral").clicked() {
                self.apply_gray_scott_preset(0.0545, 0.062, 0.2, 0.1, 1.0);
            }
            if ui.button("Solitons").clicked() {
                self.apply_gray_scott_preset(0.030, 0.060, 0.2, 0.1, 1.0);
            }
        });
        ui.separator();
        let _ = self.atlas.show_toggle_button(ui);
    }

    fn draw_fault_and_recovery(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(action) = render_worker_fault_banner(
            ui,
            self.worker.state(),
            self.worker.diagnostic_report(),
            true,
        ) {
            match action {
                WorkerFaultAction::Reset => self.reset(),
                WorkerFaultAction::ResetParameters => FiniteStateGuard::reset_parameters(self),
                WorkerFaultAction::Recovery(RecoveryAction::AutoScaleStepSize) => {
                    FiniteStateGuard::auto_scale_step_size(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::ClampParameters) => {
                    FiniteStateGuard::clamp_parameters(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::ResetParameters) => {
                    FiniteStateGuard::reset_parameters(self);
                }
            }
            ctx.request_repaint();
            ui.separator();
        }
    }

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "morphogenesis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Morphogenesis Controls");
                ui.separator();

                self.draw_fault_and_recovery(ui, &ctx);
                self.draw_kinetics_and_atlas(ui);
                ui.separator();
                self.draw_parameter_sliders(ui);
                ui.separator();
                self.draw_brush_controls(ui);
                ui.separator();

                if let Some(action) = PlaybackTransport::from_worker_state(
                    "morphogenesis_transport",
                    "Morphogenesis",
                    self.worker.state(),
                )
                .can_reset(true)
                .show(ui)
                {
                    match action {
                        PlaybackAction::Play => {
                            let _ = self.worker.send_command(SimCommand::Start);
                        }
                        PlaybackAction::Pause => {
                            let _ = self.worker.send_command(SimCommand::Pause);
                        }
                        PlaybackAction::Step => {
                            let _ = self.worker.send_command(SimCommand::Step(1));
                        }
                        PlaybackAction::Reset => {
                            self.reset();
                            ctx.request_repaint();
                        }
                    }
                }

                self.draw_presets(ui);
            });
        });
    }

    fn draw_field_plot(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(texture) = self.texture.clone() else {
            render_empty_plot_guidance(
                ui,
                "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve patterns.",
            );
            return;
        };

        let w = self.width as f32;
        let h = self.height as f32;
        let navigate = self.canvas_mode == CanvasInteractionMode::Navigate;
        let brush_r = self.brush_radius;
        let mut brush_cmd: Option<SimCommand> = None;

        Plot::new("morphogenesis_plot")
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .show_axes([false, false])
            .allow_drag(navigate)
            .allow_zoom(navigate)
            .allow_scroll(navigate)
            .show(ui, |plot_ui| {
                plot_ui.image(PlotImage::new(
                    "morphogenesis_image",
                    texture.id(),
                    PlotPoint::new(f64::from(w) / 2.0, f64::from(h) / 2.0),
                    [w, h],
                ));

                if !navigate
                    && plot_ui.response().hovered()
                    && ctx.input(|i| i.pointer.primary_down() || i.pointer.secondary_down())
                {
                    if let Some(pos) = plot_ui.pointer_coordinate() {
                        let cx = pos.x.floor() as i32;
                        let cy = (f64::from(h) - pos.y).floor() as i32;
                        if cx >= 0 && cx < self.width as i32 && cy >= 0 && cy < self.height as i32 {
                            brush_cmd = Some(SimCommand::ApplyBrush {
                                cx,
                                cy,
                                r: brush_r,
                                is_obstacle: ctx.input(|i| i.pointer.primary_down()),
                            });
                        }
                    }
                }
            });

        if let Some(cmd) = brush_cmd {
            if self.worker.send_command(cmd).is_ok() {
                self.pending_brush_frames = 4;
                ctx.request_repaint();
            }
        }
    }

    fn build_progressive_cartographer(
        &self,
        keys: (&'static str, &'static str),
    ) -> std::sync::Arc<MorphogenesisCartographer> {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let theory_params = self.theory.theory_parameters();
        std::sync::Arc::new(
            MorphogenesisCartographer::new(&theory_params, keys, p.a, p.b, p.d_u, p.d_v, p.dt)
                .with_kinetics(p.kinetics, p.f, p.k, p.auto_dt),
        )
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.sync_texture_from_snapshot(&ctx);
            if !self.atlas.is_visible() {
                self.draw_field_plot(ui, &ctx);
                return;
            }
            let (ax, ay) = self.active_axis_values();
            let sig = self.non_axis_context_sig();
            let p_snap = self
                .params
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            let theory_params = self.theory.theory_parameters();
            let build = |keys: (&'static str, &'static str)| {
                std::sync::Arc::new(
                    MorphogenesisCartographer::new(
                        &theory_params,
                        keys,
                        p_snap.a,
                        p_snap.b,
                        p_snap.d_u,
                        p_snap.d_v,
                        p_snap.dt,
                    )
                    .with_kinetics(
                        p_snap.kinetics,
                        p_snap.f,
                        p_snap.k,
                        p_snap.auto_dt,
                    ),
                ) as std::sync::Arc<dyn crate::async_sim::atlas::RegimeCartographer>
            };

            let mut teleport = None;
            if ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_field_plot(&mut cols[0], &ctx);
                    teleport =
                        self.atlas
                            .show_panel(&mut cols[1], "morphogenesis", ax, ay, sig, build);
                });
            } else {
                let _ = self.build_progressive_cartographer(self.atlas.axis_keys());
                teleport = self
                    .atlas
                    .show_panel(ui, "morphogenesis", ax, ay, sig, build);
                ui.separator();
                self.draw_field_plot(ui, &ctx);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev);
                ctx.request_repaint();
            }
        });
    }

    #[cfg(test)]
    pub(crate) fn initialize_system(
        system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        width: usize,
        height: usize,
    ) {
        initialize_schnakenberg_system(system, width, height);
    }

    #[cfg(test)]
    pub(crate) fn plot_concentration(data: &[f64], width: usize, height: usize) -> ColorImage {
        plot_concentration_with_contrast(data, width, height, false)
    }

    /// Maps a 2D concentration slice onto a normalized heatmap [`ColorImage`] with optional high-contrast tokens.
    #[cfg(test)]
    #[must_use]
    pub fn plot_concentration_with_contrast(
        data: &[f64],
        width: usize,
        height: usize,
        high_contrast: bool,
    ) -> ColorImage {
        plot_concentration_with_contrast(data, width, height, high_contrast)
    }
}
