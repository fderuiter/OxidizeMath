#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::counterfactual::{
    render_morphogenesis_heatmap, MorphogenesisHeatmapMode, MorphogenesisSnapshotData,
};
use super::sculpt::{draw_morpho_sculpt_controls, overlay_cell_mask_colors};
use super::MorphogenesisTool;
use crate::accessibility::{announce_status, AccessibleHoverText};
use crate::app::ThemeMode;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{
    fork_marker_callout, render_divergence_strip, BrushMode, CounterfactualTransport,
    InteractionContext, InteractiveTool, PlaybackAction, PlaybackTransport,
};
use crate::presets::MorphogenesisKineticsMode;
use eframe::egui;
use eframe::egui::ColorImage;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_explorer::biology::morphogenesis::GrayScottKinetics;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

impl MorphogenesisTool {
    fn render_snapshot_image(
        &self,
        theme: ThemeMode,
        mode: MorphogenesisHeatmapMode,
    ) -> Option<ColorImage> {
        let num_pixels = self.width * self.height;
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MorphogenesisSnapshotData>())
                .filter(|d| {
                    d.u_b.len() == num_pixels && d.step_count >= self.local_runner.step_count
                })
            {
                let mut img = render_morphogenesis_heatmap(
                    &data.u_b,
                    data.u_a.as_deref(),
                    self.width,
                    self.height,
                    mode,
                    theme,
                );
                if mode == MorphogenesisHeatmapMode::ActiveUb && !data.mask_roles.is_empty() {
                    overlay_cell_mask_colors(
                        &mut img,
                        &data.mask_roles,
                        theme == ThemeMode::HighContrast,
                    );
                }
                return Some(img);
            }
            if let Ok(px) = snapshot.pixels.try_read() {
                if px.len() == num_pixels && !self.is_forked() {
                    return Some(ColorImage {
                        size: [self.width, self.height],
                        pixels: px.clone(),
                        ..Default::default()
                    });
                }
            }
            return None;
        }
        self.is_forked()
            .then(|| self.render_current_heatmap_image())
    }

    fn sync_texture_from_snapshot(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let theme = self
            .last_theme_mode
            .unwrap_or_else(|| ThemeMode::from_visuals(&ctx.style().visuals));
        let mode = self.heatmap_mode();
        let num_pixels = self.width * self.height;
        let Some(image) = self.render_snapshot_image(theme, mode) else {
            return;
        };
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        self.pixels.copy_from_slice(&image.pixels);

        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("morphogenesis_tex", image, texture_opts));
        }
        self.dirty = false;
    }

    fn draw_kinetics_selector(&mut self, ui: &mut egui::Ui) {
        let current_mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        ui.label("Kinetics Model:");
        ui.horizontal_wrapped(|ui| {
            if ui
                .selectable_label(
                    current_mode == MorphogenesisKineticsMode::Schnakenberg,
                    "Schnakenberg",
                )
                .clicked()
                && current_mode != MorphogenesisKineticsMode::Schnakenberg
            {
                self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
            }
            if ui
                .selectable_label(
                    current_mode == MorphogenesisKineticsMode::GrayScott,
                    "Gray-Scott",
                )
                .clicked()
                && current_mode != MorphogenesisKineticsMode::GrayScott
            {
                let mitosis = GrayScottKinetics::mitosis();
                self.apply_gray_scott_preset(mitosis.f, mitosis.k);
            }
        });
    }

    fn stabilize_cfl_step(&mut self, ctx: &egui::Context, min_dt: f64, max_dt: f64) {
        let safe_dt = {
            let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
            let dt = (0.2485 / params.d_u.max(params.d_v).max(1e-6)).clamp(min_dt, max_dt);
            params.dt = dt;
            dt
        };
        self.reset_system();
        crate::accessibility::announce_live_status(
            ctx,
            format!("Time step stabilized to CFL bound dt = {safe_dt:.4}"),
        );
    }

    fn draw_schnakenberg_sliders(&mut self, ui: &mut egui::Ui) {
        let constraints = self.theory.theory_parameters();
        let bound = |key: &str, min: f64, max: f64, step: f64| {
            constraints
                .get(key)
                .cloned()
                .unwrap_or(ParameterConstraint { min, max, step })
        };
        let ca = bound("a", 0.0, 1.0, 0.01);
        let cb = bound("b", 0.0, 2.0, 0.01);
        let cdu = bound("d_u", 0.1, 5.0, 0.05);
        let cdv = bound("d_v", 10.0, 200.0, 1.0);
        let cdt = bound("dt", 1e-4, 0.1, 5e-4);

        let mut changed = false;
        let cfl_exceeded = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.a, ca.min..=ca.max)
                        .step_by(ca.step)
                        .text("a"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.b, cb.min..=cb.max)
                        .step_by(cb.step)
                        .text("b"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.d_u, cdu.min..=cdu.max)
                        .step_by(cdu.step)
                        .text("D_u"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.d_v, cdv.min..=cdv.max)
                        .step_by(cdv.step)
                        .text("D_v"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut p.dt, cdt.min..=cdt.max)
                        .logarithmic(true)
                        .text("dt"),
                )
                .changed();
            4.0 * p.d_u.max(p.d_v) * p.dt >= 1.0
        };
        if changed {
            let _ = self.reseed_if_washed_out_entering_pattern();
        }
        if cfl_exceeded && ui.button("⚡ Stabilize Step (CFL)").clicked() {
            self.stabilize_cfl_step(ui.ctx(), cdt.min, cdt.max);
        }
    }

    fn draw_heatmap_mode_selector(&mut self, ui: &mut egui::Ui) {
        if !self.is_forked() {
            return;
        }
        ui.separator();
        ui.label("Shadow Twin Heatmap:");
        let active_mode = self.heatmap_mode();
        ui.horizontal_wrapped(|ui| {
            for mode in MorphogenesisHeatmapMode::ALL {
                if ui
                    .selectable_label(active_mode == mode, mode.label())
                    .accessible_hover_text("Switch counterfactual Turing spatial heatmap mode")
                    .clicked()
                {
                    self.set_heatmap_mode(mode);
                    announce_status(&format!("Heatmap mode set to {}", mode.label()));
                }
            }
        });
    }

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        let mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        match mode {
            MorphogenesisKineticsMode::Schnakenberg => self.draw_schnakenberg_sliders(ui),
            MorphogenesisKineticsMode::GrayScott => {
                let mut ks = self
                    .kinetics_state
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
                ui.add(egui::Slider::new(&mut ks.f, 0.01..=0.09).text("f (feed)"));
                ui.add(egui::Slider::new(&mut ks.k, 0.03..=0.08).text("k (kill)"));
                ui.add(egui::Slider::new(&mut params.d_u, 0.05..=0.40).text("D_u"));
                ui.add(egui::Slider::new(&mut params.d_v, 0.02..=0.25).text("D_v"));
                ui.add(egui::Slider::new(&mut params.dt, 0.1..=1.5).text("dt"));
            }
        }
    }

    fn draw_presets(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Schnakenberg Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Spots").clicked() {
                self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
            }
            if ui.button("Stripes").clicked() {
                self.apply_named_preset(0.14, 0.86, 1.0, 50.0, 0.02);
            }
        });
        ui.label("Gray-Scott Presets:");
        ui.horizontal_wrapped(|ui| {
            if ui.button("Mitosis").clicked() {
                let m = GrayScottKinetics::mitosis();
                self.apply_gray_scott_preset(m.f, m.k);
            }
            if ui.button("Coral").clicked() {
                let c = GrayScottKinetics::coral();
                self.apply_gray_scott_preset(c.f, c.k);
            }
        });
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if let Some(u_data) = snapshot.structured_data.as_ref().and_then(|d| {
                d.downcast_ref::<MorphogenesisSnapshotData>()
                    .map(|s| s.u_b.as_slice())
                    .or_else(|| d.downcast_ref::<Vec<f64>>().map(Vec::as_slice))
            }) {
                let splat_count =
                    crate::volumetric_scene::presets::build_scene_from_2d_morphogenesis_slice(
                        u_data,
                        snapshot.width,
                        snapshot.height,
                    )
                    .gaussians
                    .len();
                ui.small(format!("Extruded 3D Gaussian splats: {splat_count}"));
            }
        }
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

    fn handle_playback_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
        match action {
            PlaybackAction::Play => {
                let _ = self.worker.send_command(SimCommand::Start);
            }
            PlaybackAction::Pause => {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            PlaybackAction::Step => {
                self.local_runner.step();
                self.sync_branch_engine_from_local();
                let _ = self.worker.send_command(SimCommand::Step(1));
                self.dirty = true;
            }
            PlaybackAction::Reset => {
                self.reset();
                ctx.request_repaint();
            }
        }
    }

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "morphogenesis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Morphogenesis Controls");
            ui.separator();

            self.draw_fault_and_recovery(ui, &ctx);
            self.draw_kinetics_selector(ui);
            ui.separator();
            self.draw_parameter_sliders(ui);
            ui.separator();

            if let Some(action) = PlaybackTransport::from_worker_state(
                "morphogenesis_transport",
                "Morphogenesis",
                self.worker.state(),
            )
            .can_reset(true)
            .show(ui)
            {
                self.handle_playback_action(action, &ctx);
            }

            let can_fork =
                self.local_runner.step_count > 0 && FiniteStateGuard::is_finite_state(self);
            let is_forked = self.is_forked();
            if let Some(cf_action) =
                CounterfactualTransport::new(&mut self.branch_engine, can_fork, is_forked).show(ui)
            {
                let _ = self.apply_morpho_branch_action(cf_action);
                ctx.request_repaint();
            }

            self.draw_heatmap_mode_selector(ui);
            if let Some(cmd) = draw_morpho_sculpt_controls(ui, &mut self.brush_kit) {
                self.apply_sculpt_quick_action(cmd, &ctx);
            }
            self.draw_presets(ui);
        });
    }

    fn draw_turing_field_plot(&mut self, ui: &mut egui::Ui) {
        if let Some(texture) = self.texture.clone() {
            let w = self.width as f32;
            let h = self.height as f32;
            let forked = self.local_runner.twin.as_ref().map(|t| t.fork_step);
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let allow_pan = !self.brush_kit.is_sculpting();
            let mut strokes = Vec::new();
            let plot_resp = Plot::new("morphogenesis_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .show_grid(false)
                .show_axes([false, false])
                .allow_drag(allow_pan)
                .allow_zoom(allow_pan)
                .show(ui, |plot_ui| {
                    plot_ui.image(PlotImage::new(
                        "morphogenesis_image",
                        texture.id(),
                        PlotPoint::new(f64::from(w) / 2.0, f64::from(h) / 2.0),
                        [w, h],
                    ));
                    if let Some(step) = forked {
                        let dt = self.params.read().unwrap_or_else(|e| e.into_inner()).dt;
                        plot_ui.callout(fork_marker_callout(
                            [f64::from(w) * 0.5, f64::from(h) * 0.94],
                            (step as f64) * dt,
                            &palette,
                        ));
                    }
                    strokes = self.brush_kit.sample_plot(
                        plot_ui,
                        [0.5, 0.5],
                        [f64::from(w) - 0.5, f64::from(h) - 0.5],
                        Some(BrushMode::Erase),
                    );
                });
            for stroke in strokes {
                let ic = InteractionContext::from_brush_stroke(&plot_resp.response, stroke);
                self.on_brush(&ic);
            }
        } else {
            render_empty_plot_guidance(
                ui,
                "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve Turing patterns.",
            );
        }
    }

    pub(crate) fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if let Some(snap) = self.counterfactual_snapshot() {
                render_divergence_strip(ui, self.name(), &snap);
                ui.add_space(4.0);
            }
            self.sync_texture_from_snapshot(&ctx);
            if !self.atlas.is_visible() {
                self.draw_turing_field_plot(ui);
                return;
            }
            let (ax, ay) = self.active_axis_values();
            let sig = self.non_axis_context_sig();
            let cartographer_params = {
                let p = self.params.read().unwrap_or_else(|e| e.into_inner());
                (p.a, p.b, p.d_u, p.d_v, p.dt)
            };
            let theory_params = self.theory.theory_parameters();
            let mut teleport = None;
            if ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_turing_field_plot(&mut cols[0]);
                    teleport =
                        self.atlas
                            .show_panel(&mut cols[1], "morphogenesis", ax, ay, sig, |keys| {
                                std::sync::Arc::new(super::atlas::MorphogenesisCartographer::new(
                                    &theory_params,
                                    keys,
                                    cartographer_params.0,
                                    cartographer_params.1,
                                    cartographer_params.2,
                                    cartographer_params.3,
                                    cartographer_params.4,
                                ))
                            });
                });
            } else {
                teleport = self
                    .atlas
                    .show_panel(ui, "morphogenesis", ax, ay, sig, |keys| {
                        std::sync::Arc::new(super::atlas::MorphogenesisCartographer::new(
                            &theory_params,
                            keys,
                            cartographer_params.0,
                            cartographer_params.1,
                            cartographer_params.2,
                            cartographer_params.3,
                            cartographer_params.4,
                        ))
                    });
                ui.separator();
                self.draw_turing_field_plot(ui);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev);
                ctx.request_repaint();
            }
        });
    }
}
