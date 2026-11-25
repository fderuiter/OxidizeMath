#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::sculpt::draw_morpho_sculpt_controls;
use super::MorphogenesisTool;
use crate::accessibility::AccessibleTheoryHover;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{
    fork_marker_callout, render_divergence_strip, BrushMode, CounterfactualTransport,
    InteractiveTool, PlaybackAction, PlaybackTransport, PlotBrushController, PlotBrushFrame,
    RasterBrushCanvas,
};
use crate::presets::MorphogenesisKineticsMode;
use eframe::egui;
use egui_plot::{Plot, PlotImage, PlotPoint};
use math_explorer::biology::morphogenesis::GrayScottKinetics;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

impl MorphogenesisTool {
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
        {
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
        }
        if changed {
            let _ = self.reseed_if_washed_out_entering_pattern();
        }
        let (_, _, _, combined_cfl) = self.advection_diagnostics();
        if combined_cfl >= 1.0 && ui.button("⚡ Stabilize Step (CFL)").clicked() {
            self.stabilize_combined_cfl_step(ui.ctx(), cdt.min, cdt.max);
        }
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
                let mut changed = false;
                {
                    let mut ks = self
                        .kinetics_state
                        .write()
                        .unwrap_or_else(|e| e.into_inner());
                    let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
                    changed |= ui
                        .add(egui::Slider::new(&mut ks.f, 0.01..=0.09).text("f (feed)"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut ks.k, 0.03..=0.08).text("k (kill)"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.d_u, 0.05..=0.40).text("D_u"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.d_v, 0.02..=0.25).text("D_v"))
                        .changed();
                    changed |= ui
                        .add(egui::Slider::new(&mut params.dt, 0.1..=1.5).text("dt"))
                        .changed();
                }
                if changed {
                    let _ = self.reseed_if_washed_out_entering_pattern();
                }
                let (_, _, _, combined_cfl) = self.advection_diagnostics();
                if combined_cfl >= 1.0 && ui.button("⚡ Stabilize Step (CFL)").clicked() {
                    self.stabilize_combined_cfl_step(ui.ctx(), 0.1, 1.5);
                }
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
            if ui.button("Solitons").clicked() {
                self.apply_gray_scott_preset(0.030, 0.060);
            }
        });
        super::sculpt::draw_morpho_3d_bridges(self, ui);
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

    fn draw_playback_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let canvas_focused = self
            .focused_canvas_id()
            .is_some_and(|id| ctx.memory(|m| m.has_focus(id)));
        let Some(action) = PlaybackTransport::from_worker_state(
            "morphogenesis_transport",
            "Morphogenesis",
            self.worker.state(),
        )
        .timeline(self.worker.timeline_metadata())
        .can_reset(true)
        .allow_space_shortcut(!canvas_focused)
        .show(ui) else {
            return;
        };
        match action {
            PlaybackAction::Play => {
                let _ = self.worker.send_command(SimCommand::Start);
            }
            PlaybackAction::Pause => {
                let _ = self.worker.send_command(SimCommand::Pause);
            }
            PlaybackAction::Step => {
                self.local_runner.step();
                self.live_velocity_peaks = [
                    self.local_runner.wind_tunnel.max_abs_ux,
                    self.local_runner.wind_tunnel.max_abs_uy,
                ];
                self.sync_branch_engine_from_local();
                let _ = self.worker.send_command(SimCommand::Step(1));
                self.dirty = true;
            }
            PlaybackAction::StepBack => {
                let _ = self.local_runner.clear_fork();
                self.branch_engine.clear();
                let _ = self.worker.send_command(SimCommand::StepBack(1));
            }
            PlaybackAction::ScrubTo(step) => {
                let _ = self.local_runner.clear_fork();
                self.branch_engine.clear();
                let _ = self.worker.send_command(SimCommand::SeekStep(step));
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
            self.draw_cfl_and_minimap(ui);
            ui.separator();
            self.draw_wind_tunnel_controls(ui);
            ui.separator();
            self.draw_playback_transport(ui, &ctx);

            let current_step = self
                .worker
                .timeline_metadata()
                .map_or(self.local_runner.step_count, |m| {
                    m.current_step.max(self.local_runner.step_count)
                });
            let can_fork = current_step > 0 && FiniteStateGuard::is_finite_state(self);
            let is_forked = self.is_forked();
            if let Some(cf_action) =
                CounterfactualTransport::new(&mut self.branch_engine, can_fork, is_forked).show(ui)
            {
                let _ = self.apply_morpho_branch_action(cf_action);
                ctx.request_repaint();
            }

            self.draw_heatmap_mode_selector(ui);
            if let Some(cmd) = draw_morpho_sculpt_controls(
                ui,
                &mut self.brush_kit,
                &mut self.sculpt_microfluidic_layer,
            ) {
                self.apply_sculpt_quick_action(cmd, &ctx);
            }
            self.draw_presets(ui);
            self.draw_brush_section(ui, &ctx);
        });
    }

    fn build_morphogenesis_a11y_summary(&self) -> String {
        let mode = self
            .kinetics_state
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .mode;
        let kinetics_lbl = match mode {
            MorphogenesisKineticsMode::Schnakenberg => "Schnakenberg",
            MorphogenesisKineticsMode::GrayScott => "Gray-Scott",
        };
        let masked = self.mask_cells.iter().filter(|&&c| c != 0).count();
        let fork_note = if self.is_forked() {
            ", counterfactual twin branch active"
        } else {
            ""
        };
        format!(
            "{}x{} Turing {kinetics_lbl} reaction-diffusion grid, {masked} sculpted boundary/sink/emitter cells{fork_note}; Color/shape key: blue-cyan-green-yellow-red concentration ramp from low to high morphogen density; sculpted walls and sinks rendered as dark masked cells",
            self.width, self.height
        )
    }

    fn draw_turing_field_plot(&mut self, ui: &mut egui::Ui, tex: &egui::TextureHandle) {
        let (w, h) = (self.width as f32, self.height as f32);
        let forked = self.local_runner.twin.as_ref().map(|t| t.fork_step);
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let allow_pan = !self.brush_kit.is_sculpting();
        let summary = self.build_morphogenesis_a11y_summary();
        let mut brush_frame: Option<PlotBrushFrame> = None;
        let plot_resp = Plot::new("morphogenesis_plot")
            .title("Turing Morphogenesis Field")
            .description(&summary)
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .show_axes([false, false])
            .allow_drag(allow_pan)
            .allow_zoom(true)
            .show(ui, |plot_ui| {
                plot_ui.image(PlotImage::new(
                    "morphogenesis_image",
                    tex.id(),
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
                brush_frame = Some(PlotBrushController::capture_plot(
                    &mut self.brush_kit,
                    plot_ui,
                    [0.5, 0.5],
                    [f64::from(w) - 0.5, f64::from(h) - 0.5],
                    Some(BrushMode::Erase),
                ));
            });
        if let Some(frame) = brush_frame {
            frame.dispatch(ui, &plot_resp.response, self);
        }
    }

    fn draw_turing_field_canvas(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(tex) = self.texture.clone() else {
            render_empty_plot_guidance(
                ui,
                "Reaction-diffusion grid ready — click ▶ Play or ⏭ Step to evolve Turing patterns.",
            );
            return;
        };
        if !self.brush_kit.is_sculpting() || self.is_forked() {
            self.draw_turing_field_plot(ui, &tex);
            return;
        }
        let summary = self.build_morphogenesis_a11y_summary();
        let (resp, stroke) = RasterBrushCanvas::new(
            "morphogenesis_canvas",
            self.width,
            self.height,
            "Turing Morphogenesis reaction-diffusion canvas",
        )
        .value_summary(&summary)
        .show_texture(ui, &tex, &mut self.brush_state);
        let _ = resp.accessible_theory_hover(&self.theory);
        if let Some(stroke) = stroke {
            if self.sculpt_microfluidic_layer {
                self.apply_microfluidic_raster_stroke(stroke, Some(ctx));
            } else {
                self.apply_brush_stroke(stroke, Some(ctx));
            }
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
                self.draw_turing_field_canvas(ui, &ctx);
                return;
            }
            let current_mode = self
                .kinetics_state
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .mode;
            self.sync_atlas_options_for_mode(current_mode);
            let (ax, ay) = self.active_axis_values();
            let sig = self.non_axis_context_sig();
            let cart = self.build_active_morpho_cartographer(self.atlas.axis_keys());
            let mut teleport = None;
            if ui.available_width() >= 620.0 {
                ui.columns(2, |cols| {
                    self.draw_turing_field_canvas(&mut cols[0], &ctx);
                    teleport =
                        self.atlas
                            .show_panel(&mut cols[1], "morphogenesis", ax, ay, sig, |_| {
                                std::sync::Arc::clone(&cart)
                            });
                });
            } else {
                teleport = self
                    .atlas
                    .show_panel(ui, "morphogenesis", ax, ay, sig, |_| {
                        std::sync::Arc::clone(&cart)
                    });
                ui.separator();
                self.draw_turing_field_canvas(ui, &ctx);
            }
            if let Some(ev) = teleport {
                self.apply_atlas_teleport(ev);
                ctx.request_repaint();
            }
        });
    }
}
