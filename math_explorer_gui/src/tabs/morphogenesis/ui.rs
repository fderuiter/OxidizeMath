//! Side-panel controls, 2D stability atlas integration, counterfactual twin transport,
//! and accessible brushable plot canvas for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::draw_regime_atlas;
use super::runner::{MorphoSnapshotData, MAX_CFL_SUBSTEPS};
use super::sculpt::{build_maze_rle, draw_morpho_sculpt_controls};
use super::state::MorphogenesisKinetics;
use super::MorphogenesisTool;
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{CounterfactualTransport, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::biology::morphogenesis::FusedEulerSolver;

impl MorphogenesisTool {
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
            for (mode, label) in [
                (MorphogenesisKinetics::Schnakenberg, "Schnakenberg"),
                (MorphogenesisKinetics::GrayScott, "Gray-Scott"),
            ] {
                if ui.selectable_label(kinetics == mode, label).clicked() && kinetics != mode {
                    self.switch_kinetics(mode);
                }
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

    fn draw_kinetics_sliders(
        ui: &mut egui::Ui,
        params: &mut super::runner::MorphogenesisSharedParams,
    ) -> (bool, Option<(f64, f64)>) {
        let mut changed = false;
        let mut gs_sync = None;
        match params.kinetics {
            MorphogenesisKinetics::Schnakenberg => {
                for (val, range, label) in [
                    (&mut params.a, 0.0..=1.0, "a"),
                    (&mut params.b, 0.0..=2.0, "b"),
                    (&mut params.d_u, 0.1..=5.0, "D_u"),
                    (&mut params.d_v, 10.0..=200.0, "D_v"),
                ] {
                    changed |= ui.add(egui::Slider::new(val, range).text(label)).changed();
                }
            }
            MorphogenesisKinetics::GrayScott => {
                for (val, range, label) in [
                    (&mut params.f, 0.005..=0.085, "F"),
                    (&mut params.k, 0.030..=0.070, "k"),
                    (&mut params.d_u, 0.05..=1.0, "D_u"),
                    (&mut params.d_v, 0.02..=0.5, "D_v"),
                ] {
                    changed |= ui.add(egui::Slider::new(val, range).text(label)).changed();
                }
                gs_sync = Some((params.f, params.k));
            }
        }
        (changed, gs_sync)
    }

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) {
        let (changed, gs_sync, cfl_exceeded) = {
            let mut params = self.params.write().unwrap_or_else(|e| e.into_inner());
            let (mut ch, gs) = Self::draw_kinetics_sliders(ui, &mut params);
            ui.checkbox(&mut params.auto_dt, "Auto CFL Time Step");
            let (substeps, sub_dt) = FusedEulerSolver::cfl_substeps_2d(
                params.dt,
                1.0,
                params.d_u,
                params.d_v,
                MAX_CFL_SUBSTEPS,
            );
            let mut exceeded = false;
            if params.auto_dt {
                ui.small(format!("CFL sub_dt = {sub_dt:.4} ({substeps}x/step)"));
            } else {
                let max_dt = FusedEulerSolver::cfl_dt_2d(1.0, params.d_u, params.d_v);
                ch |= ui
                    .add(egui::Slider::new(&mut params.dt, 1e-4..=1.5).text("Manual dt"))
                    .changed();
                ui.small(format!("2D CFL bound: {max_dt:.4}"));
                exceeded = 4.0 * params.d_u.max(params.d_v) * params.dt >= 1.0;
            }
            (ch, gs, exceeded)
        };
        if let Some((f, k)) = gs_sync {
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
            ks.mode = MorphogenesisKinetics::GrayScott;
            ks.f = f;
            ks.k = k;
        }
        if changed {
            let _ = self.reseed_if_washed_out_entering_pattern();
        }
        if cfl_exceeded && ui.button("⚡ Stabilize Step (CFL)").clicked() {
            self.stabilize_cfl_step(ui.ctx());
        }
    }

    fn handle_sculpt_command(&mut self, cmd: &str, ctx: &egui::Context) {
        match cmd {
            "stamp_maze" => {
                *self.mask_rle.write().unwrap_or_else(|e| e.into_inner()) =
                    build_maze_rle(self.width, self.height);
                self.local_runner
                    .system
                    .diffusion
                    .mask
                    .stamp_microfluidic_maze();
                self.local_runner.enforce_mask_concentrations();
                self.sync_branch_engine_from_local();
                let _ = self
                    .worker
                    .send_command(SimCommand::Custom("stamp_maze".to_string()));
            }
            "clear_mask" => {
                self.mask_rle
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
                self.local_runner.system.diffusion.mask.clear();
                let _ = self.worker.send_command(SimCommand::ClearObstacles);
            }
            _ => return,
        }
        self.stepped_pixels = None;
        self.step_runner = None;
        self.pending_brush_frames = 4;
        self.dirty = true;
        ctx.request_repaint();
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
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if let Some(u_data) = snapshot.structured_data.as_ref().and_then(|d| {
                d.downcast_ref::<MorphoSnapshotData>()
                    .map(|m| m.u_data.as_slice())
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
                WorkerFaultAction::ResetParameters
                | WorkerFaultAction::Recovery(RecoveryAction::ResetParameters) => {
                    FiniteStateGuard::reset_parameters(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::AutoScaleStepSize) => {
                    FiniteStateGuard::auto_scale_step_size(self);
                }
                WorkerFaultAction::Recovery(RecoveryAction::ClampParameters) => {
                    FiniteStateGuard::clamp_parameters(self);
                }
            }
            ctx.request_repaint();
            ui.separator();
        }
    }

    fn draw_playback_transport(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let canvas_sculpt_focused = self.brush_kit.is_sculpting()
            && self
                .brush_kit
                .focused_plot_id()
                .is_some_and(|id| ctx.memory(|m| m.has_focus(id)));
        let Some(action) = PlaybackTransport::from_worker_state(
            "morphogenesis_transport",
            "Morphogenesis",
            self.worker.state(),
        )
        .timeline(self.worker.timeline_metadata())
        .can_reset(true)
        .allow_space_shortcut(!canvas_sculpt_focused)
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
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Morphogenesis Controls");
                ui.separator();

                self.draw_fault_and_recovery(ui, &ctx);
                self.draw_kinetics_and_atlas(ui);
                ui.separator();
                self.draw_parameter_sliders(ui);
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
                    CounterfactualTransport::new(&mut self.branch_engine, can_fork, is_forked)
                        .show(ui)
                {
                    let _ = self.apply_morpho_branch_action(cf_action);
                    ctx.request_repaint();
                }

                self.draw_heatmap_mode_selector(ui);
                if let Some(cmd) =
                    draw_morpho_sculpt_controls(ui, &mut self.brush_kit, &mut self.reagent_brush)
                {
                    self.handle_sculpt_command(cmd, &ctx);
                }
                self.draw_presets(ui);
            });
        });
    }
}
