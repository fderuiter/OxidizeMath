#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{PRESET_GRAY_SCOTT_SPIRAL_FLOW, PRESET_KARMAN_TURING_WAKE};
use super::types::{
    compute_peclet_damkohler, MorphogenesisBrushMode, MorphogenesisFlowMode,
    MorphogenesisKineticsMode, MorphogenesisOverlayMode, MorphogenesisSharedParams,
    MorphogenesisSnapshotData,
};
use super::MorphogenesisTool;
use crate::app::SemanticPalette;
use crate::async_sim::ui::{render_worker_fault_banner, WorkerFaultAction};
use crate::async_sim::SimCommand;
use crate::framework::recovery::{FiniteStateGuard, RecoveryAction};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::biology::morphogenesis::GrayScottKinetics;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

impl MorphogenesisTool {
    pub(crate) fn current_peclet_damkohler(&self) -> (f64, f64) {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let u_est = match p.flow_mode {
            MorphogenesisFlowMode::None => 0.0,
            MorphogenesisFlowMode::LbmVortexWake => p.inlet_velocity,
            MorphogenesisFlowMode::PotentialFlow => {
                let r = ((self.height as f64) / 10.0).max(2.0);
                p.inlet_velocity + p.circulation.abs() / (2.0 * std::f64::consts::PI * r)
            }
        };
        if self.worker.is_running()
            && (self.live_peclet > 0.0 || p.flow_mode == MorphogenesisFlowMode::None)
        {
            (self.live_peclet, self.live_damkohler)
        } else {
            let length = ((self.height as f64) / 6.0).max(1.0);
            let (r1, r2) = match p.kinetics_mode {
                MorphogenesisKineticsMode::Schnakenberg => (p.a, p.b),
                MorphogenesisKineticsMode::GrayScott => (p.f, p.k),
            };
            compute_peclet_damkohler(u_est, length, p.d_u, r1, r2, p.kinetics_mode)
        }
    }

    fn draw_dimensionless_readouts(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let (pe, da) = self.current_peclet_damkohler();
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(palette.info, format!("Péclet Pe = {pe:.2}"));
            ui.separator();
            ui.colored_label(palette.accent, format!("Damköhler Da = {da:.2}"));
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

    fn render_schnakenberg_sliders(
        ui: &mut egui::Ui,
        p: &mut MorphogenesisSharedParams,
        constraints: &std::collections::HashMap<String, ParameterConstraint>,
    ) -> bool {
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
        let mut changed = false;
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
        changed
    }

    fn render_gray_scott_sliders(ui: &mut egui::Ui, p: &mut MorphogenesisSharedParams) -> bool {
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut p.f, 0.01..=0.09).text("f (feed)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut p.k, 0.03..=0.08).text("k (kill)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut p.d_u, 0.05..=0.40).text("D_u"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut p.d_v, 0.02..=0.25).text("D_v"))
            .changed();
        changed
    }

    fn render_reaction_param_sliders(
        ui: &mut egui::Ui,
        p: &mut MorphogenesisSharedParams,
        constraints: &std::collections::HashMap<String, ParameterConstraint>,
    ) -> (bool, f64, f64) {
        let cdt = constraints
            .get("dt")
            .cloned()
            .unwrap_or(ParameterConstraint {
                min: 1e-4,
                max: 0.1,
                step: 5e-4,
            });
        let (mut changed, max_dt) = match p.kinetics_mode {
            MorphogenesisKineticsMode::Schnakenberg => (
                Self::render_schnakenberg_sliders(ui, p, constraints),
                cdt.max,
            ),
            MorphogenesisKineticsMode::GrayScott => (Self::render_gray_scott_sliders(ui, p), 1.5),
        };
        changed |= ui
            .add(
                egui::Slider::new(&mut p.dt, cdt.min..=max_dt)
                    .logarithmic(true)
                    .text("dt"),
            )
            .changed();
        (changed, cdt.min, max_dt)
    }

    fn draw_kinetics_and_reaction_sliders(&mut self, ui: &mut egui::Ui) {
        let constraints = self.theory.theory_parameters();
        let mut switched_to = None;
        let (changed, cfl_exceeded, dt_bounds) = {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            let current_mode = p.kinetics_mode;
            ui.horizontal_wrapped(|ui| {
                ui.label("Kinetics:");
                for mode in [
                    MorphogenesisKineticsMode::Schnakenberg,
                    MorphogenesisKineticsMode::GrayScott,
                ] {
                    if ui
                        .selectable_label(current_mode == mode, mode.label())
                        .clicked()
                        && current_mode != mode
                    {
                        switched_to = Some(mode);
                    }
                }
            });
            let (changed, min_dt, max_dt) =
                Self::render_reaction_param_sliders(ui, &mut p, &constraints);
            let cfl_bad = p.flow_mode == MorphogenesisFlowMode::None
                && 4.0 * p.d_u.max(p.d_v) * p.dt >= 1.0;
            (changed, cfl_bad, (min_dt, max_dt))
        };

        if let Some(new_mode) = switched_to {
            match new_mode {
                MorphogenesisKineticsMode::Schnakenberg => {
                    self.apply_named_preset(0.1, 0.9, 1.0, 100.0, 0.05);
                }
                MorphogenesisKineticsMode::GrayScott => {
                    let mitosis = GrayScottKinetics::mitosis();
                    self.apply_gray_scott_preset(mitosis.f, mitosis.k);
                }
            }
        } else if changed {
            let _ = self.reseed_if_washed_out_entering_pattern();
        }
        if cfl_exceeded && ui.button("⚡ Stabilize Step (CFL)").clicked() {
            self.stabilize_cfl_step(ui.ctx(), dt_bounds.0, dt_bounds.1);
        }
    }

    fn draw_flow_controls(&mut self, ui: &mut egui::Ui) {
        let mut flow_changed = false;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            egui::ComboBox::from_label("Flow Field")
                .selected_text(p.flow_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [
                        MorphogenesisFlowMode::None,
                        MorphogenesisFlowMode::LbmVortexWake,
                        MorphogenesisFlowMode::PotentialFlow,
                    ] {
                        if ui
                            .selectable_value(&mut p.flow_mode, mode, mode.label())
                            .changed()
                        {
                            if mode != MorphogenesisFlowMode::None && p.inlet_velocity <= 0.0 {
                                p.inlet_velocity = 0.08;
                            }
                            flow_changed = true;
                        }
                    }
                });
            if p.flow_mode != MorphogenesisFlowMode::None {
                ui.add(egui::Slider::new(&mut p.inlet_velocity, 0.0..=0.22).text("Inlet U"));
            }
            if p.flow_mode == MorphogenesisFlowMode::LbmVortexWake {
                ui.add(egui::Slider::new(&mut p.viscosity, 0.005..=0.15).text("Viscosity ν"));
            }
            if p.flow_mode == MorphogenesisFlowMode::PotentialFlow {
                ui.add(egui::Slider::new(&mut p.circulation, -15.0..=15.0).text("Circulation Γ"));
            }
        }
        if flow_changed {
            self.dirty = true;
        }
    }

    fn draw_overlay_and_brush_controls(&mut self, ui: &mut egui::Ui) {
        let mut clear_obstacles = false;
        let mut overlay_changed = false;
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            egui::ComboBox::from_label("Overlay")
                .selected_text(p.overlay_mode.label())
                .show_ui(ui, |ui| {
                    for ov in [
                        MorphogenesisOverlayMode::Concentration,
                        MorphogenesisOverlayMode::Vorticity,
                        MorphogenesisOverlayMode::DualOverlay,
                    ] {
                        overlay_changed |= ui
                            .selectable_value(&mut p.overlay_mode, ov, ov.label())
                            .changed();
                    }
                });
            egui::ComboBox::from_label("Sculpt Brush")
                .selected_text(p.brush_mode.label())
                .show_ui(ui, |ui| {
                    for bm in [
                        MorphogenesisBrushMode::Obstacle,
                        MorphogenesisBrushMode::EraseObstacle,
                        MorphogenesisBrushMode::Reagent,
                        MorphogenesisBrushMode::PanZoom,
                    ] {
                        ui.selectable_value(&mut p.brush_mode, bm, bm.label());
                    }
                });
            ui.add(egui::Slider::new(&mut p.brush_radius, 1..=12).text("Brush r"));
            if ui.button("Clear Obstacles").clicked() {
                clear_obstacles = true;
            }
        }
        if overlay_changed {
            self.dirty = true;
        }
        if clear_obstacles {
            let _ = self.worker.send_command(SimCommand::ClearObstacles);
            self.dirty = true;
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
                let m = GrayScottKinetics::mitosis();
                self.apply_gray_scott_preset(m.f, m.k);
            }
            if ui.button("Coral").clicked() {
                let c = GrayScottKinetics::coral();
                self.apply_gray_scott_preset(c.f, c.k);
            }
            if ui.button("Kármán Wake").clicked() {
                let _ = self.import_preset_json(PRESET_KARMAN_TURING_WAKE);
            }
            if ui.button("GS Swirl").clicked() {
                let _ = self.import_preset_json(PRESET_GRAY_SCOTT_SPIRAL_FLOW);
            }
        });
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<MorphogenesisSnapshotData>())
            {
                let splat_count =
                    crate::volumetric_scene::presets::build_scene_from_2d_morphogenesis_slice(
                        &data.concentration,
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

    pub(crate) fn draw_controls(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(ui, "morphogenesis_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Morphogenesis Controls");
                self.draw_dimensionless_readouts(ui);
                ui.separator();

                self.draw_fault_and_recovery(ui, &ctx);
                self.draw_kinetics_and_reaction_sliders(ui);
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
                ui.separator();
                self.draw_flow_controls(ui);
                self.draw_overlay_and_brush_controls(ui);
            });
        });
    }
}
