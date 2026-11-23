//! Side-panel controls, `PlaybackTransport`, `CustomSurfaceSandbox`, and CFL/Laplace-Beltrami
//! readouts for [`super::ManifoldFlowStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::custom_surface::SurfaceType;
use super::{ManifoldFlowMode, ManifoldFlowStudio};
use crate::app::SemanticPalette;
use crate::framework::bounded_drag_value;
use crate::framework::playback::{PlaybackAction, PlaybackTransport};
use eframe::egui;

impl ManifoldFlowStudio {
    pub(super) fn render_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "manifold_flow_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(330.0)
            .show_inside(parent_ui, |ui| {
                self.render_transport_and_mode(ui);
                ui.separator();
                self.render_surface_parameters(ui);
                ui.separator();
                self.render_solver_and_probe_controls(ui);
                ui.separator();
                self.render_flow_diagnostics(ui);
            });
    }

    fn render_transport_and_mode(&mut self, ui: &mut egui::Ui) {
        ui.heading("Flow Playback & Mode");
        if let Some(action) = PlaybackTransport::in_thread(
            "manifold_flow_transport",
            "Manifold Flow Studio",
            self.is_playing,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => self.is_playing = true,
                PlaybackAction::Pause => self.is_playing = false,
                PlaybackAction::Step => self.advance_substeps(1),
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                PlaybackAction::Reset => {
                    self.is_playing = false;
                    self.reset_simulation();
                }
            }
        }

        let mut selected_mode = self.mode;
        egui::ComboBox::from_label("PDE Law")
            .selected_text(selected_mode.label())
            .show_ui(ui, |ui| {
                for m in [
                    ManifoldFlowMode::MeanCurvature,
                    ManifoldFlowMode::HeatDiffusion,
                ] {
                    ui.selectable_value(&mut selected_mode, m, m.label());
                }
            });
        if selected_mode != self.mode {
            self.mode = selected_mode;
            self.is_playing = false;
            self.reset_simulation();
        }
    }

    fn render_surface_parameters(&mut self, ui: &mut egui::Ui) {
        ui.heading("Manifold Geometry");
        let mut surf = self.surface_type;
        egui::ComboBox::from_label("Surface")
            .selected_text(format!("{surf:?}"))
            .show_ui(ui, |ui| {
                for candidate in [
                    SurfaceType::Sphere,
                    SurfaceType::Torus,
                    SurfaceType::KleinBottle,
                    SurfaceType::CustomHeightfield,
                    SurfaceType::CustomParametric,
                ] {
                    ui.selectable_value(&mut surf, candidate, format!("{candidate:?}"));
                }
            });

        let mut needs_reset = surf != self.surface_type;
        self.surface_type = surf;
        needs_reset |= self.render_surface_type_sliders(ui);

        if self.mode == ManifoldFlowMode::MeanCurvature {
            needs_reset |= ui
                .add(
                    egui::Slider::new(&mut self.initial_perturbation, 0.0..=0.45)
                        .text("Normal Ripple"),
                )
                .changed();
        }

        let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
        ui.horizontal(|ui| {
            ui.label("Grid (nu x nv):");
            needs_reset |= ui
                .add(egui::DragValue::new(&mut self.u_res).range(8..=max_res))
                .changed();
            needs_reset |= ui
                .add(egui::DragValue::new(&mut self.v_res).range(8..=max_res))
                .changed();
        });

        if needs_reset {
            self.reset_simulation();
        }
    }

    fn render_surface_type_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut needs_reset = false;
        match self.surface_type {
            SurfaceType::Sphere => {
                needs_reset |= ui
                    .add(egui::Slider::new(&mut self.sphere_radius, 0.4..=3.5).text("Radius"))
                    .changed();
            }
            SurfaceType::Torus => {
                needs_reset |= ui
                    .add(egui::Slider::new(&mut self.torus_major_radius, 1.2..=3.8).text("Major R"))
                    .changed();
                needs_reset |= ui
                    .add(
                        egui::Slider::new(&mut self.torus_minor_radius, 0.25..=1.5).text("Minor r"),
                    )
                    .changed();
            }
            SurfaceType::KleinBottle => {
                needs_reset |= ui
                    .add(egui::Slider::new(&mut self.klein_radius, 0.6..=3.0).text("Scale"))
                    .changed();
            }
            SurfaceType::CustomHeightfield | SurfaceType::CustomParametric => {
                let fp_before = self.custom_sandbox.state_fingerprint(self.surface_type);
                self.custom_sandbox.ui_for_type(ui, self.surface_type);
                if self.custom_sandbox.state_fingerprint(self.surface_type) != fp_before {
                    needs_reset = true;
                }
            }
        }
        needs_reset
    }

    fn render_solver_and_probe_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("CFL Step & Laplace-Beltrami Probe");
        let mut param_changed = false;
        param_changed |= ui
            .add(
                egui::Slider::new(&mut self.dt, 0.0001..=0.05)
                    .logarithmic(true)
                    .text("Requested dt"),
            )
            .changed();
        if self.mode == ManifoldFlowMode::HeatDiffusion {
            param_changed |= ui
                .add(egui::Slider::new(&mut self.alpha, 0.005..=0.5).text("Diffusivity alpha"))
                .changed();
        }
        param_changed |= ui
            .add(egui::Slider::new(&mut self.substeps_per_frame, 1..=16).text("Substeps / Frame"))
            .changed();
        if ui
            .checkbox(&mut self.show_wireframe, "Show Mesh Wireframe")
            .changed()
        {
            self.state_rev = self.state_rev.wrapping_add(1);
        }

        let mut probe_changed = false;
        ui.horizontal(|ui| {
            ui.label("Probe (u, v):");
            probe_changed |= ui
                .add(bounded_drag_value(
                    &mut self.probe_uv_frac[0],
                    0.05..=0.95,
                    0.02,
                ))
                .changed();
            probe_changed |= ui
                .add(bounded_drag_value(
                    &mut self.probe_uv_frac[1],
                    0.05..=0.95,
                    0.02,
                ))
                .changed();
        });

        if param_changed || probe_changed {
            self.effective_dt = self.dt.min(self.cfl_dt);
            self.refresh_probe_only();
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn render_flow_diagnostics(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Geometric Flow Diagnostics");
        ui.label(format!(
            "Time t = {:.4} ({} steps)",
            self.sim_time, self.step_count
        ));
        let cfl_clamped = self.dt > self.cfl_dt + 1e-9;
        if cfl_clamped {
            ui.colored_label(
                palette.warning,
                format!(
                    "CFL Clamp Active: dt_eff = {:.4e} (limit {:.4e})",
                    self.effective_dt, self.cfl_dt
                ),
            );
        } else {
            ui.colored_label(
                palette.success,
                format!(
                    "CFL Stable: dt_eff = {:.4e} <= {:.4e}",
                    self.effective_dt, self.cfl_dt
                ),
            );
        }

        let metric_name = match self.mode {
            ManifoldFlowMode::MeanCurvature => "Surface Area",
            ManifoldFlowMode::HeatDiffusion => "Total Thermal Energy",
        };
        let pct = 100.0 * self.current_metric / self.initial_metric.max(1e-9);
        ui.label(format!(
            "{metric_name}: {:.4} ({:.2}% of initial)",
            self.current_metric, pct
        ));
        ui.label(format!(
            "Probe Laplace-Beltrami Delta_S = {:.4}",
            self.probe_laplace_beltrami
        ));
        ui.small("3D Solid Exports: Watertight .stl, vertex-colored .ply, and .obj available in Export menu.");

        if let Some(ref note) = self.recovery_note {
            ui.colored_label(palette.warning, note);
        }
    }
}
