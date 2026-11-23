#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::{BrushMode, ColormapMode, PropagationMode, ScenarioPreset};
use super::ExcitableSheetTool;
use crate::framework::AccessibleHoverText;
use eframe::egui;

impl ExcitableSheetTool {
    /// Renders accessible left-panel controls for scenarios, brushes, kinetics, and 3-lead placement.
    pub fn render_side_panel(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.render_transport_and_scenarios(ui);
                ui.separator();
                self.render_mode_and_brush_controls(ui);
                ui.separator();
                self.render_kinetics_controls(ui);
                ui.separator();
                self.render_lead_and_colormap_controls(ui);
            });
    }

    fn render_transport_and_scenarios(&mut self, ui: &mut egui::Ui) {
        self.render_transport_row(ui);
        ui.add_space(4.0);
        self.render_scenario_buttons(ui);
    }

    fn render_transport_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let run_lbl = if self.cfg.running {
                "⏸ Pause"
            } else {
                "▶ Run"
            };
            if ui
                .button(run_lbl)
                .accessible_hover_text("Toggle real-time tissue electrophysiology stepping")
                .clicked()
            {
                self.cfg.running = !self.cfg.running;
                self.bump_revision();
            }
            if ui
                .button("⏭ Step")
                .accessible_hover_text("Advance excitable tissue and 3-lead readout by one step")
                .clicked()
            {
                self.step_once_guarded();
                self.bump_revision();
            }
            if ui
                .button("⚡ Defibrillate")
                .accessible_hover_text(
                    "Deliver synchronous whole-sheet defibrillation shock (Key D)",
                )
                .clicked()
            {
                self.trigger_defibrillation();
            }
        });
    }

    fn render_scenario_buttons(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Arrhythmia & Ablation Scenarios").strong());
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Dual-Edge Collision")
                .accessible_hover_text(
                    "Launch opposing planar wavefronts that annihilate on collision",
                )
                .clicked()
            {
                self.apply_scenario(ScenarioPreset::DualEdgeCollision);
                self.cfg.running = true;
            }
            if ui
                .button("S1-S2 Spiral")
                .accessible_hover_text(
                    "Launch planar S1 wave with timed S2 pulse in vulnerable tail",
                )
                .clicked()
            {
                self.apply_scenario(ScenarioPreset::S1S2SpiralInduction);
                self.cfg.running = true;
            }
            if ui
                .button("Pin Scar Rotor")
                .accessible_hover_text(
                    "Initialize central zero-flux ischemic scar with anchored rotor",
                )
                .clicked()
            {
                self.apply_scenario(ScenarioPreset::ScarPinnedRotor);
                self.cfg.running = true;
            }
            if ui
                .button("Ablate to Edge")
                .accessible_hover_text(
                    "Extend insulating catheter ablation line from scar to boundary",
                )
                .clicked()
            {
                self.sheet.ablate_scar_to_boundary();
                self.cfg.ablated_to_edge = true;
                self.record_lead_sample();
                self.bump_revision();
            }
        });
    }

    fn render_mode_and_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Tissue Propagation Scale").strong());
        ui.horizontal_wrapped(|ui| {
            let mut mode = self.cfg.propagation_mode;
            ui.selectable_value(
                &mut mode,
                PropagationMode::FitzHughNagumo2D,
                "2D FHN Sheet (96×96)",
            )
            .accessible_hover_text("2D FitzHugh-Nagumo reaction-diffusion sheet (Key M)");
            ui.selectable_value(
                &mut mode,
                PropagationMode::HodgkinHuxleyCable1D,
                "1D HH Axon (96)",
            )
            .accessible_hover_text("1D diffusive Hodgkin-Huxley squid-axon cable (Key M)");
            if mode != self.cfg.propagation_mode {
                self.cfg.propagation_mode = mode;
                self.clear_trace_history();
                self.record_lead_sample();
                self.bump_revision();
            }
        });

        ui.add_space(4.0);
        ui.label(egui::RichText::new("Virtual Electrode & Scar Brush").strong());
        ui.horizontal_wrapped(|ui| {
            let mut bm = self.cfg.brush_mode;
            ui.selectable_value(&mut bm, BrushMode::InjectCurrent, "⚡ Stimulate");
            ui.selectable_value(&mut bm, BrushMode::ClampVoltage, "📌 V-Clamp");
            ui.selectable_value(&mut bm, BrushMode::PaintScar, "🧱 Scar/Ablate");
            ui.selectable_value(&mut bm, BrushMode::EraseScar, "🧹 Erase Scar");
            ui.selectable_value(&mut bm, BrushMode::MoveLead, "📡 Move Lead");
            if bm != self.cfg.brush_mode {
                self.cfg.brush_mode = bm;
                self.bump_revision();
            }
        });

        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.brush_radius, 0.5..=12.0).text("Brush Radius"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.brush_current, -2.0..=4.5).text("I_stim"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.clamp_voltage, -70.0..=30.0).text("V_clamp"))
            .changed();
        if ui
            .small_button("Clear All Scars")
            .accessible_hover_text("Remove all zero-flux scar obstacles")
            .clicked()
        {
            self.sheet.clear_scars();
            self.cable.clear_scars();
            self.cfg.ablated_to_edge = false;
            changed = true;
        }
        if changed {
            self.bump_revision();
        }
    }

    fn render_kinetics_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Membrane Kinetics & Diffusion").strong());
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.a, 0.3..=1.1).text("Threshold a"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.b, 0.4..=1.1).text("Recovery b"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.epsilon, 0.01..=0.15).text("Timescale ε"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.d_u, 0.1..=1.0).text("Diffusivity D_u"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.dt, 0.01..=0.22).text("Step dt"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.cfg.steps_per_frame, 1..=12).text("Substeps/Frame"))
            .changed();
        if changed {
            self.sync_engine_parameters();
            self.bump_revision();
        }
    }

    fn render_lead_and_colormap_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("3-Lead Dipole & Colormap").strong());
        ui.horizontal_wrapped(|ui| {
            let mut bipolar = self.cfg.readout_bipolar;
            ui.selectable_value(&mut bipolar, true, "3-Lead Pseudo-ECG");
            ui.selectable_value(&mut bipolar, false, "Unipolar LFP");
            if bipolar != self.cfg.readout_bipolar {
                self.cfg.readout_bipolar = bipolar;
                self.clear_trace_history();
                self.record_lead_sample();
                self.bump_revision();
            }
        });
        ui.horizontal_wrapped(|ui| {
            let mut cm = self.cfg.colormap;
            ui.selectable_value(&mut cm, ColormapMode::Perceptual, "Perceptual");
            ui.selectable_value(&mut cm, ColormapMode::HighContrast, "High-Contrast");
            if cm != self.cfg.colormap {
                self.cfg.colormap = cm;
                self.bump_revision();
            }
        });

        ui.horizontal(|ui| {
            ui.label("Active Lead:");
            for (idx, tag) in [(0, "L1 (RA)"), (1, "L2 (LA)"), (2, "L3 (LL)")] {
                if ui
                    .selectable_label(self.cfg.active_lead == idx, tag)
                    .clicked()
                {
                    self.cfg.active_lead = idx;
                    self.bump_revision();
                }
            }
        });
        let idx = self.cfg.active_lead.min(2);
        let mut lx = self.cfg.leads[idx].x;
        let mut ly = self.cfg.leads[idx].y;
        let mut lz = self.cfg.leads[idx].z;
        let mut lead_changed = false;
        lead_changed |= ui
            .add(egui::Slider::new(&mut lx, 0.0..=96.0).text("Lead X"))
            .changed();
        lead_changed |= ui
            .add(egui::Slider::new(&mut ly, 0.0..=96.0).text("Lead Y"))
            .changed();
        lead_changed |= ui
            .add(egui::Slider::new(&mut lz, 1.0..=32.0).text("Stand-off Z"))
            .changed();
        if lead_changed {
            self.cfg.leads[idx].z = lz;
            self.move_lead_to(idx, lx, ly);
            self.bump_revision();
        }
    }
}
