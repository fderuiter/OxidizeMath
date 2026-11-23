//! Side-panel controls and analytic readout cards for [`super::RiemannSurfaceStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    CompanionInspectorMode, MeromorphicPreset, PolygonPreset, RiemannBranchFamily,
    RiemannSurfaceStudio, SurfaceHeightChannel,
};
use eframe::egui;

impl RiemannSurfaceStudio {
    pub(super) fn render_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "riemann_surface_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(335.0)
            .show_inside(parent_ui, |ui| {
                self.render_branch_controls(ui);
                ui.separator();
                self.render_monodromy_controls(ui);
                ui.separator();
                self.render_companion_controls(ui);
                ui.separator();
                self.render_diagnostics_cards(ui);
            });
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
        self.state_rev = self.state_rev.wrapping_add(1);
    }

    fn render_branch_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Riemann Surface Geometry");
        let mut changed = false;
        egui::ComboBox::from_label("Surface Family")
            .selected_text(self.branch_family.label())
            .show_ui(ui, |ui| {
                for fam in [RiemannBranchFamily::NthRoot, RiemannBranchFamily::Logarithm] {
                    changed |= ui
                        .selectable_value(&mut self.branch_family, fam, fam.label())
                        .changed();
                }
            });

        match self.branch_family {
            RiemannBranchFamily::NthRoot => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.root_n, 2..=6).text("Sheets n"))
                    .changed();
            }
            RiemannBranchFamily::Logarithm => {
                changed |= ui
                    .add(egui::Slider::new(&mut self.log_sheet_span, 1..=3).text("Sheet Span +-K"))
                    .changed();
            }
        }

        egui::ComboBox::from_label("3D Height Axis")
            .selected_text(self.height_channel.label())
            .show_ui(ui, |ui| {
                for ch in [
                    SurfaceHeightChannel::ImaginaryPart,
                    SurfaceHeightChannel::RealPart,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.height_channel, ch, ch.label())
                        .changed();
                }
            });

        changed |= ui
            .add(egui::Slider::new(&mut self.radial_max, 0.6..=2.5).text("Disk Radius R_max"))
            .changed();
        let max_res = crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION;
        ui.horizontal(|ui| {
            ui.label("Mesh (r x theta):");
            changed |= ui
                .add(egui::DragValue::new(&mut self.radial_steps).range(8..=max_res))
                .changed();
            changed |= ui
                .add(egui::DragValue::new(&mut self.angular_steps).range(16..=max_res))
                .changed();
        });
        ui.horizontal(|ui| {
            let mut layer_ch = ui.checkbox(&mut self.show_wireframe, "Wireframe").changed();
            layer_ch |= ui
                .checkbox(&mut self.show_monodromy_path, "3D Winding Lift")
                .changed();
            if layer_ch {
                self.state_rev = self.state_rev.wrapping_add(1);
            }
        });
        if changed {
            self.mark_dirty();
        }
    }

    fn render_monodromy_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Monodromy & Winding Loop");
        let mut changed = false;
        changed |= ui
            .add(egui::Slider::new(&mut self.start_sheet, -3..=5).text("Initial Sheet k_0"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.winding_turns, -4..=4).text("Winding Turns W"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.loop_radius, 0.2..=2.0).text("Loop Radius R"))
            .changed();
        ui.horizontal(|ui| {
            ui.label("Center z_c:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.loop_center[0])
                        .speed(0.05)
                        .range(-2.5..=2.5)
                        .prefix("Re: "),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.loop_center[1])
                        .speed(0.05)
                        .range(-2.5..=2.5)
                        .prefix("Im: "),
                )
                .changed();
        });
        if changed {
            self.mark_dirty();
        }
    }

    fn render_companion_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Conformal & Cauchy Inspector");
        let mut changed = false;
        egui::ComboBox::from_label("2D View")
            .selected_text(self.companion_mode.label())
            .show_ui(ui, |ui| {
                for m in [
                    CompanionInspectorMode::SchwarzChristoffelPolygon,
                    CompanionInspectorMode::CauchyResidueContour,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.companion_mode, m, m.label())
                        .changed();
                }
            });
        egui::ComboBox::from_label("SC Polygon")
            .selected_text(self.polygon_preset.label())
            .show_ui(ui, |ui| {
                for p in [
                    PolygonPreset::EquilateralTriangle,
                    PolygonPreset::SteppedChannel,
                    PolygonPreset::SymmetricQuad,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.polygon_preset, p, p.label())
                        .changed();
                }
            });
        changed |= ui
            .add(egui::Slider::new(&mut self.sc_scale, 0.3..=2.5).text("SC Scale |A|"))
            .changed();
        changed |= ui
            .checkbox(
                &mut self.enable_schwarz_reflection,
                "Schwarz Reflection Continuation",
            )
            .changed();

        egui::ComboBox::from_label("Cauchy f(z)")
            .selected_text(self.meromorphic_preset.label())
            .show_ui(ui, |ui| {
                for f_kind in [
                    MeromorphicPreset::SimplePoleRational,
                    MeromorphicPreset::ExponentialOverPole,
                    MeromorphicPreset::EntireCubic,
                    MeromorphicPreset::CustomReflectedPole,
                ] {
                    changed |= ui
                        .selectable_value(&mut self.meromorphic_preset, f_kind, f_kind.label())
                        .changed();
                }
            });
        if self.meromorphic_preset == MeromorphicPreset::CustomReflectedPole
            && self.custom_numerator_bar.ui(ui)
        {
            changed = true;
        }
        changed |= self.render_cauchy_probe_inputs(ui);
        if changed {
            self.mark_dirty();
        }
    }

    fn render_cauchy_probe_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Probe z_0:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.probe_z[0])
                        .speed(0.05)
                        .range(-2.5..=2.5),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.probe_z[1])
                        .speed(0.05)
                        .range(-2.5..=2.5),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Pole z_p:");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.pole_z[0])
                        .speed(0.05)
                        .range(-2.5..=2.5),
                )
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.pole_z[1])
                        .speed(0.05)
                        .range(-2.5..=2.5),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            changed |= ui
                .add(egui::Slider::new(&mut self.derivative_order, 1..=3).text("Order n"))
                .changed();
            changed |= ui
                .add(
                    egui::DragValue::new(&mut self.contour_steps)
                        .range(32..=400)
                        .prefix("Steps: "),
                )
                .changed();
        });
        changed
    }

    fn render_diagnostics_cards(&self, ui: &mut egui::Ui) {
        ui.heading("Live Analytic Readouts");
        let d = &self.diagnostics;
        let sheets_txt = d
            .sheet_count
            .map_or_else(|| "Infinite (Z)".to_string(), |n| format!("{n} (Z/{n}Z)"));
        ui.label(format!(
            "Sheets: {sheets_txt} | k_0 = {} -> k_end = {} | |Delta w| = {:.4}",
            self.start_sheet, d.final_sheet, d.monodromy_jump
        ));
        ui.label(format!(
            "SC Conformal Scale |f'(z_0)| = {:.4} | Area det J = {:.4}",
            d.conformal_scale, d.area_mag
        ));
        ui.label(format!(
            "Schwarz Reflect f(conj(z_0)) = {:.4} + {:.4}i",
            d.reflected_value.re, d.reflected_value.im
        ));
        ui.label(format!(
            "Path Integral oint f dz = {:.4} + {:.4}i",
            d.contour_integral.re, d.contour_integral.im
        ));
        ui.label(format!(
            "Cauchy f(z_0) = {:.4}+{:.4}i (Exact {:.4}+{:.4}i)",
            d.cauchy_value.re, d.cauchy_value.im, d.exact_probe_value.re, d.exact_probe_value.im
        ));
        ui.label(format!(
            "Cauchy f^({})(z_0) = {:.4} + {:.4}i",
            self.derivative_order, d.cauchy_nth_deriv.re, d.cauchy_nth_deriv.im
        ));
        ui.label(format!(
            "Residue(z_p): Integral {:.4}+{:.4}i | Limit {:.4}+{:.4}i",
            d.residue_integral.re, d.residue_integral.im, d.residue_limit.re, d.residue_limit.im
        ));
    }
}
