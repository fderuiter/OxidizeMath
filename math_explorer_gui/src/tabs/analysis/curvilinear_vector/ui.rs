//! Side-panel controls and Gauss/Stokes/Helmholtz readout cards for [`super::CurvilinearVectorStudio`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CoordinateFrameKind, CurvilinearVectorStudio, MAX_HELMHOLTZ_GRID, MIN_HELMHOLTZ_GRID};
use crate::app::SemanticPalette;
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use eframe::egui;

impl CurvilinearVectorStudio {
    pub(super) fn render_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "curvilinear_vector_controls");
        crate::framework::PaneSidePanel::left(panel_id)
            .default_width(330.0)
            .show_inside(parent_ui, |ui| {
                self.render_frame_and_field_controls(ui);
                ui.separator();
                self.render_domain_and_stokes_controls(ui);
                ui.separator();
                self.render_helmholtz_and_layer_controls(ui);
                ui.separator();
                self.render_diagnostics_cards(ui);
            });
    }

    fn render_frame_and_field_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Coordinate Frame & Fields");
        let mut selected = self.frame;
        egui::ComboBox::from_label("Frame")
            .selected_text(selected.label())
            .show_ui(ui, |ui| {
                for candidate in [
                    CoordinateFrameKind::Cartesian,
                    CoordinateFrameKind::Cylindrical,
                    CoordinateFrameKind::Spherical,
                ] {
                    ui.selectable_value(&mut selected, candidate, candidate.label());
                }
            });
        if selected != self.frame {
            self.switch_frame(selected);
        }

        let vars = self.frame.axis_vars();
        let basis = self.frame.basis_labels();
        ui.small(format!(
            "Variables: ({}, {}, {}) | Basis: ({}, {}, {})",
            vars[0], vars[1], vars[2], basis[0], basis[1], basis[2]
        ));

        let mut expr_changed = false;
        expr_changed |= self.f1_bar.ui(ui);
        expr_changed |= self.f2_bar.ui(ui);
        expr_changed |= self.f3_bar.ui(ui);
        expr_changed |= self.scalar_bar.ui(ui);
        if expr_changed {
            self.recompile_expressions();
            self.mark_dirty();
        }
        if self.param_sliders.ui(ui) {
            self.mark_dirty();
        }
        if let Some(ref err) = self.compile_error {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            ui.colored_label(palette.error, format!("Formula error: {err}"));
        }
    }

    fn render_domain_and_stokes_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Gauss Domain & Stokes Loop");
        let vars = self.frame.axis_vars();
        let mut changed = self.render_domain_axis_bounds(ui, &vars);
        changed |= self.render_stokes_and_probe_inputs(ui, &vars);
        if changed {
            self.mark_dirty();
        }
    }

    fn render_domain_axis_bounds(&mut self, ui: &mut egui::Ui, vars: &[&str; 3]) -> bool {
        let mut changed = false;
        let min_radial = if self.frame == CoordinateFrameKind::Cartesian {
            -6.0
        } else {
            0.15
        };
        for (axis, &var_label) in vars.iter().enumerate() {
            let lo = if axis == 0 { min_radial } else { -6.0 };
            let hi = if self.frame == CoordinateFrameKind::Spherical && axis == 1 {
                std::f64::consts::PI - 0.1
            } else {
                6.0
            };
            let lo_bound = if self.frame == CoordinateFrameKind::Spherical && axis == 1 {
                0.1
            } else {
                lo
            };
            let (mut lo_val, mut hi_val) = (self.domain_min[axis], self.domain_max[axis]);
            let (mut min_ch, mut max_ch) = (false, false);
            ui.horizontal(|ui| {
                ui.label(format!("{var_label} bounds:"));
                min_ch = ui
                    .add(bounded_drag_value(&mut lo_val, lo_bound..=(hi - 0.1), 0.05))
                    .changed();
                max_ch = ui
                    .add(bounded_drag_value(&mut hi_val, (lo_bound + 0.1)..=hi, 0.05))
                    .changed();
            });
            enforce_ordered_drag_range(&mut lo_val, &mut hi_val, lo_bound, hi, 0.1, min_ch, max_ch);
            changed |= min_ch || max_ch;
            self.domain_min[axis] = lo_val;
            self.domain_max[axis] = hi_val;
        }
        changed
    }

    fn render_stokes_and_probe_inputs(&mut self, ui: &mut egui::Ui, vars: &[&str; 3]) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("Stokes normal:");
            for (ax, &var_label) in vars.iter().enumerate() {
                if ui
                    .selectable_label(self.stokes_axis == ax, var_label)
                    .clicked()
                {
                    self.stokes_axis = ax;
                    changed = true;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Fixed slice:");
            changed |= ui
                .add(bounded_drag_value(&mut self.stokes_fixed, -4.0..=4.0, 0.05))
                .changed();
            ui.label("Steps:");
            changed |= ui
                .add(egui::DragValue::new(&mut self.integration_steps).range(6..=48))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Probe:");
            for (ax, &var_label) in vars.iter().enumerate() {
                changed |= ui
                    .add(
                        bounded_drag_value(
                            &mut self.probe_point[ax],
                            self.domain_min[ax]..=self.domain_max[ax],
                            0.05,
                        )
                        .prefix(format!("{var_label}:")),
                    )
                    .changed();
            }
        });
        changed
    }

    fn render_helmholtz_and_layer_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Helmholtz Split & 3D Layers");
        let mut changed = false;
        changed |= ui
            .add(
                egui::Slider::new(&mut self.grid_res, MIN_HELMHOLTZ_GRID..=MAX_HELMHOLTZ_GRID)
                    .text("Helmholtz Grid N"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.poisson_iters, 20..=240).text("Poisson Sweeps"))
            .changed();
        if changed {
            self.mark_dirty();
        }

        let mut layer_changed = false;
        layer_changed |= ui
            .add(egui::Slider::new(&mut self.quiver_scale, 0.04..=0.8).text("Arrow Scale"))
            .changed();
        ui.horizontal_wrapped(|ui| {
            layer_changed |= ui.checkbox(&mut self.show_total_field, "Total F").changed();
            layer_changed |= ui
                .checkbox(&mut self.show_irrotational, "Irrotational (-grad Phi)")
                .changed();
            layer_changed |= ui
                .checkbox(&mut self.show_solenoidal, "Solenoidal F_sol")
                .changed();
            layer_changed |= ui
                .checkbox(&mut self.show_domain_wireframe, "Gauss Domain")
                .changed();
            layer_changed |= ui
                .checkbox(&mut self.show_stokes_loop, "Stokes Loop")
                .changed();
        });
        if layer_changed {
            self.state_rev = self.state_rev.wrapping_add(1);
        }
    }

    fn render_diagnostics_cards(&self, ui: &mut egui::Ui) {
        let d = &self.diagnostics;
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.heading("Theorem & Operator Diagnostics");
        ui.label(format!(
            "Gauss Vol = {:.4} | Flux = {:.4}",
            d.gauss_volume, d.gauss_flux
        ));
        ui.colored_label(
            palette.success,
            format!("Gauss |Vol - Flux| = {:.3e}", d.gauss_residual),
        );
        ui.label(format!(
            "Stokes Curl = {:.4} | Loop = {:.4}",
            d.stokes_surface_curl, d.stokes_loop_circulation
        ));
        ui.colored_label(
            palette.success,
            format!("Stokes |Surf - Loop| = {:.3e}", d.stokes_residual),
        );
        ui.small(format!(
            "Helmholtz ({} sweeps): max |div F_sol| = {:.3e}, max |curl F_irr| = {:.3e}, recon err = {:.2e}",
            d.helmholtz.iterations,
            d.helmholtz.max_solenoidal_divergence,
            d.helmholtz.max_irrotational_curl,
            d.helmholtz.max_reconstruction_error
        ));
        ui.small(format!(
            "Probe: div F = {:.4}, lap Phi = {:.4}, grad Phi = [{:.3}, {:.3}, {:.3}], curl F = [{:.3}, {:.3}, {:.3}], h = [{:.2}, {:.2}, {:.2}]",
            d.probe_divergence,
            d.probe_laplacian,
            d.probe_gradient[0],
            d.probe_gradient[1],
            d.probe_gradient[2],
            d.probe_curl[0],
            d.probe_curl[1],
            d.probe_curl[2],
            d.probe_scale_factors[0],
            d.probe_scale_factors[1],
            d.probe_scale_factors[2]
        ));
    }
}
