#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::black_hole_geodesic_tool::{
    BlackHoleGeodesicWorkbench, GeodesicWorkbenchMode, GEODESIC_TRAIL_CAPACITY,
};
use crate::framework::{PlaybackAction, PlaybackTransport};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, LineStyle, MarkerShape, Plot, PlotPoints, Points};
use math_explorer::physics::high_energy::{critical_impact_parameter, GeodesicKind};

impl BlackHoleGeodesicWorkbench {
    pub(super) fn render_transport_and_metrics(&mut self, ui: &mut egui::Ui) {
        if let Some(action) = PlaybackTransport::in_thread(
            "bh_geodesic_transport",
            "Black Hole Geodesic Workbench",
            self.cfg.is_playing,
        )
        .reset_label("↺ Re-Launch")
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    self.rebuild_bundle(0);
                    self.cfg.is_playing = true;
                }
                PlaybackAction::Pause => self.cfg.is_playing = false,
                PlaybackAction::Step => {
                    self.cfg.is_playing = false;
                    self.advance_burst();
                }
                PlaybackAction::Reset => {
                    self.cfg.is_playing = false;
                    self.rebuild_bundle(800);
                }
            }
            self.revision = self.revision.wrapping_add(1).max(1);
        }

        let b_crit = critical_impact_parameter(1.0).unwrap_or(2.598_076);
        render_copyable_metric(ui, "Photon Sphere r_ph", "1.500 R_s");
        render_copyable_metric(ui, "Critical Impact b_crit", &format!("{b_crit:.4} R_s"));
        render_copyable_metric(ui, "ISCO Radius r_ISCO", "3.000 R_s");
        self.render_trajectory_readouts(ui);
    }

    fn render_trajectory_readouts(&self, ui: &mut egui::Ui) {
        if let Some(primary) = self.bundle.trajectories().first() {
            match self.cfg.mode {
                GeodesicWorkbenchMode::PhotonRingLensing => {
                    let defl_txt = match self.bundle.probe_photon_deflection_rad() {
                        Some(rad) => format!("{rad:.2} rad ({:.1} deg)", rad.to_degrees()),
                        None => "Captured by Horizon (r <= R_s)".to_string(),
                    };
                    render_copyable_metric(ui, "Probe Deflection Δφ", &defl_txt);
                }
                GeodesicWorkbenchMode::IscoPlungeComparison => {
                    let status = if primary.halted_horizon {
                        format!("{:.2} R_s/c (Horizon Plunge)", primary.affine_elapsed)
                    } else {
                        format!("{:.2} R_s/c (Orbiting)", primary.affine_elapsed)
                    };
                    render_copyable_metric(ui, "Proper Time cτ", &status);
                    render_copyable_metric(
                        ui,
                        "Newtonian Periapsis r_min",
                        &format!("{:.3} R_s", self.bundle.newtonian_state().min_radius),
                    );
                    render_copyable_metric(
                        ui,
                        "Analytic Newt. Periapsis",
                        &format!("{:.3} R_s", self.analytical_newtonian_periapsis()),
                    );
                }
                GeodesicWorkbenchMode::MixedEnsemble64 => {
                    render_copyable_metric(
                        ui,
                        "Active Geodesic Rays",
                        &format!("{}", self.bundle.trajectories().len()),
                    );
                }
            }
        }
        render_copyable_metric(
            ui,
            "Max Killing Drift",
            &format!("{:.2e}", self.bundle.max_killing_drift()),
        );
    }

    pub(super) fn render_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let mut ch = false;
        ui.heading("Scenario & Initial Data");
        egui::ComboBox::from_label("Workbench Mode")
            .selected_text(self.cfg.mode.label())
            .show_ui(ui, |ui| {
                for m in [
                    GeodesicWorkbenchMode::PhotonRingLensing,
                    GeodesicWorkbenchMode::IscoPlungeComparison,
                    GeodesicWorkbenchMode::MixedEnsemble64,
                ] {
                    ch |= ui
                        .selectable_value(&mut self.cfg.mode, m, m.label())
                        .changed();
                }
            });

        ch |= self.render_mode_sliders(ui);
        ui.separator();
        ui.heading("Integrator & Overlays");
        ch |= ui
            .add(egui::Slider::new(&mut self.cfg.d_lambda, 0.005..=0.25).text("Step dλ (R_s)"))
            .changed();
        if ui
            .add(
                egui::Slider::new(&mut self.cfg.substeps_per_frame, 1..=32)
                    .text("Substeps / Frame"),
            )
            .changed()
        {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        if ui
            .checkbox(
                &mut self.cfg.show_newtonian_overlay,
                "Overlay Newtonian (-GM/r^2) Orbit",
            )
            .changed()
            || ui
                .checkbox(
                    &mut self.cfg.show_effective_potential,
                    "Inspect Effective Potential V_eff(r)",
                )
                .changed()
        {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
        ch
    }

    fn render_mode_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let c = &mut self.cfg;
        let mut ch = false;
        match c.mode {
            GeodesicWorkbenchMode::PhotonRingLensing => {
                ch |= ui
                    .add(
                        egui::Slider::new(&mut c.probe_b_over_rs, 0.3..=12.0)
                            .text("Impact Parameter b / R_s"),
                    )
                    .changed();
                ch |= ui
                    .add(
                        egui::Slider::new(&mut c.launch_x_over_rs, 5.0..=40.0)
                            .text("Launch Distance |x_0| / R_s"),
                    )
                    .changed();
                ch |= ui
                    .add(
                        egui::Slider::new(&mut c.fan_half_span_over_rs, 0.0..=6.0)
                            .text("Beam Half-Span Δb / R_s"),
                    )
                    .changed();
                ch |= ui
                    .add(egui::Slider::new(&mut c.ray_count, 1..=64).text("Photon Fan Rays"))
                    .changed();
            }
            GeodesicWorkbenchMode::IscoPlungeComparison => {
                ch |= ui
                    .add(
                        egui::Slider::new(&mut c.r0_over_rs, 1.6..=20.0)
                            .text("Launch Radius r_0 / R_s"),
                    )
                    .changed();
                ch |= ui
                    .add(
                        egui::Slider::new(&mut c.l_over_l_isco, 0.2..=2.0)
                            .text("Angular Mom. l / l_ISCO"),
                    )
                    .changed();
                ch |= ui
                    .add(
                        egui::Slider::new(&mut c.inward_nudge, 0.0..=0.4)
                            .text("Inward Radial Nudge |u^r_0|"),
                    )
                    .changed();
            }
            GeodesicWorkbenchMode::MixedEnsemble64 => {
                ch |= ui
                    .add(egui::Slider::new(&mut c.ray_count, 2..=64).text("Total Geodesics"))
                    .changed();
            }
        }
        ch
    }

    pub(super) fn render_main_plot(&self, ui: &mut egui::Ui) {
        if self.cfg.show_effective_potential {
            self.render_potential_plot(ui);
        } else {
            self.render_equatorial_plot(ui);
        }
    }

    fn render_potential_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        Plot::new("schwarzschild_effective_potential_plot")
            .view_aspect(1.75)
            .x_axis_label("Schwarzschild Coordinate Radius r / R_s")
            .y_axis_label("Effective Potential V_eff(r)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(
                        "GR Effective Potential V_eff(r)",
                        PlotPoints::new(self.potential_gr.clone()),
                    )
                    .color(palette.accent)
                    .width(2.5_f32),
                );
                if self.cfg.show_newtonian_overlay {
                    plot_ui.line(
                        Line::new(
                            "Newtonian Centrifugal Barrier",
                            PlotPoints::new(self.potential_newt.clone()),
                        )
                        .color(palette.info)
                        .style(LineStyle::Dashed { length: 7.0 }),
                    );
                }
            });
    }

    fn render_equatorial_plot(&self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let horizon = Self::circle_ring_points(1.0, 64);
        let photon_ring = Self::circle_ring_points(1.5, 64);
        let isco_ring = Self::circle_ring_points(3.0, 64);

        Plot::new("schwarzschild_equatorial_geodesic_plot")
            .data_aspect(1.0)
            .view_aspect(1.4)
            .x_axis_label("Equatorial x / R_s")
            .y_axis_label("Equatorial y / R_s")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("Event Horizon (R_s)", PlotPoints::new(horizon))
                        .color(palette.error)
                        .width(2.4_f32),
                );
                plot_ui.line(
                    Line::new("Photon Sphere (1.5 R_s)", PlotPoints::new(photon_ring))
                        .color(palette.warning)
                        .style(LineStyle::Dashed { length: 5.0 }),
                );
                plot_ui.line(
                    Line::new("ISCO (3.0 R_s)", PlotPoints::new(isco_ring))
                        .color(palette.axis_y)
                        .style(LineStyle::Dashed { length: 8.0 }),
                );
                self.draw_geodesic_trails(plot_ui, &palette);
            });
    }

    fn draw_geodesic_trails(
        &self,
        plot_ui: &mut egui_plot::PlotUi<'_>,
        palette: &crate::app::SemanticPalette,
    ) {
        let mut buf = vec![[0.0_f64; 2]; GEODESIC_TRAIL_CAPACITY];
        let mut heads = Vec::with_capacity(self.bundle.trajectories().len());
        for (idx, tr) in self.bundle.trajectories().iter().enumerate() {
            let n = self.bundle.copy_trajectory_trail(idx, &mut buf);
            if n >= 2 {
                let (name, col, w) = if idx == 0 {
                    ("Probe Relativistic Geodesic", palette.accent, 2.5_f32)
                } else if tr.kind == GeodesicKind::Null {
                    ("", palette.axis_z, 1.3_f32)
                } else {
                    ("", palette.axis_x, 1.4_f32)
                };
                plot_ui.line(
                    Line::new(name, PlotPoints::new(buf[..n].to_vec()))
                        .color(col)
                        .width(w),
                );
                heads.push(buf[n - 1]);
            }
        }
        if !heads.is_empty() {
            plot_ui.points(
                Points::new("Geodesic Wavefront", PlotPoints::new(heads))
                    .color(palette.accent)
                    .radius(4.0_f32)
                    .shape(MarkerShape::Circle),
            );
        }
        if self.cfg.show_newtonian_overlay {
            let nn = self.bundle.copy_newtonian_trail(&mut buf);
            if nn >= 2 {
                plot_ui.line(
                    Line::new(
                        "Newtonian Comparison (-GM/r^2)",
                        PlotPoints::new(buf[..nn].to_vec()),
                    )
                    .color(palette.info)
                    .style(LineStyle::Dashed { length: 6.0 })
                    .width(2.0_f32),
                );
            }
        }
    }
}

// [cite:general_relativity_geodesics]
