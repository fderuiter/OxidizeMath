#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, 3D Flamm/wormhole `Plot3DViewport`, $V_{\text{eff}}(r)$ with $\frac{1}{2}e^2$,
//! twin worldlines, Kerr/ISCO overlays, and 2D lensing atlas UI.

use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::camera::Plot3DViewport;
use crate::framework::{scoped_panel_id, PaneSidePanel, PlaybackAction, PlaybackTransport};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, PlotPoints, Points, VLine};

use super::{GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset, TrajectorySample};

impl GeodesicPlaygroundTool {
    pub(super) fn render_tool(&mut self, parent_ui: &mut egui::Ui) {
        if self.samples.is_empty() {
            self.rebuild_trajectory(560);
        }
        let ctx = parent_ui.ctx().clone();
        if self.state.is_playing {
            let now = PlatformInstant::now();
            if now.duration_since(self.last_tick).as_millis() >= 16 {
                self.last_tick = now;
                self.advance_steps(self.state.steps_per_frame);
                self.revision = self.revision.wrapping_add(1);
            }
            ctx.request_repaint();
        }
        self.draw_side_panel(parent_ui, &ctx);
        self.draw_bottom_diagnostics(parent_ui);
        self.draw_central_workspace(parent_ui);
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = scoped_panel_id(parent_ui, "relativity_playground_side");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Geodesic Playground");
            ui.separator();
            let mut recompute = self.draw_mode_selectors(ui);
            recompute |= self.draw_sandbox_controls(ui);
            recompute |= self.draw_parameter_sliders(ui);
            ui.separator();
            if self.atlas.show_toggle_button(ui) {
                self.state.atlas_visible = self.atlas.is_visible();
                self.revision = self.revision.wrapping_add(1);
            }
            ui.separator();
            self.draw_transport_and_scrubber(ui, ctx, recompute);
            ui.separator();
            egui::CollapsingHeader::new("3D Camera")
                .default_open(false)
                .show(ui, |ui| self.state.camera.ui(ui));
        });
    }

    fn draw_mode_selectors(&mut self, ui: &mut egui::Ui) -> bool {
        let mut recompute = false;
        egui::ComboBox::from_label("Spacetime")
            .selected_text(self.state.spacetime.label())
            .show_ui(ui, |ui| {
                for candidate in SpacetimePreset::ALL {
                    recompute |= ui
                        .selectable_value(&mut self.state.spacetime, candidate, candidate.label())
                        .changed();
                }
            });
        egui::ComboBox::from_label("Worldline")
            .selected_text(self.state.mode.label())
            .show_ui(ui, |ui| {
                for candidate in [GeodesicMode::NullPhoton, GeodesicMode::TimelikeParticle] {
                    recompute |= ui
                        .selectable_value(&mut self.state.mode, candidate, candidate.label())
                        .changed();
                }
            });
        recompute
    }

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let is_wh = self.state.spacetime.is_wormhole();
        let scale_lbl = if is_wh {
            "Throat Radius b0"
        } else {
            "Schwarzschild Radius Rs"
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.state.scale_radius, 0.5..=5.0).text(scale_lbl))
            .changed();
        if self.state.spacetime == SpacetimePreset::KerrEquatorial {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.state.spin_chi, -0.95..=0.95).text("Kerr Spin chi"),
                )
                .changed();
        }
        let min_r = self.min_launch_radius();
        let (r_range, r_lbl) = if is_wh {
            (-12.0..=12.0, "Launch Distance l0")
        } else {
            (min_r..=18.0, "Launch Radius r0")
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.state.initial_radius, r_range).text(r_lbl))
            .changed();
        let b_lbl = match self.state.mode {
            GeodesicMode::NullPhoton => "Impact Parameter b",
            GeodesicMode::TimelikeParticle => "Angular Momentum l",
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.state.impact_or_momentum, -14.0..=14.0).text(b_lbl))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.state.twin_delta_b, -1.5..=1.5)
                    .step_by(0.005)
                    .text("Twin Offset delta_b"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.radial_bias, -0.98..=0.98).text("Radial Launch"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.step_size, 0.005..=0.12).text("Step dlambda"))
            .changed();
        changed
    }

    fn draw_transport_and_scrubber(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        recompute: bool,
    ) {
        if recompute {
            self.clamp_state_in_place();
            self.rebuild_trajectory(560);
            self.state.scrub_index = self.samples.len().saturating_sub(1);
            self.revision = self.revision.wrapping_add(1);
        }
        if let Some(action) = PlaybackTransport::in_thread(
            "geodesic_playground_transport",
            "Geodesic Playground",
            self.state.is_playing,
        )
        .show(ui)
        {
            match action {
                PlaybackAction::Play => {
                    if self.samples.len() >= super::MAX_TRAJECTORY_SAMPLES
                        || self.halted_near_singularity
                    {
                        self.rebuild_trajectory(80);
                    }
                    self.state.is_playing = true;
                    self.last_tick = PlatformInstant::now();
                }
                PlaybackAction::Pause => self.state.is_playing = false,
                PlaybackAction::Step => {
                    self.state.is_playing = false;
                    self.advance_steps(1);
                }
                PlaybackAction::Reset => {
                    self.state.is_playing = false;
                    self.rebuild_trajectory(560);
                    self.state.scrub_index = self.samples.len().saturating_sub(1);
                }
                PlaybackAction::StepBack => {
                    self.state.is_playing = false;
                    self.state.scrub_index = self.state.scrub_index.saturating_sub(1);
                }
                PlaybackAction::ScrubTo(step) => {
                    self.state.is_playing = false;
                    self.state.scrub_index = step.min(self.samples.len().saturating_sub(1));
                }
            }
            self.revision = self.revision.wrapping_add(1);
            ctx.request_repaint();
        }
        let max_idx = self.samples.len().saturating_sub(1);
        if max_idx > 0
            && ui
                .add(egui::Slider::new(&mut self.state.scrub_index, 0..=max_idx).text("Scrub Step"))
                .changed()
        {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    fn draw_bottom_diagnostics(&mut self, parent_ui: &mut egui::Ui) {
        let bottom_id = scoped_panel_id(parent_ui, "relativity_playground_bottom");
        egui::TopBottomPanel::bottom(bottom_id)
            .exact_height(205.0)
            .show_inside(parent_ui, |ui| {
                let palette = SemanticPalette::from_visuals(ui.visuals());
                ui.columns(2, |cols| {
                    self.draw_effective_potential_panel(&mut cols[0], &palette);
                    self.draw_twin_clocks_panel(&mut cols[1], &palette);
                });
            });
    }

    fn draw_effective_potential_panel(&mut self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        let e_half_sq = 0.5 * self.launch_energy * self.launch_energy;
        ui.strong(format!(
            "Effective Potential V_eff(r)  |  0.5 e^2 = {:.4} (e = {:.3})",
            e_half_sq, self.launch_energy
        ));
        let curve = self.effective_potential_curve(72);
        let x_bounds = match (curve.first(), curve.last()) {
            (Some(f), Some(l)) => [f[0], l[0]],
            _ => [1.0, 18.0],
        };
        let active = self.active_sample();
        let twin = self.active_twin_sample();
        let (_, r_ergo, _, isco, _) = self.relativistic_radii_and_omega(
            active.map_or(self.state.initial_radius, |s| s.radial_coord),
        );
        let is_wh = self.state.spacetime.is_wormhole();
        let plot_id = scoped_panel_id(ui, "relativity_veff_plot");
        Plot::new(plot_id)
            .height(162.0)
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("V_eff(r)", PlotPoints::new(curve)).color(palette.series_color(2)),
                );
                plot_ui.line(
                    Line::new(
                        "0.5 * e^2",
                        PlotPoints::new(vec![[x_bounds[0], e_half_sq], [x_bounds[1], e_half_sq]]),
                    )
                    .color(palette.warning)
                    .style(LineStyle::Dashed { length: 6.0 }),
                );
                if !is_wh && isco > 0.0 {
                    plot_ui.vline(
                        VLine::new("ISCO", isco)
                            .color(palette.info)
                            .style(LineStyle::Dotted { spacing: 5.0 }),
                    );
                }
                if self.state.spacetime == SpacetimePreset::KerrEquatorial && r_ergo > 0.0 {
                    plot_ui.vline(VLine::new("Ergosphere r_ergo", r_ergo).color(palette.error));
                }
                if let Some(sample) = active {
                    plot_ui.points(
                        Points::new("Current Position", vec![[sample.radial_coord, e_half_sq]])
                            .radius(5.0_f32)
                            .color(palette.accent),
                    );
                }
                if let Some(t_sample) = twin {
                    let t_e = 0.5 * self.twin_launch_energy * self.twin_launch_energy;
                    plot_ui.points(
                        Points::new("Twin Position", vec![[t_sample.radial_coord, t_e]])
                            .radius(4.5_f32)
                            .color(palette.series_color(1)),
                    );
                }
                self.callouts.render_on_plot(plot_ui);
            });
    }

    fn draw_twin_clock_readout(
        &self,
        ui: &mut egui::Ui,
        palette: &SemanticPalette,
        primary: TrajectorySample,
        twin: TrajectorySample,
    ) {
        let status = if self.twin_halted_near_singularity {
            "captured"
        } else {
            "propagating"
        };
        ui.colored_label(
            palette.series_color(1),
            format!(
                "Twin (b{:+.3}): ct {:.3} (dct {:+.3}) | tau {:.3} (dtau {:+.3}) | r {:.3} [{}]",
                self.state.twin_delta_b,
                twin.coord_ct,
                twin.coord_ct - primary.coord_ct,
                twin.proper_tau,
                twin.proper_tau - primary.proper_tau,
                twin.radial_coord,
                status
            ),
        );
    }

    fn draw_twin_clocks_panel(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong("Twin Clocks, Horizons & Invariants");
        let Some(sample) = self.active_sample() else {
            ui.label("No active trajectory sample.");
            return;
        };
        ui.label(format!(
            "Primary ct: {:.3}  |  tau/lambda: {:.3}  |  r: {:.3}",
            sample.coord_ct, sample.proper_tau, sample.radial_coord
        ));
        if let Some(twin) = self.active_twin_sample() {
            self.draw_twin_clock_readout(ui, palette, sample, twin);
        }
        let (r_plus, r_ergo, omega, isco, _) =
            self.relativistic_radii_and_omega(sample.radial_coord);
        if !self.state.spacetime.is_wormhole() {
            ui.colored_label(
                palette.info,
                format!(
                    "r+ = {:.4}  |  r_ergo = {:.3}  |  ISCO = {:.3}  |  omega(r) = {:+.4}",
                    r_plus, r_ergo, isco, omega
                ),
            );
        } else {
            ui.colored_label(
                palette.info,
                format!(
                    "Clock Ratio dtau/dt: {:.4}  |  Lorentz gamma: {:.4}",
                    sample.clock_rate, sample.lorentz_gamma
                ),
            );
        }
        ui.colored_label(
            palette.success,
            format!(
                "Metric Norm g_munu u^mu u^nu: {:.6}  |  Lapse: {:.4}",
                sample.metric_norm, sample.grav_dilation
            ),
        );
        if self.halted_near_singularity || self.twin_halted_near_singularity {
            ui.colored_label(
                palette.warning,
                "Horizon/singularity guard triggered cleanly with zero NaN propagation.",
            );
        }
    }

    pub(super) fn active_sample(&self) -> Option<TrajectorySample> {
        let idx = self
            .state
            .scrub_index
            .min(self.samples.len().saturating_sub(1));
        self.samples.get(idx).copied()
    }

    pub(super) fn active_twin_sample(&self) -> Option<TrajectorySample> {
        if self.twin_samples.is_empty() {
            return None;
        }
        let idx = self
            .state
            .scrub_index
            .min(self.twin_samples.len().saturating_sub(1));
        self.twin_samples.get(idx).copied()
    }

    fn draw_central_workspace(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.atlas.is_visible() {
                ui.columns(2, |cols| {
                    self.render_3d_viewport_inside(&mut cols[0]);
                    self.render_relativity_atlas_panel(&mut cols[1]);
                });
            } else {
                self.render_3d_viewport_inside(ui);
            }
        });
    }

    fn render_3d_viewport_inside(&mut self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        let wireframe = self.build_embedding_wireframe();
        let ref_rings = self.build_3d_reference_rings(&palette);
        let (traj, head) = self.projected_worldline(&self.samples, self.active_sample());
        let twin_proj = (!self.twin_samples.is_empty())
            .then(|| self.projected_worldline(&self.twin_samples, self.active_twin_sample()));

        let mut camera = self.state.camera;
        let plot_id = scoped_panel_id(ui, "relativity_flamm_3d");
        let response = Plot3DViewport::new(plot_id, 16.0)
            .show(ui, &mut camera, |plot_ui| {
                for (idx, ring) in wireframe.into_iter().enumerate() {
                    plot_ui.line(
                        Line::new(format!("Embedding Mesh {idx}"), PlotPoints::new(ring))
                            .color(palette.graph_edge),
                    );
                }
                for line in ref_rings {
                    plot_ui.line(line);
                }
                plot_ui.line(
                    Line::new("Geodesic Worldline", PlotPoints::new(traj))
                        .color(palette.series_color(0))
                        .width(2.4_f32),
                );
                if let Some((t_line, t_head)) = twin_proj {
                    plot_ui.line(
                        Line::new("Twin Worldline", PlotPoints::new(t_line))
                            .color(palette.series_color(1))
                            .width(2.0_f32)
                            .style(LineStyle::Dashed { length: 7.0 }),
                    );
                    if let Some(tp) = t_head {
                        plot_ui.points(
                            Points::new("Twin Ray", vec![tp])
                                .radius(4.5_f32)
                                .color(palette.series_color(1)),
                        );
                    }
                }
                if let Some(pt) = head {
                    plot_ui.points(
                        Points::new("Particle / Photon", vec![pt])
                            .radius(5.0_f32)
                            .color(palette.accent),
                    );
                }
            })
            .response;
        if camera != self.state.camera {
            self.state.camera = camera;
            self.revision = self.revision.wrapping_add(1);
        }
        response.accessible_theory_hover(self);
    }
}
