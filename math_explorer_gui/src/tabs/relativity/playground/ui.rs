#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, $V_{\text{eff}}(r)$, exact Kerr telemetry, Schwarzschild ISCO/photon-sphere
//! markers, and Doppler sonification UI for [`GeodesicPlaygroundTool`].

use crate::app::SemanticPalette;
use crate::audio::render_sonification_panel;
use crate::framework::{scoped_panel_id, PaneSidePanel, PlaybackAction, PlaybackTransport};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, PlotPoints, Points, VLine};
use math_explorer::physics::high_energy::{KerrEquatorialMetric, SchwarzschildBlackHole};

use super::{
    GeodesicInteractionMode, GeodesicMode, GeodesicPlaygroundTool, LensingBackgroundKind,
    SpacetimePreset, TrajectorySample, ViewportViewMode,
};

impl GeodesicPlaygroundTool {
    pub(super) fn render_tool(&mut self, parent_ui: &mut egui::Ui) {
        if self.samples.is_empty() {
            self.rebuild_trajectory(260);
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
        match self.state.view_mode {
            ViewportViewMode::Embedding3D => self.draw_central_3d_viewport(parent_ui),
            ViewportViewMode::LensingCamera2D => {
                egui::CentralPanel::default().show_inside(parent_ui, |ui| {
                    self.draw_lensing_2d_viewport(ui);
                });
            }
        }
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = scoped_panel_id(parent_ui, "relativity_playground_side");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Geodesic Playground");
                self.draw_mode_and_view_selectors(ui);
                ui.separator();
                let recompute = self.draw_metric_and_sliders(ui);
                ui.separator();
                self.draw_transport_and_scrubber(ui, ctx, recompute);
                ui.separator();
                let audio_frame = self.build_audio_frame();
                let is_paused = !self.state.is_playing || self.halted_near_singularity;
                render_sonification_panel(ui, "geodesic_playground", &audio_frame, is_paused);
                ui.separator();
                egui::CollapsingHeader::new("3D Camera")
                    .default_open(false)
                    .show(ui, |ui| self.state.camera.ui(ui));
            });
        });
    }

    fn draw_mode_and_view_selectors(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label("Gesture:");
            for mode in [
                GeodesicInteractionMode::Orbit,
                GeodesicInteractionMode::Slingshot,
                GeodesicInteractionMode::PhotonFan,
            ] {
                if ui
                    .selectable_value(&mut self.state.interaction_mode, mode, mode.label())
                    .changed()
                {
                    self.rebuild_trajectory(260);
                    self.state.scrub_index = self.samples.len().saturating_sub(1);
                    self.revision = self.revision.wrapping_add(1);
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("View:");
            for view in [
                ViewportViewMode::Embedding3D,
                ViewportViewMode::LensingCamera2D,
            ] {
                if ui
                    .selectable_value(&mut self.state.view_mode, view, view.label())
                    .changed()
                {
                    self.lensing_dirty = true;
                    self.refresh_lensing_pixels_in_place();
                    self.revision = self.revision.wrapping_add(1);
                }
            }
        });
        if self.state.view_mode == ViewportViewMode::LensingCamera2D {
            egui::ComboBox::from_label("Lensing Source")
                .selected_text(self.state.lensing_background.label())
                .show_ui(ui, |ui| {
                    for bg in [
                        LensingBackgroundKind::GalacticDisk,
                        LensingBackgroundKind::LeniaPortal,
                        LensingBackgroundKind::Starfield,
                    ] {
                        if ui
                            .selectable_value(&mut self.state.lensing_background, bg, bg.label())
                            .changed()
                        {
                            self.refresh_lensing_pixels_in_place();
                            self.revision = self.revision.wrapping_add(1);
                        }
                    }
                });
        }
    }

    fn draw_metric_and_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut recompute = false;
        egui::ComboBox::from_label("Spacetime")
            .selected_text(self.state.spacetime.label())
            .show_ui(ui, |ui| {
                for candidate in [
                    SpacetimePreset::Schwarzschild,
                    SpacetimePreset::KerrEquatorial,
                    SpacetimePreset::MorrisThorneWormhole,
                ] {
                    if ui
                        .selectable_value(&mut self.state.spacetime, candidate, candidate.label())
                        .changed()
                    {
                        self.lensing_dirty = true;
                        recompute = true;
                    }
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

        recompute |= self.draw_parameter_sliders(ui);
        recompute
    }

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let scale_label = match self.state.spacetime {
            SpacetimePreset::MorrisThorneWormhole => "Throat Radius b0",
            _ => "Schwarzschild Radius Rs",
        };
        if ui
            .add(egui::Slider::new(&mut self.state.scale_radius, 0.5..=5.0).text(scale_label))
            .changed()
        {
            self.lensing_dirty = true;
            changed = true;
        }
        if self.state.spacetime == SpacetimePreset::KerrEquatorial
            && ui
                .add(
                    egui::Slider::new(&mut self.state.spin_chi, -0.92..=0.92).text("Kerr Spin chi"),
                )
                .changed()
        {
            self.lensing_dirty = true;
            changed = true;
        }
        let r_range = match self.state.spacetime {
            SpacetimePreset::MorrisThorneWormhole => -12.0..=12.0,
            _ => (1.2 * self.state.scale_radius)..=18.0,
        };
        let r_label = match self.state.spacetime {
            SpacetimePreset::MorrisThorneWormhole => "Launch Distance l0",
            _ => "Launch Radius r0",
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.state.initial_radius, r_range).text(r_label))
            .changed();
        let b_label = match self.state.mode {
            GeodesicMode::NullPhoton => "Impact Parameter b",
            GeodesicMode::TimelikeParticle => "Angular Momentum l",
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.state.impact_or_momentum, -14.0..=14.0).text(b_label))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.state.radial_bias, -0.98..=0.98).text("Radial Launch"))
            .changed();
        if self.state.interaction_mode == GeodesicInteractionMode::PhotonFan {
            changed |= ui
                .add(egui::Slider::new(&mut self.state.fan_count, 2..=16).text("Fan Rays"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.state.fan_spread, 0.1..=8.0).text("Fan Spread db"))
                .changed();
        }
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
            self.rebuild_trajectory(260);
            self.refresh_lensing_pixels_in_place();
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
                PlaybackAction::Pause => {
                    self.state.is_playing = false;
                    crate::audio::notify_playback_paused();
                }
                PlaybackAction::Step => {
                    self.state.is_playing = false;
                    self.advance_steps(1);
                }
                PlaybackAction::Reset => {
                    self.state.is_playing = false;
                    crate::audio::notify_playback_reset();
                    self.rebuild_trajectory(260);
                    self.state.scrub_index = self.samples.len().saturating_sub(1);
                }
                PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
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

    fn draw_effective_potential_panel(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong("Effective Potential V_eff(r)");
        let curve = self.effective_potential_curve(72);
        let active = self.active_sample();
        let rs = self.state.scale_radius;
        let plot_id = scoped_panel_id(ui, "relativity_veff_plot");
        Plot::new(plot_id)
            .height(162.0)
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("V_eff(r)", PlotPoints::new(curve)).color(palette.series_color(2)),
                );
                if self.state.spacetime == SpacetimePreset::Schwarzschild {
                    if let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(rs) {
                        if let Ok(r_ph) = bh.photon_sphere_radius() {
                            plot_ui.vline(
                                VLine::new("Photon Sphere (1.5 Rs)", r_ph)
                                    .color(palette.warning)
                                    .style(LineStyle::Dashed { length: 5.0 }),
                            );
                        }
                        if let Ok(r_isco) = bh.isco() {
                            plot_ui.vline(
                                VLine::new("ISCO (3.0 Rs)", r_isco)
                                    .color(palette.info)
                                    .style(LineStyle::Dotted { spacing: 4.0 }),
                            );
                        }
                    }
                } else if self.state.spacetime == SpacetimePreset::KerrEquatorial {
                    if let Ok(kerr) = KerrEquatorialMetric::new(rs, 0.5 * rs * self.state.spin_chi)
                    {
                        plot_ui.vline(
                            VLine::new("Ergosphere r_ergo", kerr.ergosphere_radius())
                                .color(palette.warning)
                                .style(LineStyle::Dashed { length: 5.0 }),
                        );
                    }
                }
                if let Some(sample) = active {
                    let v_marker = 0.5 * sample.radial_vel * sample.radial_vel;
                    plot_ui.points(
                        Points::new("Current Position", vec![[sample.radial_coord, v_marker]])
                            .radius(5.0_f32)
                            .color(palette.accent),
                    );
                }
            });
    }

    fn draw_twin_clocks_panel(&self, ui: &mut egui::Ui, palette: &SemanticPalette) {
        ui.strong("Twin Clocks, Frame Dragging & Invariants");
        let Some(sample) = self.active_sample() else {
            ui.label("No active trajectory sample.");
            return;
        };
        ui.label(format!(
            "Coordinate ct: {:.3} | Proper/Affine tau: {:.3} | Active Rays: {}",
            sample.coord_ct,
            sample.proper_tau,
            self.active_ray_count.max(1)
        ));
        self.draw_spacetime_specific_readout(ui, sample, palette);
        ui.colored_label(
            palette.info,
            format!(
                "Net Clock Ratio dtau/dt: {:.4} | Lorentz gamma: {:.4} | Radius: {:.3}",
                sample.clock_rate, sample.lorentz_gamma, sample.radial_coord
            ),
        );
        ui.colored_label(
            palette.success,
            format!(
                "Metric Tangent Norm g_munu u^mu u^nu: {:.6}",
                sample.metric_norm
            ),
        );
        if self.halted_near_singularity {
            ui.colored_label(
                palette.warning,
                "Halted cleanly at horizon guard (r <= 1.02 r_+) — audio muted safely.",
            );
        }
    }

    fn draw_spacetime_specific_readout(
        &self,
        ui: &mut egui::Ui,
        sample: TrajectorySample,
        palette: &SemanticPalette,
    ) {
        let rs = self.state.scale_radius;
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild => {
                if let Ok(bh) = SchwarzschildBlackHole::from_schwarzschild_radius(rs) {
                    let r_ph = bh.photon_sphere_radius().unwrap_or(1.5 * rs);
                    let r_isco = bh.isco().unwrap_or(3.0 * rs);
                    ui.colored_label(
                        palette.warning,
                        format!(
                            "Photon Sphere r_ph: {:.2} | ISCO r_isco: {:.2} | Grav sqrt(1-Rs/r): {:.4}",
                            r_ph, r_isco, sample.grav_dilation
                        ),
                    );
                }
            }
            SpacetimePreset::KerrEquatorial => {
                if let Ok(kerr) = KerrEquatorialMetric::new(rs, 0.5 * rs * self.state.spin_chi) {
                    let r_ergo = kerr.ergosphere_radius();
                    let omega = kerr
                        .frame_dragging_omega(sample.radial_coord)
                        .unwrap_or(0.0);
                    ui.colored_label(
                        palette.warning,
                        format!(
                            "Kerr r_+: {:.2} | Ergosphere r_ergo: {:.2} | ZAMO omega(r): {:.5}",
                            kerr.outer_horizon_radius(),
                            r_ergo,
                            omega
                        ),
                    );
                }
            }
            SpacetimePreset::MorrisThorneWormhole => {
                let sheet = if sample.radial_coord < 0.0 {
                    "Lower Portal Sheet (l < 0)"
                } else {
                    "Upper Universe Sheet (l >= 0)"
                };
                ui.colored_label(palette.info, format!("Wormhole Sheet: {sheet}"));
            }
        }
    }

    pub(super) fn active_sample(&self) -> Option<TrajectorySample> {
        let idx = self
            .state
            .scrub_index
            .min(self.samples.len().saturating_sub(1));
        self.samples.get(idx).copied()
    }
}
