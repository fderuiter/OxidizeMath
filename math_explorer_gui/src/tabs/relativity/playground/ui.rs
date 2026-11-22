#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls, 3D Flamm/wormhole `Plot3DViewport`, $V_{\text{eff}}(r)$, and Twin Clocks UI.

use std::f64::consts::TAU;

use crate::accessibility::AccessibleTheoryHover;
use crate::app::SemanticPalette;
use crate::camera::Plot3DViewport;
use crate::framework::{PaneSidePanel, PlaybackAction, PlaybackTransport, scoped_panel_id};
use crate::web_io::PlatformInstant;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Points};
use math_explorer::physics::high_energy::{MorrisThorneWormhole, flamm_paraboloid_z};

use super::{GeodesicMode, GeodesicPlaygroundTool, SpacetimePreset, TrajectorySample};

impl GeodesicPlaygroundTool {
    pub(super) fn render_tool(&mut self, parent_ui: &mut egui::Ui) {
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
        self.draw_central_3d_viewport(parent_ui);
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = scoped_panel_id(parent_ui, "relativity_playground_side");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Geodesic Playground");
            ui.separator();

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
                            recompute = true;
                        }
                    }
                });

            egui::ComboBox::from_label("Worldline")
                .selected_text(self.state.mode.label())
                .show_ui(ui, |ui| {
                    for candidate in [GeodesicMode::NullPhoton, GeodesicMode::TimelikeParticle] {
                        if ui
                            .selectable_value(&mut self.state.mode, candidate, candidate.label())
                            .changed()
                        {
                            recompute = true;
                        }
                    }
                });

            recompute |= self.draw_parameter_sliders(ui);
            ui.separator();
            self.draw_transport_and_scrubber(ui, ctx, recompute);
            ui.separator();
            egui::CollapsingHeader::new("3D Camera")
                .default_open(false)
                .show(ui, |ui| self.state.camera.ui(ui));
        });
    }

    fn draw_parameter_sliders(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        let scale_label = match self.state.spacetime {
            SpacetimePreset::MorrisThorneWormhole => "Throat Radius b0",
            _ => "Schwarzschild Radius Rs",
        };
        changed |= ui
            .add(egui::Slider::new(&mut self.state.scale_radius, 0.5..=5.0).text(scale_label))
            .changed();

        if self.state.spacetime == SpacetimePreset::KerrEquatorial {
            changed |= ui
                .add(egui::Slider::new(&mut self.state.spin_chi, -0.92..=0.92).text("Kerr Spin chi"))
                .changed();
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
                    self.rebuild_trajectory(260);
                    self.state.scrub_index = self.samples.len().saturating_sub(1);
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
            .exact_height(195.0)
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
        let plot_id = scoped_panel_id(ui, "relativity_veff_plot");
        Plot::new(plot_id)
            .height(155.0)
            .allow_boxed_zoom(false)
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("V_eff(r)", PlotPoints::new(curve)).color(palette.series_color(2)),
                );
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
        ui.strong("Twin Clocks & Invariants");
        let Some(sample) = self.active_sample() else {
            ui.label("No active trajectory sample.");
            return;
        };
        ui.label(format!(
            "Distant Coordinate Clock ct: {:.3}   |   Local Proper/Affine tau: {:.3}",
            sample.coord_ct, sample.proper_tau
        ));
        ui.label(format!(
            "Gravitational sqrt(1 - Rs/r): {:.4}   |   Kinematic Lorentz gamma: {:.4}",
            sample.grav_dilation, sample.lorentz_gamma
        ));
        ui.colored_label(
            palette.info,
            format!(
                "Net Twin Clock Ratio dtau/dt: {:.4}   |   Radius/Distance: {:.3}",
                sample.clock_rate, sample.radial_coord
            ),
        );
        ui.colored_label(
            palette.success,
            format!("Metric Tangent Norm g_munu u^mu u^nu: {:.6}", sample.metric_norm),
        );
        if self.halted_near_singularity {
            ui.colored_label(
                palette.warning,
                "Halted cleanly at horizon guard (r <= 1.02 Rs) to preserve numerical stability.",
            );
        }
    }

    fn active_sample(&self) -> Option<TrajectorySample> {
        let idx = self.state.scrub_index.min(self.samples.len().saturating_sub(1));
        self.samples.get(idx).copied()
    }

    fn draw_central_3d_viewport(&mut self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = SemanticPalette::from_visuals(ui.visuals());
            let wireframe = self.build_embedding_wireframe();
            let rs = self.state.scale_radius;
            let horizon_ring = self.reference_ring_projected(rs, 0.0, 56);
            let photon_ring =
                (self.state.spacetime != SpacetimePreset::MorrisThorneWormhole).then(|| {
                    let z_ph = flamm_paraboloid_z(rs, 1.5 * rs).unwrap_or(0.0);
                    self.reference_ring_projected(1.5 * rs, z_ph, 56)
                });
            let (traj, head) = self.projected_worldline_and_head();

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
                    plot_ui.line(
                        Line::new("Horizon / Throat", PlotPoints::new(horizon_ring))
                            .color(palette.error),
                    );
                    if let Some(ph) = photon_ring {
                        plot_ui.line(
                            Line::new("Photon Sphere (1.5 Rs)", PlotPoints::new(ph))
                                .color(palette.warning),
                        );
                    }
                    plot_ui.line(
                        Line::new("Geodesic Worldline", PlotPoints::new(traj))
                            .color(palette.series_color(0))
                            .width(2.4_f32),
                    );
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
        });
    }

    fn projected_worldline_and_head(&self) -> (Vec<[f64; 2]>, Option<[f64; 2]>) {
        let limit = self
            .state
            .scrub_index
            .saturating_add(1)
            .min(self.samples.len());
        let traj = self.samples[..limit]
            .iter()
            .map(|s| self.state.camera.project(&s.embed_xyz))
            .collect();
        let head = self
            .active_sample()
            .map(|s| self.state.camera.project(&s.embed_xyz));
        (traj, head)
    }

    fn build_embedding_wireframe(&self) -> Vec<Vec<[f64; 2]>> {
        let mut lines = Vec::with_capacity(14);
        let rs = self.state.scale_radius;
        match self.state.spacetime {
            SpacetimePreset::Schwarzschild | SpacetimePreset::KerrEquatorial => {
                for ring_idx in 1..=5 {
                    let r = rs * (1.0 + 1.35 * f64::from(ring_idx));
                    let z = flamm_paraboloid_z(rs, r).unwrap_or(0.0);
                    lines.push(self.reference_ring_projected(r, z, 40));
                }
                for spoke in 0..8 {
                    let phi = TAU * f64::from(spoke) / 8.0;
                    let (s, c) = phi.sin_cos();
                    let mut meridian = Vec::with_capacity(18);
                    for k in 0..18 {
                        let r = rs + 7.0 * rs * (k as f64) / 17.0;
                        let z = flamm_paraboloid_z(rs, r).unwrap_or(0.0);
                        meridian.push(self.state.camera.project(&[r * c, r * s, z]));
                    }
                    lines.push(meridian);
                }
            }
            SpacetimePreset::MorrisThorneWormhole => {
                let Ok(wh) = MorrisThorneWormhole::new(rs) else {
                    return lines;
                };
                for level in [-6.0, -3.0, 3.0, 6.0] {
                    let r = wh.areal_radius(level).unwrap_or(rs);
                    let z = wh.embedding_z(level).unwrap_or(0.0);
                    lines.push(self.reference_ring_projected(r, z, 40));
                }
                for spoke in 0..8 {
                    let phi = TAU * f64::from(spoke) / 8.0;
                    let mut meridian = Vec::with_capacity(21);
                    for k in 0..=20 {
                        let l = -8.0 + 16.0 * (k as f64) / 20.0;
                        if let Ok(xyz) = wh.embedding_xyz(l, phi) {
                            meridian.push(self.state.camera.project(&xyz));
                        }
                    }
                    lines.push(meridian);
                }
            }
        }
        lines
    }
}
