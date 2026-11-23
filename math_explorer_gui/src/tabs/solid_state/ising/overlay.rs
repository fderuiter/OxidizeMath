#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::{FermiGamePreset, IsingInteractionMode};
use super::{DEFAULT_ISING_SIZE, IsingModelTool, MAX_SPATIAL_HISTORY, ising_spin_color};
use crate::async_sim::SimCommand;
use eframe::egui;
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints};
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;
use nalgebra::{DMatrix, DVector};
use std::sync::atomic::Ordering;

impl IsingModelTool {
    /// Returns `true` when the lattice is trapped in an all-`+1` or all-`-1` absorbing state.
    #[inline]
    pub(super) fn is_lattice_absorbed(&self) -> bool {
        self.uniform_magnetization.abs() >= 0.9998
            && !self.defector_sea
            && self.droplet_seeds.is_empty()
            && self.active_challenge.is_none()
    }

    /// Auto-reseeds an absorbed `FermiGame` lattice to a deterministic 50/50 configuration.
    pub(super) fn reseed_if_absorbed_fermi_lattice(&mut self) -> bool {
        if !self.is_lattice_absorbed() {
            return false;
        }
        self.defector_sea = false;
        self.droplet_seeds.clear();
        self.magnetization = 0.0;
        self.uniform_magnetization = 0.0;
        self.ode_initial_cooperator = 0.5;
        self.ode_anchor_sweep = 0.0;
        self.pending_ode_resync = true;
        self.spatial_history.clear();
        self.spatial_history.push_back([0.0, 0.5]);
        let _ = self
            .worker
            .send_command(SimCommand::Custom("reseed_random".to_string()));
        self.dirty = true;
        true
    }

    /// Pushes `defector_sea` and `droplet_seeds` to the background worker deterministically.
    pub(super) fn push_seed_layout_to_worker(&mut self) {
        self.sync_shared_params();
        let _ = self
            .worker
            .send_command(SimCommand::Custom("apply_seed_layout".to_string()));
        self.pending_ode_resync = true;
        self.dirty = true;
    }

    /// Applies one of the built-in 2x2 evolutionary game presets and updates the 0D ODE curve.
    pub fn apply_game_preset(&mut self, preset: FermiGamePreset, ctx: &egui::Context) {
        if self.active_challenge.is_some() {
            return;
        }
        self.mode = IsingInteractionMode::FermiGame;
        self.game_preset = preset;
        if preset != FermiGamePreset::Custom {
            self.game_payoffs = preset.payoffs();
            self.temperature = preset.default_temperature();
        }
        let reseeded = self.reseed_if_absorbed_fermi_lattice();
        self.sync_shared_params();
        if !reseeded {
            self.ode_anchor_sweep = self.current_sweeps();
            self.ode_initial_cooperator = self.cooperator_fraction();
        }
        self.recompute_replicator_curve();
        ctx.request_repaint();
    }

    #[inline]
    pub(super) fn cooperator_fraction(&self) -> f64 {
        ((self.uniform_magnetization + 1.0) * 0.5).clamp(0.0, 1.0)
    }

    pub(super) fn current_sweeps(&self) -> f64 {
        let trials = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .total_trials
            .load(Ordering::Relaxed);
        trials as f64 / (DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE) as f64
    }

    pub(super) fn sync_shared_params(&self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.temperature = self.temperature;
        p.j_coupling = self.j_coupling;
        p.h_field = self.h_field;
        p.mutation_rate = self.mutation_rate;
        p.mode = self.mode;
        p.game_payoffs = self.game_payoffs;
        p.defector_sea = self.defector_sea;
        p.droplet_seeds = self.droplet_seeds.clone();
        p.steps_per_frame = self.steps_per_frame;
    }

    pub(super) fn reset_trajectory_overlays(&mut self) {
        let coop = self.cooperator_fraction();
        self.spatial_history.clear();
        self.spatial_history.push_back([0.0, coop]);
        self.ode_anchor_sweep = 0.0;
        self.ode_initial_cooperator = coop;
        self.recompute_replicator_curve();
    }

    pub(super) fn recompute_replicator_curve(&mut self) {
        if !self.game_payoffs.iter().all(|v| v.is_finite()) {
            self.replicator_curve.clear();
            return;
        }
        let mat = DMatrix::from_row_slice(2, 2, &self.game_payoffs);
        let Ok(system) = ReplicatorDynamics::new(mat) else {
            self.replicator_curve.clear();
            return;
        };
        let x0 = self.ode_initial_cooperator.clamp(0.0, 1.0);
        let init = DVector::from_vec(vec![x0, 1.0 - x0]);
        let span = (self.current_sweeps() - self.ode_anchor_sweep + 40.0).clamp(40.0, 400.0);
        let dt = (span / 240.0).max(0.05);
        let anchor = self.ode_anchor_sweep;
        self.replicator_curve = system
            .simulate(init, span, dt)
            .into_iter()
            .map(|(t, state)| [anchor + t, state[0]])
            .collect();
    }

    pub(super) fn extract_snapshot_metrics(
        snapshot: &crate::async_sim::StateSnapshot,
        fallback_m: f64,
        fallback_e: f64,
    ) -> (f64, f64, f64, f64) {
        let (m, e) = if snapshot.custom_data.len() >= 2 {
            (snapshot.custom_data[0], snapshot.custom_data[1])
        } else {
            (fallback_m, fallback_e)
        };
        let n = (snapshot.width * snapshot.height).max(1);
        if let Some(spins) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<i8>>())
            .filter(|s| s.len() == n)
        {
            let mut sum_u: i64 = 0;
            let mut sum_stag: i64 = 0;
            for (idx, &s) in spins.iter().enumerate() {
                let x = idx % snapshot.width.max(1);
                let y = idx / snapshot.width.max(1);
                let sign = if (x + y).is_multiple_of(2) {
                    1_i64
                } else {
                    -1_i64
                };
                sum_u += i64::from(s);
                sum_stag += sign * i64::from(s);
            }
            (m, e, sum_u as f64 / n as f64, sum_stag as f64 / n as f64)
        } else {
            (m, e, m, m)
        }
    }

    /// Applies a 2D [`crate::async_sim::atlas::AtlasTeleportEvent`], switching mode and auto-reseeding absorbed lattices.
    pub fn apply_atlas_teleport(
        &mut self,
        ev: crate::async_sim::atlas::AtlasTeleportEvent,
        ctx: Option<&egui::Context>,
    ) {
        if self.active_challenge.is_some() {
            return;
        }
        let mut entered_game = false;
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "temperature" => {
                    self.mode = IsingInteractionMode::Ferromagnetic;
                    self.temperature = val.clamp(0.1, 10.0);
                }
                "h_field" => {
                    self.mode = IsingInteractionMode::Ferromagnetic;
                    self.h_field = val.clamp(-2.0, 2.0);
                }
                "j_coupling" => {
                    self.mode = IsingInteractionMode::Ferromagnetic;
                    self.j_coupling = val.clamp(-2.0, 2.0);
                }
                "T" => {
                    entered_game = true;
                    self.game_payoffs[2] = val.clamp(-20.0, 20.0);
                }
                "S" => {
                    entered_game = true;
                    self.game_payoffs[1] = val.clamp(-20.0, 20.0);
                }
                _ => {}
            }
        }
        if entered_game {
            self.mode = IsingInteractionMode::FermiGame;
            self.game_preset = FermiGamePreset::Custom;
            self.reseed_if_absorbed_fermi_lattice();
            self.ode_anchor_sweep = self.current_sweeps();
            self.ode_initial_cooperator = self.cooperator_fraction();
            self.recompute_replicator_curve();
        }
        self.sync_shared_params();
        if let Some(egui_ctx) = ctx {
            self.request_metric_refresh(egui_ctx);
        }
    }

    pub(super) fn render_ising_atlas_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        use scientific_metadata::theory::TheoryDescribable;
        use std::hash::{Hash, Hasher};
        use std::sync::Arc;
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let get_val = |k: &str| match k {
            "temperature" => self.temperature,
            "h_field" => self.h_field,
            "j_coupling" => self.j_coupling,
            "T" => self.game_payoffs[2],
            "S" => self.game_payoffs[1],
            _ => 0.0,
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.mode.hash(&mut h);
        if xk == "T" {
            self.temperature.to_bits().hash(&mut h);
            self.game_payoffs[0].to_bits().hash(&mut h);
            self.game_payoffs[3].to_bits().hash(&mut h);
        } else if yk == "j_coupling" {
            self.h_field.to_bits().hash(&mut h);
        } else {
            self.j_coupling.to_bits().hash(&mut h);
        }
        let theory_params = self.theory_parameters();
        let (temp, j_c, h_f, payoffs) = (
            self.temperature,
            self.j_coupling,
            self.h_field,
            self.game_payoffs,
        );
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "ising_model",
            get_val(xk),
            get_val(yk),
            h.finish(),
            move |keys| {
                Arc::new(super::atlas::IsingCartographer::new(
                    &theory_params,
                    keys,
                    temp,
                    j_c,
                    h_f,
                    payoffs,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev, Some(ctx));
        }
    }

    pub(super) fn sync_metrics_from_latest(&mut self) {
        if let Some((m, e, mu, ms)) = self
            .worker
            .latest_snapshot()
            .map(|s| Self::extract_snapshot_metrics(s, self.magnetization, self.energy))
        {
            self.magnetization = m;
            self.energy = e;
            self.uniform_magnetization = mu;
            self.staggered_magnetization = ms;
        }
    }

    pub(super) fn record_snapshot_progress(&mut self) {
        let sweeps = self.current_sweeps();
        let coop = self.cooperator_fraction();
        let escaped_absorption = self.mutation_rate > 0.0
            && (self.ode_initial_cooperator <= 1e-4 || self.ode_initial_cooperator >= 1.0 - 1e-4)
            && (0.01..=0.99).contains(&coop);
        if self.pending_ode_resync || sweeps == 0.0 || escaped_absorption {
            self.pending_ode_resync = false;
            if sweeps == 0.0 {
                self.spatial_history.clear();
            }
            self.spatial_history.push_back([sweeps, coop]);
            self.ode_anchor_sweep = sweeps;
            self.ode_initial_cooperator = coop;
            self.recompute_replicator_curve();
            return;
        }
        if self.spatial_history.len() >= MAX_SPATIAL_HISTORY {
            self.spatial_history.pop_front();
        }
        self.spatial_history.push_back([sweeps, coop]);
        if self
            .replicator_curve
            .last()
            .is_none_or(|last| sweeps > last[0] - 5.0)
        {
            self.recompute_replicator_curve();
        }
    }

    pub(super) fn request_metric_refresh(&mut self, ctx: &egui::Context) {
        if self.worker.state().is_ready()
            && self
                .worker
                .send_command(SimCommand::Custom("recompute_metrics".to_string()))
                .is_ok()
        {
            ctx.request_repaint();
        }
    }

    pub(super) fn sync_lattice_texture(&mut self, ctx: &egui::Context) {
        if !self.dirty && self.texture.is_some() {
            return;
        }
        let Some(snapshot) = self.worker.latest_snapshot() else {
            return;
        };
        let Ok(snap_pixels) = snapshot.pixels.try_read() else {
            return;
        };
        if snap_pixels.is_empty() {
            return;
        }
        let num_pixels = snapshot.width * snapshot.height;
        if self.pixels.len() != num_pixels {
            self.pixels.resize(num_pixels, egui::Color32::BLACK);
        }
        let palette = crate::app::SemanticPalette::from_visuals(&ctx.style().visuals);
        if let Some(spins) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<Vec<i8>>())
            .filter(|s| s.len() == num_pixels)
        {
            for (dst, &s) in self.pixels.iter_mut().zip(spins.iter()) {
                *dst = ising_spin_color(s, &palette);
            }
        } else if snap_pixels.len() == num_pixels {
            self.pixels.copy_from_slice(&snap_pixels);
        }

        let image = egui::ColorImage {
            size: [snapshot.width, snapshot.height],
            pixels: self.pixels.clone(),
            ..Default::default()
        };
        let texture_opts = egui::TextureOptions::NEAREST;
        if let Some(texture) = &mut self.texture {
            texture.set(image, texture_opts);
        } else {
            self.texture = Some(ctx.load_texture("ising_lattice", image, texture_opts));
        }
        self.dirty = false;
    }

    pub(super) fn draw_replicator_overlay_plot(&self, ui: &mut egui::Ui, height: f32) {
        ui.strong(format!(
            "Spatial Reciprocity (100×100 Fermi) vs 0D Replicator ODE — {}",
            self.game_preset.label()
        ));
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        let spatial_pts: PlotPoints = self
            .spatial_history
            .iter()
            .copied()
            .collect::<Vec<[f64; 2]>>()
            .into();
        let ode_pts: PlotPoints = self.replicator_curve.clone().into();

        Plot::new("ising_fermi_replicator_overlay")
            .height(height.max(140.0))
            .legend(Legend::default())
            .include_y(0.0)
            .include_y(1.0)
            .x_axis_label("Monte Carlo Sweeps / Replicator Time t")
            .y_axis_label("Cooperator Fraction x_C")
            .show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new("2D Spatial Lattice (100×100 Fermi)", spatial_pts)
                        .color(palette.axis_x)
                        .width(2.5_f32),
                );
                plot_ui.line(
                    Line::new("0D Replicator Dynamics (Well-Mixed ODE)", ode_pts)
                        .color(palette.axis_z)
                        .style(LineStyle::Dashed { length: 7.0_f32 })
                        .width(2.0_f32),
                );
            });
    }
}
