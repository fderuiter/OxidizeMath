#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::preset::{FermiGamePreset, IsingInteractionMode};
use super::{
    ising_spin_color, IsingBrushMode, IsingModelTool, DEFAULT_ISING_SIZE, MAX_SPATIAL_HISTORY,
};
use crate::async_sim::SimCommand;
use crate::framework::{encode_u8_grid, RasterBrushStroke};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::applied::game_theory::evolutionary::ReplicatorDynamics;
use nalgebra::{DMatrix, DVector};
use std::sync::atomic::Ordering;

impl IsingModelTool {
    /// Applies one of the built-in 2x2 evolutionary game presets and updates the 0D ODE curve.
    pub fn apply_game_preset(&mut self, preset: FermiGamePreset, ctx: &egui::Context) {
        self.mode = IsingInteractionMode::FermiGame;
        self.game_preset = preset;
        if preset != FermiGamePreset::Custom {
            self.game_payoffs = preset.payoffs();
            self.temperature = preset.default_temperature();
        }
        self.sync_shared_params();
        self.ode_anchor_sweep = self.current_sweeps();
        self.ode_initial_cooperator = self.cooperator_fraction();
        self.recompute_replicator_curve();
        ctx.request_repaint();
    }

    #[inline]
    pub(super) fn cooperator_fraction(&self) -> f64 {
        ((self.magnetization + 1.0) * 0.5).clamp(0.0, 1.0)
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
        p.mode = self.mode;
        p.game_payoffs = self.game_payoffs;
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

    pub(super) fn sync_metrics_from_latest(&mut self) {
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if snapshot.custom_data.len() >= 2 {
                self.magnetization = snapshot.custom_data[0];
                self.energy = snapshot.custom_data[1];
            }
        }
    }

    pub(super) fn record_snapshot_progress(&mut self) {
        let sweeps = self.current_sweeps();
        let coop = self.cooperator_fraction();
        if self.pending_ode_resync || sweeps == 0.0 {
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

    pub(crate) fn sync_brush_fields_before_frame(&mut self) {
        if self.brush_radius != self.last_synced_brush_radius {
            self.brush_state.radius = self.brush_radius.clamp(0, 99) as usize;
            self.last_synced_brush_radius = self.brush_radius;
        }
        if self.brush_cooperator != self.last_synced_brush_cooperator {
            self.brush_mode = if self.brush_cooperator {
                IsingBrushMode::SpinUp
            } else {
                IsingBrushMode::SpinDown
            };
            self.last_synced_brush_cooperator = self.brush_cooperator;
        }
    }

    pub(crate) fn sync_brush_fields_from_palette(&mut self) {
        self.brush_radius = (self.brush_state.radius as i32).clamp(0, 99);
        self.last_synced_brush_radius = self.brush_radius;
        match self.brush_mode {
            IsingBrushMode::SpinUp | IsingBrushMode::PinUp => self.brush_cooperator = true,
            IsingBrushMode::SpinDown | IsingBrushMode::PinDown => self.brush_cooperator = false,
            IsingBrushMode::Unpin => {}
        }
        self.last_synced_brush_cooperator = self.brush_cooperator;
    }

    /// Applies a [`RasterBrushStroke`] to spin domains and/or the boundary pin mask.
    pub fn apply_brush_stroke(&mut self, stroke: RasterBrushStroke, ctx: Option<&egui::Context>) {
        let effective_mode = if stroke.secondary {
            match self.brush_mode {
                IsingBrushMode::SpinUp => IsingBrushMode::SpinDown,
                IsingBrushMode::SpinDown => IsingBrushMode::SpinUp,
                IsingBrushMode::PinUp | IsingBrushMode::PinDown | IsingBrushMode::Unpin => {
                    IsingBrushMode::Unpin
                }
            }
        } else {
            self.brush_mode
        };
        if self.pin_cells.len() != DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE {
            self.pin_cells
                .resize(DEFAULT_ISING_SIZE * DEFAULT_ISING_SIZE, 0);
        }
        let pin_sym = match effective_mode {
            IsingBrushMode::PinUp => Some(1u8),
            IsingBrushMode::PinDown => Some(2u8),
            IsingBrushMode::Unpin => Some(0u8),
            IsingBrushMode::SpinUp | IsingBrushMode::SpinDown => None,
        };
        if let Some(sym) = pin_sym {
            for &(x, y) in &stroke.cells {
                if x < DEFAULT_ISING_SIZE && y < DEFAULT_ISING_SIZE {
                    self.pin_cells[flatten_2d_index(x, y, DEFAULT_ISING_SIZE)] = sym;
                }
            }
            self.sync_pinned_rle();
        }
        let _ = self.worker.send_command(SimCommand::ApplyBrushCells {
            cells: stroke.cells,
            mode: effective_mode.to_code(),
        });
        if !self.worker.is_running() {
            self.pending_ode_resync = true;
        }
        self.stepped_pixels = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    fn sync_pinned_rle(&mut self) {
        let next_rle = if self.pin_cells.iter().any(|&c| c != 0) {
            encode_u8_grid(DEFAULT_ISING_SIZE, DEFAULT_ISING_SIZE, &self.pin_cells, 2).ok()
        } else {
            None
        };
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle = next_rle;
    }

    /// Clears all pinned boundary sites across UI and worker threads.
    pub fn clear_pins(&mut self, ctx: Option<&egui::Context>) {
        self.pin_cells.fill(0);
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .pinned_rle = None;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_pins".to_string()));
        self.stepped_pixels = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }
}
