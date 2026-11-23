#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::kspace_runner::MriKSpaceRunner;
use super::kspace_types::{
    CompactBrushStroke, MriKSpaceSharedParams, MriKSpaceSnapshot, MriKSpaceTheory,
    MriKSpaceToolState, StudioBrushMode, MAX_STUDIO_STROKES,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState, SimulationRunner};
use crate::framework::{ExportCapabilities, InteractionContext, InteractiveTool};
use eframe::egui;
use eframe::egui::ColorImage;
use math_commons::math_kernel::types::flatten_2d_index;
use scientific_metadata::theory::TheoryDescribable;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Interactive 2D MRI k-space, RF pulse, spin-echo, and Turing tissue studio.
pub struct MriKSpaceTool {
    /// Active validated studio configuration.
    pub config: MriKSpaceToolState,
    /// Shared configuration handle synchronized with [`MriKSpaceRunner`].
    pub params: Arc<std::sync::RwLock<MriKSpaceSharedParams>>,
    /// Asynchronous worker controller.
    pub worker: BackgroundWorker<MriKSpaceRunner>,
    /// Most recent frame snapshot rendered across spatial, k-space, recon, and 3D Bloch views.
    pub latest: MriKSpaceSnapshot,
    pub(crate) phantom_tex: Option<egui::TextureHandle>,
    pub(crate) kspace_tex: Option<egui::TextureHandle>,
    pub(crate) recon_tex: Option<egui::TextureHandle>,
    pub(crate) textures_dirty: bool,
    pub(crate) last_high_contrast: Option<bool>,
    pub(crate) theory: MriKSpaceTheory,
}

impl Default for MriKSpaceTool {
    fn default() -> Self {
        let config = MriKSpaceToolState::default();
        let params = Arc::new(std::sync::RwLock::new(MriKSpaceSharedParams {
            config: config.clone(),
            high_contrast: false,
        }));
        let runner = MriKSpaceRunner::new(Arc::clone(&params));
        let initial_snap = runner.get_snapshot();
        let latest = initial_snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MriKSpaceSnapshot>())
            .cloned()
            .expect("MriKSpaceRunner must emit MriKSpaceSnapshot");
        let worker = BackgroundWorker::new(runner);

        Self {
            config,
            params,
            worker,
            latest,
            phantom_tex: None,
            kspace_tex: None,
            recon_tex: None,
            textures_dirty: true,
            last_high_contrast: None,
            theory: MriKSpaceTheory,
        }
    }
}

impl MriKSpaceTool {
    /// Synchronizes `self.config` into `self.params` and sends a custom worker command.
    pub fn sync_and_notify(&mut self, action: &'static str) {
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .config = self.config.clone();
        let _ = self
            .worker
            .send_command(SimCommand::Custom(action.to_string()));
    }

    /// Fires a $180^\circ$ Hahn/CPMG spin-echo refocusing pulse across all voxels.
    pub fn trigger_spin_echo_180(&mut self) {
        self.sync_and_notify("refocus_180");
    }

    /// Re-excites spins with the configured RF flip angle and $T_R$ recovery.
    pub fn trigger_excite_90(&mut self) {
        self.sync_and_notify("excite_90");
    }

    /// Clears painted $\Delta B_0$ pockets, local RF flip offsets, and k-space strokes.
    pub fn clear_painted_strokes(&mut self) {
        self.config.strokes.clear();
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .config = self.config.clone();
        let _ = self.worker.send_command(SimCommand::ClearObstacles);
    }

    /// Restores factory defaults and resets the background worker cleanly.
    pub fn reset(&mut self) {
        let target_state = MriKSpaceToolState::default();
        self.config = target_state.clone();
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .config = target_state;
        let params = Arc::clone(&self.params);
        let fresh_runner = MriKSpaceRunner::new(Arc::clone(&params));
        if let Some(snap) = fresh_runner
            .get_snapshot()
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MriKSpaceSnapshot>())
        {
            self.latest = snap.clone();
        }
        self.worker
            .reset_or_respawn(move || MriKSpaceRunner::new(params));
        self.textures_dirty = true;
    }

    /// Applies a brush stroke at grid coordinates `(cx, cy)` and dispatches `SimCommand::ApplyBrush`.
    pub fn apply_brush_at_grid(&mut self, mode: StudioBrushMode, cx: i32, cy: i32) {
        let n = self.config.grid_size as i32;
        let clamped_x = cx.clamp(0, n - 1);
        let clamped_y = cy.clamp(0, n - 1);
        let r = self.config.brush_radius.clamp(1, 16);
        self.config.brush_mode = mode;

        let stroke = CompactBrushStroke::new(mode, clamped_x, clamped_y, r);
        if self.config.strokes.last().copied() != Some(stroke) {
            if self.config.strokes.len() >= MAX_STUDIO_STROKES {
                self.config.strokes.remove(0);
            }
            self.config.strokes.push(stroke);
        }
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .config = self.config.clone();
        let _ = self.worker.send_command(SimCommand::ApplyBrush {
            cx: clamped_x,
            cy: clamped_y,
            r,
            is_obstacle: true,
        });
    }

    pub(crate) fn map_pointer_to_grid(&self, rect: egui::Rect, pos: egui::Pos2) -> (i32, i32) {
        let n = self.config.grid_size as f32;
        let u = ((pos.x - rect.min.x) / rect.width().max(1.0)).clamp(0.0, 0.999_99);
        let v = ((pos.y - rect.min.y) / rect.height().max(1.0)).clamp(0.0, 0.999_99);
        ((u * n) as i32, (v * n) as i32)
    }

    fn apply_worker_state(&mut self, target: SimWorkerState) {
        if target == SimWorkerState::Running {
            if !self.worker.is_running() {
                let _ = self.worker.send_command(SimCommand::Start);
            }
        } else if self.worker.is_running() {
            let _ = self.worker.send_command(SimCommand::Pause);
        }
    }
}

impl InteractiveTool for MriKSpaceTool {
    fn name(&self) -> &'static str {
        "MRI k-Space & RF Studio"
    }

    fn domain(&self) -> &'static str {
        "mri"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let n = self.latest.grid_size;
        if n == 0 || self.latest.recon_rgba.len() != n * n {
            return None;
        }
        Some(ColorImage {
            size: [n, n],
            pixels: self.latest.recon_rgba.clone(),
            ..Default::default()
        })
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let n = self.latest.grid_size;
        if n == 0 || self.latest.recon_mag.len() != n * n {
            return Vec::new();
        }
        let mid_r = self.config.selected_voxel[0].min(n - 1);
        let profile: Vec<[f64; 2]> = (0..n)
            .map(|c| {
                [
                    c as f64,
                    self.latest.recon_mag[flatten_2d_index(c, mid_r, n)],
                ]
            })
            .collect();
        vec![("Reconstructed Slice Row Profile".to_string(), profile)]
    }

    fn step_simulation(&mut self) {
        let _ = self.worker.send_command(SimCommand::Step(1));
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.show_studio_inside(ui);
    }

    fn on_brush(&mut self, ctx: &InteractionContext) {
        if let Some(pos) = ctx.pointer_pos {
            let (cx, cy) = self.map_pointer_to_grid(ctx.response.rect, pos);
            self.apply_brush_at_grid(self.config.brush_mode, cx, cy);
        }
    }

    fn on_click(&mut self, ctx: &InteractionContext) {
        if let Some(pos) = ctx.pointer_pos {
            let (cx, cy) = self.map_pointer_to_grid(ctx.response.rect, pos);
            if self.config.brush_mode.targets_kspace() {
                self.apply_brush_at_grid(self.config.brush_mode, cx, cy);
            } else {
                self.config.selected_voxel = [cy as usize, cx as usize];
                self.sync_and_notify("sync_view");
            }
        }
    }

    fn on_keyboard(&mut self, ctx: &InteractionContext) {
        let max_idx = self.config.grid_size.saturating_sub(1);
        let mut moved = false;
        if ctx.keys_down.contains(&egui::Key::ArrowUp) && self.config.selected_voxel[0] > 0 {
            self.config.selected_voxel[0] -= 1;
            moved = true;
        }
        if ctx.keys_down.contains(&egui::Key::ArrowDown) && self.config.selected_voxel[0] < max_idx
        {
            self.config.selected_voxel[0] += 1;
            moved = true;
        }
        if ctx.keys_down.contains(&egui::Key::ArrowLeft) && self.config.selected_voxel[1] > 0 {
            self.config.selected_voxel[1] -= 1;
            moved = true;
        }
        if ctx.keys_down.contains(&egui::Key::ArrowRight) && self.config.selected_voxel[1] < max_idx
        {
            self.config.selected_voxel[1] += 1;
            moved = true;
        }
        if moved {
            self.sync_and_notify("sync_view");
        }
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.config.flip_angle_deg.to_bits().hash(&mut h);
        self.config.tr.to_bits().hash(&mut h);
        self.config.te.to_bits().hash(&mut h);
        self.config.mask_cutoff.to_bits().hash(&mut h);
        self.config.grad_gx_integral.to_bits().hash(&mut h);
        self.config.grad_gy_integral.to_bits().hash(&mut h);
        self.config.grid_size.hash(&mut h);
        self.config.radial_spokes.hash(&mut h);
        self.config.cartesian_stride.hash(&mut h);
        self.config.brush_radius.hash(&mut h);
        self.config.selected_voxel.hash(&mut h);
        self.config.phantom.hash(&mut h);
        self.config.mask_preset.hash(&mut h);
        self.config.brush_mode.hash(&mut h);
        self.config.evolve_turing.hash(&mut h);
        self.config.camera.pitch.to_bits().hash(&mut h);
        self.config.camera.yaw.to_bits().hash(&mut h);
        self.config.camera.zoom.to_bits().hash(&mut h);
        self.config.strokes.hash(&mut h);
        std::mem::discriminant(&self.worker.state()).hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        let mut persisted = self.config.clone();
        persisted.state = Some(self.worker.state());
        serde_json::to_string(&persisted).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let mut parsed: MriKSpaceToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid MRI Studio state: {e}"))?;
        parsed.validate()?;
        let target_worker = parsed.state.unwrap_or(SimWorkerState::Ready);
        parsed.state = Some(target_worker);
        self.config = parsed.clone();
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .config = parsed;

        let fresh_runner = MriKSpaceRunner::new(Arc::clone(&self.params));
        if let Some(snap) = fresh_runner
            .get_snapshot()
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<MriKSpaceSnapshot>())
        {
            self.latest = snap.clone();
        }
        let params = Arc::clone(&self.params);
        self.worker
            .reset_or_respawn(move || MriKSpaceRunner::new(params));
        self.apply_worker_state(target_worker);
        self.textures_dirty = true;
        Ok(())
    }

    fn has_finite_parameters(&self) -> bool {
        self.config.validate().is_ok() && self.worker.state() != SimWorkerState::Faulted
    }

    fn clamp_parameters(&mut self) -> bool {
        self.config.flip_angle_deg = self.config.flip_angle_deg.clamp(5.0, 180.0);
        self.config.tr = self.config.tr.clamp(0.1, 5.0);
        self.config.te = self.config.te.clamp(0.005, 0.4);
        self.config.mask_cutoff = self.config.mask_cutoff.clamp(0.05, 1.0);
        self.config.grad_gx_integral = self.config.grad_gx_integral.clamp(-5.0e-5, 5.0e-5);
        self.config.grad_gy_integral = self.config.grad_gy_integral.clamp(-5.0e-5, 5.0e-5);
        if !matches!(self.config.grid_size, 64 | 128) {
            self.config.grid_size = 64;
        }
        let max_v = self.config.grid_size - 1;
        self.config.selected_voxel[0] = self.config.selected_voxel[0].min(max_v);
        self.config.selected_voxel[1] = self.config.selected_voxel[1].min(max_v);
        self.sync_and_notify("sync_config");
        true
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "MriKSpaceStudio",
        display_name: "MRI k-Space & RF Studio",
        domain: "mri",
        tags: &[
            "mri",
            "k-space",
            "fft",
            "turing",
            "spin-echo",
            "bloch",
            "phantom",
        ],
        build: || Box::new(MriKSpaceTool::default()),
    }
}
