#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::{AtlasPassMeta, AtlasRenderRequest, AtlasRunner, DEFAULT_ATLAS_SIDE};
use super::{AtlasCell, RegimeCartographer, ATLAS_STRIDES};
#[cfg(target_arch = "wasm32")]
use crate::async_sim::SimulationRunner;
use crate::async_sim::StateSnapshot;
#[cfg(not(target_arch = "wasm32"))]
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Parameter update emitted when the user clicks or activates a coordinate in the 2D atlas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasTeleportEvent {
    /// Horizontal parameter key (`p1`).
    pub x_key: &'static str,
    /// Clamped target value for `x_key`.
    pub x_val: f64,
    /// Vertical parameter key (`p2`).
    pub y_key: &'static str,
    /// Clamped target value for `y_key`.
    pub y_val: f64,
    /// Regime label at the teleported coordinate.
    pub regime_label: &'static str,
}

/// Lazy asynchronous controller coordinating [`AtlasRunner`] passes, textures, and navigation.
pub struct ParameterAtlasController {
    axis_pair_options: Vec<(&'static str, &'static str)>,
    selected_pair_idx: usize,
    visible: bool,
    suspended: bool,
    dirty: bool,
    raster_size: (usize, usize),
    context_signature: Option<u64>,
    high_contrast: bool,
    request_generation: u64,
    completed_generation: u64,
    current_stride: usize,
    pub(super) active_cartographer: Option<Arc<dyn RegimeCartographer>>,
    pub(super) latest_meta: Option<AtlasPassMeta>,
    pub(super) texture: Option<egui::TextureHandle>,
    pub(super) keyboard_cursor: Option<[f64; 2]>,
    shared_request: Option<Arc<RwLock<AtlasRenderRequest>>>,
    latest_generation: Arc<AtomicU64>,
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<BackgroundWorker<AtlasRunner>>,
    #[cfg(target_arch = "wasm32")]
    wasm_runner: Option<AtlasRunner>,
}

impl ParameterAtlasController {
    /// Creates a new hidden [`ParameterAtlasController`] for the declared `(p1, p2)` axis options.
    #[must_use]
    pub fn new(axis_pair_options: Vec<(&'static str, &'static str)>) -> Self {
        Self {
            axis_pair_options,
            selected_pair_idx: 0,
            visible: false,
            suspended: false,
            dirty: true,
            raster_size: (DEFAULT_ATLAS_SIDE, DEFAULT_ATLAS_SIDE),
            context_signature: None,
            high_contrast: false,
            request_generation: 0,
            completed_generation: 0,
            current_stride: ATLAS_STRIDES[0],
            active_cartographer: None,
            latest_meta: None,
            texture: None,
            keyboard_cursor: None,
            shared_request: None,
            latest_generation: Arc::new(AtomicU64::new(0)),
            #[cfg(not(target_arch = "wasm32"))]
            worker: None,
            #[cfg(target_arch = "wasm32")]
            wasm_runner: None,
        }
    }

    /// Returns whether Atlas Mode is currently open.
    #[inline]
    #[must_use]
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Opens or closes Atlas Mode, pausing any in-flight worker when hidden.
    pub fn set_visible(&mut self, visible: bool) {
        if self.visible == visible {
            return;
        }
        self.visible = visible;
        if visible {
            self.dirty = true;
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(worker) = &mut self.worker {
                if worker.is_running() {
                    let _ = worker.send_command(SimCommand::Pause);
                }
            }
        }
    }

    /// Suspends or resumes any background [`AtlasRunner`] worker when the host pane/tool hides or shows.
    pub fn set_suspended(&mut self, suspended: bool) {
        self.suspended = suspended;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(worker) = &mut self.worker {
            worker.set_suspended(suspended);
        }
    }

    /// Available `(x_key, y_key)` axis pairs supported by the host tool.
    #[inline]
    #[must_use]
    pub fn axis_pair_options(&self) -> &[(&'static str, &'static str)] {
        &self.axis_pair_options
    }

    /// Replaces the supported `(x_key, y_key)` axis pairs when a multi-mode tool changes kinetics,
    /// preserving the active pair if still valid or resetting to index `0`.
    pub fn replace_axis_pair_options(&mut self, options: Vec<(&'static str, &'static str)>) {
        if self.axis_pair_options == options {
            return;
        }
        let prev = self.axis_keys();
        let next_idx = options.iter().position(|&pair| pair == prev).unwrap_or(0);
        self.axis_pair_options = options;
        self.selected_pair_idx = next_idx;
        self.keyboard_cursor = None;
        self.dirty = true;
    }

    /// Currently selected axis pair index.
    #[inline]
    #[must_use]
    pub fn selected_pair_index(&self) -> usize {
        self.selected_pair_idx
    }

    /// Selects an axis pair by index and invalidates the current raster pass.
    pub fn select_axis_pair_index(&mut self, idx: usize) {
        if idx < self.axis_pair_options.len() && idx != self.selected_pair_idx {
            self.selected_pair_idx = idx;
            self.keyboard_cursor = None;
            self.dirty = true;
        }
    }

    /// Returns the active `(x_key, y_key)` parameter identifiers.
    #[must_use]
    pub fn axis_keys(&self) -> (&'static str, &'static str) {
        self.axis_pair_options
            .get(self.selected_pair_idx)
            .copied()
            .unwrap_or(("", ""))
    }

    /// Restores persisted `(x_key, y_key)` axis keys, returning `Err` if unsupported by this tool.
    pub fn set_axis_keys(&mut self, x_key: &str, y_key: &str) -> Result<(), String> {
        let Some(idx) = self
            .axis_pair_options
            .iter()
            .position(|&(xk, yk)| xk == x_key && yk == y_key)
        else {
            return Err(format!(
                "Unsupported parameter atlas axis pair ('{x_key}', '{y_key}')"
            ));
        };
        if self.selected_pair_idx != idx {
            self.selected_pair_idx = idx;
            self.keyboard_cursor = None;
            self.dirty = true;
        }
        Ok(())
    }

    /// Returns `true` when visibility is `false` and the default axis pair (`0`) is selected.
    #[inline]
    #[must_use]
    pub fn has_default_persisted_state(&self) -> bool {
        !self.visible && self.selected_pair_idx == 0
    }

    /// Current coarse-to-fine stride (`8`, `4`, `2`, or `1`) of the latest uploaded snapshot.
    #[inline]
    #[must_use]
    pub fn current_stride(&self) -> usize {
        self.current_stride
    }

    /// Returns `true` once the `1x` pass for the latest requested generation has been uploaded.
    #[inline]
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.request_generation > 0 && self.completed_generation == self.request_generation
    }

    pub(super) fn lookup_or_evaluate_cell(&self, p1: f64, p2: f64) -> Option<AtlasCell> {
        let cartographer = self.active_cartographer.as_ref()?;
        if let Some(meta) = &self.latest_meta {
            if meta.generation == self.request_generation
                && meta.is_final
                && meta.width > 0
                && meta.height > 0
                && meta.cells.len() == meta.width * meta.height
            {
                let (sx, sy) = meta.axes.coords_to_pixel(p1, p2, meta.width, meta.height);
                return Some(meta.cells[flatten_2d_index(sx, sy, meta.width)]);
            }
        }
        let axes = cartographer.axes();
        Some(cartographer.evaluate_cell(axes.x.clamp(p1), axes.y.clamp(p2)))
    }

    /// Clamps `(p1, p2)` to the active [`super::AtlasAxisPair`] bounds, announces via WAI-ARIA
    /// live regions, and returns an [`AtlasTeleportEvent`].
    pub fn teleport_to(
        &mut self,
        ctx: Option<&egui::Context>,
        p1: f64,
        p2: f64,
    ) -> Option<AtlasTeleportEvent> {
        let cartographer = self.active_cartographer.as_ref()?;
        let axes = cartographer.axes();
        let x_val = axes.x.clamp(p1);
        let y_val = axes.y.clamp(p2);
        let cell = cartographer.evaluate_cell(x_val, y_val);
        let regime_label = cartographer.regime_label(cell.regime);
        self.keyboard_cursor = Some([x_val, y_val]);

        let msg = format!(
            "Parameter atlas teleported: {} = {:.3}, {} = {:.3} ({})",
            axes.x.label, x_val, axes.y.label, y_val, regime_label
        );
        if let Some(egui_ctx) = ctx {
            crate::accessibility::announce_live_status(egui_ctx, msg);
        } else {
            crate::accessibility::announce_status(&msg);
        }

        Some(AtlasTeleportEvent {
            x_key: axes.x.key,
            x_val,
            y_key: axes.y.key,
            y_val,
            regime_label,
        })
    }

    fn upload_snapshot(&mut self, ctx: &egui::Context, snapshot: &StateSnapshot) {
        if snapshot.width == 0 || snapshot.height == 0 {
            return;
        }
        let Ok(px) = snapshot.pixels.try_read() else {
            return;
        };
        if px.len() != snapshot.width.saturating_mul(snapshot.height) {
            return;
        }
        let image = egui::ColorImage::new([snapshot.width, snapshot.height], px.clone());
        let opts = egui::TextureOptions::NEAREST;
        if let Some(tex) = &mut self.texture {
            tex.set(image, opts);
        } else {
            self.texture = Some(ctx.load_texture("parameter_atlas_tex", image, opts));
        }
        if let Some(meta) = snapshot
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        {
            self.current_stride = meta.stride;
            if meta.is_final {
                self.completed_generation = meta.generation;
            }
            self.latest_meta = Some(meta.clone());
        }
    }

    fn dispatch_generation(&mut self, cartographer: Arc<dyn RegimeCartographer>) {
        self.request_generation = self.request_generation.wrapping_add(1).max(1);
        self.current_stride = ATLAS_STRIDES[0];
        let req = AtlasRenderRequest {
            generation: self.request_generation,
            width: self.raster_size.0,
            height: self.raster_size.1,
            high_contrast: self.high_contrast,
            cartographer: Arc::clone(&cartographer),
        };
        self.active_cartographer = Some(cartographer);

        if let Some(shared) = &self.shared_request {
            *shared.write().unwrap_or_else(|e| e.into_inner()) = req;
        } else {
            self.shared_request = Some(Arc::new(RwLock::new(req)));
        }
        self.latest_generation
            .store(self.request_generation, Ordering::Release);

        let shared_ref = Arc::clone(self.shared_request.as_ref().expect("initialized above"));
        let latest_ref = Arc::clone(&self.latest_generation);
        self.dispatch_platform_runner(shared_ref, latest_ref);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn dispatch_platform_runner(
        &mut self,
        shared_ref: Arc<RwLock<AtlasRenderRequest>>,
        latest_ref: Arc<AtomicU64>,
    ) {
        match &mut self.worker {
            None => {
                let mut w = BackgroundWorker::new_with_suspended(
                    AtlasRunner::new(shared_ref, latest_ref),
                    self.suspended,
                );
                let _ = w.send_command(SimCommand::Start);
                self.worker = Some(w);
            }
            Some(w) => match w.state() {
                SimWorkerState::Ready => {
                    let _ = w.send_command(SimCommand::Start);
                }
                SimWorkerState::Running => {
                    let _ = w.send_command(SimCommand::Custom("cancel_pass".to_string()));
                }
                SimWorkerState::Faulted | SimWorkerState::Terminated => {
                    w.replace_runner(AtlasRunner::new(shared_ref, latest_ref));
                    let _ = w.send_command(SimCommand::Start);
                }
                SimWorkerState::Uninitialized | SimWorkerState::Stepping => {}
            },
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    fn dispatch_platform_runner(
        &mut self,
        shared_ref: Arc<RwLock<AtlasRenderRequest>>,
        latest_ref: Arc<AtomicU64>,
    ) {
        match &mut self.wasm_runner {
            None => {
                self.wasm_runner = Some(AtlasRunner::new(shared_ref, latest_ref));
            }
            Some(runner) => {
                let _ = runner.sync_request();
            }
        }
    }

    pub(super) fn sync_and_poll<F>(
        &mut self,
        ctx: &egui::Context,
        context_sig: u64,
        high_contrast: bool,
        build_cartographer: F,
    ) where
        F: FnOnce((&'static str, &'static str)) -> Arc<dyn RegimeCartographer>,
    {
        if self.context_signature != Some(context_sig) || self.high_contrast != high_contrast {
            self.context_signature = Some(context_sig);
            self.high_contrast = high_contrast;
            self.dirty = true;
        }
        if self.dirty || self.active_cartographer.is_none() {
            self.dirty = false;
            let keys = self.axis_keys();
            let cartographer = build_cartographer(keys);
            self.dispatch_generation(cartographer);
        }
        self.poll_platform_runner(ctx);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn poll_platform_runner(&mut self, ctx: &egui::Context) {
        let new_snapshot = self
            .worker
            .as_mut()
            .and_then(|w| w.update().map(clone_snapshot_lightweight));
        if let Some(snap) = new_snapshot {
            self.upload_snapshot(ctx, &snap);
        } else if self.texture.is_none() {
            let seed = self
                .worker
                .as_ref()
                .and_then(|w| w.latest_snapshot().map(clone_snapshot_lightweight));
            if let Some(snap) = seed {
                self.upload_snapshot(ctx, &snap);
            }
        }

        if let Some(w) = &mut self.worker {
            if self.completed_generation != self.request_generation && w.state().is_ready() {
                let _ = w.send_command(SimCommand::Start);
            }
            if self.completed_generation != self.request_generation || w.is_running() {
                ctx.request_repaint();
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    // theory_verification!
    fn poll_platform_runner(&mut self, ctx: &egui::Context) {
        let mut snap_to_upload = None;
        if let Some(runner) = &mut self.wasm_runner {
            if !self.suspended && self.completed_generation != self.request_generation {
                let (budget, shared_with_workers) =
                    crate::async_sim::wasm_scheduler::allocate_inline_slice_budget();
                let slice_start = crate::web_io::PlatformInstant::now();
                runner.step_sliced(Some(budget));
                if shared_with_workers {
                    crate::async_sim::wasm_scheduler::record_worker_slice_elapsed(
                        slice_start.elapsed(),
                    );
                }
                snap_to_upload = Some(runner.get_snapshot());
            } else if self.texture.is_none() {
                snap_to_upload = Some(runner.get_snapshot());
            }
        }
        if let Some(snap) = snap_to_upload {
            self.upload_snapshot(ctx, &snap);
        }
        if !self.suspended && self.completed_generation != self.request_generation {
            ctx.request_repaint();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn clone_snapshot_lightweight(s: &StateSnapshot) -> StateSnapshot {
    let meta = s
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<AtlasPassMeta>())
        .cloned();
    StateSnapshot {
        width: s.width,
        height: s.height,
        pixels: Arc::clone(&s.pixels),
        custom_data: Vec::new(),
        structured_data: meta.map(|m| Box::new(m) as Box<dyn std::any::Any + Send>),
        sonification: s.sonification,
    }
}
