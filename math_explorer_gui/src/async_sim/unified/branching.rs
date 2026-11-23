//! Preallocated timeline ring-buffer and lockstep shadow-twin runtime for [`super::UnifiedSimRunner`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::UnifiedModel;
use crate::async_sim::StateSnapshot;
use crate::framework::{
    compute_slice_divergence, render_perceptual_difference_rgb, BranchDivergenceAccumulator,
    BranchDivergenceMetrics, RasterBranchViewMode, TimelineRingBuffer,
};
use eframe::egui::Color32;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock};

/// Preallocated reusable checkpoint slot for [`UnifiedModel`] state snapshots.
#[derive(Debug, Clone, Default)]
pub struct UnifiedCheckpointSlot {
    /// Fixed scalar registers (e.g. time, step, membrane voltage, gating variables).
    pub scalars: [f64; 8],
    /// Contiguous numerical field state (e.g. `u_prev` + `u_curr` or D2Q9 populations).
    pub field: Vec<f64>,
    /// Auxiliary obstacle or boundary mask bytes.
    pub mask: Vec<u8>,
}

impl UnifiedCheckpointSlot {
    /// Total heap + struct byte footprint used to enforce the `2 MiB` ring-buffer cap.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.field.capacity() * std::mem::size_of::<f64>()
            + self.mask.capacity()
    }
}

/// Thread-safe UI snapshot of timeline bounds, divergence metrics, and shadow overlays.
#[derive(Debug, Clone, Default)]
pub struct UnifiedBranchSharedState {
    /// Current step index $k$.
    pub current_step: usize,
    /// Earliest retained step in the ring buffer.
    pub min_step: usize,
    /// Latest reachable step in the ring buffer.
    pub max_step: usize,
    /// Current physical time $t_k$.
    pub current_time: f64,
    /// Whether a shadow-twin branch is currently forked.
    pub fork_active: bool,
    /// Fork step index $k$.
    pub fork_step: Option<usize>,
    /// Fork simulation time $t_k$.
    pub fork_time: Option<f64>,
    /// Live divergence metrics since $t_k$.
    pub divergence: BranchDivergenceMetrics,
    /// Active raster display mode when forked.
    pub view_mode: RasterBranchViewMode,
    /// Frozen baseline parameter summary at $t_k$.
    pub baseline_summary: String,
    /// Active counterfactual parameter summary.
    pub branch_summary: String,
    /// Interleaved `[t, y]` shadow curve samples for 1D curve `UnifiedModel`s.
    pub shadow_custom_data: Vec<f64>,
    /// Horizontal centerline `[x, u_primary]` profile for raster dataset exports.
    pub primary_profile: Vec<[f64; 2]>,
    /// Horizontal centerline `[x, u_shadow]` profile for raster dataset exports.
    pub shadow_profile: Vec<[f64; 2]>,
    /// Horizontal centerline `[x, |u_primary - u_shadow|]` profile for raster dataset exports.
    pub diff_profile: Vec<[f64; 2]>,
}

pub(crate) struct UnifiedShadowTwin<M: UnifiedModel> {
    pub(crate) model: M,
    pub(crate) baseline_params: HashMap<String, f64>,
    pub(crate) baseline_summary: String,
    pub(crate) fork_step: usize,
    pub(crate) fork_time: f64,
    pub(crate) fork_slot: UnifiedCheckpointSlot,
    pub(crate) divergence: BranchDivergenceAccumulator,
    pub(crate) primary_field_buf: Vec<f64>,
    pub(crate) shadow_field_buf: Vec<f64>,
    pub(crate) diff_rgb_buf: Vec<u8>,
}

/// Worker-owned timeline ring-buffer and shadow-twin coordinator inside [`super::UnifiedSimRunner`].
pub struct UnifiedBranchRuntime<M: UnifiedModel> {
    pub(crate) current_step: usize,
    pub(crate) current_time: f64,
    pub(crate) timeline: TimelineRingBuffer<UnifiedCheckpointSlot>,
    pub(crate) shadow: Option<UnifiedShadowTwin<M>>,
    pub(crate) view_mode: RasterBranchViewMode,
    pub(crate) shared: Arc<RwLock<UnifiedBranchSharedState>>,
}

impl<M: UnifiedModel> UnifiedBranchRuntime<M> {
    pub(crate) fn new(model: &M, shared: Arc<RwLock<UnifiedBranchSharedState>>) -> Self {
        let timeline = if M::supports_timeline_branch() {
            let template = model.checkpoint_template();
            let elem_bytes = template.estimated_bytes();
            let mut ring = TimelineRingBuffer::new_preallocated(
                elem_bytes,
                4096,
                M::branch_checkpoint_stride(),
                template,
            );
            ring.write_checkpoint_in_place(0, 0.0, |slot| model.write_branch_checkpoint(slot));
            ring
        } else {
            TimelineRingBuffer::new_default(64, 1, 1)
        };
        let rt = Self {
            current_step: 0,
            current_time: 0.0,
            timeline,
            shadow: None,
            view_mode: RasterBranchViewMode::CounterfactualBranch,
            shared,
        };
        rt.publish_shared_metadata(model, &HashMap::new());
        rt
    }

    pub(crate) fn reset(&mut self, model: &M, params: &HashMap<String, f64>) {
        self.current_step = 0;
        self.current_time = 0.0;
        self.shadow = None;
        self.view_mode = RasterBranchViewMode::CounterfactualBranch;
        self.timeline.clear();
        if M::supports_timeline_branch() {
            self.timeline
                .write_checkpoint_in_place(0, 0.0, |slot| model.write_branch_checkpoint(slot));
        }
        self.publish_shared_metadata(model, params);
    }

    /// Advances lockstep shadow twin + ring checkpointing with zero heap allocations in hot loops.
    pub(crate) fn on_step_in_place(&mut self, primary: &M, params: &HashMap<String, f64>) {
        if !M::supports_timeline_branch() {
            return;
        }
        let dt = primary.time_step_dt(params);
        self.current_step = self.current_step.saturating_add(1);
        self.current_time += dt;

        if let Some(ref mut twin) = self.shadow {
            twin.model.step(&twin.baseline_params);
            let n1 = primary.read_scalar_field(&mut twin.primary_field_buf);
            let n2 = twin.model.read_scalar_field(&mut twin.shadow_field_buf);
            let count = n1.min(n2);
            let (l2, peak) = if M::is_raster_model() && twin.diff_rgb_buf.len() >= count * 3 {
                render_perceptual_difference_rgb(
                    &twin.primary_field_buf[..count],
                    &twin.shadow_field_buf[..count],
                    &mut twin.diff_rgb_buf[..count * 3],
                )
            } else {
                compute_slice_divergence(
                    &twin.primary_field_buf[..count],
                    &twin.shadow_field_buf[..count],
                )
            };
            twin.divergence.record_step(dt, l2, peak);
        }

        if self.timeline.should_record_step(self.current_step) {
            self.timeline
                .write_checkpoint_in_place(self.current_step, self.current_time, |slot| {
                    primary.write_branch_checkpoint(slot);
                });
        }

        self.sync_shared_scalars_in_place();
    }

    fn sync_shared_scalars_in_place(&self) {
        if let Ok(mut guard) = self.shared.write() {
            let (min_s, max_s) = self
                .timeline
                .step_bounds()
                .unwrap_or((self.current_step, self.current_step));
            guard.current_step = self.current_step;
            guard.min_step = min_s;
            guard.max_step = max_s.max(self.current_step);
            guard.current_time = self.current_time;
            if let Some(ref twin) = self.shadow {
                guard.fork_active = true;
                guard.fork_step = Some(twin.fork_step);
                guard.fork_time = Some(twin.fork_time);
                guard.divergence = twin.divergence.metrics();
            } else {
                guard.fork_active = false;
                guard.fork_step = None;
                guard.fork_time = None;
                guard.divergence = BranchDivergenceMetrics::default();
            }
        }
    }

    pub(crate) fn fork_at_current(&mut self, primary: &M, params: &HashMap<String, f64>) {
        if !M::supports_timeline_branch() {
            return;
        }
        let Some(twin_model) = primary.fork_shadow_twin() else {
            return;
        };
        let field_len = primary.scalar_field_len();
        let mut fork_slot = primary.checkpoint_template();
        primary.write_branch_checkpoint(&mut fork_slot);
        let mut primary_field_buf = vec![0.0_f64; field_len];
        let mut shadow_field_buf = vec![0.0_f64; field_len];
        let mut diff_rgb_buf = vec![18_u8; field_len * 3];
        let count = primary
            .read_scalar_field(&mut primary_field_buf)
            .min(twin_model.read_scalar_field(&mut shadow_field_buf));
        if M::is_raster_model() && diff_rgb_buf.len() >= count * 3 {
            let _ = render_perceptual_difference_rgb(
                &primary_field_buf[..count],
                &shadow_field_buf[..count],
                &mut diff_rgb_buf[..count * 3],
            );
        }
        self.shadow = Some(UnifiedShadowTwin {
            model: twin_model,
            baseline_params: params.clone(),
            baseline_summary: summarize_params(params),
            fork_step: self.current_step,
            fork_time: self.current_time,
            fork_slot,
            divergence: BranchDivergenceAccumulator::default(),
            primary_field_buf,
            shadow_field_buf,
            diff_rgb_buf,
        });
        if M::is_raster_model() {
            self.view_mode = RasterBranchViewMode::DifferenceHeatmap;
        }
        self.publish_shared_metadata(primary, params);
    }

    pub(crate) fn clear_branch(&mut self, primary: &M, params: &HashMap<String, f64>) {
        self.shadow = None;
        self.view_mode = RasterBranchViewMode::CounterfactualBranch;
        self.publish_shared_metadata(primary, params);
    }

    pub(crate) fn scrub_to_step(
        &mut self,
        target_step: usize,
        primary: &mut M,
        params: &HashMap<String, f64>,
    ) {
        if !M::supports_timeline_branch() {
            return;
        }
        let Some((cp_step, cp_time, cp_slot)) = self
            .timeline
            .checkpoint_at_or_before(target_step)
            .map(|cp| (cp.step, cp.time, cp.state.clone()))
        else {
            return;
        };
        primary.restore_branch_checkpoint(&cp_slot, params);
        primary.truncate_history_to_step(cp_step, cp_time);
        self.timeline.truncate_after(target_step);
        self.current_step = cp_step;
        self.current_time = cp_time;

        let dt = primary.time_step_dt(params);
        while self.current_step < target_step {
            primary.step(params);
            self.current_step += 1;
            self.current_time += dt;
            if self.timeline.should_record_step(self.current_step) {
                self.timeline.write_checkpoint_in_place(
                    self.current_step,
                    self.current_time,
                    |slot| {
                        primary.write_branch_checkpoint(slot);
                    },
                );
            }
        }
        primary.truncate_history_to_step(self.current_step, self.current_time);
        self.resync_shadow_after_scrub(primary, params);
        self.publish_shared_metadata(primary, params);
    }

    pub(crate) fn sync_from_time_travel_restore(
        &mut self,
        primary: &M,
        params: &HashMap<String, f64>,
        step: usize,
    ) {
        if !M::supports_timeline_branch() {
            return;
        }
        self.current_step = step;
        self.current_time = step as f64 * primary.time_step_dt(params);
        self.timeline.truncate_after(step);
        self.resync_shadow_after_scrub(primary, params);
        self.sync_shared_scalars_in_place();
    }

    fn resync_shadow_after_scrub(&mut self, primary: &M, params: &HashMap<String, f64>) {
        let Some(ref mut twin) = self.shadow else {
            return;
        };
        if self.current_step < twin.fork_step {
            self.shadow = None;
            self.view_mode = RasterBranchViewMode::CounterfactualBranch;
            return;
        }
        twin.model
            .restore_branch_checkpoint(&twin.fork_slot, &twin.baseline_params);
        twin.model
            .truncate_history_to_step(twin.fork_step, twin.fork_time);
        twin.divergence.reset();
        let dt = primary.time_step_dt(params);
        for _ in twin.fork_step..self.current_step {
            twin.model.step(&twin.baseline_params);
            let n = primary
                .read_scalar_field(&mut twin.primary_field_buf)
                .min(twin.model.read_scalar_field(&mut twin.shadow_field_buf));
            let (l2, peak) =
                compute_slice_divergence(&twin.primary_field_buf[..n], &twin.shadow_field_buf[..n]);
            twin.divergence.record_step(dt, l2, peak);
        }
        if M::is_raster_model() {
            let n = primary
                .read_scalar_field(&mut twin.primary_field_buf)
                .min(twin.model.read_scalar_field(&mut twin.shadow_field_buf));
            if twin.diff_rgb_buf.len() >= n * 3 {
                let _ = render_perceptual_difference_rgb(
                    &twin.primary_field_buf[..n],
                    &twin.shadow_field_buf[..n],
                    &mut twin.diff_rgb_buf[..n * 3],
                );
            }
        }
    }

    pub(crate) fn step_back(&mut self, primary: &mut M, params: &HashMap<String, f64>) {
        if self.current_step == 0 {
            return;
        }
        let prev = self
            .timeline
            .previous_step_before(self.current_step)
            .unwrap_or_else(|| self.current_step.saturating_sub(1));
        self.scrub_to_step(prev, primary, params);
    }

    pub(crate) fn decorate_snapshot(
        &self,
        primary: &M,
        params: &HashMap<String, f64>,
        mut snap: StateSnapshot,
    ) -> StateSnapshot {
        self.publish_shared_metadata(primary, params);
        if let Some(ref twin) = self.shadow {
            if M::is_raster_model() {
                match self.view_mode {
                    RasterBranchViewMode::CounterfactualBranch => {}
                    RasterBranchViewMode::ShadowBaseline => {
                        let shadow_snap = twin.model.get_snapshot();
                        snap.pixels = shadow_snap.pixels;
                    }
                    RasterBranchViewMode::DifferenceHeatmap => {
                        let px_count = snap.width * snap.height;
                        if twin.diff_rgb_buf.len() >= px_count * 3 {
                            let mut diff_px = Vec::with_capacity(px_count);
                            for idx in 0..px_count {
                                let b = idx * 3;
                                diff_px.push(Color32::from_rgb(
                                    twin.diff_rgb_buf[b],
                                    twin.diff_rgb_buf[b + 1],
                                    twin.diff_rgb_buf[b + 2],
                                ));
                            }
                            snap.pixels = Arc::new(RwLock::new(diff_px));
                        }
                    }
                }
            }
        }
        snap
    }

    pub(crate) fn publish_shared_metadata(&self, primary: &M, params: &HashMap<String, f64>) {
        let Ok(mut guard) = self.shared.write() else {
            return;
        };
        let (min_s, max_s) = self
            .timeline
            .step_bounds()
            .unwrap_or((self.current_step, self.current_step));
        guard.current_step = self.current_step;
        guard.min_step = min_s;
        guard.max_step = max_s.max(self.current_step);
        guard.current_time = self.current_time;
        guard.view_mode = self.view_mode;
        guard.branch_summary = summarize_params(params);

        if let Some(ref twin) = self.shadow {
            guard.fork_active = true;
            guard.fork_step = Some(twin.fork_step);
            guard.fork_time = Some(twin.fork_time);
            guard.divergence = twin.divergence.metrics();
            guard.baseline_summary = twin.baseline_summary.clone();
            if M::is_raster_model() {
                populate_centerline_profiles(primary, twin, &mut guard);
                guard.shadow_custom_data.clear();
            } else {
                guard.shadow_custom_data = twin.model.get_snapshot().custom_data;
                guard.primary_profile.clear();
                guard.shadow_profile.clear();
                guard.diff_profile.clear();
            }
        } else {
            guard.fork_active = false;
            guard.fork_step = None;
            guard.fork_time = None;
            guard.divergence = BranchDivergenceMetrics::default();
            guard.baseline_summary.clear();
            guard.shadow_custom_data.clear();
            guard.primary_profile.clear();
            guard.shadow_profile.clear();
            guard.diff_profile.clear();
        }
    }
}

fn populate_centerline_profiles<M: UnifiedModel>(
    primary: &M,
    twin: &UnifiedShadowTwin<M>,
    out: &mut UnifiedBranchSharedState,
) {
    let snap = primary.get_snapshot();
    let w = snap.width;
    let h = snap.height;
    out.primary_profile.clear();
    out.shadow_profile.clear();
    out.diff_profile.clear();
    if w == 0 || h == 0 {
        return;
    }
    let mid_row = (h / 2) * w;
    let available = twin
        .primary_field_buf
        .len()
        .min(twin.shadow_field_buf.len());
    for x in 0..w {
        let idx = mid_row + x;
        if idx < available {
            let pv = twin.primary_field_buf[idx];
            let sv = twin.shadow_field_buf[idx];
            out.primary_profile.push([x as f64, pv]);
            out.shadow_profile.push([x as f64, sv]);
            out.diff_profile.push([x as f64, (pv - sv).abs()]);
        }
    }
}

pub(crate) fn summarize_params(params: &HashMap<String, f64>) -> String {
    let sorted: BTreeMap<&String, &f64> = params.iter().collect();
    sorted
        .into_iter()
        .map(|(k, v)| format!("{k}={v:.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}
