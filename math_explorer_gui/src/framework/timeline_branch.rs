//! Fixed-capacity timeline ring buffer and zero-allocation shadow-twin divergence kernel.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "timeline_branch/exports.rs"]
pub mod exports;
#[path = "timeline_branch/hud.rs"]
pub mod hud;
#[cfg(test)]
#[path = "timeline_branch/tests.rs"]
mod tests;

pub use exports::{
    build_branch_metadata_dataset_2d, build_branch_metadata_dataset_3d,
    build_dashed_export_primitives, build_fork_vline_primitive, inject_png_branch_metadata,
    inject_svg_branch_metadata, render_perceptual_difference_rgb, RasterBranchViewMode,
};
pub use hud::{
    announce_branch_cleared, announce_branch_forked, BranchDivergenceHud, BranchMetadata,
    TimelineBranchAction, TimelineTransportState,
};

/// Hard per-tool memory ceiling (`2 MiB`) for retained timeline checkpoints.
pub const MAX_TIMELINE_RING_BYTES: usize = 2 * 1024 * 1024;

/// Timestamped state checkpoint recorded inside [`TimelineRingBuffer`].
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineCheckpoint<S> {
    /// Monotonic simulation step index $k$.
    pub step: usize,
    /// Physical or normalized simulation time $t_k$.
    pub time: f64,
    /// Captured model state at step $k$.
    pub state: S,
}

/// Fixed-capacity ring buffer capped strictly at [`MAX_TIMELINE_RING_BYTES`].
///
/// Preallocates all checkpoint slots up front so pushing, overwriting, and truncating
/// checkpoints inside simulation loops never reallocate the backing vector.
#[derive(Debug, Clone)]
pub struct TimelineRingBuffer<S> {
    slots: Vec<TimelineCheckpoint<S>>,
    valid: Vec<bool>,
    start: usize,
    len: usize,
    slot_bytes: usize,
    stride: usize,
}

impl<S: Clone> TimelineRingBuffer<S> {
    /// Creates a preallocated ring buffer bounded by `MAX_TIMELINE_RING_BYTES` using `template`.
    #[must_use]
    pub fn new_preallocated(
        element_bytes: usize,
        requested_slots: usize,
        stride: usize,
        template: S,
    ) -> Self {
        Self::new_in_place_with(element_bytes, requested_slots, stride, || template.clone())
    }
}

impl<S: Default> TimelineRingBuffer<S> {
    /// Creates a preallocated ring buffer bounded by `MAX_TIMELINE_RING_BYTES` using `S::default`.
    #[must_use]
    pub fn new_default(element_bytes: usize, requested_slots: usize, stride: usize) -> Self {
        Self::new_in_place_with(element_bytes, requested_slots, stride, S::default)
    }
}

impl<S> TimelineRingBuffer<S> {
    /// Creates a preallocated ring buffer whose total retained byte budget never exceeds `2 MiB`.
    #[must_use]
    pub fn new_in_place_with<F>(
        element_bytes: usize,
        requested_slots: usize,
        stride: usize,
        mut init_slot: F,
    ) -> Self
    where
        F: FnMut() -> S,
    {
        let base_bytes = std::mem::size_of::<TimelineCheckpoint<S>>() + std::mem::size_of::<bool>();
        let effective_slot_bytes = element_bytes.max(base_bytes).max(1);
        let max_by_memory = (MAX_TIMELINE_RING_BYTES / effective_slot_bytes).max(1);
        let cap = requested_slots.max(1).min(max_by_memory);
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(TimelineCheckpoint {
                step: 0,
                time: 0.0,
                state: init_slot(),
            });
        }
        Self {
            slots,
            valid: vec![false; cap],
            start: 0,
            len: 0,
            slot_bytes: effective_slot_bytes,
            stride: stride.max(1),
        }
    }

    /// Maximum number of checkpoints retained in the ring.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Number of active checkpoints currently stored.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no checkpoints have been recorded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Periodic step interval between recorded checkpoints (`>= 1`).
    #[inline]
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Total memory budget consumed by preallocated checkpoint slots (`<= MAX_TIMELINE_RING_BYTES`).
    #[inline]
    #[must_use]
    pub fn memory_budget_bytes(&self) -> usize {
        self.slots.len().saturating_mul(self.slot_bytes)
    }

    /// Returns `true` if `step` aligns with this buffer's periodic checkpoint stride.
    #[inline]
    #[must_use]
    pub fn should_record_step(&self, step: usize) -> bool {
        step.is_multiple_of(self.stride)
    }

    /// Clears all checkpoints without deallocating preallocated slot storage.
    pub fn clear(&mut self) {
        self.valid.fill(false);
        self.start = 0;
        self.len = 0;
    }

    /// Records a checkpoint by value into the next ring slot without heap reallocation.
    pub fn push_checkpoint(&mut self, step: usize, time: f64, state: S) {
        self.write_checkpoint_in_place(step, time, move |dst| {
            *dst = state;
        });
    }

    /// Mutates the target ring slot in place so `Vec`-backed states avoid heap allocations.
    pub fn write_checkpoint_in_place<F>(&mut self, step: usize, time: f64, write_fn: F)
    where
        F: FnOnce(&mut S),
    {
        let cap = self.slots.len();
        if cap == 0 {
            return;
        }
        if self.len > 0 {
            let last_idx = (self.start + self.len - 1) % cap;
            if self.valid[last_idx] && self.slots[last_idx].step == step {
                self.slots[last_idx].time = time;
                write_fn(&mut self.slots[last_idx].state);
                return;
            }
        }
        let target_idx = if self.len < cap {
            let idx = (self.start + self.len) % cap;
            self.len += 1;
            idx
        } else {
            let idx = self.start;
            self.start = (self.start + 1) % cap;
            idx
        };
        self.slots[target_idx].step = step;
        self.slots[target_idx].time = time;
        write_fn(&mut self.slots[target_idx].state);
        self.valid[target_idx] = true;
    }

    #[inline]
    fn ring_index(&self, offset: usize) -> usize {
        (self.start + offset) % self.slots.len()
    }

    /// Returns `(min_step, max_step)` across retained checkpoints, if non-empty.
    #[must_use]
    pub fn step_bounds(&self) -> Option<(usize, usize)> {
        if self.len == 0 {
            return None;
        }
        let first = &self.slots[self.ring_index(0)];
        let last = &self.slots[self.ring_index(self.len - 1)];
        Some((first.step, last.step))
    }

    /// Returns the most recently recorded checkpoint.
    #[must_use]
    pub fn latest(&self) -> Option<&TimelineCheckpoint<S>> {
        if self.len == 0 {
            None
        } else {
            Some(&self.slots[self.ring_index(self.len - 1)])
        }
    }

    /// Returns the oldest retained checkpoint.
    #[must_use]
    pub fn earliest(&self) -> Option<&TimelineCheckpoint<S>> {
        if self.len == 0 {
            None
        } else {
            Some(&self.slots[self.ring_index(0)])
        }
    }

    /// Looks up the checkpoint matching `target_step` exactly.
    #[must_use]
    pub fn checkpoint_exact(&self, target_step: usize) -> Option<&TimelineCheckpoint<S>> {
        for offset in (0..self.len).rev() {
            let cp = &self.slots[self.ring_index(offset)];
            if cp.step == target_step {
                return Some(cp);
            }
            if cp.step < target_step {
                break;
            }
        }
        None
    }

    /// Looks up the latest checkpoint at or before `target_step`.
    #[must_use]
    pub fn checkpoint_at_or_before(&self, target_step: usize) -> Option<&TimelineCheckpoint<S>> {
        for offset in (0..self.len).rev() {
            let cp = &self.slots[self.ring_index(offset)];
            if cp.step <= target_step {
                return Some(cp);
            }
        }
        None
    }

    /// Returns the step index of the checkpoint immediately preceding `current_step`, if any.
    #[must_use]
    pub fn previous_step_before(&self, current_step: usize) -> Option<usize> {
        if current_step == 0 || self.len == 0 {
            return None;
        }
        for offset in (0..self.len).rev() {
            let cp = &self.slots[self.ring_index(offset)];
            if cp.step < current_step {
                return Some(cp.step);
            }
        }
        None
    }

    /// Drops all checkpoints strictly after `max_step` in $O(N)$ without reallocating.
    pub fn truncate_after(&mut self, max_step: usize) {
        while self.len > 0 {
            let tail_idx = self.ring_index(self.len - 1);
            if self.slots[tail_idx].step > max_step {
                self.valid[tail_idx] = false;
                self.len -= 1;
            } else {
                break;
            }
        }
        if self.len == 0 {
            self.start = 0;
        }
    }
}

/// Live divergence summary between a counterfactual branch and its unperturbed shadow twin.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BranchDivergenceMetrics {
    /// Number of lockstep steps advanced since forking at $t_k$.
    pub steps_since_fork: usize,
    /// Elapsed simulation time $t - t_k$ since forking.
    pub elapsed_since_fork: f64,
    /// Instantaneous Euclidean $L_2$ distance $\|x_{\text{branch}}(t) - x_{\text{shadow}}(t)\|_2$.
    pub instantaneous_l2: f64,
    /// Peak element-wise absolute delta $\max_{\tau \in [t_k, t]} \|x_{\text{branch}}(\tau) - x_{\text{shadow}}(\tau)\|_\infty$.
    pub peak_abs_delta: f64,
    /// Cumulative trapezoidal area delta $\int_{t_k}^{t} \|x_{\text{branch}}(\tau) - x_{\text{shadow}}(\tau)\|_2 \, d\tau$.
    pub cumulative_area_delta: f64,
    /// Empirical log-divergence slope $\lambda_{\text{emp}}$ fitted over $\ln \| \Delta x(t) \|_2$ vs $t - t_k$.
    pub log_divergence_slope: f64,
}

/// Zero-allocation streaming accumulator for [`BranchDivergenceMetrics`].
#[derive(Debug, Clone, Copy, Default)]
pub struct BranchDivergenceAccumulator {
    metrics: BranchDivergenceMetrics,
    prev_l2: f64,
    pos_samples: usize,
    sum_t: f64,
    sum_ln_l2: f64,
    sum_tt: f64,
    sum_t_ln_l2: f64,
}

impl BranchDivergenceAccumulator {
    /// Resets accumulated divergence back to zero at fork initialization.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Returns the latest snapshot of [`BranchDivergenceMetrics`].
    #[inline]
    #[must_use]
    pub const fn metrics(&self) -> BranchDivergenceMetrics {
        self.metrics
    }

    /// Updates all four divergence metrics for one lockstep step with zero heap allocations.
    #[inline]
    pub fn record_step(
        &mut self,
        dt: f64,
        l2_distance: f64,
        max_abs_delta: f64,
    ) -> BranchDivergenceMetrics {
        let step_dt = if dt.is_finite() && dt > 0.0 { dt } else { 1.0 };
        let l2 = if l2_distance.is_finite() && l2_distance > 0.0 {
            l2_distance
        } else {
            0.0
        };
        let abs_peak = if max_abs_delta.is_finite() && max_abs_delta > 0.0 {
            max_abs_delta
        } else {
            0.0
        };

        self.metrics.steps_since_fork = self.metrics.steps_since_fork.saturating_add(1);
        self.metrics.elapsed_since_fork += step_dt;
        self.metrics.instantaneous_l2 = l2;
        if abs_peak > self.metrics.peak_abs_delta {
            self.metrics.peak_abs_delta = abs_peak;
        }
        self.metrics.cumulative_area_delta += 0.5 * (self.prev_l2 + l2) * step_dt;
        self.prev_l2 = l2;

        if l2 > 1.0e-15 {
            let t = self.metrics.elapsed_since_fork;
            let y = l2.ln();
            self.pos_samples = self.pos_samples.saturating_add(1);
            self.sum_t += t;
            self.sum_ln_l2 += y;
            self.sum_tt += t * t;
            self.sum_t_ln_l2 += t * y;
            if self.pos_samples >= 2 {
                let n = self.pos_samples as f64;
                let denom = n * self.sum_tt - self.sum_t * self.sum_t;
                if denom.abs() > 1.0e-18 {
                    let slope = (n * self.sum_t_ln_l2 - self.sum_t * self.sum_ln_l2) / denom;
                    if slope.is_finite() {
                        self.metrics.log_divergence_slope = slope;
                    }
                }
            }
        } else if self.pos_samples == 0 {
            self.metrics.log_divergence_slope = 0.0;
        }

        self.metrics
    }
}

/// Computes `(l2_distance, max_abs_delta)` across two equal-length scalar slices with zero heap allocations.
#[inline]
#[must_use]
pub fn compute_slice_divergence(primary: &[f64], shadow: &[f64]) -> (f64, f64) {
    let n = primary.len().min(shadow.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut sum_sq = 0.0_f64;
    let mut peak = 0.0_f64;
    for idx in 0..n {
        let diff = (primary[idx] - shadow[idx]).abs();
        if diff.is_finite() {
            sum_sq += diff * diff;
            if diff > peak {
                peak = diff;
            }
        }
    }
    (sum_sq.sqrt(), peak)
}
