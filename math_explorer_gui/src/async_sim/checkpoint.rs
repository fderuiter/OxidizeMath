//! Bounded time-travel keyframe ring buffer and deterministic sub-step replay.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimulationRunner;
use oxidize_core::rng::OxidizeRng;
use std::collections::{HashMap, VecDeque};

/// Hard `2 MiB` (`2,097,152` bytes) cap on retained time-travel keyframes and epochs per worker.
pub const MAX_CHECKPOINT_RING_BYTES: usize = crate::framework::MAX_TOOL_STATE_BYTES;

/// Maximum inline scalar slots stored without heap allocation in a checkpoint.
pub const MAX_INLINE_CHECKPOINT_SCALARS: usize = 8;

/// Snapshot of the worker's scrubbable timeline bounds exposed to UI transport controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineMetadata {
    /// Current deterministic step index of the simulation runner.
    pub current_step: usize,
    /// Oldest retained keyframe step index still available in the bounded ring.
    pub min_step: usize,
    /// Newest reachable step index prior to forward-resumption truncation.
    pub max_step: usize,
    /// Keyframe capture stride `K` (`1` for compact ODEs, `16` for 2D grids).
    pub stride: usize,
}

impl TimelineMetadata {
    /// Computes the target step when jumping backward by one keyframe stride (`[`).
    #[must_use]
    pub fn previous_keyframe_step(&self) -> usize {
        if self.current_step <= self.min_step {
            return self.min_step;
        }
        let k = self.stride.max(1);
        let prev = ((self.current_step - 1) / k) * k;
        prev.max(self.min_step)
    }
}

/// Dynamical field state captured inside a [`TimeTravelCheckpoint`].
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointPayload {
    /// Fixed-capacity inline scalar state vector (zero heap allocation).
    Scalars {
        /// Inline scalar buffer.
        values: [f64; MAX_INLINE_CHECKPOINT_SCALARS],
        /// Number of active scalar entries in `0..=MAX_INLINE_CHECKPOINT_SCALARS`.
        len: usize,
    },
    /// Two-component 2D spatial concentration grid (`u`, `v`) with exact slice capacity.
    DualField2D {
        /// Grid width in cells.
        width: usize,
        /// Grid height in cells.
        height: usize,
        /// Primary concentration field `u`.
        u: Box<[f64]>,
        /// Secondary concentration field `v`.
        v: Box<[f64]>,
    },
}

impl CheckpointPayload {
    /// Packs up to [`MAX_INLINE_CHECKPOINT_SCALARS`] `f64` values on the stack without heap allocation.
    #[must_use]
    pub fn scalars(slice: &[f64]) -> Self {
        let mut values = [0.0; MAX_INLINE_CHECKPOINT_SCALARS];
        let len = slice.len().min(MAX_INLINE_CHECKPOINT_SCALARS);
        values[..len].copy_from_slice(&slice[..len]);
        Self::Scalars { values, len }
    }

    /// Returns the inline scalar slice when this payload is [`Self::Scalars`].
    #[must_use]
    pub fn as_scalars(&self) -> Option<&[f64]> {
        match self {
            Self::Scalars { values, len } => Some(&values[..*len]),
            Self::DualField2D { .. } => None,
        }
    }

    /// Dynamic heap bytes owned by this payload.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::Scalars { .. } => 0,
            Self::DualField2D { u, v, .. } => {
                std::mem::size_of_val(&**u) + std::mem::size_of_val(&**v)
            }
        }
    }
}

/// Active model parameters captured at a keyframe or mid-run parameter epoch.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointParams {
    /// Fixed-capacity inline parameter tuple (zero heap allocation).
    Scalars {
        /// Inline parameter values.
        values: [f64; MAX_INLINE_CHECKPOINT_SCALARS],
        /// Number of active parameter entries.
        len: usize,
    },
    /// Named parameter map for dynamic `UnifiedModel` implementations.
    NamedMap(HashMap<String, f64>),
}

impl CheckpointParams {
    /// Packs up to [`MAX_INLINE_CHECKPOINT_SCALARS`] parameter scalars on the stack.
    #[must_use]
    pub fn scalars(slice: &[f64]) -> Self {
        let mut values = [0.0; MAX_INLINE_CHECKPOINT_SCALARS];
        let len = slice.len().min(MAX_INLINE_CHECKPOINT_SCALARS);
        values[..len].copy_from_slice(&slice[..len]);
        Self::Scalars { values, len }
    }

    /// Returns the inline parameter slice when stored as [`Self::Scalars`].
    #[must_use]
    pub fn as_scalars(&self) -> Option<&[f64]> {
        match self {
            Self::Scalars { values, len } => Some(&values[..*len]),
            Self::NamedMap(_) => None,
        }
    }

    /// Returns the named parameter map when stored as [`Self::NamedMap`].
    #[must_use]
    pub fn as_named_map(&self) -> Option<&HashMap<String, f64>> {
        match self {
            Self::NamedMap(map) => Some(map),
            Self::Scalars { .. } => None,
        }
    }

    /// Dynamic heap bytes owned by this parameter snapshot.
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::Scalars { .. } => 0,
            Self::NamedMap(map) => {
                let entry_stride = std::mem::size_of::<String>()
                    + std::mem::size_of::<f64>()
                    + std::mem::size_of::<u64>();
                let key_bytes: usize = map.keys().map(String::capacity).sum();
                map.capacity() * entry_stride + key_bytes
            }
        }
    }
}

/// Self-contained deterministic checkpoint preserving state fields, parameters, and RNG stream.
#[derive(Debug, Clone)]
pub struct TimeTravelCheckpoint {
    /// Exact step index at which this checkpoint was captured.
    pub step: usize,
    /// Cloned deterministic PRNG state at `step`.
    pub rng: OxidizeRng,
    /// Active simulation parameters at `step`.
    pub params: CheckpointParams,
    /// Dynamical field payload at `step`.
    pub state: CheckpointPayload,
}

impl TimeTravelCheckpoint {
    /// Total inline + heap memory footprint of this checkpoint in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.state.heap_bytes() + self.params.heap_bytes()
    }
}

/// Recorded parameter configuration effective starting at `start_step`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterEpoch {
    /// First step index (`start_step -> start_step + 1`) that uses `params`.
    pub start_step: usize,
    /// Active simulation parameters for this epoch.
    pub params: CheckpointParams,
}

impl ParameterEpoch {
    #[must_use]
    fn byte_size(&self) -> usize {
        std::mem::size_of::<Self>() + self.params.heap_bytes()
    }
}

/// Bounded `2 MiB` ring buffer of periodic keyframes and mid-run parameter epochs.
pub struct CheckpointRing {
    stride: usize,
    current_step: usize,
    max_reached_step: usize,
    keyframes: VecDeque<TimeTravelCheckpoint>,
    epochs: VecDeque<ParameterEpoch>,
    keyframe_bytes: usize,
    epoch_bytes: usize,
    max_bytes: usize,
}

impl CheckpointRing {
    /// Creates a new bounded ring with keyframe stride `stride` and [`MAX_CHECKPOINT_RING_BYTES`].
    #[must_use]
    pub fn new(stride: usize) -> Self {
        Self::with_max_bytes(stride, MAX_CHECKPOINT_RING_BYTES)
    }

    /// Creates a bounded ring with a custom byte cap (clamped to `<= MAX_CHECKPOINT_RING_BYTES`).
    #[must_use]
    pub fn with_max_bytes(stride: usize, max_bytes: usize) -> Self {
        Self {
            stride: stride.max(1),
            current_step: 0,
            max_reached_step: 0,
            keyframes: VecDeque::new(),
            epochs: VecDeque::new(),
            keyframe_bytes: 0,
            epoch_bytes: 0,
            max_bytes: max_bytes.min(MAX_CHECKPOINT_RING_BYTES),
        }
    }

    /// Seeds or resets the ring from `runner`'s current baseline state.
    pub fn reset_and_seed<R: SimulationRunner + ?Sized>(&mut self, runner: &R) {
        self.keyframes.clear();
        self.epochs.clear();
        self.keyframes.shrink_to(64);
        self.epochs.shrink_to(16);
        self.keyframe_bytes = 0;
        self.epoch_bytes = 0;
        let step = runner.current_step();
        self.current_step = step;
        self.max_reached_step = step;
        if let Some(params) = runner.capture_active_params() {
            self.push_epoch(ParameterEpoch {
                start_step: step,
                params,
            });
        }
        if let Some(cp) = runner.capture_checkpoint() {
            self.push_keyframe(cp);
        }
    }

    /// Total bytes currently used by retained keyframes and parameter epochs.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.keyframe_bytes + self.epoch_bytes
    }

    /// Number of retained keyframes currently stored in the ring.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Current timeline metadata snapshot for UI transport controls.
    #[must_use]
    pub fn metadata(&self) -> TimelineMetadata {
        let min_step = self
            .keyframes
            .front()
            .map_or(self.current_step, |kf| kf.step);
        let max_step = self.max_reached_step.max(min_step);
        TimelineMetadata {
            current_step: self.current_step.clamp(min_step, max_step),
            min_step,
            max_step,
            stride: self.stride,
        }
    }

    /// Truncates future keyframes and parameter epochs strictly `> current_step` when resuming forward stepping.
    pub fn prepare_forward_advance(&mut self) {
        if self.current_step >= self.max_reached_step {
            return;
        }
        while let Some(back) = self.keyframes.back() {
            if back.step > self.current_step {
                let removed = self.keyframes.pop_back().expect("checked back");
                self.keyframe_bytes = self.keyframe_bytes.saturating_sub(removed.byte_size());
            } else {
                break;
            }
        }
        while let Some(back) = self.epochs.back() {
            if back.start_step > self.current_step {
                let removed = self.epochs.pop_back().expect("checked back");
                self.epoch_bytes = self.epoch_bytes.saturating_sub(removed.byte_size());
            } else {
                break;
            }
        }
        self.max_reached_step = self.current_step;
    }

    /// Records a parameter epoch at `current_step` if `runner`'s active parameters changed.
    pub fn record_active_params<R: SimulationRunner + ?Sized>(&mut self, runner: &R) {
        let Some(params) = runner.capture_active_params() else {
            return;
        };
        if let Some(last) = self.epochs.back_mut() {
            if last.params == params {
                return;
            }
            if last.start_step == self.current_step {
                self.epoch_bytes = self.epoch_bytes.saturating_sub(last.byte_size());
                last.params = params;
                self.epoch_bytes += last.byte_size();
                self.evict_to_fit(0);
                return;
            }
        }
        self.push_epoch(ParameterEpoch {
            start_step: self.current_step,
            params,
        });
    }

    /// Updates step counters after one forward sub-step and records a keyframe when `step % stride == 0`.
    pub fn on_substep_advanced<R: SimulationRunner + ?Sized>(&mut self, runner: &R) {
        self.current_step = runner.current_step();
        if self.current_step > self.max_reached_step {
            self.max_reached_step = self.current_step;
        }
        if self.current_step.is_multiple_of(self.stride) {
            if let Some(cp) = runner.capture_checkpoint() {
                self.push_keyframe(cp);
            }
        }
    }

    /// Restores the newest keyframe `<= target_step` and deterministically replays sub-steps to `target_step`
    /// without truncating future keyframes.
    pub fn seek_to_step<R: SimulationRunner + ?Sized>(
        &mut self,
        runner: &mut R,
        target_step: usize,
    ) -> usize {
        if self.keyframes.is_empty() {
            return self.current_step;
        }
        let meta = self.metadata();
        let clamped_target = target_step.clamp(meta.min_step, meta.max_step);
        let kf_idx = self.find_keyframe_index_at_or_before(clamped_target);
        let kf_step = self.keyframes[kf_idx].step;

        runner.restore_checkpoint(&self.keyframes[kf_idx]);
        let (front, back) = self.prefix_slices_inclusive(kf_idx);
        runner.rebuild_history(front, back);

        self.replay_substeps(runner, kf_step, clamped_target);
        self.current_step = clamped_target;
        clamped_target
    }

    fn replay_substeps<R: SimulationRunner + ?Sized>(
        &self,
        runner: &mut R,
        from_step: usize,
        to_step: usize,
    ) {
        if from_step >= to_step {
            return;
        }
        let fallback_params = &self.keyframes[0].params;
        let mut epoch_idx = 0usize;
        for step in from_step..to_step {
            while epoch_idx + 1 < self.epochs.len() && self.epochs[epoch_idx + 1].start_step <= step
            {
                epoch_idx += 1;
            }
            let params = self
                .epochs
                .get(epoch_idx)
                .map_or(fallback_params, |ep| &ep.params);
            runner.replay_step(params);
        }
    }

    fn find_keyframe_index_at_or_before(&self, target_step: usize) -> usize {
        let mut best_idx = 0usize;
        for (idx, kf) in self.keyframes.iter().enumerate() {
            if kf.step <= target_step {
                best_idx = idx;
            } else {
                break;
            }
        }
        best_idx
    }

    fn prefix_slices_inclusive(
        &self,
        kf_idx: usize,
    ) -> (&[TimeTravelCheckpoint], &[TimeTravelCheckpoint]) {
        let (front, back) = self.keyframes.as_slices();
        let count = kf_idx + 1;
        if count <= front.len() {
            (&front[..count], &[])
        } else {
            (front, &back[..(count - front.len())])
        }
    }

    fn push_keyframe(&mut self, cp: TimeTravelCheckpoint) {
        let cp_bytes = cp.byte_size();
        if cp_bytes > self.max_bytes {
            return;
        }
        self.evict_to_fit(cp_bytes);
        if self.total_bytes() + cp_bytes > self.max_bytes {
            return;
        }
        self.keyframe_bytes += cp_bytes;
        self.keyframes.push_back(cp);
    }

    fn push_epoch(&mut self, epoch: ParameterEpoch) {
        let ep_bytes = epoch.byte_size();
        if ep_bytes > self.max_bytes {
            return;
        }
        self.evict_to_fit(ep_bytes);
        if self.total_bytes() + ep_bytes > self.max_bytes {
            return;
        }
        self.epoch_bytes += ep_bytes;
        self.epochs.push_back(epoch);
    }

    fn evict_to_fit(&mut self, incoming_bytes: usize) {
        while self.total_bytes() + incoming_bytes > self.max_bytes && !self.keyframes.is_empty() {
            let removed = self.keyframes.pop_front().expect("non-empty keyframes");
            self.keyframe_bytes = self.keyframe_bytes.saturating_sub(removed.byte_size());
            if let Some(new_front) = self.keyframes.front() {
                self.prune_epochs_before(new_front.step);
            }
        }
        while self.total_bytes() + incoming_bytes > self.max_bytes && self.epochs.len() > 1 {
            let removed = self.epochs.pop_front().expect("epochs len > 1");
            self.epoch_bytes = self.epoch_bytes.saturating_sub(removed.byte_size());
        }
    }

    fn prune_epochs_before(&mut self, min_step: usize) {
        while self.epochs.len() >= 2 && self.epochs[1].start_step <= min_step {
            let removed = self.epochs.pop_front().expect("at least 2 epochs");
            self.epoch_bytes = self.epoch_bytes.saturating_sub(removed.byte_size());
        }
        if let Some(front) = self.epochs.front_mut() {
            if front.start_step < min_step {
                front.start_step = min_step;
            }
        }
    }
}
