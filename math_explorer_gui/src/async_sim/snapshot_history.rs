//! Bounded UI-side `VecDeque<StateSnapshot>` ring buffer for opted-in `SimulationController` time travel.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{SimulationRunner, StateSnapshot, TimelineMetadata};
use std::collections::VecDeque;

pub(super) fn initial_runner_timeline<T: SimulationRunner + ?Sized>(
    runner: &T,
) -> Option<TimelineMetadata> {
    let stride = runner.checkpoint_stride()?.max(1);
    let step = runner.current_step();
    Some(TimelineMetadata {
        current_step: step,
        min_step: step,
        max_step: step,
        stride,
    })
}

pub(super) fn initial_snapshot_history(
    timeline: Option<TimelineMetadata>,
    snapshot: &StateSnapshot,
) -> Option<SnapshotHistory> {
    let meta = timeline?;
    let mut hist = SnapshotHistory::default();
    hist.record(meta.current_step, snapshot);
    Some(hist)
}

/// Default maximum number of UI `StateSnapshot` frames retained in [`SnapshotHistory`].
pub const DEFAULT_MAX_SNAPSHOT_FRAMES: usize = 512;

/// Hard `2 MiB` (`2,097,152` bytes) cap on retained UI `StateSnapshot` frames per controller.
pub const MAX_SNAPSHOT_HISTORY_BYTES: usize = crate::framework::MAX_TOOL_STATE_BYTES;

struct RetainedSnapshot {
    step: usize,
    snapshot: StateSnapshot,
    bytes: usize,
}

/// Opt-in bounded ring buffer retaining recent [`StateSnapshot`] frames indexed by deterministic step.
pub struct SnapshotHistory {
    entries: VecDeque<RetainedSnapshot>,
    total_bytes: usize,
    max_frames: usize,
    max_bytes: usize,
}

impl Default for SnapshotHistory {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SNAPSHOT_FRAMES, MAX_SNAPSHOT_HISTORY_BYTES)
    }
}

impl SnapshotHistory {
    /// Creates a bounded snapshot history ring capped at `max_frames` and `max_bytes <= 2 MiB`.
    #[must_use]
    pub fn new(max_frames: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            total_bytes: 0,
            max_frames: max_frames.max(1),
            max_bytes: max_bytes.clamp(1, MAX_SNAPSHOT_HISTORY_BYTES),
        }
    }

    pub(super) fn seeded(
        max_frames: usize,
        max_bytes: usize,
        timeline: Option<TimelineMetadata>,
        snapshot: Option<&StateSnapshot>,
    ) -> Self {
        let mut hist = Self::new(max_frames, max_bytes);
        if let (Some(meta), Some(snap)) = (timeline, snapshot) {
            hist.record(meta.current_step, snap);
        }
        hist
    }

    /// Configured `(max_frames, max_bytes)` bounds for preserving custom limits across respawns.
    #[must_use]
    pub fn limits(&self) -> (usize, usize) {
        (self.max_frames, self.max_bytes)
    }

    /// Number of retained [`StateSnapshot`] frames currently in the ring.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no [`StateSnapshot`] frames are currently retained.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Estimated total memory footprint in bytes of all retained [`StateSnapshot`] frames.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Clears all retained frames and shrinks excess `VecDeque` capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.entries.shrink_to(32);
        self.total_bytes = 0;
    }

    /// Truncates any retained frames strictly newer than `step` when resuming forward integration.
    pub fn truncate_after(&mut self, step: usize) {
        while let Some(back) = self.entries.back() {
            if back.step > step {
                let removed = self.entries.pop_back().expect("checked back");
                self.total_bytes = self.total_bytes.saturating_sub(removed.bytes);
            } else {
                break;
            }
        }
    }

    /// Looks up a retained [`StateSnapshot`] captured at `step`, if still present in the ring.
    #[must_use]
    pub fn snapshot_at_step(&self, step: usize) -> Option<&StateSnapshot> {
        self.entries
            .binary_search_by_key(&step, |entry| entry.step)
            .ok()
            .map(|idx| &self.entries[idx].snapshot)
    }

    /// Records or updates a [`StateSnapshot`] for `step`, evicting oldest frames to stay within bounds.
    pub fn record(&mut self, step: usize, snapshot: &StateSnapshot) {
        let bytes = estimate_snapshot_bytes(snapshot);
        if bytes > self.max_bytes {
            return;
        }
        if let Ok(idx) = self.entries.binary_search_by_key(&step, |e| e.step) {
            self.total_bytes = self.total_bytes.saturating_sub(self.entries[idx].bytes);
            self.entries[idx] = RetainedSnapshot {
                step,
                snapshot: clone_retained_snapshot(snapshot),
                bytes,
            };
            self.total_bytes += bytes;
            self.evict_oldest_to_fit(0);
            return;
        }
        self.evict_oldest_to_fit(bytes);
        while self.entries.len() >= self.max_frames {
            if let Some(removed) = self.entries.pop_front() {
                self.total_bytes = self.total_bytes.saturating_sub(removed.bytes);
            } else {
                break;
            }
        }
        if self.total_bytes + bytes <= self.max_bytes {
            let insert_idx = self
                .entries
                .binary_search_by_key(&step, |e| e.step)
                .unwrap_or_else(|idx| idx);
            self.total_bytes += bytes;
            self.entries.insert(
                insert_idx,
                RetainedSnapshot {
                    step,
                    snapshot: clone_retained_snapshot(snapshot),
                    bytes,
                },
            );
        }
    }

    fn evict_oldest_to_fit(&mut self, incoming_bytes: usize) {
        while self.total_bytes + incoming_bytes > self.max_bytes && !self.entries.is_empty() {
            let removed = self.entries.pop_front().expect("non-empty entries");
            self.total_bytes = self.total_bytes.saturating_sub(removed.bytes);
        }
    }
}

fn clone_retained_snapshot(snapshot: &StateSnapshot) -> StateSnapshot {
    let structured_data: Option<Box<dyn std::any::Any + Send>> =
        snapshot.structured_data.as_deref().and_then(|any| {
            #[cfg(feature = "physics")]
            if let Some(attr) =
                any.downcast_ref::<crate::tabs::chaos::attractors::AttractorSnapshot>()
            {
                return Some(Box::new(attr.clone()) as Box<dyn std::any::Any + Send>);
            }
            let _ = any;
            None
        });
    StateSnapshot {
        width: snapshot.width,
        height: snapshot.height,
        pixels: std::sync::Arc::clone(&snapshot.pixels),
        custom_data: snapshot.custom_data.clone(),
        structured_data,
        sonification: snapshot.sonification,
    }
}

fn estimate_snapshot_bytes(snapshot: &StateSnapshot) -> usize {
    let base = std::mem::size_of::<StateSnapshot>();
    let custom_bytes = snapshot.custom_data.capacity() * std::mem::size_of::<f64>();
    let pixel_bytes = snapshot.pixels.read().map_or(0, |px| {
        px.capacity() * std::mem::size_of::<eframe::egui::Color32>()
    });
    let structured_bytes = match snapshot.structured_data.as_deref() {
        Some(any) => {
            #[cfg(feature = "physics")]
            if let Some(attr) =
                any.downcast_ref::<crate::tabs::chaos::attractors::AttractorSnapshot>()
            {
                attr.estimated_heap_bytes()
            } else {
                256
            }
            #[cfg(not(feature = "physics"))]
            {
                let _ = any;
                256
            }
        }
        None => 0,
    };
    base + custom_bytes + pixel_bytes + structured_bytes
}
