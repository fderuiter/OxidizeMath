//! Pre-allocated fixed-capacity timeline ring buffer for deterministic simulation scrubbing.

/// Default pre-allocated frame capacity (`4,096` snapshots) for [`TimelineRingBuffer`].
pub const DEFAULT_TIMELINE_RING_CAPACITY: usize = 4_096;

/// Fixed-capacity ring buffer supporting zero-allocation push, backward stepping,
/// random-access seeking, and deterministic branch truncation upon forward resume.
#[derive(Debug, Clone)]
pub struct TimelineRingBuffer<T: Copy> {
    storage: Box<[T]>,
    capacity: usize,
    head: usize,
    len: usize,
    cursor: usize,
}

impl<T: Copy> TimelineRingBuffer<T> {
    /// Pre-allocates a ring buffer with [`DEFAULT_TIMELINE_RING_CAPACITY`] slots.
    #[must_use]
    pub fn new(placeholder: T) -> Self {
        Self::with_capacity(DEFAULT_TIMELINE_RING_CAPACITY, placeholder)
    }

    /// Pre-allocates a ring buffer with `capacity.max(1)` contiguous slots.
    #[must_use]
    pub fn with_capacity(capacity: usize, placeholder: T) -> Self {
        let cap = capacity.max(1);
        let storage = vec![placeholder; cap].into_boxed_slice();
        Self {
            storage,
            capacity: cap,
            head: 0,
            len: 0,
            cursor: 0,
        }
    }

    /// Returns the maximum number of frames retained without overwriting oldest frames.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of valid frames currently stored in the ring buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when zero frames are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the zero-based logical index of the active cursor within `0..self.len()`.
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Returns `true` when at least one earlier frame is available before `cursor`.
    #[must_use]
    pub fn can_step_back(&self) -> bool {
        self.len > 1 && self.cursor > 0
    }

    /// Resets logical indices to empty while retaining the pre-allocated backing slice.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.cursor = 0;
    }

    /// Discards any frames newer than `cursor` without reallocating backing storage.
    pub fn truncate_after_cursor(&mut self) {
        if self.len > 0 && self.cursor + 1 < self.len {
            self.len = self.cursor + 1;
        }
    }

    /// Appends a new frame after `cursor` (truncating any scrubbed future branch)
    /// and advances `cursor` to the newest frame with zero heap allocations.
    pub fn push(&mut self, frame: T) {
        self.truncate_after_cursor();
        if self.len < self.capacity {
            let physical = self.physical_slot(self.len);
            self.storage[physical] = frame;
            self.len += 1;
            self.cursor = self.len - 1;
        } else {
            self.storage[self.head] = frame;
            self.head += 1;
            if self.head == self.capacity {
                self.head = 0;
            }
            self.cursor = self.capacity - 1;
        }
    }

    /// Returns a reference to the frame at logical `index` (`0` = oldest retained frame).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        Some(&self.storage[self.physical_slot(index)])
    }

    /// Returns a copy of the frame at logical `index`.
    #[must_use]
    pub fn get_copied(&self, index: usize) -> Option<T> {
        self.get(index).copied()
    }

    /// Returns a reference to the frame at the active `cursor`.
    #[must_use]
    pub fn current(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        self.get(self.cursor)
    }

    /// Returns a copy of the frame at the active `cursor`.
    #[must_use]
    pub fn current_copied(&self) -> Option<T> {
        self.current().copied()
    }

    /// Moves `cursor` backward by up to `count` frames and returns the restored frame.
    pub fn step_back(&mut self, count: usize) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.cursor = self.cursor.saturating_sub(count);
        self.current_copied()
    }

    /// Moves `cursor` to `index` (clamped to `0..self.len()`) and returns the restored frame.
    pub fn seek(&mut self, index: usize) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.cursor = index.min(self.len - 1);
        self.current_copied()
    }

    /// Iterates chronologically from the oldest retained frame through `cursor` inclusive.
    pub fn iter_up_to_cursor(&self) -> impl Iterator<Item = &T> {
        let count = if self.len == 0 { 0 } else { self.cursor + 1 };
        (0..count).filter_map(|idx| self.get(idx))
    }

    /// Iterates chronologically over all `self.len()` retained frames.
    pub fn iter_all(&self) -> impl Iterator<Item = &T> {
        (0..self.len).filter_map(|idx| self.get(idx))
    }

    fn physical_slot(&self, logical_index: usize) -> usize {
        let sum = self.head + logical_index;
        if sum >= self.capacity {
            sum - self.capacity
        } else {
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_wraparound_seek_and_branch_truncation() {
        let mut ring = TimelineRingBuffer::with_capacity(4, 0_u32);
        assert!(ring.is_empty());
        assert!(!ring.can_step_back());

        for v in 1..=6 {
            ring.push(v);
        }
        assert_eq!(ring.len(), 4);
        assert_eq!(ring.cursor(), 3);
        let all: Vec<u32> = ring.iter_all().copied().collect();
        assert_eq!(all, vec![3, 4, 5, 6]);

        assert_eq!(ring.step_back(2), Some(4));
        assert_eq!(ring.cursor(), 1);
        let active: Vec<u32> = ring.iter_up_to_cursor().copied().collect();
        assert_eq!(active, vec![3, 4]);

        // Pushing after rewind truncates frames 5 and 6 before appending 99.
        ring.push(99);
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.cursor(), 2);
        let branched: Vec<u32> = ring.iter_all().copied().collect();
        assert_eq!(branched, vec![3, 4, 99]);

        assert_eq!(ring.seek(0), Some(3));
        assert!(!ring.can_step_back());
        assert_eq!(ring.seek(99), Some(99));
        assert_eq!(ring.cursor(), 2);
    }

    #[test]
    fn test_ring_buffer_operations_are_zero_allocation() {
        let mut ring = TimelineRingBuffer::new([0.0_f64; 4]);
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for i in 0..5000 {
                ring.push([i as f64, 1.0, 2.0, 3.0]);
            }
            let _ = ring.step_back(12);
            let _ = ring.seek(100);
            ring.push([42.0, 0.0, 0.0, 0.0]);
        });
        assert_eq!(ring.len(), 102);
        assert_eq!(ring.current_copied().unwrap()[0], 42.0);
    }
}
