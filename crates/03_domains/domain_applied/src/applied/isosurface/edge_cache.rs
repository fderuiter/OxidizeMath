//! Rolling two-slice edge-to-vertex cache for Marching Cubes traversal backed by [`OwnedByteArena`].

use math_memory::{ArenaLayout, ArenaRegion, OwnedByteArena};

/// Sentinel value marking an unpopulated edge slot in [`SliceEdgeCache`].
pub(super) const EMPTY_EDGE_VERTEX: u32 = u32::MAX;

/// Lazy rolling cache mapping cube edges to mesh vertex indices across two XY
/// planes and two Z-directed rows inside a single [`OwnedByteArena`].
pub(super) struct SliceEdgeCache {
    width: usize,
    height: usize,
    xy_len: usize,
    allocated: bool,
    arena: OwnedByteArena,
    curr_xy: ArenaRegion<u32>,
    next_xy: ArenaRegion<u32>,
    curr_z_row: ArenaRegion<u32>,
    next_z_row: ArenaRegion<u32>,
    curr_xy_dirty: bool,
    next_xy_dirty: bool,
    curr_z_dirty: bool,
    next_z_dirty: bool,
}

#[inline(always)]
#[verified_engine::verified]
fn clear_region_if_dirty(arena: &mut OwnedByteArena, reg: ArenaRegion<u32>, dirty: &mut bool) {
    if *dirty {
        let buf = arena.region_slice_mut(reg);
        debug_assert!(!buf.is_empty());
        buf.fill(EMPTY_EDGE_VERTEX);
        *dirty = false;
    }
}

fn build_edge_layout(
    xy_len: usize,
    width: usize,
) -> (
    ArenaLayout,
    ArenaRegion<u32>,
    ArenaRegion<u32>,
    ArenaRegion<u32>,
    ArenaRegion<u32>,
) {
    let mut layout = ArenaLayout::new();
    let curr_xy = layout.reserve::<u32>(xy_len).expect("curr_xy");
    let next_xy = layout.reserve::<u32>(xy_len).expect("next_xy");
    let curr_z_row = layout.reserve::<u32>(width).expect("curr_z_row");
    let next_z_row = layout.reserve::<u32>(width).expect("next_z_row");
    (layout, curr_xy, next_xy, curr_z_row, next_z_row)
}

impl SliceEdgeCache {
    /// Creates an unallocated rolling cache for a grid of `width x height` XY dimensions.
    #[inline]
    #[verified_engine::verified]
    pub(super) fn new(width: usize, height: usize) -> Self {
        debug_assert!(width >= 2 && height >= 2);
        let xy_len = width * height * 2;
        Self {
            width,
            height,
            xy_len,
            allocated: false,
            arena: OwnedByteArena::new(),
            curr_xy: ArenaRegion::empty(),
            next_xy: ArenaRegion::empty(),
            curr_z_row: ArenaRegion::empty(),
            next_z_row: ArenaRegion::empty(),
            curr_xy_dirty: false,
            next_xy_dirty: false,
            curr_z_dirty: false,
            next_z_dirty: false,
        }
    }

    #[cold]
    #[inline(never)]
    #[verified_engine::verified]
    fn allocate_buffers(&mut self) {
        let (layout, curr_xy, next_xy, curr_z_row, next_z_row) =
            build_edge_layout(self.xy_len, self.width);
        self.arena.ensure_capacity(layout.total_bytes());
        self.curr_xy = curr_xy;
        self.next_xy = next_xy;
        self.curr_z_row = curr_z_row;
        self.next_z_row = next_z_row;
        self.arena
            .region_slice_mut(self.curr_xy)
            .fill(EMPTY_EDGE_VERTEX);
        self.arena
            .region_slice_mut(self.next_xy)
            .fill(EMPTY_EDGE_VERTEX);
        self.arena
            .region_slice_mut(self.curr_z_row)
            .fill(EMPTY_EDGE_VERTEX);
        self.arena
            .region_slice_mut(self.next_z_row)
            .fill(EMPTY_EDGE_VERTEX);
        self.allocated = true;
    }

    /// Pre-allocates rolling slice edge buffers up to `max_width x max_height` for zero-allocation reuse.
    #[verified_engine::verified]
    pub(super) fn with_capacity(max_width: usize, max_height: usize) -> Self {
        let w = max_width.max(2);
        let h = max_height.max(2);
        let mut cache = Self::new(w, h);
        cache.allocate_buffers();
        cache
    }

    /// Resets active XY dimensions and clears any dirty edge slots in-place without reallocating if capacity suffices.
    #[verified_engine::verified]
    pub(super) fn reset_for_dimensions(&mut self, width: usize, height: usize) {
        let w = width.max(2);
        let h = height.max(2);
        let xy_len = w * h * 2;
        self.width = w;
        self.height = h;
        self.xy_len = xy_len;
        if !self.allocated {
            return;
        }
        if self.curr_xy.len() >= xy_len
            && self.next_xy.len() >= xy_len
            && self.curr_z_row.len() >= w
            && self.next_z_row.len() >= w
        {
            clear_region_if_dirty(&mut self.arena, self.curr_xy, &mut self.curr_xy_dirty);
            clear_region_if_dirty(&mut self.arena, self.next_xy, &mut self.next_xy_dirty);
            clear_region_if_dirty(&mut self.arena, self.curr_z_row, &mut self.curr_z_dirty);
            clear_region_if_dirty(&mut self.arena, self.next_z_row, &mut self.next_z_dirty);
        } else {
            self.curr_xy_dirty = false;
            self.next_xy_dirty = false;
            self.curr_z_dirty = false;
            self.next_z_dirty = false;
            self.allocate_buffers();
        }
    }

    /// Rotates XY plane buffers when advancing to `z > 0` and resets Z-edge rows.
    #[inline]
    #[verified_engine::verified]
    pub(super) fn advance_z(&mut self) {
        if !self.allocated {
            return;
        }
        std::mem::swap(&mut self.curr_xy, &mut self.next_xy);
        std::mem::swap(&mut self.curr_xy_dirty, &mut self.next_xy_dirty);
        clear_region_if_dirty(&mut self.arena, self.next_xy, &mut self.next_xy_dirty);
        clear_region_if_dirty(&mut self.arena, self.curr_z_row, &mut self.curr_z_dirty);
        clear_region_if_dirty(&mut self.arena, self.next_z_row, &mut self.next_z_dirty);
        debug_assert!(!self.next_xy_dirty && !self.curr_z_dirty && !self.next_z_dirty);
    }

    /// Rotates Z-directed row buffers when advancing to `y > 0` within the same `z` slice.
    #[inline]
    #[verified_engine::verified]
    pub(super) fn advance_y(&mut self) {
        if !self.allocated {
            return;
        }
        std::mem::swap(&mut self.curr_z_row, &mut self.next_z_row);
        std::mem::swap(&mut self.curr_z_dirty, &mut self.next_z_dirty);
        clear_region_if_dirty(&mut self.arena, self.next_z_row, &mut self.next_z_dirty);
        debug_assert!(!self.next_z_dirty);
    }

    /// Returns a mutable reference to the cached vertex slot for `edge_num` (`0..12`)
    /// of the cube at `(x, y)` in the current `z` slice.
    #[inline(always)]
    #[verified_engine::verified]
    pub(super) fn slot_mut(&mut self, x: usize, y: usize, edge_num: usize) -> &mut u32 {
        debug_assert!(x + 1 < self.width && y + 1 < self.height);
        debug_assert!(edge_num < 12);

        if !self.allocated {
            self.allocate_buffers();
        }

        let w = self.width;
        match edge_num {
            0..=3 => {
                self.curr_xy_dirty = true;
                let base_xy = (y * w + x) * 2;
                let idx = match edge_num {
                    0 => base_xy,
                    1 => base_xy + 3,
                    2 => base_xy + w * 2,
                    _ => base_xy + 1,
                };
                &mut self.arena.region_slice_mut(self.curr_xy)[idx]
            }
            4..=7 => {
                self.next_xy_dirty = true;
                let base_xy = (y * w + x) * 2;
                let idx = match edge_num {
                    4 => base_xy,
                    5 => base_xy + 3,
                    6 => base_xy + w * 2,
                    _ => base_xy + 1,
                };
                &mut self.arena.region_slice_mut(self.next_xy)[idx]
            }
            8 | 9 => {
                self.curr_z_dirty = true;
                let idx = if edge_num == 8 { x } else { x + 1 };
                &mut self.arena.region_slice_mut(self.curr_z_row)[idx]
            }
            10 | 11 => {
                self.next_z_dirty = true;
                let idx = if edge_num == 11 { x } else { x + 1 };
                &mut self.arena.region_slice_mut(self.next_z_row)[idx]
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EMPTY_EDGE_VERTEX, SliceEdgeCache};

    #[test]
    #[verified_engine::verified]
    fn test_lazy_zero_allocation_until_active_edge() {
        let mut cache = SliceEdgeCache::new(8, 6);
        assert!(!cache.allocated);
        assert_eq!(cache.arena.byte_capacity(), 0);

        cache.advance_y();
        cache.advance_z();
        assert!(!cache.allocated);
        assert_eq!(cache.arena.byte_capacity(), 0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_xy_and_z_edge_sharing_and_slice_rotation() {
        let mut cache = SliceEdgeCache::new(4, 3);

        *cache.slot_mut(1, 0, 1) = 11;
        *cache.slot_mut(1, 0, 2) = 22;
        *cache.slot_mut(1, 0, 4) = 44;
        *cache.slot_mut(1, 0, 9) = 99;
        *cache.slot_mut(1, 0, 10) = 101;

        assert_eq!(*cache.slot_mut(2, 0, 3), 11);
        assert_eq!(*cache.slot_mut(2, 0, 8), 99);

        cache.advance_y();
        assert_eq!(*cache.slot_mut(1, 1, 0), 22);
        assert_eq!(*cache.slot_mut(2, 1, 8), 101);
        assert_eq!(*cache.slot_mut(2, 1, 10), EMPTY_EDGE_VERTEX);

        cache.advance_z();
        assert_eq!(*cache.slot_mut(1, 0, 0), 44);
        assert_eq!(*cache.slot_mut(1, 0, 4), EMPTY_EDGE_VERTEX);
        assert_eq!(*cache.slot_mut(2, 0, 8), EMPTY_EDGE_VERTEX);
        assert_eq!(*cache.slot_mut(1, 0, 10), EMPTY_EDGE_VERTEX);
    }

    #[test]
    #[verified_engine::verified]
    fn test_dirty_flags_reset_after_empty_slices() {
        let mut cache = SliceEdgeCache::new(2, 2);
        *cache.slot_mut(0, 0, 0) = 1;
        *cache.slot_mut(0, 0, 4) = 2;
        *cache.slot_mut(0, 0, 8) = 3;
        *cache.slot_mut(0, 0, 11) = 4;

        cache.advance_z();
        assert!(cache.curr_xy_dirty);
        assert!(!cache.next_xy_dirty);
        assert!(!cache.curr_z_dirty);
        assert!(!cache.next_z_dirty);

        cache.advance_z();
        assert!(!cache.curr_xy_dirty);
        assert!(!cache.next_xy_dirty);
    }
}
