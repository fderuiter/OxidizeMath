//! Rolling two-slice edge-to-vertex cache for Marching Cubes traversal.
//!
//! Because `MarchingCubes::extract` visits voxel cubes in `z -> y -> x` order,
//! cube `(x, y, z)` only shares edges with cubes in the active XY plane `z`,
//! the next XY plane `z + 1`, the active Z-edge row `(y, z)`, and the next
//! Z-edge row `(y + 1, z)`. Storing only those rolling slices replaces global
//! `HashMap` hashing and rehashing with `O(1)` direct array indexing while
//! deferring buffer allocation until the first intersected surface cube.

/// Sentinel value marking an unpopulated edge slot in [`SliceEdgeCache`].
pub(super) const EMPTY_EDGE_VERTEX: u32 = u32::MAX;

/// Lazy rolling cache mapping cube edges to mesh vertex indices across two XY
/// planes and two Z-directed rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SliceEdgeCache {
    width: usize,
    height: usize,
    xy_len: usize,
    allocated: bool,
    curr_xy: Vec<u32>,
    next_xy: Vec<u32>,
    curr_z_row: Vec<u32>,
    next_z_row: Vec<u32>,
    curr_xy_dirty: bool,
    next_xy_dirty: bool,
    curr_z_dirty: bool,
    next_z_dirty: bool,
}

#[inline(always)]
#[verified_engine::verified]
fn clear_if_dirty(buf: &mut [u32], dirty: &mut bool) {
    if *dirty {
        debug_assert!(!buf.is_empty());
        buf.fill(EMPTY_EDGE_VERTEX);
        *dirty = false;
    }
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
            curr_xy: Vec::new(),
            next_xy: Vec::new(),
            curr_z_row: Vec::new(),
            next_z_row: Vec::new(),
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
        self.curr_xy.resize(self.xy_len, EMPTY_EDGE_VERTEX);
        self.next_xy.resize(self.xy_len, EMPTY_EDGE_VERTEX);
        self.curr_z_row.resize(self.width, EMPTY_EDGE_VERTEX);
        self.next_z_row.resize(self.width, EMPTY_EDGE_VERTEX);
        self.allocated = true;
        debug_assert_eq!(self.curr_xy.len(), self.xy_len);
        debug_assert_eq!(self.curr_z_row.len(), self.width);
    }

    /// Pre-allocates rolling slice edge buffers up to `max_width x max_height` for zero-allocation reuse.
    #[verified_engine::verified]
    pub(super) fn with_capacity(max_width: usize, max_height: usize) -> Self {
        let w = max_width.max(2);
        let h = max_height.max(2);
        let xy_len = w * h * 2;
        Self {
            width: w,
            height: h,
            xy_len,
            allocated: true,
            curr_xy: vec![EMPTY_EDGE_VERTEX; xy_len],
            next_xy: vec![EMPTY_EDGE_VERTEX; xy_len],
            curr_z_row: vec![EMPTY_EDGE_VERTEX; w],
            next_z_row: vec![EMPTY_EDGE_VERTEX; w],
            curr_xy_dirty: false,
            next_xy_dirty: false,
            curr_z_dirty: false,
            next_z_dirty: false,
        }
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
            clear_if_dirty(&mut self.curr_xy, &mut self.curr_xy_dirty);
            clear_if_dirty(&mut self.next_xy, &mut self.next_xy_dirty);
            clear_if_dirty(&mut self.curr_z_row, &mut self.curr_z_dirty);
            clear_if_dirty(&mut self.next_z_row, &mut self.next_z_dirty);
        } else {
            self.curr_xy.fill(EMPTY_EDGE_VERTEX);
            self.next_xy.fill(EMPTY_EDGE_VERTEX);
            self.curr_z_row.fill(EMPTY_EDGE_VERTEX);
            self.next_z_row.fill(EMPTY_EDGE_VERTEX);
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
        clear_if_dirty(&mut self.next_xy, &mut self.next_xy_dirty);
        clear_if_dirty(&mut self.curr_z_row, &mut self.curr_z_dirty);
        clear_if_dirty(&mut self.next_z_row, &mut self.next_z_dirty);
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
        clear_if_dirty(&mut self.next_z_row, &mut self.next_z_dirty);
        debug_assert!(!self.next_z_dirty);
    }

    /// Returns a mutable reference to the cached vertex slot for `edge_num` (`0..12`)
    /// of the cube at `(x, y)` in the current `z` slice, lazily allocating the slice
    /// buffers with [`EMPTY_EDGE_VERTEX`] on first active cube access.
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
                &mut self.curr_xy[idx]
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
                &mut self.next_xy[idx]
            }
            8 | 9 => {
                self.curr_z_dirty = true;
                let idx = if edge_num == 8 { x } else { x + 1 };
                &mut self.curr_z_row[idx]
            }
            10 | 11 => {
                self.next_z_dirty = true;
                let idx = if edge_num == 11 { x } else { x + 1 };
                &mut self.next_z_row[idx]
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
        assert_eq!(cache.curr_xy.capacity(), 0);
        assert_eq!(cache.next_xy.capacity(), 0);
        assert_eq!(cache.curr_z_row.capacity(), 0);
        assert_eq!(cache.next_z_row.capacity(), 0);

        cache.advance_y();
        cache.advance_z();
        assert!(!cache.allocated);
        assert_eq!(cache.curr_xy.capacity(), 0);
        assert_eq!(cache.next_xy.capacity(), 0);
        assert_eq!(cache.curr_z_row.capacity(), 0);
        assert_eq!(cache.next_z_row.capacity(), 0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_xy_and_z_edge_sharing_and_slice_rotation() {
        let mut cache = SliceEdgeCache::new(4, 3);

        // Cube (1, 0, 0): populate shared edges across X, Y, and Z neighbors
        *cache.slot_mut(1, 0, 1) = 11; // shared with (2, 0, 0) edge 3
        *cache.slot_mut(1, 0, 2) = 22; // shared with (1, 1, 0) edge 0
        *cache.slot_mut(1, 0, 4) = 44; // shared with (1, 0, 1) edge 0
        *cache.slot_mut(1, 0, 9) = 99; // shared with (2, 0, 0) edge 8
        *cache.slot_mut(1, 0, 10) = 101; // shared with (2, 1, 0) edge 8

        assert_eq!(*cache.slot_mut(2, 0, 3), 11);
        assert_eq!(*cache.slot_mut(2, 0, 8), 99);

        // Advance Y to y = 1 within z = 0
        cache.advance_y();
        assert_eq!(*cache.slot_mut(1, 1, 0), 22);
        assert_eq!(*cache.slot_mut(2, 1, 8), 101);
        assert_eq!(*cache.slot_mut(2, 1, 10), EMPTY_EDGE_VERTEX);

        // Advance Z to z = 1: next_xy becomes curr_xy and Z rows clear
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

        // First advance_z promotes next_xy -> curr_xy (still dirty) and clears Z rows
        cache.advance_z();
        assert!(cache.curr_xy_dirty);
        assert!(!cache.next_xy_dirty);
        assert!(!cache.curr_z_dirty);
        assert!(!cache.next_z_dirty);

        // Second advance_z with no active cubes clears remaining dirty plane
        cache.advance_z();
        assert!(!cache.curr_xy_dirty);
        assert!(!cache.next_xy_dirty);
    }
}
