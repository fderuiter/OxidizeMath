//! Internal Ring-Buffer Wavefront, Lexicographic Sweep, and Dhar Burning Helpers.

use math_commons::math_kernel::types::flatten_2d_index;

use super::{
    AvalancheStats, BurningTestResult, CRITICAL_HEIGHT, MAX_STABLE_HEIGHT, SandpileError,
    SandpileGrid,
};

impl SandpileGrid {
    #[inline]
    #[verified_engine::verified]
    pub(super) fn clear_ring_queue(&mut self) {
        self.ring_head = 0;
        self.ring_tail = 0;
        self.in_queue.fill(0);
    }

    #[inline]
    pub(super) const fn ring_len(&self) -> usize {
        self.ring_tail.wrapping_sub(self.ring_head)
    }

    #[inline]
    pub(super) fn ring_push(&mut self, idx: u32) {
        self.ring_buf[self.ring_tail & self.ring_mask] = idx;
        self.ring_tail = self.ring_tail.wrapping_add(1);
    }

    #[inline]
    pub(super) fn ring_pop(&mut self) -> u32 {
        let val = self.ring_buf[self.ring_head & self.ring_mask];
        self.ring_head = self.ring_head.wrapping_add(1);
        val
    }

    pub(super) fn next_epoch(&mut self) -> u32 {
        self.epoch_counter = self.epoch_counter.wrapping_add(1);
        if self.epoch_counter == 0 {
            self.touched_epoch.fill(0);
            self.epoch_counter = 1;
        }
        self.epoch_counter
    }

    pub(super) fn seed_unstable_into_ring(&mut self) {
        for idx in 0..self.heights.len() {
            if self.heights[idx] >= CRITICAL_HEIGHT && self.in_queue[idx] == 0 {
                self.in_queue[idx] = 1;
                self.ring_push(idx as u32);
            }
        }
    }

    pub(super) fn ensure_unstable_seeded(&mut self) {
        if !self.has_active_wavefront() {
            self.seed_unstable_into_ring();
        }
    }

    #[inline]
    fn enqueue_neighbor(&mut self, n_idx: usize, grains: u32) {
        self.heights[n_idx] += grains;
        if self.heights[n_idx] >= CRITICAL_HEIGHT && self.in_queue[n_idx] == 0 {
            self.in_queue[n_idx] = 1;
            self.ring_push(n_idx as u32);
        }
    }

    pub(super) fn distribute_single_topple(&mut self, idx: usize) -> u64 {
        self.distribute_multi_topple(idx, 1)
    }

    pub(super) fn distribute_multi_topple(&mut self, idx: usize, topples: u32) -> u64 {
        let x = idx % self.width;
        let y = idx / self.width;
        let mut boundary_exits = 0_u64;
        if x > 0 {
            self.enqueue_neighbor(flatten_2d_index(x - 1, y, self.width), topples);
        } else {
            boundary_exits += u64::from(topples);
        }
        if x + 1 < self.width {
            self.enqueue_neighbor(flatten_2d_index(x + 1, y, self.width), topples);
        } else {
            boundary_exits += u64::from(topples);
        }
        if y > 0 {
            self.enqueue_neighbor(flatten_2d_index(x, y - 1, self.width), topples);
        } else {
            boundary_exits += u64::from(topples);
        }
        if y + 1 < self.height {
            self.enqueue_neighbor(flatten_2d_index(x, y + 1, self.width), topples);
        } else {
            boundary_exits += u64::from(topples);
        }
        boundary_exits
    }

    pub(super) fn topple_site_immediate(
        &mut self,
        idx: usize,
        epoch: u32,
        stats: &mut AvalancheStats,
    ) {
        let topples = self.heights[idx] / CRITICAL_HEIGHT;
        self.heights[idx] %= CRITICAL_HEIGHT;
        stats.topples += u64::from(topples);
        if self.touched_epoch[idx] != epoch {
            self.touched_epoch[idx] = epoch;
            stats.area += 1;
        }
        let x = idx % self.width;
        let y = idx / self.width;
        if x > 0 {
            self.heights[flatten_2d_index(x - 1, y, self.width)] += topples;
        } else {
            stats.dissipated_grains += u64::from(topples);
        }
        if x + 1 < self.width {
            self.heights[flatten_2d_index(x + 1, y, self.width)] += topples;
        } else {
            stats.dissipated_grains += u64::from(topples);
        }
        if y > 0 {
            self.heights[flatten_2d_index(x, y - 1, self.width)] += topples;
        } else {
            stats.dissipated_grains += u64::from(topples);
        }
        if y + 1 < self.height {
            self.heights[flatten_2d_index(x, y + 1, self.width)] += topples;
        } else {
            stats.dissipated_grains += u64::from(topples);
        }
    }

    pub(super) fn stabilize_silent_wavefront(&mut self) -> u64 {
        self.ensure_unstable_seeded();
        let mut topples_sum = 0_u64;
        while self.has_active_wavefront() {
            let wave_len = self.ring_len();
            for slot in 0..wave_len {
                let idx = self.ring_pop() as usize;
                self.in_queue[idx] = 0;
                self.scratch_a[slot] = idx as u32;
                let topples = self.heights[idx] / CRITICAL_HEIGHT;
                self.scratch_b[slot] = topples;
                self.heights[idx] %= CRITICAL_HEIGHT;
                topples_sum += u64::from(topples);
            }
            for slot in 0..wave_len {
                let idx = self.scratch_a[slot] as usize;
                let topples = self.scratch_b[slot];
                let _ = self.distribute_multi_topple(idx, topples);
            }
        }
        topples_sum
    }

    pub(super) fn init_neighbor_degrees(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let mut deg = 0_u8;
                if x > 0 {
                    deg += 1;
                }
                if x + 1 < self.width {
                    deg += 1;
                }
                if y > 0 {
                    deg += 1;
                }
                if y + 1 < self.height {
                    deg += 1;
                }
                self.rem_deg[flatten_2d_index(x, y, self.width)] = deg;
            }
        }
    }

    pub(super) fn propagate_burning_bfs(&mut self) -> (usize, u16) {
        let mut burned = 0usize;
        let mut max_depth = 0_u16;
        while self.has_active_wavefront() {
            let idx = self.ring_pop() as usize;
            burned += 1;
            let next_depth = self.burn_depth[idx].saturating_add(1);
            max_depth = max_depth.max(self.burn_depth[idx]);
            let x = idx % self.width;
            let y = idx / self.width;
            if x > 0 {
                self.try_burn_neighbor(flatten_2d_index(x - 1, y, self.width), next_depth);
            }
            if x + 1 < self.width {
                self.try_burn_neighbor(flatten_2d_index(x + 1, y, self.width), next_depth);
            }
            if y > 0 {
                self.try_burn_neighbor(flatten_2d_index(x, y - 1, self.width), next_depth);
            }
            if y + 1 < self.height {
                self.try_burn_neighbor(flatten_2d_index(x, y + 1, self.width), next_depth);
            }
        }
        (burned, max_depth)
    }

    #[inline]
    fn try_burn_neighbor(&mut self, n_idx: usize, next_depth: u16) {
        if self.burn_depth[n_idx] == 0 {
            self.rem_deg[n_idx] = self.rem_deg[n_idx].saturating_sub(1);
            if self.heights[n_idx] >= u32::from(self.rem_deg[n_idx]) {
                self.burn_depth[n_idx] = next_depth;
                self.ring_push(n_idx as u32);
            }
        }
    }

    /// Stabilizes the lattice using deterministic row-major forward sweeps (for Abelian commutativity proofs).
    #[verified_engine::verified]
    pub fn stabilize_forward(&mut self) -> AvalancheStats {
        self.clear_ring_queue();
        let epoch = self.next_epoch();
        let mut stats = AvalancheStats::default();
        loop {
            let mut pass_sites = 0usize;
            for y in 0..self.height {
                for x in 0..self.width {
                    let idx = flatten_2d_index(x, y, self.width);
                    if self.heights[idx] >= CRITICAL_HEIGHT {
                        pass_sites += 1;
                        self.topple_site_immediate(idx, epoch, &mut stats);
                    }
                }
            }
            if pass_sites == 0 {
                break;
            }
            stats.duration += 1;
            stats.peak_wavefront = stats.peak_wavefront.max(pass_sites);
        }
        self.telemetry.record(stats);
        stats
    }

    /// Stabilizes the lattice using deterministic reverse row-major sweeps (for Abelian commutativity proofs).
    #[verified_engine::verified]
    pub fn stabilize_reverse(&mut self) -> AvalancheStats {
        self.clear_ring_queue();
        let epoch = self.next_epoch();
        let mut stats = AvalancheStats::default();
        loop {
            let mut pass_sites = 0usize;
            for y in (0..self.height).rev() {
                for x in (0..self.width).rev() {
                    let idx = flatten_2d_index(x, y, self.width);
                    if self.heights[idx] >= CRITICAL_HEIGHT {
                        pass_sites += 1;
                        self.topple_site_immediate(idx, epoch, &mut stats);
                    }
                }
            }
            if pass_sites == 0 {
                break;
            }
            stats.duration += 1;
            stats.peak_wavefront = stats.peak_wavefront.max(pass_sites);
        }
        self.telemetry.record(stats);
        stats
    }

    /// Computes Creutz's recurrent identity configuration $e = (2 z_{\max} - (2 z_{\max})^\circ)^\circ$
    /// in place with zero heap allocations, returning total topples across both relaxations.
    #[verified_engine::verified]
    pub fn load_creutz_identity(&mut self) -> u64 {
        let two_z_max = 2 * MAX_STABLE_HEIGHT;
        self.fill(two_z_max);
        let first = self.stabilize_silent_wavefront();
        for h in &mut self.heights {
            *h = two_z_max - *h;
        }
        self.seed_unstable_into_ring();
        let second = self.stabilize_silent_wavefront();
        first + second
    }

    /// Adds `other` (`width * height` heights) cellwise into `self` and stabilizes $(z + \text{other})^\circ$
    /// without altering the avalanche power-law histogram.
    #[verified_engine::verified]
    pub fn compose_with_slice(&mut self, other: &[u32]) -> Result<u64, SandpileError> {
        if other.len() != self.heights.len() {
            return Err(SandpileError::BufferLengthMismatch {
                expected: self.heights.len(),
                actual: other.len(),
            });
        }
        for (dst, &src) in self.heights.iter_mut().zip(other.iter()) {
            *dst = dst.saturating_add(src);
        }
        self.clear_ring_queue();
        self.seed_unstable_into_ring();
        Ok(self.stabilize_silent_wavefront())
    }

    /// Replaces `self` with its recurrent group inverse $r^{-1} = (2 z_0 - r^\circ)^\circ$
    /// using pre-allocated buffers and zero heap allocations.
    #[verified_engine::verified]
    pub fn invert_in_place(&mut self) -> u64 {
        let mut total_topples = self.stabilize_silent_wavefront();
        for (dst, &h) in self.burn_depth.iter_mut().zip(self.heights.iter()) {
            *dst = h as u16;
        }
        let two_z_max = 2 * MAX_STABLE_HEIGHT;
        self.fill(two_z_max);
        total_topples += self.stabilize_silent_wavefront();
        for (h, &r_val) in self.heights.iter_mut().zip(self.burn_depth.iter()) {
            let z0 = two_z_max - *h;
            *h = 2 * z0 - u32::from(r_val);
        }
        self.seed_unstable_into_ring();
        total_topples + self.stabilize_silent_wavefront()
    }

    /// Executes Dhar's burning algorithm in $O(W \cdot H)$ time using the pre-allocated ring buffer,
    /// classifying recurrence and recording per-cell burning wave depths in [`Self::burn_depths`].
    #[verified_engine::verified]
    pub fn dhar_burning_test(&mut self) -> BurningTestResult {
        let total_cells = self.heights.len();
        self.burn_depth.fill(0);
        if self.heights.iter().any(|&h| h >= CRITICAL_HEIGHT) {
            return BurningTestResult {
                is_recurrent: false,
                burned_cells: 0,
                total_cells,
                max_wave_depth: 0,
            };
        }
        self.init_neighbor_degrees();
        self.clear_ring_queue();
        for idx in 0..total_cells {
            if self.heights[idx] >= u32::from(self.rem_deg[idx]) {
                self.burn_depth[idx] = 1;
                self.ring_push(idx as u32);
            }
        }
        let (burned_cells, max_wave_depth) = self.propagate_burning_bfs();
        self.clear_ring_queue();
        BurningTestResult {
            is_recurrent: burned_cells == total_cells,
            burned_cells,
            total_cells,
            max_wave_depth,
        }
    }
}
