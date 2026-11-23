//! Zero-Allocation 2D Bak-Tang-Wiesenfeld (BTW) Abelian Sandpile Engine & Group Algebra.
//!
//! Implements parameter-free self-organized criticality (Bak, Tang & Wiesenfeld, 1987),
//! Abelian sandpile group arithmetic and Dhar's burning recurrence test (Dhar, 1990),
//! Creutz's iterative recurrent identity solver $e = (2 z_{\max} - (2 z_{\max})^\circ)^\circ$
//! (Creutz, 1991), and exact Bareiss Kirchhoff Laplacian determinants.
//!
//! # Key Guarantees
//! - **Zero-allocation hot paths**: [`SandpileGrid`] pre-allocates integer wavefront ring
//!   buffers, epoch markers, Dhar burning queues, and power-law histogram buckets at
//!   construction time so grain drops, wavefront stabilization, Creutz identity steps,
//!   Dhar burning tests, and [`math_commons::fit::FitModel::PowerLaw`] fits perform zero heap
//!   allocations.
//! - **Abelian order independence**: Forward lexicographic sweeps, reverse lexicographic
//!   sweeps, and synchronous wavefront ring-buffer stabilization produce identical final
//!   heights and identical topple counts for every initial configuration.

mod algebra;
mod kirchhoff;
mod telemetry;
mod wavefront;

#[cfg(test)]
mod tests;

use math_commons::error::FitError;
use math_commons::fit::CurveFit;
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

pub use algebra::SandpileElement;
pub use kirchhoff::{MAX_BAREISS_CELLS, SandpileError, kirchhoff_determinant_bareiss};
pub use telemetry::{
    AvalancheObservable, AvalancheStats, AvalancheTelemetry, EXACT_AVALANCHE_BUCKET_CAP,
    MAX_POWER_LAW_FIT_POINTS, WAVEFRONT_TRACE_CAP,
};

/// Critical grain threshold $z_c = 4$ on the 2D square von Neumann lattice.
pub const CRITICAL_HEIGHT: u32 = 4;

/// Maximum stable grain height $z_{\max} = z_c - 1 = 3$ on the 2D square lattice.
pub const MAX_STABLE_HEIGHT: u32 = 3;

/// Maximum interactive lattice width or height (`128 x 128`).
pub const MAX_SANDPILE_DIM: usize = 128;

/// Outcome of Dhar's burning recurrence algorithm on a 2D sandpile configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BurningTestResult {
    /// `true` if and only if every cell in the lattice burned (configuration is recurrent).
    pub is_recurrent: bool,
    /// Number of lattice sites burned before the fire extinguished.
    pub burned_cells: usize,
    /// Total number of lattice sites (`width * height`).
    pub total_cells: usize,
    /// Deepest synchronous burning wave reached (`0` if no cell burned).
    pub max_wave_depth: u16,
}

/// Pre-allocated 2D Bak-Tang-Wiesenfeld Abelian sandpile engine.
///
/// # Examples
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::stat_mech::sandpile::SandpileGrid;
///
/// let mut grid = SandpileGrid::new(5, 5, Some(42)).unwrap();
/// grid.fill(4);
/// grid.add_grains(0, 0, 18).unwrap();
/// let stats = grid.stabilize_wavefront();
/// assert_eq!(stats.topples, 105);
/// assert!(grid.dhar_burning_test().is_recurrent);
/// ```
#[derive(Debug, Clone)]
pub struct SandpileGrid {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) heights: Vec<u32>,
    pub(super) scratch_a: Vec<u32>,
    pub(super) scratch_b: Vec<u32>,
    pub(super) ring_buf: Vec<u32>,
    pub(super) ring_mask: usize,
    pub(super) ring_head: usize,
    pub(super) ring_tail: usize,
    pub(super) in_queue: Vec<u8>,
    pub(super) touched_epoch: Vec<u32>,
    pub(super) epoch_counter: u32,
    pub(super) burn_depth: Vec<u16>,
    pub(super) rem_deg: Vec<u8>,
    rng: OxidizeRng,
    telemetry: AvalancheTelemetry,
}

impl SandpileGrid {
    /// Allocates a new `width x height` open-boundary sandpile grid (`1..=MAX_SANDPILE_DIM`).
    #[verified_engine::verified]
    pub fn new(width: usize, height: usize, seed: Option<u64>) -> Result<Self, SandpileError> {
        if width == 0 || height == 0 || width > MAX_SANDPILE_DIM || height > MAX_SANDPILE_DIM {
            return Err(SandpileError::InvalidDimensions {
                width,
                height,
                max_dim: MAX_SANDPILE_DIM,
            });
        }
        let cells = width * height;
        let ring_cap = (cells * 2).next_power_of_two().max(4);
        let rng = seed.map_or_else(OxidizeRng::default, OxidizeRng::from_seed);
        Ok(Self {
            width,
            height,
            heights: vec![0; cells],
            scratch_a: vec![0; cells],
            scratch_b: vec![0; cells],
            ring_buf: vec![0; ring_cap],
            ring_mask: ring_cap - 1,
            ring_head: 0,
            ring_tail: 0,
            in_queue: vec![0; cells],
            touched_epoch: vec![0; cells],
            epoch_counter: 1,
            burn_depth: vec![0; cells],
            rem_deg: vec![0; cells],
            rng,
            telemetry: AvalancheTelemetry::new(),
        })
    }

    /// Lattice width in cells.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Lattice height in cells.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Row-major slice of grain heights indexed via `flatten_2d_index(x, y, width)`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn heights(&self) -> &[u32] {
        &self.heights
    }

    /// Per-cell synchronous wave depths (`0` = unburned) from the most recent [`Self::dhar_burning_test`].
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn burn_depths(&self) -> &[u16] {
        &self.burn_depth
    }

    /// Borrows the zero-allocation avalanche scaling telemetry accumulator.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn telemetry(&self) -> &AvalancheTelemetry {
        &self.telemetry
    }

    /// Mutable access to the avalanche telemetry accumulator (e.g., to clear histograms).
    #[inline]
    #[verified_engine::verified]
    pub fn telemetry_mut(&mut self) -> &mut AvalancheTelemetry {
        &mut self.telemetry
    }

    /// Reseeds the deterministic [`OxidizeRng`] stream without allocating.
    #[inline]
    #[verified_engine::verified]
    pub fn reseed(&mut self, seed: u64) {
        self.rng = OxidizeRng::from_seed(seed);
    }

    /// Sets every cell in the lattice to `height` and clears the active wavefront queue.
    #[verified_engine::verified]
    pub fn fill(&mut self, height: u32) {
        self.heights.fill(height);
        self.clear_ring_queue();
        if height >= CRITICAL_HEIGHT {
            self.seed_unstable_into_ring();
        }
    }

    /// Copies an external row-major height slice into the lattice and refreshes unstable wavefronts.
    #[verified_engine::verified]
    pub fn set_heights(&mut self, values: &[u32]) -> Result<(), SandpileError> {
        if values.len() != self.heights.len() {
            return Err(SandpileError::BufferLengthMismatch {
                expected: self.heights.len(),
                actual: values.len(),
            });
        }
        self.heights.copy_from_slice(values);
        self.clear_ring_queue();
        self.seed_unstable_into_ring();
        Ok(())
    }

    /// Reads the grain height at `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn height_at(&self, x: usize, y: usize) -> Result<u32, SandpileError> {
        if x >= self.width || y >= self.height {
            return Err(SandpileError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        Ok(self.heights[flatten_2d_index(x, y, self.width)])
    }

    /// Adds `grains` to `(x, y)` and enqueues the cell into the wavefront ring buffer if unstable.
    #[inline]
    #[verified_engine::verified]
    pub fn add_grains(&mut self, x: usize, y: usize, grains: u32) -> Result<(), SandpileError> {
        if x >= self.width || y >= self.height {
            return Err(SandpileError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        let idx = flatten_2d_index(x, y, self.width);
        self.heights[idx] = self.heights[idx].saturating_add(grains);
        if self.heights[idx] >= CRITICAL_HEIGHT && self.in_queue[idx] == 0 {
            self.in_queue[idx] = 1;
            self.ring_push(idx as u32);
        }
        Ok(())
    }

    /// Drops a single grain at a uniformly sampled cell using [`OxidizeRng`], returning `(x, y)`.
    #[inline]
    #[verified_engine::verified]
    pub fn drop_random_grain(&mut self) -> (usize, usize) {
        let x = sample_bounded_index(&mut self.rng, self.width);
        let y = sample_bounded_index(&mut self.rng, self.height);
        let idx = flatten_2d_index(x, y, self.width);
        self.heights[idx] += 1;
        if self.heights[idx] >= CRITICAL_HEIGHT && self.in_queue[idx] == 0 {
            self.in_queue[idx] = 1;
            self.ring_push(idx as u32);
        }
        (x, y)
    }

    /// Returns `true` when at least one lattice site is queued for toppling ($z \ge 4$).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn has_active_wavefront(&self) -> bool {
        self.ring_head != self.ring_tail
    }

    /// Advances exactly one synchronous integer wavefront pass, returning `(sites_toppled, grains_dissipated)`.
    #[verified_engine::verified]
    pub fn step_wavefront(&mut self) -> (usize, u64) {
        let wave_len = self.ring_len();
        if wave_len == 0 {
            return (0, 0);
        }
        let mut dissipated = 0_u64;
        for slot in 0..wave_len {
            let idx = self.ring_pop() as usize;
            self.in_queue[idx] = 0;
            self.scratch_a[slot] = idx as u32;
            self.heights[idx] -= CRITICAL_HEIGHT;
        }
        for slot in 0..wave_len {
            let idx = self.scratch_a[slot] as usize;
            dissipated += self.distribute_single_topple(idx);
            if self.heights[idx] >= CRITICAL_HEIGHT && self.in_queue[idx] == 0 {
                self.in_queue[idx] = 1;
                self.ring_push(idx as u32);
            }
        }
        (wave_len, dissipated)
    }

    /// Stabilizes all unstable cells via integer wavefront ring-buffer passes and records telemetry.
    #[verified_engine::verified]
    pub fn stabilize_wavefront(&mut self) -> AvalancheStats {
        self.ensure_unstable_seeded();
        let epoch = self.next_epoch();
        self.telemetry.reset_wavefront_trace();
        let mut stats = AvalancheStats::default();

        while self.has_active_wavefront() {
            let wave_len = self.ring_len();
            stats.duration += 1;
            stats.peak_wavefront = stats.peak_wavefront.max(wave_len);
            self.telemetry.push_wavefront_step(wave_len);

            for slot in 0..wave_len {
                let idx = self.ring_pop() as usize;
                self.in_queue[idx] = 0;
                self.scratch_a[slot] = idx as u32;
                let topples = self.heights[idx] / CRITICAL_HEIGHT;
                self.scratch_b[slot] = topples;
                self.heights[idx] %= CRITICAL_HEIGHT;
                stats.topples += u64::from(topples);
                if self.touched_epoch[idx] != epoch {
                    self.touched_epoch[idx] = epoch;
                    stats.area += 1;
                }
            }
            for slot in 0..wave_len {
                let idx = self.scratch_a[slot] as usize;
                let topples = self.scratch_b[slot];
                stats.dissipated_grains += self.distribute_multi_topple(idx, topples);
            }
        }

        self.telemetry.record(stats);
        stats
    }

    /// Bulk scaling cutoff prior to open-boundary dissipation rolloff for `observable`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn scaling_cutoff(&self, observable: AvalancheObservable) -> usize {
        let l_min = self.width.min(self.height);
        match observable {
            AvalancheObservable::Size | AvalancheObservable::Area => {
                (l_min * 2).clamp(12, EXACT_AVALANCHE_BUCKET_CAP / 2)
            }
            AvalancheObservable::Duration => l_min.clamp(8, 64),
        }
    }

    /// Fits [`math_commons::fit::FitModel::PowerLaw`] to the accumulated avalanche survival distribution.
    #[verified_engine::verified]
    pub fn fit_avalanche_power_law(
        &self,
        observable: AvalancheObservable,
    ) -> Result<CurveFit, FitError> {
        self.telemetry
            .fit_power_law(observable, self.scaling_cutoff(observable))
    }
}

#[inline]
fn sample_bounded_index(rng: &mut OxidizeRng, bound: usize) -> usize {
    debug_assert!(bound > 0 && bound <= MAX_SANDPILE_DIM);
    let b = bound as u32;
    let threshold = u32::MAX - (u32::MAX % b);
    loop {
        let r = rng.next_u32();
        if r < threshold {
            return (r % b) as usize;
        }
    }
}
