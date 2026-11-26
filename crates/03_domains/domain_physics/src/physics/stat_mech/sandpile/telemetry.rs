//! Zero-Allocation Avalanche Telemetry, Complementary Survival Curves, and Power-Law Fitting.
//!
//! Accumulates exact integer avalanche histograms on stack/fixed-size buffers during critical
//! sandpile cascades and fits parameter-free `1/f` scaling laws via
//! [`math_commons::fit::FitModel::PowerLaw`] without heap allocation.

use math_commons::error::FitError;
use math_commons::fit::{CurveFit, FitModel, fit_model};
use serde::{Deserialize, Serialize};

/// Maximum exact integer bucket tracked before accumulating into the overflow tail counter.
pub const EXACT_AVALANCHE_BUCKET_CAP: usize = 256;

/// Maximum number of log-spaced points emitted for zero-allocation power-law curve fitting.
pub const MAX_POWER_LAW_FIT_POINTS: usize = 20;

/// Maximum number of wavefront topple counts retained for the most recent cascade profile.
pub const WAVEFRONT_TRACE_CAP: usize = 256;

/// Observable channel for avalanche scaling distributions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AvalancheObservable {
    /// Total topple count $S$ per avalanche.
    Size,
    /// Distinct toppled site count $A$ per avalanche.
    Area,
    /// Synchronous wavefront step count $T$ per avalanche.
    Duration,
}

/// Summary telemetry produced by stabilizing a single avalanche or multi-site perturbation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvalancheStats {
    /// Total number of individual site topplings ($S$).
    pub topples: u64,
    /// Number of distinct lattice sites that toppled at least once ($A$).
    pub area: usize,
    /// Number of synchronous wavefront steps or passes ($T$).
    pub duration: usize,
    /// Total grains dissipated across the open sink boundary.
    pub dissipated_grains: u64,
    /// Maximum number of sites toppling simultaneously within a single wavefront.
    pub peak_wavefront: usize,
}

/// Fixed-capacity zero-allocation accumulator for critical avalanche scaling statistics.
#[derive(Debug, Clone)]
pub struct AvalancheTelemetry {
    size_counts: [u64; EXACT_AVALANCHE_BUCKET_CAP],
    size_tail: u64,
    area_counts: [u64; EXACT_AVALANCHE_BUCKET_CAP],
    area_tail: u64,
    duration_counts: [u64; EXACT_AVALANCHE_BUCKET_CAP],
    duration_tail: u64,
    total_avalanches: u64,
    zero_topple_drops: u64,
    last_stats: AvalancheStats,
    wavefront_trace: [u32; WAVEFRONT_TRACE_CAP],
    wavefront_len: usize,
}

impl Default for AvalancheTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl AvalancheTelemetry {
    /// Creates an empty fixed-capacity avalanche telemetry accumulator.
    #[must_use]
    #[verified_engine::verified]
    pub const fn new() -> Self {
        Self {
            size_counts: [0; EXACT_AVALANCHE_BUCKET_CAP],
            size_tail: 0,
            area_counts: [0; EXACT_AVALANCHE_BUCKET_CAP],
            area_tail: 0,
            duration_counts: [0; EXACT_AVALANCHE_BUCKET_CAP],
            duration_tail: 0,
            total_avalanches: 0,
            zero_topple_drops: 0,
            last_stats: AvalancheStats {
                topples: 0,
                area: 0,
                duration: 0,
                dissipated_grains: 0,
                peak_wavefront: 0,
            },
            wavefront_trace: [0; WAVEFRONT_TRACE_CAP],
            wavefront_len: 0,
        }
    }

    /// Resets all accumulated histograms and cascade traces to zero.
    #[verified_engine::verified]
    pub fn clear(&mut self) {
        self.size_counts.fill(0);
        self.size_tail = 0;
        self.area_counts.fill(0);
        self.area_tail = 0;
        self.duration_counts.fill(0);
        self.duration_tail = 0;
        self.total_avalanches = 0;
        self.zero_topple_drops = 0;
        self.last_stats = AvalancheStats::default();
        self.wavefront_len = 0;
    }

    /// Records a completed stabilization event into the size, area, and duration histograms.
    #[inline]
    pub fn record(&mut self, stats: AvalancheStats) {
        self.last_stats = stats;
        if stats.topples == 0 {
            self.zero_topple_drops += 1;
            return;
        }
        self.total_avalanches += 1;
        record_bucket(
            stats.topples as usize,
            &mut self.size_counts,
            &mut self.size_tail,
        );
        record_bucket(stats.area, &mut self.area_counts, &mut self.area_tail);
        record_bucket(
            stats.duration,
            &mut self.duration_counts,
            &mut self.duration_tail,
        );
    }

    /// Clears the wavefront trace prior to a new cascade.
    #[inline]
    pub(crate) fn reset_wavefront_trace(&mut self) {
        self.wavefront_len = 0;
    }

    /// Appends a single wavefront size to the bounded trace buffer.
    #[inline]
    pub(crate) fn push_wavefront_step(&mut self, wavefront_sites: usize) {
        if self.wavefront_len < WAVEFRONT_TRACE_CAP {
            self.wavefront_trace[self.wavefront_len] = wavefront_sites as u32;
            self.wavefront_len += 1;
        }
    }

    /// Total number of non-trivial avalanches ($S \ge 1$) recorded since the last reset.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn total_avalanches(&self) -> u64 {
        self.total_avalanches
    }

    /// Number of grain drops that stabilized immediately without toppling ($S = 0$).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn zero_topple_drops(&self) -> u64 {
        self.zero_topple_drops
    }

    /// Summary statistics of the most recent stabilization event.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn last_stats(&self) -> AvalancheStats {
        self.last_stats
    }

    /// Slice of synchronous wavefront sizes recorded during the most recent cascade.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn wavefront_trace(&self) -> &[u32] {
        &self.wavefront_trace[..self.wavefront_len]
    }

    /// Writes quasi-logarithmically spaced complementary cumulative survival points
    /// `[x, P(X >= x)]` into `out` with zero heap allocations, returning the number of points written.
    #[verified_engine::verified]
    pub fn survival_curve_into(
        &self,
        observable: AvalancheObservable,
        max_cutoff: usize,
        out: &mut [[f64; 2]],
    ) -> usize {
        if self.total_avalanches == 0 || out.is_empty() {
            return 0;
        }
        let (counts, tail) = match observable {
            AvalancheObservable::Size => (&self.size_counts, self.size_tail),
            AvalancheObservable::Area => (&self.area_counts, self.area_tail),
            AvalancheObservable::Duration => (&self.duration_counts, self.duration_tail),
        };
        const THRESHOLDS: [usize; MAX_POWER_LAW_FIT_POINTS] = [
            1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 36, 48, 64, 85, 112, 144, 180, 212, 236, 256,
        ];
        let cap = max_cutoff.clamp(2, EXACT_AVALANCHE_BUCKET_CAP);
        let total = self.total_avalanches as f64;
        let mut written = 0usize;

        for &thr in &THRESHOLDS {
            if thr > cap || written == out.len() {
                break;
            }
            let mut geq = tail;
            for &c in &counts[(thr - 1)..EXACT_AVALANCHE_BUCKET_CAP] {
                geq += c;
            }
            if geq >= 4 {
                out[written] = [thr as f64, (geq as f64) / total];
                written += 1;
            }
        }
        written
    }

    /// Fits [`FitModel::PowerLaw`] ($P(X \ge x) = a \cdot x^b$) to the empirical complementary
    /// cumulative distribution in log-log mode with zero heap allocations.
    #[verified_engine::verified]
    pub fn fit_power_law(
        &self,
        observable: AvalancheObservable,
        max_cutoff: usize,
    ) -> Result<CurveFit, FitError> {
        let mut pts = [[0.0_f64; 2]; MAX_POWER_LAW_FIT_POINTS];
        let len = self.survival_curve_into(observable, max_cutoff, &mut pts);
        fit_model(&pts[..len], FitModel::PowerLaw, true)
    }
}

#[inline]
fn record_bucket(value: usize, counts: &mut [u64; EXACT_AVALANCHE_BUCKET_CAP], tail: &mut u64) {
    if value == 0 {
        return;
    }
    if value <= EXACT_AVALANCHE_BUCKET_CAP {
        counts[value - 1] += 1;
    } else {
        *tail += 1;
    }
}
