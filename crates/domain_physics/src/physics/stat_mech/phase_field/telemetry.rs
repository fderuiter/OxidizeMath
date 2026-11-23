//! Pre-allocated radial structure factor $S(k, t)$ and domain-coarsening history store.

// [cite:phase_field]

use pure_math::pure_math::analysis::pde::spde::{RadialSpectrumSummary, SemiImplicitSpectral2D};

use crate::physics::stat_mech::interface_growth::scaling::fit_log_log_slope;

/// Fixed-capacity zero-allocation telemetry buffers for radial $S(k, t)$, $k_{\max}(t)$, and $F_h[\phi]$.
#[derive(Clone, Debug)]
pub struct PhaseFieldTelemetry {
    num_bins: usize,
    k_bins: Vec<f64>,
    s_k: Vec<f64>,
    bin_counts: Vec<usize>,
    latest_summary: RadialSpectrumSummary,
    capacity: usize,
    len: usize,
    times: Vec<f64>,
    k_max_history: Vec<f64>,
    k_moment_history: Vec<f64>,
    free_energy_history: Vec<f64>,
}

impl PhaseFieldTelemetry {
    /// Pre-allocates radial $S(k, t)$ bins and coarsening time-series storage.
    #[verified_engine::verified]
    pub fn new(num_bins: usize, capacity: usize) -> Self {
        assert!(num_bins >= 2, "Radial bin count must be >= 2");
        assert!(capacity >= 4, "History capacity must be >= 4");
        Self {
            num_bins,
            k_bins: vec![0.0; num_bins],
            s_k: vec![0.0; num_bins],
            bin_counts: vec![0usize; num_bins],
            latest_summary: RadialSpectrumSummary::default(),
            capacity,
            len: 0,
            times: vec![0.0; capacity],
            k_max_history: vec![0.0; capacity],
            k_moment_history: vec![0.0; capacity],
            free_energy_history: vec![0.0; capacity],
        }
    }

    /// Resets recorded coarsening history in place without deallocating buffers.
    #[verified_engine::verified]
    pub fn clear(&mut self) {
        self.len = 0;
        self.s_k.fill(0.0);
        self.latest_summary = RadialSpectrumSummary::default();
    }

    /// Number of recorded coarsening time-series samples.
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether zero coarsening samples have been recorded.
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Radial wavenumber shell centers $k$.
    #[verified_engine::verified]
    pub fn k_bins(&self) -> &[f64] {
        &self.k_bins[..self.num_bins]
    }

    /// Circularly averaged radial structure factor $S(k, t)$ from the latest evaluation.
    #[verified_engine::verified]
    pub fn s_k(&self) -> &[f64] {
        &self.s_k[..self.num_bins]
    }

    /// Summary metrics (`k_max`, `s_max`, `k_first_moment`) from the latest $S(k, t)$ evaluation.
    #[verified_engine::verified]
    pub fn latest_summary(&self) -> RadialSpectrumSummary {
        self.latest_summary
    }

    /// Recorded sample times $t$.
    #[verified_engine::verified]
    pub fn times(&self) -> &[f64] {
        &self.times[..self.len]
    }

    /// Recorded peak wavenumber trajectory $k_{\max}(t)$.
    #[verified_engine::verified]
    pub fn k_max_history(&self) -> &[f64] {
        &self.k_max_history[..self.len]
    }

    /// Recorded first spectral moment trajectory $k_1(t)$.
    #[verified_engine::verified]
    pub fn k_moment_history(&self) -> &[f64] {
        &self.k_moment_history[..self.len]
    }

    /// Recorded discrete Ginzburg-Landau free energy trajectory $F_h[\phi(t)]$.
    #[verified_engine::verified]
    pub fn free_energy_history(&self) -> &[f64] {
        &self.free_energy_history[..self.len]
    }

    /// Computes radial $S(k, t)$ and appends `(time, k_max, k_first_moment, free_energy)` without heap allocation.
    #[verified_engine::verified]
    pub fn update_from_field(
        &mut self,
        spectral: &mut SemiImplicitSpectral2D,
        field: &[f64],
        time: f64,
        free_energy: f64,
    ) -> RadialSpectrumSummary {
        let summary = spectral.compute_radial_structure_factor(
            field,
            &mut self.k_bins,
            &mut self.s_k,
            &mut self.bin_counts,
        );
        self.latest_summary = summary;
        if self.len == self.capacity {
            decimate_four_series(
                &mut self.times,
                &mut self.k_max_history,
                &mut self.k_moment_history,
                &mut self.free_energy_history,
                &mut self.len,
            );
        }
        let idx = self.len;
        self.times[idx] = time;
        self.k_max_history[idx] = summary.k_max;
        self.k_moment_history[idx] = summary.k_first_moment;
        self.free_energy_history[idx] = free_energy;
        self.len += 1;
        summary
    }

    /// Fits the log-log coarsening slope of $k_1(t)$ over $t \in [t_{\min}, t_{\max}]$ (e.g., $-1/3$ for Model B).
    #[verified_engine::verified]
    pub fn fit_coarsening_slope(&self, t_min: f64, t_max: f64) -> Option<(f64, f64)> {
        fit_log_log_slope(
            &self.times[..self.len],
            &self.k_moment_history[..self.len],
            t_min,
            t_max,
        )
    }
}

fn decimate_four_series(
    a: &mut [f64],
    b: &mut [f64],
    c: &mut [f64],
    d: &mut [f64],
    len: &mut usize,
) {
    let mut write = 0usize;
    let mut read = 0usize;
    while read < *len {
        a[write] = a[read];
        b[write] = b[read];
        c[write] = c[read];
        d[write] = d[read];
        write += 1;
        read += 2;
    }
    *len = write;
}
