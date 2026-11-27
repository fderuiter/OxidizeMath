//! Zero-allocation `#![no_std]` stochastic solver kernel (`stochastic`).
//!
//! Provides `#![no_std]` [`SolverWorkspace`](math_memory::SolverWorkspace)-driven kernels and
//! (with `alloc`) a single-arena [`StochasticWorkspace`](crate::stochastic::StochasticWorkspace) backing:
//! - Iterative Tarjan strongly connected component (SCC) graph decomposition and closed sink
//!   class isolation over discrete-time transition ($P$) and continuous-time generator ($G$)
//!   matrices.
//! - Scale-normalized Householder QR constrained linear solvers for unique stationary and
//!   steady-state distributions ($(P^T - I)\pi = 0$ and $G^T\pi = 0$ subject to $\sum_i \pi_i = 1$),
//!   returning `false` / `None` when multiple closed sink classes make stationary weights
//!   non-unique.
//! - Exact discrete-time AR(1) conditional Gaussian likelihood Ornstein-Uhlenbeck (OU) parameter
//!   calibration with stationary $\sqrt{2\theta \cdot \mathrm{Var}}$ volatility scaling, plus
//!   zero-allocation quantile and ensemble summary statistics.
// [cite:math_commons]

mod ou;
mod qr;
mod tarjan;
mod views;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use math_memory::{ArenaError, SolverWorkspace};
#[cfg(feature = "alloc")]
use math_memory::{ArenaLayout, OwnedByteArena};
#[cfg(feature = "alloc")]
use views::StochasticRegions;
use views::StochasticViews;

pub use ou::{OuCalibration, SampleSummary, calibrate_ou};

/// Summary of a Tarjan strongly connected component (SCC) decomposition over an $n$-state graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SccSummary {
    /// Total number of strongly connected components discovered.
    pub num_components: usize,
    /// Number of closed sink components (recurrent classes with zero outgoing edges to other SCCs).
    pub num_sink_components: usize,
    /// Number of states belonging to the unique closed sink class when `num_sink_components == 1`.
    pub unique_sink_size: usize,
}

/// Decomposes a flat row-major $n \times n$ matrix into strongly connected components using
/// `#![no_std]` scratch carved from `ws`.
///
/// # Errors
/// Returns [`ArenaError`] if `ws` lacks sufficient aligned byte capacity for $n$-state scratch.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn analyze_scc_with_workspace(
    matrix: &[f64],
    num_states: usize,
    ws: &mut (impl SolverWorkspace + ?Sized),
) -> Result<Option<SccSummary>, ArenaError> {
    debug_assert!(matrix.len() >= num_states.saturating_mul(num_states) || matrix.is_empty());
    let Some(nn) = num_states.checked_mul(num_states) else {
        return Ok(None);
    };
    if num_states == 0 || matrix.len() < nn {
        return Ok(None);
    }
    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let mut views = StochasticViews::carve(&mut cur, num_states)?;
    Ok(Some(tarjan::decompose_tarjan(
        &mut views,
        &matrix[..nn],
        num_states,
    )))
}

/// Solves $(P^T - I)\pi = 0$ subject to $\sum_{i=1}^n \pi_i = 1$ for a flat row-major $n \times n$
/// discrete-time Markov transition matrix $P$ using `#![no_std]` scratch carved from `ws`.
///
/// Reference: Stewart, W. J. (1994). *Introduction to the Numerical Solution of Markov Chains*.
/// Princeton University Press. DOI: `10.1515/9780691223384`.
///
/// # Stability
/// Isolates the unique closed recurrent sink class via iterative Tarjan SCC traversal before
/// solving the scale-normalized $(m + 1) \times m$ constrained least-squares system via
/// Householder QR reflections and verifying $\|(P^T - I)\pi\|_2 / \|P^T - I\|_\infty < 10^{-12}$.
///
/// # Domain of Validity
/// Requires `num_states >= 1`, `transition.len() >= num_states * num_states` with finite entries,
/// `out_pi.len() >= num_states`, and a unique closed sink component (`num_sink_components == 1`).
///
/// # Errors
/// Returns [`ArenaError`] if `ws` lacks sufficient byte capacity for $n$-state Tarjan/QR scratch.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn solve_dtmc_stationary_with_workspace(
    transition: &[f64],
    num_states: usize,
    out_pi: &mut [f64],
    ws: &mut (impl SolverWorkspace + ?Sized),
) -> Result<bool, ArenaError> {
    debug_assert!(out_pi.len() >= num_states || out_pi.is_empty());
    let Some(nn) = num_states.checked_mul(num_states) else {
        return Ok(false);
    };
    if num_states == 0 || out_pi.len() < num_states || transition.len() < nn {
        return Ok(false);
    }
    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let mut views = StochasticViews::carve(&mut cur, num_states)?;
    Ok(qr::solve_stationary_slice(
        &mut views,
        &transition[..nn],
        num_states,
        false,
        &mut out_pi[..num_states],
    ))
}

/// Solves $G^T \pi = 0$ subject to $\sum_{i=1}^n \pi_i = 1$ for a flat row-major $n \times n$
/// continuous-time Markov generator matrix $G$ using `#![no_std]` scratch carved from `ws`.
///
/// Reference: Stewart, W. J. (1994). *Introduction to the Numerical Solution of Markov Chains*.
/// Princeton University Press. DOI: `10.1515/9780691223384`.
///
/// # Stability
/// Normalizes the isolated recurrent sub-generator $G_C^T$ by $\max |G_{ij}|$ so ultra-slow rates
/// ($10^{-7}$) and stiff generators maintain Householder QR pivot stability and scale-invariant
/// residual verification $\|G^T \pi\|_2 / \|G\|_\infty < 10^{-12}$.
///
/// # Domain of Validity
/// Requires `num_states >= 1`, `generator.len() >= num_states * num_states` with finite entries,
/// `out_pi.len() >= num_states`, and a unique closed sink component (`num_sink_components == 1`).
///
/// # Errors
/// Returns [`ArenaError`] if `ws` lacks sufficient byte capacity for $n$-state Tarjan/QR scratch.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn solve_ctmc_steady_state_with_workspace(
    generator: &[f64],
    num_states: usize,
    out_pi: &mut [f64],
    ws: &mut (impl SolverWorkspace + ?Sized),
) -> Result<bool, ArenaError> {
    debug_assert!(out_pi.len() >= num_states || out_pi.is_empty());
    let Some(nn) = num_states.checked_mul(num_states) else {
        return Ok(false);
    };
    if num_states == 0 || out_pi.len() < num_states || generator.len() < nn {
        return Ok(false);
    }
    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let mut views = StochasticViews::carve(&mut cur, num_states)?;
    Ok(qr::solve_stationary_slice(
        &mut views,
        &generator[..nn],
        num_states,
        true,
        &mut out_pi[..num_states],
    ))
}

/// Computes sample mean, standard deviation, median, 5th/95th percentiles, and optional exceedance
/// probability in `#![no_std]` using sorting scratch carved from `ws`.
///
/// # Errors
/// Returns [`ArenaError`] if `ws` lacks `samples.len() * size_of::<f64>()` aligned bytes.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn summarize_samples_with_workspace(
    samples: &[f64],
    threshold: Option<f64>,
    ws: &mut (impl SolverWorkspace + ?Sized),
) -> Result<Option<SampleSummary>, ArenaError> {
    debug_assert!(samples.is_empty() || !samples.is_empty());
    if samples.is_empty() {
        return Ok(None);
    }
    let mut sub = ws.reborrow();
    let scratch = sub.alloc_slice::<f64>(samples.len())?;
    Ok(ou::summarize_samples_into(samples, threshold, scratch))
}

/// Preallocated single-arena scratch workspace (`alloc` feature) for zero-allocation Markov
/// stationary/steady-state solvers, Tarjan SCC traversal, and Ornstein-Uhlenbeck sample summaries.
///
/// # Example
/// ```
/// use math_commons::stochastic::StochasticWorkspace;
///
/// let p = [
///     0.0, 0.5, 0.5,
///     1.0, 0.0, 0.0,
///     1.0, 0.0, 0.0,
/// ];
/// let mut ws = StochasticWorkspace::new(3, 0);
/// let mut pi = [0.0; 3];
/// assert!(ws.solve_dtmc_stationary(&p, 3, &mut pi));
/// assert!((pi[0] - 0.5).abs() < 1e-12);
/// assert!((pi[1] - 0.25).abs() < 1e-12);
/// assert!((pi[2] - 0.25).abs() < 1e-12);
/// ```
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default)]
pub struct StochasticWorkspace {
    arena: OwnedByteArena,
    state_capacity: usize,
    sample_capacity: usize,
    regions: StochasticRegions,
}

#[cfg(feature = "alloc")]
impl StochasticWorkspace {
    /// Fallibly allocates a single-buffer workspace pre-sized for `state_capacity` Markov states
    /// and `sample_capacity` Monte Carlo terminal samples.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on capacity arithmetic overflow or allocator failure.
    pub fn try_new(state_capacity: usize, sample_capacity: usize) -> Result<Self, ArenaError> {
        let mut ws = Self::default();
        ws.try_ensure_capacities(state_capacity, sample_capacity)?;
        Ok(ws)
    }

    /// Allocates a single-buffer workspace pre-sized for `state_capacity` Markov states and
    /// `sample_capacity` Monte Carlo terminal samples.
    #[must_use]
    pub fn new(state_capacity: usize, sample_capacity: usize) -> Self {
        Self::try_new(state_capacity, sample_capacity).expect("StochasticWorkspace::new failed")
    }

    /// Returns the currently allocated Markov state capacity.
    #[must_use]
    #[inline]
    pub const fn state_capacity(&self) -> usize {
        self.state_capacity
    }

    /// Returns the currently allocated Monte Carlo sample capacity.
    #[must_use]
    #[inline]
    pub const fn sample_capacity(&self) -> usize {
        self.sample_capacity
    }

    fn try_ensure_capacities(
        &mut self,
        min_states: usize,
        min_samples: usize,
    ) -> Result<(), ArenaError> {
        let target_states = self.state_capacity.max(min_states);
        let target_samples = self.sample_capacity.max(min_samples);
        if target_states == self.state_capacity && target_samples == self.sample_capacity {
            return Ok(());
        }
        let mut layout = ArenaLayout::new();
        let regions = StochasticRegions::reserve(&mut layout, target_states, target_samples)?;
        self.arena.try_ensure_capacity(layout.total_bytes())?;
        self.arena.clear_zero();
        self.state_capacity = target_states;
        self.sample_capacity = target_samples;
        self.regions = regions;
        Ok(())
    }

    /// Fallibly ensures internal graph and Householder QR buffers accommodate `num_states` states.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on arithmetic overflow or allocation failure.
    pub fn try_ensure_state_capacity(&mut self, num_states: usize) -> Result<(), ArenaError> {
        self.try_ensure_capacities(num_states, self.sample_capacity)
    }

    /// Ensures internal graph and Householder QR buffers accommodate `num_states` states without
    /// reallocating on subsequent calls of size `<= num_states`.
    pub fn ensure_state_capacity(&mut self, num_states: usize) {
        self.try_ensure_state_capacity(num_states)
            .expect("StochasticWorkspace::ensure_state_capacity failed");
    }

    /// Fallibly ensures the quantile sorting buffer accommodates `num_samples` observations.
    ///
    /// # Errors
    /// Returns [`ArenaError`] on arithmetic overflow or allocation failure.
    pub fn try_ensure_sample_capacity(&mut self, num_samples: usize) -> Result<(), ArenaError> {
        self.try_ensure_capacities(self.state_capacity, num_samples)
    }

    /// Ensures the quantile sorting buffer accommodates `num_samples` observations without
    /// reallocating on subsequent calls of size `<= num_samples`.
    pub fn ensure_sample_capacity(&mut self, num_samples: usize) {
        self.try_ensure_sample_capacity(num_samples)
            .expect("StochasticWorkspace::ensure_sample_capacity failed");
    }

    /// Mutable view of the preallocated $n \times n$ flat row-major operator staging buffer.
    pub fn operator_staging_mut(&mut self, num_states: usize) -> &mut [f64] {
        self.ensure_state_capacity(num_states);
        &mut self.arena.region_slice_mut(self.regions.operator_buf)[..num_states * num_states]
    }

    /// Solves the DTMC stationary distribution staged in [`Self::operator_staging_mut`].
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_dtmc_stationary(&mut self, num_states: usize, out_pi: &mut [f64]) -> bool {
        debug_assert!(out_pi.len() >= num_states);
        if num_states == 0 || out_pi.len() < num_states || self.state_capacity < num_states {
            return false;
        }
        let regions = self.regions;
        let mut raw = self.arena.raw_cursor();
        let (op, _, mut views) = regions.split_all(&mut raw, num_states);
        qr::solve_stationary_slice(&mut views, op, num_states, false, &mut out_pi[..num_states])
    }

    /// Solves the CTMC steady-state distribution staged in [`Self::operator_staging_mut`].
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_ctmc_steady_state(
        &mut self,
        num_states: usize,
        out_pi: &mut [f64],
    ) -> bool {
        debug_assert!(out_pi.len() >= num_states);
        if num_states == 0 || out_pi.len() < num_states || self.state_capacity < num_states {
            return false;
        }
        let regions = self.regions;
        let mut raw = self.arena.raw_cursor();
        let (op, _, mut views) = regions.split_all(&mut raw, num_states);
        qr::solve_stationary_slice(&mut views, op, num_states, true, &mut out_pi[..num_states])
    }

    /// Solves the staged DTMC stationary distribution into internal arena memory and returns a
    /// borrowed `&[f64]` slice with zero heap allocations.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_dtmc_stationary_slice(&mut self, num_states: usize) -> Option<&[f64]> {
        debug_assert!(self.state_capacity >= num_states || num_states == 0);
        if num_states == 0 || self.state_capacity < num_states {
            return None;
        }
        let regions = self.regions;
        let ok = {
            let mut raw = self.arena.raw_cursor();
            let (op, sol, mut views) = regions.split_all(&mut raw, num_states);
            qr::solve_stationary_slice(&mut views, op, num_states, false, sol)
        };
        ok.then(|| &self.arena.region_slice(regions.solution_buf)[..num_states])
    }

    /// Solves the staged CTMC steady-state distribution into internal arena memory and returns a
    /// borrowed `&[f64]` slice with zero heap allocations.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_ctmc_steady_state_slice(&mut self, num_states: usize) -> Option<&[f64]> {
        debug_assert!(self.state_capacity >= num_states || num_states == 0);
        if num_states == 0 || self.state_capacity < num_states {
            return None;
        }
        let regions = self.regions;
        let ok = {
            let mut raw = self.arena.raw_cursor();
            let (op, sol, mut views) = regions.split_all(&mut raw, num_states);
            qr::solve_stationary_slice(&mut views, op, num_states, true, sol)
        };
        ok.then(|| &self.arena.region_slice(regions.solution_buf)[..num_states])
    }

    /// Decomposes a flat row-major $n \times n$ matrix into SCCs without heap allocation.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn analyze_scc(&mut self, matrix: &[f64], num_states: usize) -> Option<SccSummary> {
        debug_assert!(matrix.len() >= num_states.saturating_mul(num_states));
        if num_states == 0 || matrix.len() < num_states.checked_mul(num_states)? {
            return None;
        }
        self.ensure_state_capacity(num_states);
        let regions = self.regions;
        let mut raw = self.arena.raw_cursor();
        let (_, _, mut views) = regions.split_all(&mut raw, num_states);
        Some(tarjan::decompose_tarjan(
            &mut views,
            &matrix[..num_states * num_states],
            num_states,
        ))
    }

    /// Returns the state indices belonging to the unique closed sink class from the most recent
    /// [`Self::analyze_scc`] or stationary solve call when `num_sink_components == 1`.
    #[must_use]
    pub fn unique_sink_members(&self, unique_sink_size: usize) -> &[usize] {
        let sink = self.arena.region_slice(self.regions.sink_members);
        &sink[..unique_sink_size.min(sink.len())]
    }

    /// Solves $(P^T - I)\pi = 0$ with $\sum_i \pi_i = 1$ for `transition` ($n \times n$).
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_dtmc_stationary(
        &mut self,
        transition: &[f64],
        num_states: usize,
        out_pi: &mut [f64],
    ) -> bool {
        debug_assert!(out_pi.len() >= num_states);
        if num_states == 0
            || out_pi.len() < num_states
            || transition.len() < num_states.saturating_mul(num_states)
        {
            return false;
        }
        self.ensure_state_capacity(num_states);
        let regions = self.regions;
        let mut raw = self.arena.raw_cursor();
        let (_, _, mut views) = regions.split_all(&mut raw, num_states);
        qr::solve_stationary_slice(
            &mut views,
            &transition[..num_states * num_states],
            num_states,
            false,
            &mut out_pi[..num_states],
        )
    }

    /// Solves $G^T \pi = 0$ with $\sum_i \pi_i = 1$ for `generator` ($n \times n$).
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_ctmc_steady_state(
        &mut self,
        generator: &[f64],
        num_states: usize,
        out_pi: &mut [f64],
    ) -> bool {
        debug_assert!(out_pi.len() >= num_states);
        if num_states == 0
            || out_pi.len() < num_states
            || generator.len() < num_states.saturating_mul(num_states)
        {
            return false;
        }
        self.ensure_state_capacity(num_states);
        let regions = self.regions;
        let mut raw = self.arena.raw_cursor();
        let (_, _, mut views) = regions.split_all(&mut raw, num_states);
        qr::solve_stationary_slice(
            &mut views,
            &generator[..num_states * num_states],
            num_states,
            true,
            &mut out_pi[..num_states],
        )
    }

    /// Calibrates Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ with zero allocations.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn calibrate_ou(&self, observations: &[f64], dt: f64) -> Option<OuCalibration> {
        debug_assert!(dt.is_finite() || dt <= 0.0 || dt.is_nan());
        ou::calibrate_ou(observations, dt)
    }

    /// Computes sample summary statistics in-place using the preallocated sorting region.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn summarize_samples(
        &mut self,
        samples: &[f64],
        threshold: Option<f64>,
    ) -> Option<SampleSummary> {
        debug_assert!(samples.is_empty() || !samples.is_empty());
        if samples.is_empty() {
            return None;
        }
        self.ensure_sample_capacity(samples.len());
        let scratch =
            &mut self.arena.region_slice_mut(self.regions.sample_scratch)[..samples.len()];
        ou::summarize_samples_into(samples, threshold, scratch)
    }
}

/// Convenience allocating wrapper (`alloc` feature) solving the unique DTMC stationary distribution.
#[cfg(feature = "alloc")]
#[must_use]
pub fn dtmc_stationary_distribution(transition: &[f64], num_states: usize) -> Option<Vec<f64>> {
    if num_states == 0 || transition.len() < num_states.checked_mul(num_states)? {
        return None;
    }
    let mut ws = StochasticWorkspace::new(num_states, 0);
    let mut out = vec![0.0; num_states];
    ws.solve_dtmc_stationary(transition, num_states, &mut out)
        .then_some(out)
}

/// Convenience allocating wrapper (`alloc` feature) solving the unique CTMC steady-state distribution.
#[cfg(feature = "alloc")]
#[must_use]
pub fn ctmc_steady_state(generator: &[f64], num_states: usize) -> Option<Vec<f64>> {
    if num_states == 0 || generator.len() < num_states.checked_mul(num_states)? {
        return None;
    }
    let mut ws = StochasticWorkspace::new(num_states, 0);
    let mut out = vec![0.0; num_states];
    ws.solve_ctmc_steady_state(generator, num_states, &mut out)
        .then_some(out)
}
