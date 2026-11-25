//! Zero-allocation `#![no_std]` stochastic solver kernel (`stochastic`).
//!
//! Provides a preallocated [`SolverWorkspace`] backing:
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

use alloc::vec;
use alloc::vec::Vec;

pub use ou::{OuCalibration, SampleSummary, calibrate_ou};

/// Summary of a Tarjan strongly connected component (SCC) decomposition over an $n$-state graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SccSummary {
    /// Total number of strongly connected components discovered.
    pub num_components: usize,
    /// Number of closed sink components (recurrent classes with zero outgoing edges to other SCCs).
    pub num_sink_components: usize,
    /// Number of states belonging to the unique closed sink class when `num_sink_components == 1`.
    pub unique_sink_size: usize,
}

/// Preallocated `#![no_std]` scratch workspace for zero-allocation Markov stationary/steady-state
/// solvers, Tarjan SCC graph traversal, and Ornstein-Uhlenbeck quantile/summary routines.
///
/// # Example
/// ```
/// use math_commons::stochastic::SolverWorkspace;
///
/// // Periodic 3-state bipartite chain: 0 -> {1, 2} w.p. 0.5 each; {1, 2} -> 0 w.p. 1.0.
/// let p = [
///     0.0, 0.5, 0.5,
///     1.0, 0.0, 0.0,
///     1.0, 0.0, 0.0,
/// ];
/// let mut ws = SolverWorkspace::new(3, 0);
/// let mut pi = [0.0; 3];
/// assert!(ws.solve_dtmc_stationary(&p, 3, &mut pi));
/// assert!((pi[0] - 0.5).abs() < 1e-12);
/// assert!((pi[1] - 0.25).abs() < 1e-12);
/// assert!((pi[2] - 0.25).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SolverWorkspace {
    pub(crate) state_capacity: usize,
    pub(crate) sample_capacity: usize,
    pub(crate) tarjan_index: Vec<usize>,
    pub(crate) tarjan_lowlink: Vec<usize>,
    pub(crate) tarjan_on_stack: Vec<bool>,
    pub(crate) tarjan_stack: Vec<usize>,
    pub(crate) call_node: Vec<usize>,
    pub(crate) call_next_edge: Vec<usize>,
    pub(crate) component_id: Vec<usize>,
    pub(crate) component_is_sink: Vec<bool>,
    pub(crate) sink_members: Vec<usize>,
    pub(crate) row_scales: Vec<f64>,
    pub(crate) operator_buf: Vec<f64>,
    pub(crate) qr_matrix: Vec<f64>,
    pub(crate) qr_rhs: Vec<f64>,
    pub(crate) qr_sol: Vec<f64>,
    pub(crate) solution_buf: Vec<f64>,
    pub(crate) sample_scratch: Vec<f64>,
}

impl SolverWorkspace {
    /// Allocates a workspace pre-sized for up to `state_capacity` Markov states and
    /// `sample_capacity` Monte Carlo terminal samples.
    #[must_use]
    pub fn new(state_capacity: usize, sample_capacity: usize) -> Self {
        let mut ws = Self::default();
        ws.ensure_state_capacity(state_capacity);
        ws.ensure_sample_capacity(sample_capacity);
        ws
    }

    /// Returns the currently allocated Markov state capacity.
    #[must_use]
    #[inline]
    pub fn state_capacity(&self) -> usize {
        self.state_capacity
    }

    /// Returns the currently allocated Monte Carlo sample capacity.
    #[must_use]
    #[inline]
    pub fn sample_capacity(&self) -> usize {
        self.sample_capacity
    }

    /// Ensures internal graph and Householder QR buffers can accommodate `num_states` states
    /// without reallocating on subsequent calls of size `<= num_states`.
    pub fn ensure_state_capacity(&mut self, num_states: usize) {
        if num_states <= self.state_capacity {
            return;
        }
        self.state_capacity = num_states;
        self.tarjan_index.resize(num_states, 0);
        self.tarjan_lowlink.resize(num_states, 0);
        self.tarjan_on_stack.resize(num_states, false);
        self.tarjan_stack.resize(num_states, 0);
        self.call_node.resize(num_states, 0);
        self.call_next_edge.resize(num_states, 0);
        self.component_id.resize(num_states, 0);
        self.component_is_sink.resize(num_states, false);
        self.sink_members.resize(num_states, 0);
        self.row_scales.resize(num_states, 0.0);
        self.operator_buf.resize(num_states * num_states, 0.0);
        self.qr_matrix.resize((num_states + 1) * num_states, 0.0);
        self.qr_rhs.resize(num_states + 1, 0.0);
        self.qr_sol.resize(num_states, 0.0);
        self.solution_buf.resize(num_states, 0.0);
    }

    /// Ensures the quantile sorting scratch buffer can accommodate `num_samples` observations
    /// without reallocating on subsequent calls of size `<= num_samples`.
    pub fn ensure_sample_capacity(&mut self, num_samples: usize) {
        if num_samples <= self.sample_capacity {
            return;
        }
        self.sample_capacity = num_samples;
        self.sample_scratch.resize(num_samples, 0.0);
    }

    /// Mutable view of the preallocated $n \times n$ flat row-major operator staging buffer,
    /// allowing callers (e.g. generic `nalgebra::DMatrix<T>` wrappers) to populate entries
    /// without allocating a temporary `Vec<f64>`.
    pub fn operator_staging_mut(&mut self, num_states: usize) -> &mut [f64] {
        self.ensure_state_capacity(num_states);
        &mut self.operator_buf[..num_states * num_states]
    }

    /// Solves the DTMC stationary distribution staged in [`Self::operator_staging_mut`].
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_dtmc_stationary(&mut self, num_states: usize, out_pi: &mut [f64]) -> bool {
        debug_assert!(out_pi.len() >= num_states);
        if num_states == 0 || out_pi.len() < num_states || self.state_capacity < num_states {
            return false;
        }
        qr::solve_staged_stationary(self, num_states, false, &mut out_pi[..num_states])
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
        qr::solve_staged_stationary(self, num_states, true, &mut out_pi[..num_states])
    }

    /// Solves the staged DTMC stationary distribution into internal workspace memory and returns
    /// a borrowed slice `&[f64]` with zero heap allocations.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_dtmc_stationary_slice(&mut self, num_states: usize) -> Option<&[f64]> {
        debug_assert!(self.state_capacity >= num_states || num_states == 0);
        if num_states == 0 || self.state_capacity < num_states {
            return None;
        }
        let mut sol = core::mem::take(&mut self.solution_buf);
        let ok = qr::solve_staged_stationary(self, num_states, false, &mut sol[..num_states]);
        self.solution_buf = sol;
        ok.then_some(&self.solution_buf[..num_states])
    }

    /// Solves the staged CTMC steady-state distribution into internal workspace memory and returns
    /// a borrowed slice `&[f64]` with zero heap allocations.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_ctmc_steady_state_slice(&mut self, num_states: usize) -> Option<&[f64]> {
        debug_assert!(self.state_capacity >= num_states || num_states == 0);
        if num_states == 0 || self.state_capacity < num_states {
            return None;
        }
        let mut sol = core::mem::take(&mut self.solution_buf);
        let ok = qr::solve_staged_stationary(self, num_states, true, &mut sol[..num_states]);
        self.solution_buf = sol;
        ok.then_some(&self.solution_buf[..num_states])
    }

    /// Decomposes a flat row-major $n \times n$ transition or generator matrix into strongly
    /// connected components and isolates closed sink classes without heap allocation.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn analyze_scc(&mut self, matrix: &[f64], num_states: usize) -> Option<SccSummary> {
        debug_assert!(matrix.len() >= num_states.saturating_mul(num_states));
        if num_states == 0 || matrix.len() < num_states.checked_mul(num_states)? {
            return None;
        }
        self.ensure_state_capacity(num_states);
        Some(tarjan::decompose_tarjan(
            self,
            &matrix[..num_states * num_states],
            num_states,
        ))
    }

    /// Returns the state indices belonging to the unique closed sink class from the most recent
    /// [`Self::analyze_scc`] or stationary solve call when `num_sink_components == 1`.
    #[must_use]
    pub fn unique_sink_members(&self, unique_sink_size: usize) -> &[usize] {
        &self.sink_members[..unique_sink_size.min(self.sink_members.len())]
    }

    /// Solves $(P^T - I)\pi = 0$ with $\sum_i \pi_i = 1$ for a flat row-major $n \times n$
    /// discrete-time transition matrix `transition`, writing the unique stationary distribution
    /// into `out_pi[..num_states]`. Returns `false` if multiple closed sink classes exist.
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
        qr::solve_stationary_slice(
            self,
            &transition[..num_states * num_states],
            num_states,
            false,
            &mut out_pi[..num_states],
        )
    }

    /// Solves $G^T \pi = 0$ with $\sum_i \pi_i = 1$ for a flat row-major $n \times n$
    /// continuous-time generator matrix `generator`, writing the unique steady-state distribution
    /// into `out_pi[..num_states]`. Returns `false` if multiple closed sink classes exist.
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
        qr::solve_stationary_slice(
            self,
            &generator[..num_states * num_states],
            num_states,
            true,
            &mut out_pi[..num_states],
        )
    }

    /// Calibrates Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ from uniformly spaced
    /// discrete observations with zero heap allocations.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn calibrate_ou(&self, observations: &[f64], dt: f64) -> Option<OuCalibration> {
        debug_assert!(dt.is_finite() || dt <= 0.0 || dt.is_nan());
        ou::calibrate_ou(observations, dt)
    }

    /// Computes sample mean, standard deviation, median, 5th/95th percentiles, and optional
    /// threshold exceedance probability in-place using the preallocated scratch buffer.
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
        ou::summarize_samples_into(
            samples,
            threshold,
            &mut self.sample_scratch[..samples.len()],
        )
    }
}

/// Convenience allocating wrapper solving the unique DTMC stationary distribution.
#[must_use]
pub fn dtmc_stationary_distribution(transition: &[f64], num_states: usize) -> Option<Vec<f64>> {
    if num_states == 0 || transition.len() < num_states.checked_mul(num_states)? {
        return None;
    }
    let mut ws = SolverWorkspace::new(num_states, 0);
    let mut out = vec![0.0; num_states];
    ws.solve_dtmc_stationary(transition, num_states, &mut out)
        .then_some(out)
}

/// Convenience allocating wrapper solving the unique CTMC steady-state distribution.
#[must_use]
pub fn ctmc_steady_state(generator: &[f64], num_states: usize) -> Option<Vec<f64>> {
    if num_states == 0 || generator.len() < num_states.checked_mul(num_states)? {
        return None;
    }
    let mut ws = SolverWorkspace::new(num_states, 0);
    let mut out = vec![0.0; num_states];
    ws.solve_ctmc_steady_state(generator, num_states, &mut out)
        .then_some(out)
}
