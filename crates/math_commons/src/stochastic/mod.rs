//! Zero-allocation `#![no_std]` stochastic solver kernel (`stochastic`).
//!
//! Provides `ByteArena`-compatible Tarjan SCC and Householder QR stationary/steady-state solvers
//! alongside an `alloc`-gated [`StochasticWorkspace`] backed by [`math_memory::OwnedByteArena`].
// [cite:math_commons]

mod ou;
mod qr;
mod tarjan;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use math_memory::{ArenaLayout, SolverWorkspace};
#[cfg(feature = "alloc")]
use math_memory::{ArenaRegion, ArenaScope, ByteArena, OwnedByteArena};

pub use ou::{OuCalibration, SampleSummary, calibrate_ou, summarize_samples_into};

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

/// Computes the byte capacity required in an [`ArenaScope`] to run Tarjan SCC and Householder QR
/// stationary solvers on `num_states` states.
#[must_use]
pub fn stochastic_scratch_bytes(num_states: usize) -> usize {
    let mut layout = ArenaLayout::new();
    let _ = layout.add_slice::<usize>(num_states); // tarjan_index
    let _ = layout.add_slice::<usize>(num_states); // tarjan_lowlink
    let _ = layout.add_slice::<u8>(num_states); // tarjan_on_stack (FlagSlice)
    let _ = layout.add_slice::<usize>(num_states); // tarjan_stack
    let _ = layout.add_slice::<usize>(num_states); // call_node
    let _ = layout.add_slice::<usize>(num_states); // call_next_edge
    let _ = layout.add_slice::<usize>(num_states); // component_id
    let _ = layout.add_slice::<u8>(num_states); // component_is_sink (FlagSlice)
    let _ = layout.add_slice::<f64>(num_states); // row_scales
    let _ = layout.add_slice::<f64>(num_states.saturating_add(1).saturating_mul(num_states)); // qr_matrix
    let _ = layout.add_slice::<f64>(num_states.saturating_add(1)); // qr_rhs
    let _ = layout.add_slice::<f64>(num_states); // qr_sol
    layout.total_bytes().saturating_add(128)
}

/// Decomposes a flat row-major $n \times n$ matrix into SCCs in `#![no_std]` using `ws`.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn analyze_scc_with_workspace(
    matrix: &[f64],
    num_states: usize,
    sink_members_out: &mut [usize],
    ws: &mut impl SolverWorkspace,
) -> Option<SccSummary> {
    if num_states == 0
        || sink_members_out.len() < num_states
        || matrix.len() < num_states.checked_mul(num_states)?
    {
        return None;
    }
    let mut scope = ws.reborrow();
    tarjan::decompose_tarjan(
        &mut scope,
        &mut sink_members_out[..num_states],
        &matrix[..num_states * num_states],
        num_states,
    )
}

/// Solves $(P^T - I)\pi = 0$ with $\sum_{i=1}^n \pi_i = 1$ in `#![no_std]` using `ws`
/// (`docs/math_specs/math_commons.md`, DOI: `10.1007/978-1-4612-2670-3`).
///
/// # Stability
/// Isolates the unique closed recurrent sink class via iterative Tarjan SCC traversal before
/// solving the augmented overdetermined Householder QR system with equilibrium row equilibration.
///
/// # Domain of Validity
/// Requires `num_states >= 1`, `transition.len() >= num_states * num_states`, `out_pi.len() >= num_states`,
/// a row-stochastic matrix with a unique closed sink SCC, and `ws` sized via [`stochastic_scratch_bytes`].
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn solve_dtmc_stationary_with_workspace(
    transition: &[f64],
    num_states: usize,
    out_pi: &mut [f64],
    ws: &mut impl SolverWorkspace,
) -> bool {
    if num_states == 0
        || out_pi.len() < num_states
        || transition.len() < num_states.saturating_mul(num_states)
    {
        return false;
    }
    let mut scope = ws.reborrow();
    let Ok(sink) = scope.alloc_slice::<usize>(num_states) else {
        return false;
    };
    qr::solve_stationary_slice(
        &mut scope,
        sink,
        &transition[..num_states * num_states],
        num_states,
        false,
        &mut out_pi[..num_states],
    )
}

/// Solves $Q^T \pi = 0$ with $\sum_{i=1}^n \pi_i = 1$ in `#![no_std]` using `ws`
/// (`docs/math_specs/math_commons.md`, DOI: `10.1007/978-1-4612-2670-3`).
///
/// # Stability
/// Restricts the system to the unique closed recurrent sink SCC and equilibrates generator rows
/// prior to augmented Householder QR factorization to handle stiff rate separation.
///
/// # Domain of Validity
/// Requires `num_states >= 1`, `generator.len() >= num_states * num_states`, `out_pi.len() >= num_states`,
/// a conservative rate generator with a unique closed sink SCC, and `ws` sized via [`stochastic_scratch_bytes`].
#[cfg_attr(feature = "std", verified_engine::verified)]
pub fn solve_ctmc_steady_state_with_workspace(
    generator: &[f64],
    num_states: usize,
    out_pi: &mut [f64],
    ws: &mut impl SolverWorkspace,
) -> bool {
    if num_states == 0
        || out_pi.len() < num_states
        || generator.len() < num_states.saturating_mul(num_states)
    {
        return false;
    }
    let mut scope = ws.reborrow();
    let Ok(sink) = scope.alloc_slice::<usize>(num_states) else {
        return false;
    };
    qr::solve_stationary_slice(
        &mut scope,
        sink,
        &generator[..num_states * num_states],
        num_states,
        true,
        &mut out_pi[..num_states],
    )
}

/// Preallocated reusable workspace for zero-allocation Markov stationary/steady-state solvers,
/// Tarjan SCC graph traversal, and Ornstein-Uhlenbeck quantile/summary routines.
///
/// Backed by a single [`OwnedByteArena`] with zero multi-`Vec` fields.
///
/// # Example
/// ```rust
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
    state_capacity: usize,
    sample_capacity: usize,
    arena: OwnedByteArena,
    operator_reg: ArenaRegion<f64>,
    solution_reg: ArenaRegion<f64>,
    sink_members_reg: ArenaRegion<usize>,
    scratch_offset: usize,
}

#[cfg(feature = "alloc")]
impl SolverWorkspace for StochasticWorkspace {
    #[inline]
    fn as_byte_arena(&mut self) -> ByteArena<'_> {
        let start = self.scratch_offset.min(self.arena.byte_capacity());
        ByteArena::new(&mut self.arena.as_bytes_mut()[start..])
    }

    #[inline]
    fn byte_capacity(&self) -> usize {
        self.arena
            .byte_capacity()
            .saturating_sub(self.scratch_offset)
    }

    #[inline]
    fn reset(&mut self) {}
}

#[cfg(feature = "alloc")]
impl StochasticWorkspace {
    /// Allocates a workspace pre-sized for up to `state_capacity` Markov states and
    /// `sample_capacity` Monte Carlo terminal samples.
    #[must_use]
    pub fn new(state_capacity: usize, sample_capacity: usize) -> Self {
        let mut ws = Self::default();
        ws.rebuild_layout(state_capacity, sample_capacity);
        ws
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

    fn rebuild_layout(&mut self, state_cap: usize, sample_cap: usize) {
        self.state_capacity = state_cap;
        self.sample_capacity = sample_cap;
        let mut layout = ArenaLayout::new();
        self.operator_reg = layout
            .add_slice::<f64>(state_cap.saturating_mul(state_cap))
            .unwrap_or_default();
        self.solution_reg = layout.add_slice::<f64>(state_cap).unwrap_or_default();
        self.sink_members_reg = layout.add_slice::<usize>(state_cap).unwrap_or_default();
        self.scratch_offset = layout.total_bytes();
        let sample_bytes = sample_cap
            .saturating_mul(core::mem::size_of::<f64>())
            .saturating_add(16);
        let scratch_bytes = stochastic_scratch_bytes(state_cap).max(sample_bytes);
        self.arena
            .ensure_capacity(self.scratch_offset.saturating_add(scratch_bytes));
    }

    /// Ensures internal graph and Householder QR buffers can accommodate `num_states` states.
    pub fn ensure_state_capacity(&mut self, num_states: usize) {
        if num_states > self.state_capacity {
            self.rebuild_layout(num_states, self.sample_capacity);
        }
    }

    /// Ensures the quantile sorting scratch buffer can accommodate `num_samples` observations.
    pub fn ensure_sample_capacity(&mut self, num_samples: usize) {
        if num_samples > self.sample_capacity {
            self.rebuild_layout(self.state_capacity, num_samples);
        }
    }

    /// Mutable view of the preallocated $n \times n$ flat row-major operator staging buffer.
    pub fn operator_staging_mut(&mut self, num_states: usize) -> &mut [f64] {
        self.ensure_state_capacity(num_states);
        let reg = self.operator_reg;
        &mut self.arena.region_mut(reg)[..num_states * num_states]
    }

    fn split_persistent_and_scratch(
        &mut self,
        n: usize,
    ) -> (&mut [f64], &mut [f64], &mut [usize], ArenaScope<'_>) {
        let mut scope = self.arena.reborrow();
        let op_full = scope
            .retain_slice::<f64>(self.operator_reg.len())
            .unwrap_or(&mut []);
        let sol_full = scope
            .retain_slice::<f64>(self.solution_reg.len())
            .unwrap_or(&mut []);
        let sink_full = scope
            .retain_slice::<usize>(self.sink_members_reg.len())
            .unwrap_or(&mut []);
        (
            &mut op_full[..n * n],
            &mut sol_full[..n],
            &mut sink_full[..n],
            scope,
        )
    }

    /// Solves the DTMC stationary distribution staged in [`Self::operator_staging_mut`].
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_dtmc_stationary(&mut self, num_states: usize, out_pi: &mut [f64]) -> bool {
        if num_states == 0 || out_pi.len() < num_states || self.state_capacity < num_states {
            return false;
        }
        let (op, _sol, sink, mut scratch) = self.split_persistent_and_scratch(num_states);
        qr::solve_stationary_slice(
            &mut scratch,
            sink,
            op,
            num_states,
            false,
            &mut out_pi[..num_states],
        )
    }

    /// Solves the CTMC steady-state distribution staged in [`Self::operator_staging_mut`].
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_ctmc_steady_state(
        &mut self,
        num_states: usize,
        out_pi: &mut [f64],
    ) -> bool {
        if num_states == 0 || out_pi.len() < num_states || self.state_capacity < num_states {
            return false;
        }
        let (op, _sol, sink, mut scratch) = self.split_persistent_and_scratch(num_states);
        qr::solve_stationary_slice(
            &mut scratch,
            sink,
            op,
            num_states,
            true,
            &mut out_pi[..num_states],
        )
    }

    /// Solves the staged DTMC stationary distribution into internal workspace memory and returns `&[f64]`.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_dtmc_stationary_slice(&mut self, num_states: usize) -> Option<&[f64]> {
        if num_states == 0 || self.state_capacity < num_states {
            return None;
        }
        let ok = {
            let (op, sol, sink, mut scratch) = self.split_persistent_and_scratch(num_states);
            qr::solve_stationary_slice(&mut scratch, sink, op, num_states, false, sol)
        };
        ok.then(|| &self.arena.region(self.solution_reg)[..num_states])
    }

    /// Solves the staged CTMC steady-state distribution into internal workspace memory and returns `&[f64]`.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_staged_ctmc_steady_state_slice(&mut self, num_states: usize) -> Option<&[f64]> {
        if num_states == 0 || self.state_capacity < num_states {
            return None;
        }
        let ok = {
            let (op, sol, sink, mut scratch) = self.split_persistent_and_scratch(num_states);
            qr::solve_stationary_slice(&mut scratch, sink, op, num_states, true, sol)
        };
        ok.then(|| &self.arena.region(self.solution_reg)[..num_states])
    }

    /// Decomposes a flat row-major $n \times n$ transition or generator matrix into SCCs.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn analyze_scc(&mut self, matrix: &[f64], num_states: usize) -> Option<SccSummary> {
        if num_states == 0 || matrix.len() < num_states.checked_mul(num_states)? {
            return None;
        }
        self.ensure_state_capacity(num_states);
        let (_op, _sol, sink, mut scratch) = self.split_persistent_and_scratch(num_states);
        tarjan::decompose_tarjan(
            &mut scratch,
            sink,
            &matrix[..num_states * num_states],
            num_states,
        )
    }

    /// Returns the state indices belonging to the unique closed sink class from the most recent solve.
    #[must_use]
    pub fn unique_sink_members(&self, unique_sink_size: usize) -> &[usize] {
        let slice = self.arena.region(self.sink_members_reg);
        &slice[..unique_sink_size.min(slice.len())]
    }

    /// Solves $(P^T - I)\pi = 0$ with $\sum_i \pi_i = 1$ for a flat row-major $n \times n$ matrix.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_dtmc_stationary(
        &mut self,
        transition: &[f64],
        num_states: usize,
        out_pi: &mut [f64],
    ) -> bool {
        if num_states == 0
            || out_pi.len() < num_states
            || transition.len() < num_states.saturating_mul(num_states)
        {
            return false;
        }
        self.ensure_state_capacity(num_states);
        let (_op, _sol, sink, mut scratch) = self.split_persistent_and_scratch(num_states);
        qr::solve_stationary_slice(
            &mut scratch,
            sink,
            &transition[..num_states * num_states],
            num_states,
            false,
            &mut out_pi[..num_states],
        )
    }

    /// Solves $G^T \pi = 0$ with $\sum_i \pi_i = 1$ for a flat row-major $n \times n$ generator.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn solve_ctmc_steady_state(
        &mut self,
        generator: &[f64],
        num_states: usize,
        out_pi: &mut [f64],
    ) -> bool {
        if num_states == 0
            || out_pi.len() < num_states
            || generator.len() < num_states.saturating_mul(num_states)
        {
            return false;
        }
        self.ensure_state_capacity(num_states);
        let (_op, _sol, sink, mut scratch) = self.split_persistent_and_scratch(num_states);
        qr::solve_stationary_slice(
            &mut scratch,
            sink,
            &generator[..num_states * num_states],
            num_states,
            true,
            &mut out_pi[..num_states],
        )
    }

    /// Calibrates Ornstein-Uhlenbeck parameters $(\mu, \theta, \sigma)$ with zero heap allocations.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn calibrate_ou(&self, observations: &[f64], dt: f64) -> Option<OuCalibration> {
        ou::calibrate_ou(observations, dt)
    }

    /// Computes sample summary statistics in-place using the preallocated byte-arena scratch region.
    #[cfg_attr(feature = "std", verified_engine::verified)]
    pub fn summarize_samples(
        &mut self,
        samples: &[f64],
        threshold: Option<f64>,
    ) -> Option<SampleSummary> {
        if samples.is_empty() {
            return None;
        }
        self.ensure_sample_capacity(samples.len());
        let mut scratch = self.as_byte_arena().into_scope();
        let buf = scratch.alloc_slice::<f64>(samples.len()).ok()?;
        ou::summarize_samples_into(samples, threshold, buf)
    }
}

/// Convenience allocating wrapper solving the unique DTMC stationary distribution.
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

/// Convenience allocating wrapper solving the unique CTMC steady-state distribution.
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
