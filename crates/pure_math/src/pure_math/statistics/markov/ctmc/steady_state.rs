//! Rank-checked QR steady-state solver and communicating classes for CTMCs.

use super::ContinuousMarkovChain;
use crate::pure_math::statistics::markov::dtmc::CommunicatingClass;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::stochastic::SolverWorkspace;
use nalgebra::{DVector, RealField};
use num_traits::ToPrimitive;

struct CtmcTarjan {
    next_index: usize,
    indices: Vec<Option<usize>>,
    lowlink: Vec<usize>,
    on_stack: Vec<bool>,
    tarjan_stack: Vec<usize>,
    call_stack: Vec<(usize, usize)>,
    sccs: Vec<Vec<usize>>,
}

impl CtmcTarjan {
    #[verified_engine::verified(alloc_ok = "allocates Tarjan SCC workspace")]
    fn new(n: usize) -> Self {
        Self {
            next_index: 0,
            indices: vec![None; n],
            lowlink: vec![0; n],
            on_stack: vec![false; n],
            tarjan_stack: Vec::with_capacity(n),
            call_stack: Vec::with_capacity(n),
            sccs: Vec::new(),
        }
    }

    #[verified_engine::verified]
    fn discover(&mut self, v: usize) {
        let idx = self.next_index;
        self.next_index += 1;
        self.indices[v] = Some(idx);
        self.lowlink[v] = idx;
        self.tarjan_stack.push(v);
        self.on_stack[v] = true;
        self.call_stack.push((v, 0));
    }

    #[verified_engine::verified(alloc_ok = "allocates SCC vertex list")]
    fn finish_vertex(&mut self, v: usize) {
        if Some(self.lowlink[v]) == self.indices[v] {
            let mut comp = Vec::new();
            while let Some(u) = self.tarjan_stack.pop() {
                self.on_stack[u] = false;
                comp.push(u);
                if u == v {
                    break;
                }
            }
            comp.sort_unstable();
            self.sccs.push(comp);
        }
        if let Some(&(parent, _)) = self.call_stack.last() {
            self.lowlink[parent] = self.lowlink[parent].min(self.lowlink[v]);
        }
    }
}

impl<T: RealField + Copy + ToPrimitive> ContinuousMarkovChain<T> {
    /// Computes communicating classes of the CTMC from positive off-diagonal rates $G_{ij} > 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn communicating_classes(&self) -> Vec<CommunicatingClass> {
        let n = self.num_states;
        let mut ws = CtmcTarjan::new(n);
        for start in 0..n {
            if ws.indices[start].is_none() {
                ws.discover(start);
                self.drain_ctmc_tarjan(&mut ws);
            }
        }

        let mut classes = self.classify_ctmc_sccs(ws.sccs);
        classes.sort_by_key(|c| c.states.first().copied().unwrap_or(usize::MAX));
        classes
    }

    /// Returns the closed (recurrent) communicating classes of the generator.
    #[must_use]
    #[verified_engine::verified(alloc_ok = "collects closed communicating classes")]
    pub fn closed_classes(&self) -> Vec<CommunicatingClass> {
        self.communicating_classes()
            .into_iter()
            .filter(|c| c.is_closed)
            .collect()
    }

    /// Computes the unique steady-state distribution $\pi$ satisfying $\pi^T G = 0$ and
    /// $\sum_i \pi_i = 1$ via Tarjan SCC sink isolation and scale-normalized Householder QR.
    ///
    /// Returns `None` when multiple closed sink classes exist (non-unique steady state).
    #[must_use]
    #[verified_engine::verified(alloc_ok = "allocates workspace and output vector")]
    pub fn steady_state(&self) -> Option<DVector<T>> {
        assert!(self.num_states > 0);
        let mut ws = SolverWorkspace::new(self.num_states, 0);
        let mut out = DVector::zeros(self.num_states);
        self.steady_state_with_workspace(&mut ws, &mut out)
            .then_some(out)
    }

    /// Computes the unique steady-state distribution into `out` using preallocated
    /// [`SolverWorkspace`] buffers with zero heap allocations when
    /// `ws.state_capacity() >= self.num_states`.
    #[verified_engine::verified]
    pub fn steady_state_with_workspace(
        &self,
        ws: &mut SolverWorkspace,
        out: &mut DVector<T>,
    ) -> bool {
        let n = self.num_states;
        debug_assert_eq!(out.len(), n);
        if n == 0 || out.len() != n {
            return false;
        }
        {
            let staged = ws.operator_staging_mut(n);
            for r in 0..n {
                for c in 0..n {
                    let Some(v) = self.generator[(r, c)].to_f64() else {
                        return false;
                    };
                    staged[flatten_2d_index(c, r, n)] = v;
                }
            }
        }
        let Some(pi_f64) = ws.solve_staged_ctmc_steady_state_slice(n) else {
            return false;
        };
        for (i, &val) in pi_f64.iter().enumerate() {
            let Some(cast) = T::from_f64(val) else {
                return false;
            };
            out[i] = cast;
        }
        true
    }

    #[verified_engine::verified]
    fn drain_ctmc_tarjan(&self, ws: &mut CtmcTarjan) {
        let n = self.num_states;
        let zero = T::zero();
        while let Some((v, next_w)) = ws.call_stack.pop() {
            let mut advanced = false;
            for w in next_w..n {
                if v != w && self.generator[(v, w)] > zero {
                    ws.call_stack.push((v, w + 1));
                    if ws.indices[w].is_none() {
                        ws.discover(w);
                    } else if ws.on_stack[w] {
                        let w_idx = ws.indices[w].unwrap_or(ws.lowlink[v]);
                        ws.lowlink[v] = ws.lowlink[v].min(w_idx);
                    }
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                ws.finish_vertex(v);
            }
        }
    }

    #[verified_engine::verified(alloc_ok = "allocates SCC classification vectors")]
    fn classify_ctmc_sccs(&self, sccs: Vec<Vec<usize>>) -> Vec<CommunicatingClass> {
        let n = self.num_states;
        let zero = T::zero();
        let mut class_of = vec![0_usize; n];
        for (cid, states) in sccs.iter().enumerate() {
            for &s in states {
                class_of[s] = cid;
            }
        }

        sccs.into_iter()
            .enumerate()
            .map(|(cid, states)| {
                let is_closed = states.iter().all(|&u| {
                    (0..n).all(|v| u == v || class_of[v] == cid || self.generator[(u, v)] <= zero)
                });
                CommunicatingClass { states, is_closed }
            })
            .collect()
    }
}

// [cite:clinical_trials_statistics]
