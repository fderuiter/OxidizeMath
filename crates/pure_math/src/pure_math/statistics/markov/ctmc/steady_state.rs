//! Rank-checked QR steady-state solver and communicating classes for CTMCs.

use super::ContinuousMarkovChain;
use crate::pure_math::statistics::markov::dtmc::CommunicatingClass;
use nalgebra::{DMatrix, DVector, RealField};
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

    fn discover(&mut self, v: usize) {
        let idx = self.next_index;
        self.next_index += 1;
        self.indices[v] = Some(idx);
        self.lowlink[v] = idx;
        self.tarjan_stack.push(v);
        self.on_stack[v] = true;
        self.call_stack.push((v, 0));
    }

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
    #[verified_engine::verified]
    pub fn closed_classes(&self) -> Vec<CommunicatingClass> {
        self.communicating_classes()
            .into_iter()
            .filter(|c| c.is_closed)
            .collect()
    }

    /// Computes the unique steady-state distribution $\pi$ solving $G^T \pi = 0$, $\sum_i \pi_i = 1$.
    ///
    /// Uses generator rate preconditioning and a rank-checked QR linear solve so slow-rate
    /// generators (e.g., $10^{-4}$, $10^{-7}$) are solved to machine precision without time-stepping.
    /// Returns `None` when multiple closed classes exist.
    #[must_use]
    #[verified_engine::verified]
    pub fn steady_state(&self) -> Option<DVector<T>> {
        let closed = self.closed_classes();
        if closed.len() != 1 {
            return None;
        }

        let n = self.num_states;
        if closed[0].states.len() == 1 {
            let mut pi = DVector::zeros(n);
            pi[closed[0].states[0]] = T::one();
            return Some(pi);
        }

        solve_scaled_generator_nullspace(&self.generator)
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

    #[verified_engine::verified]
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

#[verified_engine::verified]
fn solve_scaled_generator_nullspace<T: RealField + Copy + ToPrimitive>(
    generator: &DMatrix<T>,
) -> Option<DVector<T>> {
    let n = generator.nrows();
    let max_rate = generator.iter().fold(T::zero(), |acc, &x| acc.max(x.abs()));
    if max_rate <= T::zero() {
        return None;
    }

    let mut a = DMatrix::zeros(n + 1, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = generator[(j, i)] / max_rate;
        }
    }
    for j in 0..n {
        a[(n, j)] = T::one();
    }

    let mut b = DVector::zeros(n + 1);
    b[n] = T::one();

    let qr = a.qr();
    let r = qr.r();
    let r_max = (0..n).fold(T::zero(), |acc, i| acc.max(r[(i, i)].abs()));
    let rank_tol = T::from_f64(1e-12).unwrap() * r_max;
    if r_max <= T::zero() || (0..n).any(|i| r[(i, i)].abs() <= rank_tol) {
        return None;
    }

    let q_t_b = qr.q().transpose() * b;
    let mut pi = r.solve_upper_triangular(&q_t_b)?;

    let clamp_tol = T::from_f64(1e-12).unwrap();
    let mut total = T::zero();
    for i in 0..n {
        if !pi[i].is_finite() || pi[i] < -clamp_tol {
            return None;
        }
        if pi[i] < T::zero() {
            pi[i] = T::zero();
        }
        total += pi[i];
    }
    if total <= T::zero() {
        return None;
    }
    pi /= total;

    let residual = (generator.transpose() * &pi).amax() / max_rate;
    if residual > T::from_f64(1e-11).unwrap() {
        return None;
    }
    Some(pi)
}

// [cite:clinical_trials_statistics]
