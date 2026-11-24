//! Iterative Tarjan strongly connected components for discrete-time Markov chains.

use super::MarkovChain;
use nalgebra::RealField;
use num_traits::ToPrimitive;

/// A strongly connected communicating class of a Markov chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommunicatingClass {
    /// State indices in this class, sorted in ascending order.
    pub states: Vec<usize>,
    /// True if no positive-probability transition leaves this class (closed / recurrent).
    pub is_closed: bool,
}

struct TarjanWorkspace {
    next_index: usize,
    indices: Vec<Option<usize>>,
    lowlink: Vec<usize>,
    on_stack: Vec<bool>,
    tarjan_stack: Vec<usize>,
    call_stack: Vec<(usize, usize)>,
    sccs: Vec<Vec<usize>>,
}

impl TarjanWorkspace {
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
            let mut component = Vec::new();
            while let Some(u) = self.tarjan_stack.pop() {
                self.on_stack[u] = false;
                component.push(u);
                if u == v {
                    break;
                }
            }
            component.sort_unstable();
            self.sccs.push(component);
        }
        if let Some(&(parent, _)) = self.call_stack.last() {
            self.lowlink[parent] = self.lowlink[parent].min(self.lowlink[v]);
        }
    }
}

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Computes all communicating classes via iterative Tarjan strongly connected components.
    #[must_use]
    #[verified_engine::verified]
    pub fn communicating_classes(&self) -> Vec<CommunicatingClass> {
        let n = self.num_states();
        let eps = T::from_f64(1e-15).unwrap();
        let mut ws = TarjanWorkspace::new(n);

        for start in 0..n {
            if ws.indices[start].is_none() {
                ws.discover(start);
                self.drain_tarjan_component(&mut ws, eps);
            }
        }

        let mut classes = self.classify_sccs(ws.sccs, eps);
        classes.sort_by_key(|c| c.states.first().copied().unwrap_or(usize::MAX));
        classes
    }

    /// Returns only the closed (recurrent) communicating classes of the chain.
    #[must_use]
    #[verified_engine::verified]
    pub fn closed_classes(&self) -> Vec<CommunicatingClass> {
        self.communicating_classes()
            .into_iter()
            .filter(|c| c.is_closed)
            .collect()
    }

    /// Returns `true` if the entire state space forms a single communicating class.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_irreducible(&self) -> bool {
        let n = self.num_states();
        n > 0 && self.communicating_classes().len() == 1
    }

    #[verified_engine::verified]
    fn drain_tarjan_component(&self, ws: &mut TarjanWorkspace, eps: T) {
        let n = self.num_states();
        while let Some((v, next_w)) = ws.call_stack.pop() {
            let mut advanced = false;
            for w in next_w..n {
                if self.transition_matrix[(v, w)] > eps {
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
    fn classify_sccs(&self, sccs: Vec<Vec<usize>>, eps: T) -> Vec<CommunicatingClass> {
        let n = self.num_states();
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
                    (0..n).all(|v| class_of[v] == cid || self.transition_matrix[(u, v)] <= eps)
                });
                CommunicatingClass { states, is_closed }
            })
            .collect()
    }
}

// [cite:clinical_trials_statistics]
