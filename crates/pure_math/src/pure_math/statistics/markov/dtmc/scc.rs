//! Iterative Tarjan strongly connected components for discrete-time Markov chains.

use super::MarkovChain;
use math_memory::{ArenaLayout, ArenaRegion, FlagSlice, OwnedByteArena};
use nalgebra::RealField;
use num_traits::ToPrimitive;

const UNVISITED: usize = usize::MAX;

/// A strongly connected communicating class of a Markov chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommunicatingClass {
    /// State indices in this class, sorted in ascending order.
    pub states: Vec<usize>,
    /// True if no positive-probability transition leaves this class (closed / recurrent).
    pub is_closed: bool,
}

struct TarjanSlices<'a> {
    indices: &'a mut [usize],
    lowlink: &'a mut [usize],
    tarjan_stack: &'a mut [usize],
    call_stack: &'a mut [[usize; 2]],
    sccs_flat: &'a mut [usize],
    scc_bounds: &'a mut [[usize; 2]],
    on_stack: FlagSlice<'a>,
}

struct TarjanWorkspace {
    arena: OwnedByteArena,
    sccs_flat_reg: ArenaRegion<usize>,
    scc_bounds_reg: ArenaRegion<[usize; 2]>,
    cap: usize,
    next_index: usize,
    tarjan_len: usize,
    call_len: usize,
    sccs_flat_len: usize,
    scc_count: usize,
}

impl TarjanWorkspace {
    fn new(n: usize) -> Self {
        let cap = n.max(1);
        let mut layout = ArenaLayout::new();
        let indices_reg = layout.add_slice::<usize>(cap).expect("tarjan indices");
        let _ = layout.add_slice::<usize>(cap).expect("tarjan lowlink");
        let _ = layout.add_slice::<usize>(cap).expect("tarjan stack");
        let _ = layout.add_slice::<[usize; 2]>(cap).expect("tarjan call");
        let sccs_flat_reg = layout.add_slice::<usize>(cap).expect("tarjan sccs_flat");
        let scc_bounds_reg = layout
            .add_slice::<[usize; 2]>(cap)
            .expect("tarjan scc_bounds");
        let _ = layout.add_slice::<u8>(cap).expect("tarjan on_stack");
        let mut arena = OwnedByteArena::with_capacity(layout.total_bytes());
        arena.region_mut(indices_reg).fill(UNVISITED);
        Self {
            arena,
            sccs_flat_reg,
            scc_bounds_reg,
            cap,
            next_index: 0,
            tarjan_len: 0,
            call_len: 0,
            sccs_flat_len: 0,
            scc_count: 0,
        }
    }

    fn slices_mut(&mut self) -> TarjanSlices<'_> {
        let c = self.cap;
        let mut scope = self.arena.reborrow();
        let indices = scope.retain_slice::<usize>(c).expect("indices");
        let lowlink = scope.retain_slice::<usize>(c).expect("lowlink");
        let tarjan_stack = scope.retain_slice::<usize>(c).expect("tarjan_stack");
        let call_stack = scope.retain_slice::<[usize; 2]>(c).expect("call_stack");
        let sccs_flat = scope.retain_slice::<usize>(c).expect("sccs_flat");
        let scc_bounds = scope.retain_slice::<[usize; 2]>(c).expect("scc_bounds");
        let on_stack_bytes = scope.retain_slice::<u8>(c).expect("on_stack");
        TarjanSlices {
            indices,
            lowlink,
            tarjan_stack,
            call_stack,
            sccs_flat,
            scc_bounds,
            on_stack: FlagSlice::from_bytes(on_stack_bytes),
        }
    }

    fn is_unvisited(&mut self, v: usize) -> bool {
        self.slices_mut().indices[v] == UNVISITED
    }

    fn discover(&mut self, v: usize) {
        let idx = self.next_index;
        self.next_index += 1;
        let t_pos = self.tarjan_len;
        self.tarjan_len += 1;
        let c_pos = self.call_len;
        self.call_len += 1;
        let mut s = self.slices_mut();
        s.indices[v] = idx;
        s.lowlink[v] = idx;
        s.tarjan_stack[t_pos] = v;
        s.on_stack.set(v, true);
        s.call_stack[c_pos] = [v, 0];
    }

    fn finish_vertex(&mut self, v: usize) {
        let mut t_len = self.tarjan_len;
        let c_len = self.call_len;
        let mut flat_len = self.sccs_flat_len;
        let mut count = self.scc_count;
        {
            let mut s = self.slices_mut();
            if s.lowlink[v] == s.indices[v] {
                let start = flat_len;
                while t_len > 0 {
                    t_len -= 1;
                    let u = s.tarjan_stack[t_len];
                    s.on_stack.set(u, false);
                    s.sccs_flat[flat_len] = u;
                    flat_len += 1;
                    if u == v {
                        break;
                    }
                }
                s.sccs_flat[start..flat_len].sort_unstable();
                s.scc_bounds[count] = [start, flat_len - start];
                count += 1;
            }
            if c_len > 0 {
                let parent = s.call_stack[c_len - 1][0];
                s.lowlink[parent] = s.lowlink[parent].min(s.lowlink[v]);
            }
        }
        self.tarjan_len = t_len;
        self.sccs_flat_len = flat_len;
        self.scc_count = count;
    }
}

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Computes all communicating classes via iterative Tarjan strongly connected components.
    #[must_use]
    #[verified_engine::verified]
    pub fn communicating_classes(&self) -> Vec<CommunicatingClass> {
        let n = self.num_states();
        if n == 0 {
            return Vec::new();
        }
        let eps = T::from_f64(1e-15).unwrap();
        let mut ws = TarjanWorkspace::new(n);

        for start in 0..n {
            if ws.is_unvisited(start) {
                ws.discover(start);
                self.drain_tarjan_component(&mut ws, eps);
            }
        }

        let mut classes = self.classify_sccs(&ws, eps);
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
        while ws.call_len > 0 {
            ws.call_len -= 1;
            let top_idx = ws.call_len;
            let [v, next_w] = ws.slices_mut().call_stack[top_idx];
            let mut advanced = false;
            for w in next_w..n {
                if self.transition_matrix[(v, w)] > eps {
                    {
                        let pos = ws.call_len;
                        ws.call_len += 1;
                        ws.slices_mut().call_stack[pos] = [v, w + 1];
                    }
                    if ws.is_unvisited(w) {
                        ws.discover(w);
                    } else {
                        let s = ws.slices_mut();
                        if s.on_stack.get(w) {
                            s.lowlink[v] = s.lowlink[v].min(s.indices[w]);
                        }
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
    fn classify_sccs(&self, ws: &TarjanWorkspace, eps: T) -> Vec<CommunicatingClass> {
        let n = self.num_states();
        let flat = ws.arena.region(ws.sccs_flat_reg);
        let bounds = &ws.arena.region(ws.scc_bounds_reg)[..ws.scc_count];
        let mut class_of = vec![0_usize; n];
        for (cid, &[start, len]) in bounds.iter().enumerate() {
            for &s in &flat[start..start + len] {
                class_of[s] = cid;
            }
        }

        bounds
            .iter()
            .enumerate()
            .map(|(cid, &[start, len])| {
                let states = flat[start..start + len].to_vec();
                let is_closed = states.iter().all(|&u| {
                    (0..n).all(|v| class_of[v] == cid || self.transition_matrix[(u, v)] <= eps)
                });
                CommunicatingClass { states, is_closed }
            })
            .collect()
    }
}

// [cite:clinical_trials_statistics]
