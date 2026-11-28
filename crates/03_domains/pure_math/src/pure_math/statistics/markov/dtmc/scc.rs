//! Iterative Tarjan strongly connected components for discrete-time Markov chains.

use super::MarkovChain;
use math_memory::{ArenaLayout, ArenaRegion, FlagRegion, OwnedByteArena};
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

struct TarjanWorkspace {
    next_index: usize,
    tarjan_len: usize,
    call_len: usize,
    scc_members_len: usize,
    scc_count: usize,
    arena: OwnedByteArena,
    indices: ArenaRegion<usize>,
    lowlink: ArenaRegion<usize>,
    on_stack: FlagRegion,
    tarjan_stack: ArenaRegion<usize>,
    call_stack: ArenaRegion<[usize; 2]>,
    scc_members: ArenaRegion<usize>,
    scc_spans: ArenaRegion<[usize; 2]>,
    class_of: ArenaRegion<usize>,
}

impl TarjanWorkspace {
    fn new(n: usize) -> Self {
        let mut layout = ArenaLayout::new();
        let indices = layout.reserve::<usize>(n).expect("Tarjan indices layout");
        let lowlink = layout.reserve::<usize>(n).expect("Tarjan lowlink layout");
        let on_stack = layout.reserve_flags(n).expect("Tarjan on_stack layout");
        let tarjan_stack = layout.reserve::<usize>(n).expect("Tarjan stack layout");
        let call_stack = layout
            .reserve::<[usize; 2]>(n)
            .expect("Tarjan call_stack layout");
        let scc_members = layout
            .reserve::<usize>(n)
            .expect("Tarjan scc_members layout");
        let scc_spans = layout
            .reserve::<[usize; 2]>(n)
            .expect("Tarjan scc_spans layout");
        let class_of = layout.reserve::<usize>(n).expect("Tarjan class_of layout");
        let mut arena = OwnedByteArena::from_layout(&layout);
        arena.region_slice_mut(indices).fill(UNVISITED);
        Self {
            next_index: 0,
            tarjan_len: 0,
            call_len: 0,
            scc_members_len: 0,
            scc_count: 0,
            arena,
            indices,
            lowlink,
            on_stack,
            tarjan_stack,
            call_stack,
            scc_members,
            scc_spans,
            class_of,
        }
    }

    #[inline]
    fn is_unvisited(&self, v: usize) -> bool {
        self.arena.region_slice(self.indices)[v] == UNVISITED
    }

    fn pop_call(&mut self) -> Option<(usize, usize)> {
        if self.call_len == 0 {
            return None;
        }
        self.call_len -= 1;
        let frame = self.arena.region_slice(self.call_stack)[self.call_len];
        Some((frame[0], frame[1]))
    }

    fn push_call(&mut self, v: usize, next_w: usize) {
        let slot = self.call_len;
        self.call_len += 1;
        self.arena.region_slice_mut(self.call_stack)[slot] = [v, next_w];
    }

    fn discover(&mut self, v: usize) {
        let idx = self.next_index;
        self.next_index += 1;
        let ind_reg = self.indices;
        let low_reg = self.lowlink;
        let flag_reg = self.on_stack;
        let t_reg = self.tarjan_stack;
        let c_reg = self.call_stack;
        let mut cur = self.arena.raw_cursor();
        let indices = cur.region_mut(ind_reg);
        let lowlink = cur.region_mut(low_reg);
        let mut on_stack = cur.flags_mut(flag_reg);
        let tarjan_stack = cur.region_mut(t_reg);
        let call_stack = cur.region_mut(c_reg);
        indices[v] = idx;
        lowlink[v] = idx;
        tarjan_stack[self.tarjan_len] = v;
        self.tarjan_len += 1;
        on_stack.set(v, true);
        call_stack[self.call_len] = [v, 0];
        self.call_len += 1;
    }

    fn update_back_edge(&mut self, v: usize, w: usize) {
        if self.arena.flag_get(self.on_stack, w) {
            let ind_reg = self.indices;
            let low_reg = self.lowlink;
            let mut cur = self.arena.raw_cursor();
            let indices = cur.region_mut(ind_reg);
            let lowlink = cur.region_mut(low_reg);
            let w_idx = if indices[w] == UNVISITED {
                lowlink[v]
            } else {
                indices[w]
            };
            lowlink[v] = lowlink[v].min(w_idx);
        }
    }

    fn finish_vertex(&mut self, v: usize) {
        let ind_reg = self.indices;
        let low_reg = self.lowlink;
        let flag_reg = self.on_stack;
        let t_reg = self.tarjan_stack;
        let c_reg = self.call_stack;
        let m_reg = self.scc_members;
        let s_reg = self.scc_spans;
        let cl_reg = self.class_of;
        let mut cur = self.arena.raw_cursor();
        let indices = cur.region_mut(ind_reg);
        let lowlink = cur.region_mut(low_reg);
        let mut on_stack = cur.flags_mut(flag_reg);
        let tarjan_stack = cur.region_mut(t_reg);
        let call_stack = cur.region_mut(c_reg);
        let scc_members = cur.region_mut(m_reg);
        let scc_spans = cur.region_mut(s_reg);
        let class_of = cur.region_mut(cl_reg);

        if lowlink[v] == indices[v] {
            let start = self.scc_members_len;
            let cid = self.scc_count;
            while self.tarjan_len > 0 {
                self.tarjan_len -= 1;
                let u = tarjan_stack[self.tarjan_len];
                on_stack.set(u, false);
                class_of[u] = cid;
                scc_members[self.scc_members_len] = u;
                self.scc_members_len += 1;
                if u == v {
                    break;
                }
            }
            let end = self.scc_members_len;
            scc_members[start..end].sort_unstable();
            scc_spans[cid] = [start, end];
            self.scc_count += 1;
        }
        if self.call_len > 0 {
            let parent = call_stack[self.call_len - 1][0];
            lowlink[parent] = lowlink[parent].min(lowlink[v]);
        }
    }
}

impl<T: RealField + Copy + ToPrimitive> MarkovChain<T> {
    /// Fallible companion to [`communicating_classes`](Self::communicating_classes).
    #[verified_engine::verified]
    pub fn try_communicating_classes(
        &self,
    ) -> core::result::Result<Vec<CommunicatingClass>, math_commons::error::MathError> {
        let n = self.num_states();
        let eps = T::from_f64(1e-15).ok_or(math_commons::error::MathError::ConversionError {
            reason: "Failed to convert Tarjan tolerance 1e-15 to scalar type",
        })?;
        let mut ws = TarjanWorkspace::new(n);

        for start in 0..n {
            if ws.is_unvisited(start) {
                ws.discover(start);
                self.drain_tarjan_component(&mut ws, eps);
            }
        }

        let mut classes = self.classify_sccs(&ws, eps);
        classes.sort_by_key(|c| c.states.first().copied().unwrap_or(usize::MAX));
        debug_assert!(!classes.is_empty() || n == 0);
        Ok(classes)
    }

    /// Computes all communicating classes via iterative Tarjan strongly connected components.
    #[must_use]
    #[verified_engine::verified]
    pub fn communicating_classes(&self) -> Vec<CommunicatingClass> {
        self.try_communicating_classes()
            .expect("communicating_classes preconditions must hold")
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
        while let Some((v, next_w)) = ws.pop_call() {
            let mut advanced = false;
            for w in next_w..n {
                if self.transition_matrix[(v, w)] > eps {
                    ws.push_call(v, w + 1);
                    if ws.is_unvisited(w) {
                        ws.discover(w);
                    } else {
                        ws.update_back_edge(v, w);
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
        let members = ws.arena.region_slice(ws.scc_members);
        let spans = &ws.arena.region_slice(ws.scc_spans)[..ws.scc_count];
        let class_of = ws.arena.region_slice(ws.class_of);

        spans
            .iter()
            .enumerate()
            .map(|(cid, &[start, end])| {
                let slice = &members[start..end];
                let is_closed = slice.iter().all(|&u| {
                    (0..n).all(|v| class_of[v] == cid || self.transition_matrix[(u, v)] <= eps)
                });
                CommunicatingClass {
                    states: slice.to_vec(),
                    is_closed,
                }
            })
            .collect()
    }
}

// [cite:clinical_trials_statistics]
