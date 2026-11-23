//! Bounded `#![no_std]` + `alloc` e-graph equality saturation and iterative minimum-cost extraction.
// [cite:engine_symbolic]

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::egraph_rules::collect_rewrites_step;
use crate::sym::{SymExpr, SymOp};

/// Hard deterministic cap on total canonical e-nodes in an [`EGraph`] (`2048`).
pub const MAX_EGRAPH_NODES: usize = 2048;

/// Hard deterministic cap on equality saturation rewrite passes (`16`).
pub const MAX_REWRITE_PASSES: usize = 16;

/// Bounded equality-saturation e-graph modeled on `egg` using strictly iterative worklists.
#[derive(Debug, Clone, Default)]
pub struct EGraph {
    pub(crate) parents: Vec<u32>,
    pub(crate) classes: Vec<Vec<SymOp>>,
    pub(crate) memo: BTreeMap<SymOp, u32>,
    pub(crate) dirty: Vec<u32>,
    pub(crate) symbols: Vec<String>,
    pub(crate) node_count: usize,
}

impl EGraph {
    /// Creates an empty bounded [`EGraph`].
    #[must_use]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns total canonical e-nodes currently stored (`<= MAX_EGRAPH_NODES`).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Returns the number of equivalence classes allocated.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Returns the interned symbol slice.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// Interns a variable or constant name and returns its `SymOp::Symbol` e-class ID.
    #[verified_engine::verified]
    pub fn add_symbol(&mut self, name: &str) -> Option<u32> {
        let mut idx = 0;
        while idx < self.symbols.len() {
            if self.symbols[idx] == name {
                return self.add(SymOp::Symbol(idx as u16));
            }
            idx += 1;
        }
        if self.symbols.len() >= u16::MAX as usize {
            return None;
        }
        let id = self.symbols.len() as u16;
        self.symbols.push(String::from(name));
        self.add(SymOp::Symbol(id))
    }

    /// Iteratively resolves the canonical representative e-class ID (with path halving).
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn find(&self, mut id: u32) -> u32 {
        while (id as usize) < self.parents.len() && self.parents[id as usize] != id {
            id = self.parents[id as usize];
        }
        id
    }

    #[inline]
    #[verified_engine::verified]
    fn find_mut(&mut self, mut id: u32) -> u32 {
        while self.parents[id as usize] != id {
            let p = self.parents[id as usize];
            let gp = self.parents[p as usize];
            self.parents[id as usize] = gp;
            id = p;
        }
        id
    }

    /// Canonicalizes `enode`'s child IDs and inserts it if below [`MAX_EGRAPH_NODES`].
    #[verified_engine::verified]
    pub fn add(&mut self, enode: SymOp) -> Option<u32> {
        let canon = enode.map_children(|c| self.find(c));
        if let Some(&existing) = self.memo.get(&canon) {
            return Some(self.find(existing));
        }
        if self.node_count >= MAX_EGRAPH_NODES {
            return None;
        }
        let id = self.parents.len() as u32;
        self.parents.push(id);
        self.classes.push(vec![canon]);
        self.memo.insert(canon, id);
        self.node_count += 1;
        Some(id)
    }

    /// Merges two e-classes (`a` and `b`) and queues congruence repair.
    #[verified_engine::verified]
    pub fn union(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find_mut(a);
        let rb = self.find_mut(b);
        if ra == rb {
            return false;
        }
        let (dst, src) = if self.classes[ra as usize].len() >= self.classes[rb as usize].len() {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parents[src as usize] = dst;
        let moved = core::mem::take(&mut self.classes[src as usize]);
        self.classes[dst as usize].extend(moved);
        self.dirty.push(dst);
        true
    }

    /// Iteratively restores e-graph congruence invariants after [`Self::union`] calls.
    #[verified_engine::verified]
    pub fn rebuild(&mut self) {
        while !self.dirty.is_empty() {
            self.dirty.clear();
            let mut pending_merges: Vec<(u32, u32)> = Vec::new();
            let mut next_memo: BTreeMap<SymOp, u32> = BTreeMap::new();

            let mut cid = 0usize;
            while cid < self.classes.len() {
                if self.parents[cid] as usize == cid {
                    let mut nodes = core::mem::take(&mut self.classes[cid]);
                    for node in &mut nodes {
                        *node = node.map_children(|c| self.find(c));
                    }
                    nodes.sort_unstable();
                    nodes.dedup();
                    for &node in &nodes {
                        if let Some(&prev_c) = next_memo.get(&node) {
                            if prev_c != cid as u32 {
                                pending_merges.push((prev_c, cid as u32));
                            }
                        } else {
                            next_memo.insert(node, cid as u32);
                        }
                    }
                    self.classes[cid] = nodes;
                }
                cid += 1;
            }
            self.memo = next_memo;
            for (u, v) in pending_merges {
                self.union(u, v);
            }
        }
    }

    /// Runs iterative equality saturation up to `min(max_passes, MAX_REWRITE_PASSES)` passes
    /// and `MAX_EGRAPH_NODES` total nodes.
    #[verified_engine::verified]
    pub fn saturate(&mut self, max_passes: usize) -> usize {
        let cap = if max_passes < MAX_REWRITE_PASSES {
            max_passes
        } else {
            MAX_REWRITE_PASSES
        };
        let mut pass = 0;
        while pass < cap && self.node_count < MAX_EGRAPH_NODES {
            let changed = collect_rewrites_step(self);
            self.rebuild();
            pass += 1;
            if !changed {
                break;
            }
        }
        pass
    }

    /// Computes minimal AST node cost per e-class iteratively and extracts the best [`SymExpr`]
    /// rooted at `root` without recursion.
    #[must_use]
    #[verified_engine::verified]
    pub fn extract(&self, root: u32) -> SymExpr {
        let n_classes = self.classes.len();
        let mut best_cost = vec![u64::MAX; n_classes];
        let mut best_node: Vec<Option<SymOp>> = vec![None; n_classes];

        let mut improved = true;
        let mut iter_guard = 0usize;
        while improved && iter_guard <= n_classes {
            improved = false;
            iter_guard += 1;
            let mut cid = 0usize;
            while cid < n_classes {
                if self.parents[cid] as usize == cid {
                    for &node in &self.classes[cid] {
                        if let Some(cost) = node_total_cost(&node, &best_cost, self)
                            && cost < best_cost[cid]
                        {
                            best_cost[cid] = cost;
                            best_node[cid] = Some(node);
                            improved = true;
                        }
                    }
                }
                cid += 1;
            }
        }

        extract_dag_iterative(self, self.find(root), &best_node)
    }
}

#[verified_engine::verified]
fn node_total_cost(node: &SymOp, best_cost: &[u64], egraph: &EGraph) -> Option<u64> {
    let mut kids = [0u32; 3];
    let n = node.children_into(&mut kids);
    let mut sum = node.local_cost();
    let mut i = 0;
    while i < n {
        let c = egraph.find(kids[i]) as usize;
        let child_cost = best_cost[c];
        if child_cost == u64::MAX {
            return None;
        }
        sum = sum.saturating_add(child_cost);
        i += 1;
    }
    Some(sum)
}

#[verified_engine::verified]
fn extract_dag_iterative(egraph: &EGraph, root_class: u32, best_node: &[Option<SymOp>]) -> SymExpr {
    let mut emitted: BTreeMap<u32, u32> = BTreeMap::new();
    let mut nodes: Vec<SymOp> = Vec::new();
    let mut stack: Vec<(u32, bool)> = vec![(root_class, false)];

    while let Some((cid, visited)) = stack.pop() {
        let canon = egraph.find(cid);
        if emitted.contains_key(&canon) {
            continue;
        }
        let chosen = best_node[canon as usize]
            .or_else(|| egraph.classes[canon as usize].first().copied())
            .unwrap_or(SymOp::num(0.0));
        if !visited {
            stack.push((canon, true));
            let mut kids = [0u32; 3];
            let n = chosen.children_into(&mut kids);
            let mut i = n;
            while i > 0 {
                i -= 1;
                let child = egraph.find(kids[i]);
                if !emitted.contains_key(&child) {
                    stack.push((child, false));
                }
            }
        } else {
            let mapped = chosen.map_children(|k| {
                let kc = egraph.find(k);
                emitted.get(&kc).copied().unwrap_or(0)
            });
            let slot = nodes.len() as u32;
            nodes.push(mapped);
            emitted.insert(canon, slot);
        }
    }

    let root_slot = emitted.get(&root_class).copied().unwrap_or(0);
    SymExpr {
        nodes,
        symbols: egraph.symbols.clone(),
        root: root_slot,
    }
}
