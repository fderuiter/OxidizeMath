//! Arena-indexed finite decision trees / acyclic decision graphs and backward induction.

use super::{DecisionTheoryError, RiskAttitude};
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};

/// Directed branch connecting a parent decision or chance node to a child node in the arena.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionEdge {
    /// Target child node index in `nodes`.
    pub target: usize,
    /// Branch probability (validated on `Chance` nodes; ignored on `Decision` nodes).
    pub probability: f64,
}

/// Arena-indexed node in a finite sequential decision tree or DAG.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecisionNode {
    /// Terminal leaf node yielding deterministic wealth `payoff`.
    Terminal {
        /// Terminal wealth / payoff outcome.
        payoff: f64,
    },
    /// Decision maker chooses the outgoing branch in `edges[edge_start .. edge_start + edge_len]` maximizing utility.
    Decision {
        /// Start offset in `edges`.
        edge_start: usize,
        /// Number of available actions (`>= 1`).
        edge_len: usize,
    },
    /// Nature resolves uncertainty across `edges[edge_start .. edge_start + edge_len]`.
    Chance {
        /// Start offset in `edges`.
        edge_start: usize,
        /// Number of stochastic outcomes (`>= 1`).
        edge_len: usize,
    },
}

/// Borrowed view over flat arena slices representing a finite decision tree or DAG.
#[derive(Debug, Clone, Copy)]
pub struct DecisionTreeView<'a> {
    /// Flat node arena.
    pub nodes: &'a [DecisionNode],
    /// Contiguous outgoing edge ranges referenced by `Decision` and `Chance` nodes.
    pub edges: &'a [DecisionEdge],
    /// Root node index in `nodes`.
    pub root: usize,
}

impl<'a> DecisionTreeView<'a> {
    /// Validates arena bounds and chance-node probability distributions.
    #[verified_engine::verified]
    pub fn new(
        nodes: &'a [DecisionNode],
        edges: &'a [DecisionEdge],
        root: usize,
    ) -> Result<Self, DecisionTheoryError> {
        if nodes.is_empty() {
            return Err(DecisionTheoryError::EmptyTree);
        }
        if root >= nodes.len() {
            return Err(DecisionTheoryError::InvalidNodeIndex {
                index: root,
                node_count: nodes.len(),
            });
        }
        for (idx, &node) in nodes.iter().enumerate() {
            match node {
                DecisionNode::Terminal { payoff } => {
                    if !payoff.is_finite() {
                        return Err(DecisionTheoryError::InvalidRiskParameter { value: payoff });
                    }
                }
                DecisionNode::Decision {
                    edge_start,
                    edge_len,
                } => {
                    validate_edge_slice(idx, edge_start, edge_len, nodes.len(), edges)?;
                }
                DecisionNode::Chance {
                    edge_start,
                    edge_len,
                } => {
                    validate_edge_slice(idx, edge_start, edge_len, nodes.len(), edges)?;
                    let mut sum = 0.0;
                    for e in &edges[edge_start..edge_start + edge_len] {
                        let p = e.probability;
                        if !p.is_finite() || p < -TOLERANCE_HIGH || p > 1.0 + TOLERANCE_FAST {
                            return Err(DecisionTheoryError::InvalidBranchProbabilities {
                                node: idx,
                                sum: f64::NAN,
                            });
                        }
                        sum += p.max(0.0);
                    }
                    if (sum - 1.0).abs() > TOLERANCE_FAST {
                        return Err(DecisionTheoryError::InvalidBranchProbabilities {
                            node: idx,
                            sum,
                        });
                    }
                }
            }
        }
        Ok(Self { nodes, edges, root })
    }
}

#[verified_engine::verified]
fn validate_edge_slice(
    node_idx: usize,
    start: usize,
    len: usize,
    node_count: usize,
    edges: &[DecisionEdge],
) -> Result<(), DecisionTheoryError> {
    if len == 0 {
        return Err(DecisionTheoryError::EmptyBranches { node: node_idx });
    }
    let end = start.saturating_add(len);
    if end > edges.len() {
        return Err(DecisionTheoryError::InvalidEdgeRange {
            start,
            len,
            edge_count: edges.len(),
        });
    }
    for e in &edges[start..end] {
        if e.target >= node_count {
            return Err(DecisionTheoryError::InvalidNodeIndex {
                index: e.target,
                node_count,
            });
        }
    }
    Ok(())
}

/// Owned builder convenience around [`DecisionTreeView`].
#[derive(Clone, Debug, Default)]
pub struct DecisionTreeArena {
    /// Owned node arena.
    pub nodes: Vec<DecisionNode>,
    /// Owned edge table.
    pub edges: Vec<DecisionEdge>,
    /// Root node index.
    pub root: usize,
}

impl DecisionTreeArena {
    /// Creates a validated [`DecisionTreeView`] borrowing this arena.
    pub fn view(&self) -> Result<DecisionTreeView<'_>, DecisionTheoryError> {
        DecisionTreeView::new(&self.nodes, &self.edges, self.root)
    }
}

/// Summary returned by [`backward_induction`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackwardInductionResult {
    /// Certainty equivalent at `tree.root`.
    pub root_certainty_equivalent: f64,
    /// Expected utility $u(CE_{\mathrm{root}})$ at `tree.root`.
    pub root_expected_utility: f64,
    /// Number of nodes evaluated in reverse topological order.
    pub evaluated_nodes: usize,
}

/// Preallocated scratch buffers for zero-allocation [`backward_induction`].
#[derive(Clone, Debug)]
pub struct DecisionTreeWorkspace {
    in_degree: Vec<usize>,
    topo_order: Vec<usize>,
    node_ce: Vec<f64>,
    optimal_choice: Vec<Option<usize>>,
    scratch_payoffs: Vec<f64>,
    scratch_probs: Vec<f64>,
}

impl DecisionTreeWorkspace {
    /// Allocates workspace buffers for arenas with up to `max_nodes` nodes and `max_edges` edges.
    #[must_use]
    pub fn new(max_nodes: usize, max_edges: usize) -> Self {
        let cap_edges = max_edges.max(1);
        Self {
            in_degree: vec![0; max_nodes],
            topo_order: vec![0; max_nodes],
            node_ce: vec![0.0; max_nodes],
            optimal_choice: vec![None; max_nodes],
            scratch_payoffs: vec![0.0; cap_edges],
            scratch_probs: vec![0.0; cap_edges],
        }
    }

    /// Reads the computed certainty equivalent at `node`.
    #[must_use]
    pub fn node_certainty_equivalent(&self, node: usize) -> Option<f64> {
        self.node_ce.get(node).copied()
    }

    /// Reads the optimal local branch offset (`0..edge_len`) selected at a `Decision` node.
    #[must_use]
    pub fn optimal_branch(&self, node: usize) -> Option<usize> {
        self.optimal_choice.get(node).copied().flatten()
    }
}

#[verified_engine::verified]
fn compute_topological_order(
    tree: DecisionTreeView<'_>,
    in_degree: &mut [usize],
    topo_order: &mut [usize],
) -> Result<(), DecisionTheoryError> {
    let n = tree.nodes.len();
    in_degree.fill(0);
    for &node in tree.nodes {
        match node {
            DecisionNode::Terminal { .. } => {}
            DecisionNode::Decision {
                edge_start,
                edge_len,
            }
            | DecisionNode::Chance {
                edge_start,
                edge_len,
            } => {
                for e in &tree.edges[edge_start..edge_start + edge_len] {
                    in_degree[e.target] += 1;
                }
            }
        }
    }

    let mut head = 0usize;
    let mut tail = 0usize;
    for (u, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            topo_order[tail] = u;
            tail += 1;
        }
    }

    while head < tail {
        let u = topo_order[head];
        head += 1;
        match tree.nodes[u] {
            DecisionNode::Terminal { .. } => {}
            DecisionNode::Decision {
                edge_start,
                edge_len,
            }
            | DecisionNode::Chance {
                edge_start,
                edge_len,
            } => {
                for e in &tree.edges[edge_start..edge_start + edge_len] {
                    in_degree[e.target] -= 1;
                    if in_degree[e.target] == 0 {
                        topo_order[tail] = e.target;
                        tail += 1;
                    }
                }
            }
        }
    }

    if tail != n {
        return Err(DecisionTheoryError::CyclicTree {
            visited: tail,
            total: n,
        });
    }
    Ok(())
}

/// Solves a finite decision tree or DAG by iterative reverse-topological certainty-equivalent folding.
#[verified_engine::verified]
pub fn backward_induction(
    tree: DecisionTreeView<'_>,
    risk: RiskAttitude,
    workspace: &mut DecisionTreeWorkspace,
) -> Result<BackwardInductionResult, DecisionTheoryError> {
    risk.validate()?;
    let n = tree.nodes.len();
    let m = tree.edges.len().max(1);
    if workspace.in_degree.len() < n
        || workspace.topo_order.len() < n
        || workspace.node_ce.len() < n
        || workspace.optimal_choice.len() < n
    {
        return Err(DecisionTheoryError::WorkspaceTooSmall {
            required: n,
            actual: workspace.node_ce.len(),
        });
    }
    if workspace.scratch_payoffs.len() < m || workspace.scratch_probs.len() < m {
        return Err(DecisionTheoryError::WorkspaceTooSmall {
            required: m,
            actual: workspace.scratch_payoffs.len(),
        });
    }

    compute_topological_order(
        tree,
        &mut workspace.in_degree[..n],
        &mut workspace.topo_order[..n],
    )?;

    for idx in (0..n).rev() {
        let u = workspace.topo_order[idx];
        match tree.nodes[u] {
            DecisionNode::Terminal { payoff } => {
                let _ = risk.utility(payoff)?;
                workspace.node_ce[u] = payoff;
                workspace.optimal_choice[u] = None;
            }
            DecisionNode::Decision {
                edge_start,
                edge_len,
            } => {
                let mut best_offset = 0usize;
                let mut best_ce = workspace.node_ce[tree.edges[edge_start].target];
                for offset in 1..edge_len {
                    let cand_ce = workspace.node_ce[tree.edges[edge_start + offset].target];
                    if cand_ce > best_ce {
                        best_ce = cand_ce;
                        best_offset = offset;
                    }
                }
                workspace.node_ce[u] = best_ce;
                workspace.optimal_choice[u] = Some(best_offset);
            }
            DecisionNode::Chance {
                edge_start,
                edge_len,
            } => {
                for offset in 0..edge_len {
                    let e = tree.edges[edge_start + offset];
                    workspace.scratch_payoffs[offset] = workspace.node_ce[e.target];
                    workspace.scratch_probs[offset] = e.probability;
                }
                let ce = risk.certainty_equivalent(
                    &workspace.scratch_payoffs[..edge_len],
                    &workspace.scratch_probs[..edge_len],
                )?;
                workspace.node_ce[u] = ce;
                workspace.optimal_choice[u] = None;
            }
        }
    }

    let root_ce = workspace.node_ce[tree.root];
    let root_eu = risk.utility(root_ce)?;
    Ok(BackwardInductionResult {
        root_certainty_equivalent: root_ce,
        root_expected_utility: root_eu,
        evaluated_nodes: n,
    })
}

// [cite:graph_parameters_rust]
