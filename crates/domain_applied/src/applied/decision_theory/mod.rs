//! # Quantitative Decision Theory & Multi-Armed Bandits
//!
//! Zero-allocation `f64` algorithms for sequential decision analysis under uncertainty:
//! - Expected utility & certainty equivalents (`Neutral`, `CARA`, `CRRA`) (`RiskAttitude`)
//! - Arena-indexed backward induction on finite decision trees and DAGs (`backward_induction`)
//! - Expected Value of Perfect (`EVPI`) and Sample (`EVSI`) Information (`evpi`, `evsi`)
//! - Deterministic `OxidizeRng`-seeded `UCB1` and Beta-Bernoulli Thompson Sampling bandits
//!   (`Ucb1Bandit`, `ThompsonBandit`)
//!
//! ## Quick Start
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::decision_theory::{
//!     DecisionEdge, DecisionNode, DecisionTreeView, DecisionTreeWorkspace, RiskAttitude,
//!     backward_induction,
//! };
//!
//! let nodes = [
//!     DecisionNode::Decision { edge_start: 0, edge_len: 2 },
//!     DecisionNode::Terminal { payoff: 40.0 },
//!     DecisionNode::Chance { edge_start: 2, edge_len: 2 },
//!     DecisionNode::Terminal { payoff: 100.0 },
//!     DecisionNode::Terminal { payoff: 0.0 },
//! ];
//! let edges = [
//!     DecisionEdge { target: 1, probability: 1.0 },
//!     DecisionEdge { target: 2, probability: 1.0 },
//!     DecisionEdge { target: 3, probability: 0.5 },
//!     DecisionEdge { target: 4, probability: 0.5 },
//! ];
//! let tree = DecisionTreeView::new(&nodes, &edges, 0).unwrap();
//! let mut ws = DecisionTreeWorkspace::new(5, 4);
//! let res = backward_induction(tree, RiskAttitude::Neutral, &mut ws).unwrap();
//! assert!((res.root_certainty_equivalent - 50.0).abs() < 1e-12);
//! assert_eq!(ws.optimal_branch(0), Some(1));
//! ```

mod bandits;
mod tree;
mod utility;
mod voi;

pub use bandits::{
    ThompsonBandit, Ucb1Bandit, simulate_bernoulli_thompson, simulate_bernoulli_ucb1,
};
pub use tree::{
    BackwardInductionResult, DecisionEdge, DecisionNode, DecisionTreeArena, DecisionTreeView,
    DecisionTreeWorkspace, backward_induction,
};
pub use utility::RiskAttitude;
pub use voi::{EvpiResult, EvsiResult, VoiWorkspace, evpi, evsi};

use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

/// Stack-only error type for decision trees, risk utilities, value of information, and bandits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecisionTheoryError {
    /// Decision tree cannot be empty.
    EmptyTree,
    /// Referenced node index exceeds the arena node slice length.
    InvalidNodeIndex {
        /// Offending node index.
        index: usize,
        /// Arena node count.
        node_count: usize,
    },
    /// Node edge slice `[start .. start + len]` exceeds the arena edge table.
    InvalidEdgeRange {
        /// Edge start offset.
        start: usize,
        /// Edge count for the node.
        len: usize,
        /// Total arena edge count.
        edge_count: usize,
    },
    /// Decision or chance node has zero outgoing branches.
    EmptyBranches {
        /// Offending node index.
        node: usize,
    },
    /// Chance node outgoing probabilities are invalid or do not sum to `1.0`.
    InvalidBranchProbabilities {
        /// Offending chance node index.
        node: usize,
        /// Observed probability sum.
        sum: f64,
    },
    /// Decision graph contains a directed cycle (`visited < total` in topological sort).
    CyclicTree {
        /// Number of acyclic nodes visited.
        visited: usize,
        /// Total nodes in arena.
        total: usize,
    },
    /// Risk-aversion parameter (`alpha` for CARA or `gamma` for CRRA) is out of domain.
    InvalidRiskParameter {
        /// Offending parameter value.
        value: f64,
    },
    /// CRRA utility requires strictly positive wealth outcomes (`wealth > 0`).
    NonPositiveWealth {
        /// Offending wealth outcome.
        value: f64,
    },
    /// Lottery or payoff slice cannot be empty.
    EmptyOutcomes,
    /// Slice or matrix dimensions did not match expected sizes.
    DimensionMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// Prior state distribution is invalid or unnormalized.
    InvalidPrior {
        /// Observed prior sum.
        sum: f64,
    },
    /// Signal likelihood row $P(z \mid \theta)$ is invalid or unnormalized.
    InvalidLikelihoodMatrix {
        /// Offending state row index.
        state: usize,
        /// Observed row sum.
        sum: f64,
    },
    /// Multi-armed bandit requires at least one arm.
    EmptyArms,
    /// Arm index exceeds configured arm count.
    InvalidArmIndex {
        /// Offending arm index.
        index: usize,
        /// Configured arm count.
        arm_count: usize,
    },
    /// Bandit reward or Bernoulli arm mean lies outside the valid range.
    InvalidReward {
        /// Offending reward or probability value.
        value: f64,
    },
    /// UCB1 exploration constant or Beta prior hyperparameter is invalid.
    InvalidExplorationParameter {
        /// Offending hyperparameter value.
        value: f64,
    },
    /// Preallocated workspace or output buffer is smaller than required.
    WorkspaceTooSmall {
        /// Minimum required capacity.
        required: usize,
        /// Provided capacity.
        actual: usize,
    },
}

impl std::fmt::Display for DecisionTheoryError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTree => write!(f, "decision tree cannot be empty"),
            Self::InvalidNodeIndex { index, node_count } => {
                write!(f, "node index {index} out of bounds for {node_count} nodes")
            }
            Self::InvalidEdgeRange {
                start,
                len,
                edge_count,
            } => write!(
                f,
                "edge slice {start}..{} exceeds arena of {edge_count} edges",
                start + len
            ),
            Self::EmptyBranches { node } => {
                write!(f, "non-terminal node {node} must have at least one branch")
            }
            Self::InvalidBranchProbabilities { node, sum } => {
                write!(
                    f,
                    "chance node {node} probabilities must be valid and sum to 1.0 (got {sum})"
                )
            }
            Self::CyclicTree { visited, total } => {
                write!(
                    f,
                    "decision graph contains a cycle (topologically visited {visited} of {total})"
                )
            }
            Self::InvalidRiskParameter { value } => {
                write!(f, "invalid risk aversion parameter (got {value})")
            }
            Self::NonPositiveWealth { value } => {
                write!(
                    f,
                    "crra utility requires strictly positive wealth (got {value})"
                )
            }
            Self::EmptyOutcomes => write!(f, "outcome slice cannot be empty"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch (expected {expected}, got {actual})")
            }
            Self::InvalidPrior { sum } => {
                write!(f, "prior state probabilities must sum to 1.0 (got {sum})")
            }
            Self::InvalidLikelihoodMatrix { state, sum } => {
                write!(
                    f,
                    "signal likelihood row for state {state} must sum to 1.0 (got {sum})"
                )
            }
            Self::EmptyArms => write!(f, "bandit must have at least one arm"),
            Self::InvalidArmIndex { index, arm_count } => {
                write!(f, "arm index {index} out of bounds for {arm_count} arms")
            }
            Self::InvalidReward { value } => {
                write!(f, "invalid bandit reward or probability ({value})")
            }
            Self::InvalidExplorationParameter { value } => {
                write!(f, "invalid bandit exploration or prior parameter ({value})")
            }
            Self::WorkspaceTooSmall { required, actual } => {
                write!(
                    f,
                    "decision workspace too small (required {required}, got {actual})"
                )
            }
        }
    }
}

impl std::error::Error for DecisionTheoryError {}

impl Diagnostic for DecisionTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "DecisionTheoryError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

#[cfg(test)]
mod tests;

// [cite:graph_parameters_rust]
