/// Submodule providing `dijkstra` types, traits, and algorithms.
pub mod dijkstra;
/// Error types for graph solvers and topology interchange codecs.
pub mod error;
/// Submodule providing `graph` types, traits, and algorithms.
pub mod graph;
/// Deterministic `.dot`, `.graphml`, and `.csv` graph interchange codecs.
pub mod interchange;
/// Submodule providing `layout` types, traits, and algorithms.
pub mod layout;
/// Preallocated zero-allocation Edmonds-Karp `s-t` max-flow and min-cut solver.
pub mod max_flow_min_cut;
/// Preallocated zero-allocation Kruskal minimum spanning tree solver.
pub mod minimum_spanning_tree;
/// Submodule providing `parameters` types, traits, and algorithms.
pub mod parameters;

pub use error::GraphTheoryError;
pub use max_flow_min_cut::max_flow_min_cut as solve_max_flow_min_cut;
pub use minimum_spanning_tree::minimum_spanning_tree as solve_minimum_spanning_tree;

// [cite:graph_parameters_rust]
// [cite:dijkstra]
