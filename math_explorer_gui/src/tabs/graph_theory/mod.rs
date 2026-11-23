// @explorer_feature = "pure_math"
// @explorer_domain = "Pure Math"
// @explorer_slug = "graph_theory"
// @explorer_order = 23

pub mod algorithm_visualizer;
pub mod graph_editor;
pub mod network_metrics;

crate::declare_simulation_tab!(
    pub struct GraphTheoryTab,
    "Graph Theory",
);

// [cite:graph_parameters_rust]
