#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod algorithm_visualizer;
pub mod graph_editor;
pub mod network_metrics;

crate::declare_simulation_tab!(
    pub struct GraphTheoryTab,
    name: "Graph Theory",
    domain: "Pure Math",
    slug: "graph_theory",
    order: 23,
);

// [cite:graph_parameters_rust]
