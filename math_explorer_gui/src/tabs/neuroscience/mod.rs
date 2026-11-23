#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod cortical_morphometry;
pub mod hodgkin_huxley;
pub mod neural_network_viz;
pub mod spike_analysis;

crate::declare_simulation_tab!(
    pub struct NeuroscienceTab,
    name: "Neuroscience",
    domain: "Biology",
    slug: "neuroscience",
    order: 41,
);

// [cite:graph_parameters_rust]
