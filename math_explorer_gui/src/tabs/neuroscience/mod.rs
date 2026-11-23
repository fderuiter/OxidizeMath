// @explorer_feature = "biology"
// @explorer_domain = "Biology"
// @explorer_slug = "neuroscience"
// @explorer_order = 41

pub mod hodgkin_huxley;
pub mod neural_network_viz;
pub mod spike_analysis;

crate::declare_simulation_tab!(
    pub struct NeuroscienceTab,
    "Neuroscience",
);

// [cite:graph_parameters_rust]
