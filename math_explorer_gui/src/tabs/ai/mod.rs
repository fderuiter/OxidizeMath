// @explorer_feature = "ai"
// @explorer_domain = "AI"
// @explorer_slug = "ai"
// @explorer_order = 60

pub mod activation_functions;
pub mod attention_maps;
pub mod gaussian_splatting;
pub mod grid_world;
pub mod loss_landscape;
pub mod q_table_inspector;
pub mod reward_plots;
pub mod tokenization;
pub mod training_monitor;

crate::declare_simulation_tab!(
    pub struct AiTab,
    "Artificial Intelligence",
);

// [cite:graph_parameters_rust]
