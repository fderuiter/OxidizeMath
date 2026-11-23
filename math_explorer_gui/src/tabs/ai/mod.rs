#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
    name: "Artificial Intelligence",
    domain: "AI",
    slug: "ai",
    order: 60,
);

// [cite:graph_parameters_rust]
