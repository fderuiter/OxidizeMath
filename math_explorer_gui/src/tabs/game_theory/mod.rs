#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod glicko2;
pub mod replicator;

crate::declare_simulation_tab!(
    pub struct GameTheoryTab,
    name: "Game Theory",
    domain: "Applied Math",
    slug: "game_theory",
    order: 30,
);

// [cite:graph_parameters_rust]
