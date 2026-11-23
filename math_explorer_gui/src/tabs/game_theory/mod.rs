// @explorer_feature = "applied"
// @explorer_domain = "Applied Math"
// @explorer_slug = "game_theory"
// @explorer_order = 30

pub mod glicko2;
pub mod replicator;

crate::declare_simulation_tab!(
    pub struct GameTheoryTab,
    "Game Theory",
);

// [cite:graph_parameters_rust]
