#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod pharmacokinetics;
pub mod randomization;
pub mod randomization_io;
pub mod sample_size;
pub mod sample_size_cards;
pub mod simpson_atlas;
pub mod simpson_paradox;
pub mod survival;
pub mod survival_io;
pub mod win_ratio;
pub mod win_ratio_io;
pub mod win_ratio_weibull;
pub mod win_ratio_workbench;

crate::declare_simulation_tab!(
    pub struct ClinicalTrialsTab,
    name: "Clinical Trials",
    domain: "Applied Math",
    slug: "clinical_trials",
    order: 31,
);

// [cite:clinical_trials_statistics]
