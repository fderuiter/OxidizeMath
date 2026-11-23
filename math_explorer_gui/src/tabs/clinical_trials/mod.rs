// @explorer_feature = "applied"
// @explorer_domain = "Applied Math"
// @explorer_slug = "clinical_trials"
// @explorer_order = 31

pub mod pharmacokinetics;
pub mod randomization;
pub mod sample_size;
pub mod simpson_paradox;
pub mod survival;

crate::declare_simulation_tab!(
    pub struct ClinicalTrialsTab,
    "Clinical Trials",
);

// [cite:clinical_trials_statistics]
