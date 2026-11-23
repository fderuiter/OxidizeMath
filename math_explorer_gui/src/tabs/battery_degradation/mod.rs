// @explorer_feature = "applied"
// @explorer_domain = "Applied Math"
// @explorer_slug = "battery_degradation"
// @explorer_order = 33

pub mod capacity_fade;
pub mod lifetime_estimator;

crate::declare_simulation_tab!(
    pub struct BatteryDegradationTab,
    "Battery Degradation",
);

// [cite:algorithmic_information_rust]
