#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod capacity_fade;
pub mod lifetime_estimator;

crate::declare_simulation_tab!(
    pub struct BatteryDegradationTab,
    name: "Battery Degradation",
    domain: "Applied Math",
    slug: "battery_degradation",
    order: 33,
);

// [cite:algorithmic_information_rust]
