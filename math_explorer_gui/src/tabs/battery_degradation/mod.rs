#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod capacity_fade;
pub mod capacity_fade_atlas;
pub mod cell_card;
pub mod lifetime_estimator;

crate::declare_simulation_tab!(
    pub struct BatteryDegradationTab,
    name: "Battery Degradation",
    domain: "Applied Math",
    slug: "battery_degradation",
    order: 33,
);

// [cite:battery_degradation]
