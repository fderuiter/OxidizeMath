#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod elliptic_curve_audio;
pub mod elliptic_curve_canvas;
pub mod elliptic_curve_hasse;
pub mod elliptic_curve_playground;
#[cfg(test)]
mod elliptic_curve_tests;
pub mod factorization;
pub mod fractran_machine;
pub mod partitions_widget;
pub mod prime_spiral;
pub mod ualbf_widget;

crate::declare_simulation_tab!(
    pub struct NumberTheoryTab,
    name: "Number Theory",
    domain: "Pure Math",
    slug: "number_theory",
    order: 21,
);

// [cite:graph_parameters_rust]
