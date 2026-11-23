// @explorer_feature = "pure_math"
// @explorer_domain = "Pure Math"
// @explorer_slug = "number_theory"

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
    "Number Theory",
);

// [cite:graph_parameters_rust]
