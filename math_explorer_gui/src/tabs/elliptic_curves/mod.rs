// @explorer_feature = "pure_math"
// @explorer_domain = "Pure Math"
// @explorer_slug = "elliptic_curves"
// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]

mod chord_tangent_canvas;
mod chord_tangent_scene;
pub mod chord_tangent_tool;
mod chord_tangent_ui;
pub mod modular_invariants_tool;
mod modular_invariants_ui;

#[cfg(test)]
mod tests;

crate::declare_simulation_tab!(
    /// Auto-discovered Pure Math tab for short-Weierstrass elliptic curve geometry and modular invariants.
    pub struct EllipticCurvesTab,
    "Elliptic Curves",
);
