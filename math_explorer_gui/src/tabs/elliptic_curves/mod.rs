#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
    /// Pure Math tab for short-Weierstrass elliptic curve geometry and modular invariants.
    pub struct EllipticCurvesTab,
    name: "Elliptic Curves",
    domain: "Pure Math",
    slug: "elliptic_curves",
    order: 26,
);
