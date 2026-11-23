//! Torsion subgroup sonification helper for [`super::elliptic_curve_playground::EllipticCurvePlayground`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;

use super::elliptic_curve_playground::EllipticCurvePlayground;
use crate::audio::mappings::{elliptic_torsion_frame, EllipticTorsionSample};
use crate::audio::{render_sonification_panel, SonificationFrame};

impl EllipticCurvePlayground {
    /// Extracts the current stateless torsion `SonificationFrame` (`SonificationFrame::silent()` when singular).
    #[must_use]
    pub fn current_sonification_frame(&self) -> SonificationFrame {
        elliptic_torsion_frame(EllipticTorsionSample {
            modulus_p: self.p,
            is_singular: self.is_singular,
            kp_coords: self.scalar_kp.coordinates(),
            p_order: self.p_order,
            scalar_k: self.scalar_k,
            frobenius_trace: self.frobenius_trace,
            torus_wraps: self.ray_segments.len(),
        })
    }

    pub(super) fn render_torsion_sonification(&self, ui: &mut egui::Ui) {
        let frame = self.current_sonification_frame();
        render_sonification_panel(ui, "elliptic_curve_playground", &frame, self.is_singular);
    }
}

// [cite:modular_polynomials_review]
