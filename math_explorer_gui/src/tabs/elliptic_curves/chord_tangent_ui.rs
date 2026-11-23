// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]

use super::chord_tangent_canvas::{fmt_fp_pt, fmt_real_pt};
use super::chord_tangent_tool::{
    ClickTarget, CurveFieldMode, WeierstrassPlaygroundTool, snap_prime_i64, snap_real_point,
};
use crate::accessibility::AccessibleHoverText;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport, bounded_drag_value};
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{
    MAX_LATTICE_PRIME, PrimeWeierstrassCurve, WeierstrassCurve, j_invariant_mod_p,
    j_invariant_real,
};

pub(crate) fn render_side_panel(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) {
    let mut changed = false;
    ui.horizontal(|ui| {
        changed |= ui
            .selectable_value(
                &mut tool.state.field_mode,
                CurveFieldMode::FiniteField,
                "Finite Torus F_p",
            )
            .accessible_hover_text("Switch to finite prime-field torus F_p x F_p")
            .changed();
        changed |= ui
            .selectable_value(
                &mut tool.state.field_mode,
                CurveFieldMode::Real,
                "Continuous ℝ",
            )
            .accessible_hover_text("Switch to continuous real curve y^2 = x^3 + ax + b")
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("Click sets:");
        ui.radio_value(&mut tool.click_target, ClickTarget::PointP, "Point P");
        ui.radio_value(&mut tool.click_target, ClickTarget::PointQ, "Point Q");
    });
    ui.separator();

    match tool.state.field_mode {
        CurveFieldMode::Real => changed |= render_real_controls(tool, ui),
        CurveFieldMode::FiniteField => changed |= render_fp_controls(tool, ui),
    }

    ui.separator();
    changed |= render_orbit_and_transport(tool, ui);
    if changed {
        tool.rebuild_lattice();
    }
    ui.separator();
    render_group_summary(tool, ui);
}

fn render_real_controls(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("a:");
        changed |= ui
            .add(bounded_drag_value(&mut tool.state.real_a, -10.0..=10.0, 0.05))
            .changed();
        ui.label("b:");
        changed |= ui
            .add(bounded_drag_value(&mut tool.state.real_b, -10.0..=10.0, 0.05))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("x_P:");
        changed |= ui
            .add(bounded_drag_value(&mut tool.state.real_px, -4.0..=4.0, 0.05))
            .changed();
        changed |= ui
            .checkbox(&mut tool.state.real_p_upper, "+y_P")
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("x_Q:");
        changed |= ui
            .add(bounded_drag_value(&mut tool.state.real_qx, -4.0..=4.0, 0.05))
            .changed();
        changed |= ui
            .checkbox(&mut tool.state.real_q_upper, "+y_Q")
            .changed();
    });
    changed
}

fn render_fp_controls(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    let mut cand_p = tool.state.prime_p;
    if ui
        .add(egui::Slider::new(&mut cand_p, 5..=MAX_LATTICE_PRIME).text("Prime p"))
        .changed()
    {
        tool.state.prime_p = snap_prime_i64(cand_p);
        changed = true;
    }
    ui.horizontal_wrapped(|ui| {
        for preset_p in [7_i64, 17, 23, 47, 97, 251, 997] {
            if ui
                .selectable_label(tool.state.prime_p == preset_p, format!("p={preset_p}"))
                .clicked()
            {
                tool.state.prime_p = preset_p;
                changed = true;
            }
        }
    });
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.mod_a, -15..=15).text("a (mod p)"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.mod_b, -15..=15).text("b (mod p)"))
        .changed();
    if !tool.lattice_points.is_empty() {
        let max_idx = tool.lattice_points.len() - 1;
        changed |= ui
            .add(egui::Slider::new(&mut tool.state.selected_p_idx, 0..=max_idx).text("P index"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.state.selected_q_idx, 0..=max_idx).text("Q index"))
            .changed();
    }
    changed
}

fn render_orbit_and_transport(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.scalar_k, 1..=64).text("Orbit [k]P"))
        .changed();
    changed |= ui
        .checkbox(&mut tool.state.show_secant, "Show Secant & Toroidal Wrap")
        .changed();
    changed |= ui
        .checkbox(&mut tool.state.show_orbit, "Show [k]P Subgroup Orbit")
        .changed();

    if let Some(action) =
        PlaybackTransport::in_thread("weierstrass_orbit_transport", tool.name(), tool.is_playing)
            .show(ui)
    {
        match action {
            PlaybackAction::Play => tool.is_playing = true,
            PlaybackAction::Pause => tool.is_playing = false,
            PlaybackAction::Step => tool.step_simulation(),
            PlaybackAction::Reset => {
                tool.is_playing = false;
                tool.state.scalar_k = 1;
                changed = true;
            }
        }
    }
    changed
}

fn render_group_summary(tool: &WeierstrassPlaygroundTool, ui: &mut egui::Ui) {
    let err_color = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
    match tool.state.field_mode {
        CurveFieldMode::Real => match WeierstrassCurve::new(tool.state.real_a, tool.state.real_b) {
            Ok(curve) => {
                let p = snap_real_point(&curve, tool.state.real_px, tool.state.real_p_upper);
                let q = snap_real_point(&curve, tool.state.real_qx, tool.state.real_q_upper);
                let sum = curve.add(p, q);
                let kp = curve.scalar_mul(tool.state.scalar_k, p);
                ui.label(format!(
                    "Δ = {:.3}, j(E) = {:.3}",
                    curve.discriminant(),
                    j_invariant_real(&curve)
                ));
                ui.label(format!("P + Q = {}", fmt_real_pt(sum)));
                ui.label(format!(
                    "[{}]P = {}",
                    tool.state.scalar_k,
                    fmt_real_pt(kp)
                ));
            }
            Err(err) => {
                ui.colored_label(err_color, err.to_string());
            }
        },
        CurveFieldMode::FiniteField => {
            match PrimeWeierstrassCurve::new(
                tool.state.mod_a,
                tool.state.mod_b,
                tool.state.prime_p,
            ) {
                Ok(curve) => {
                    let (p, q) = tool.current_fp_points();
                    let sum = curve.add(p, q);
                    let kp = curve.scalar_mul(tool.state.scalar_k, p);
                    let order = curve.point_count().unwrap_or(1);
                    let trace = curve.frobenius_trace().unwrap_or(0);
                    ui.label(format!(
                        "#E(F_{}) = {order}, a_p = {trace}, j = {}",
                        curve.p,
                        j_invariant_mod_p(&curve)
                    ));
                    ui.label(format!("P = {}, Q = {}", fmt_fp_pt(p), fmt_fp_pt(q)));
                    ui.label(format!(
                        "P + Q = {}, [{}]P = {}",
                        fmt_fp_pt(sum),
                        tool.state.scalar_k,
                        fmt_fp_pt(kp)
                    ));
                }
                Err(err) => {
                    ui.colored_label(err_color, err.to_string());
                }
            }
        }
    }
}
