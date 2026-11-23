// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]

use super::atlas::WeierstrassRegimeCartographer;
use super::chord_tangent_canvas::{fmt_fp_pt, fmt_real_pt};
use super::chord_tangent_tool::{
    snap_prime_i64, snap_real_point, ClickTarget, CurveFieldMode, WeierstrassPlaygroundTool,
};
use super::handoff::{
    curve_point_to_affine_fp, render_handoff_buttons, EllipticHandoffTarget, EllipticTorusHandoff,
};
use super::singular_scene::snap_real_singular_point;
use crate::accessibility::AccessibleHoverText;
use crate::framework::{bounded_drag_value, InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::pure_math::elliptic_curves::{
    j_invariant_mod_p, j_invariant_real, CurvePoint, FpSingularCubic, PrimeWeierstrassCurve,
    RealSingularCubic, WeierstrassCurve, MAX_LATTICE_PRIME,
};
use std::sync::Arc;

pub(crate) fn render_side_panel(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) {
    let mut changed = render_mode_and_handoff_header(tool, ui);
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

fn render_mode_and_handoff_header(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.add_enabled_ui(tool.active_challenge.is_none(), |ui| {
        ui.horizontal(|ui| {
            if ui
                .selectable_value(
                    &mut tool.state.field_mode,
                    CurveFieldMode::FiniteField,
                    "Finite Torus F_p",
                )
                .accessible_hover_text("Switch to finite prime-field torus F_p x F_p")
                .changed()
            {
                let _ = tool.atlas.set_axis_keys("mod_a", "mod_b");
                changed = true;
            }
            if ui
                .selectable_value(
                    &mut tool.state.field_mode,
                    CurveFieldMode::Real,
                    "Continuous ℝ",
                )
                .accessible_hover_text("Switch to continuous real curve y^2 = x^3 + ax + b")
                .changed()
            {
                let _ = tool.atlas.set_axis_keys("real_a", "real_b");
                changed = true;
            }
        });
    });
    ui.horizontal(|ui| {
        ui.label("Click sets:");
        ui.radio_value(&mut tool.click_target, ClickTarget::PointP, "Point P");
        ui.radio_value(&mut tool.click_target, ClickTarget::PointQ, "Point Q");
    });
    if tool.atlas.show_toggle_button(ui) {
        if tool.atlas.is_visible() {
            let (xk, yk) = match tool.state.field_mode {
                CurveFieldMode::Real => ("real_a", "real_b"),
                CurveFieldMode::FiniteField => ("mod_a", "mod_b"),
            };
            let _ = tool.atlas.set_axis_keys(xk, yk);
        }
        changed = true;
    }
    render_playground_handoff_bar(tool, ui);
    changed
}

fn render_playground_handoff_bar(tool: &WeierstrassPlaygroundTool, ui: &mut egui::Ui) {
    let (fp_p, fp_q) = tool.current_fp_points();
    let (ha, hb) = match tool.state.field_mode {
        CurveFieldMode::Real => (
            tool.state.real_a.round() as i64,
            tool.state.real_b.round() as i64,
        ),
        CurveFieldMode::FiniteField => (tool.state.mod_a, tool.state.mod_b),
    };
    render_handoff_buttons(
        ui,
        EllipticHandoffTarget::WeierstrassPlayground,
        EllipticTorusHandoff {
            target: EllipticHandoffTarget::WeierstrassPlayground,
            p: tool.state.prime_p,
            a: ha,
            b: hb,
            point_p: curve_point_to_affine_fp(fp_p),
            point_q: curve_point_to_affine_fp(fp_q),
            scalar_k: tool.state.scalar_k,
        },
    );
}

fn render_real_controls(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("a:");
        changed |= ui
            .add(bounded_drag_value(
                &mut tool.state.real_a,
                -20.0..=20.0,
                0.05,
            ))
            .changed();
        ui.label("b:");
        changed |= ui
            .add(bounded_drag_value(
                &mut tool.state.real_b,
                -20.0..=20.0,
                0.05,
            ))
            .changed();
    });
    ui.horizontal(|ui| {
        ui.label("x_P:");
        changed |= ui
            .add(bounded_drag_value(
                &mut tool.state.real_px,
                -4.0..=4.0,
                0.05,
            ))
            .changed();
        changed |= ui.checkbox(&mut tool.state.real_p_upper, "+y_P").changed();
    });
    ui.horizontal(|ui| {
        ui.label("x_Q:");
        changed |= ui
            .add(bounded_drag_value(
                &mut tool.state.real_qx,
                -4.0..=4.0,
                0.05,
            ))
            .changed();
        changed |= ui.checkbox(&mut tool.state.real_q_upper, "+y_Q").changed();
    });
    changed
}

fn render_fp_controls(tool: &mut WeierstrassPlaygroundTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.add_enabled_ui(tool.active_challenge.is_none(), |ui| {
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
            PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
        }
    }
    changed
}

fn render_group_summary(tool: &WeierstrassPlaygroundTool, ui: &mut egui::Ui) {
    let err_color = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
    match tool.state.field_mode {
        CurveFieldMode::Real => render_real_group_summary(tool, ui, err_color),
        CurveFieldMode::FiniteField => render_fp_group_summary(tool, ui, err_color),
    }
}

fn render_real_group_summary(
    tool: &WeierstrassPlaygroundTool,
    ui: &mut egui::Ui,
    err_color: egui::Color32,
) {
    match WeierstrassCurve::new(tool.state.real_a, tool.state.real_b) {
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
            ui.label(format!("[{}]P = {}", tool.state.scalar_k, fmt_real_pt(kp)));
        }
        Err(err) => {
            if let Some(cubic) = RealSingularCubic::classify(tool.state.real_a, tool.state.real_b) {
                let p =
                    snap_real_singular_point(&cubic, tool.state.real_px, tool.state.real_p_upper);
                let q =
                    snap_real_singular_point(&cubic, tool.state.real_qx, tool.state.real_q_upper);
                let sum = cubic.add_smooth(p, q).unwrap_or(CurvePoint::Infinity);
                ui.colored_label(
                    err_color,
                    format!(
                        "Δ = 0 Singular {}: vertex ({:.2}, 0)",
                        cubic.kind.label(),
                        cubic.singular_x
                    ),
                );
                ui.label(format!("Smooth P + Q = {}", fmt_real_pt(sum)));
            } else {
                ui.colored_label(err_color, err.to_string());
            }
        }
    }
}

fn render_fp_group_summary(
    tool: &WeierstrassPlaygroundTool,
    ui: &mut egui::Ui,
    err_color: egui::Color32,
) {
    match PrimeWeierstrassCurve::new(tool.state.mod_a, tool.state.mod_b, tool.state.prime_p) {
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
            if let Ok(Some(sing)) =
                FpSingularCubic::classify(tool.state.mod_a, tool.state.mod_b, tool.state.prime_p)
            {
                ui.colored_label(
                    err_color,
                    format!(
                        "Singular {} over F_{}: vertex ({}, 0)",
                        sing.kind.label(),
                        sing.p,
                        sing.singular_x
                    ),
                );
                ui.label(format!(
                    "Affine pts = {}, #E_ns(F_{}) = {}",
                    sing.affine_point_count(),
                    sing.p,
                    sing.smooth_group_order()
                ));
            } else {
                ui.colored_label(err_color, err.to_string());
            }
        }
    }
}

pub(crate) fn render_weierstrass_atlas_split(
    tool: &mut WeierstrassPlaygroundTool,
    ui: &mut egui::Ui,
) {
    if !tool.atlas.is_visible() {
        return;
    }
    let (xk, _) = tool.atlas.axis_keys();
    let (ax, ay) = if xk == "mod_a" {
        (tool.state.mod_a as f64, tool.state.mod_b as f64)
    } else {
        (tool.state.real_a, tool.state.real_b)
    };
    let prime_p = tool.state.prime_p;
    let sig = if xk == "mod_a" { prime_p as u64 } else { 0_u64 };
    let atlas_h = (ui.available_height() * 0.44_f32).clamp(180.0_f32, 300.0_f32);
    let mut teleport = None;
    ui.allocate_ui(egui::vec2(ui.available_width(), atlas_h), |atlas_ui| {
        teleport = tool
            .atlas
            .show_panel(atlas_ui, "weierstrass_play", ax, ay, sig, move |keys| {
                Arc::new(WeierstrassRegimeCartographer::new(keys, prime_p, 0.25))
            });
    });
    ui.separator();
    if let Some(ev) = teleport {
        tool.apply_atlas_teleport(ev);
    }
}
