// [cite:modular_polynomials_review]
// [cite:hurwitz_kronecker]

use super::atlas::WeierstrassRegimeCartographer;
use super::handoff::{render_handoff_buttons, EllipticHandoffTarget, EllipticTorusHandoff};
use super::modular_invariants_tool::{
    snap_prime_u64, ModularInvariantsTool, ModularPlotMode, ModularQSeriesKind,
};
use crate::accessibility::AccessibleHoverText;
use crate::framework::scoped_panel_id;
use eframe::egui;
use egui_plot::{Bar, BarChart, Legend, Line, Plot, Points};
use math_explorer::pure_math::elliptic_curves::{
    hurwitz_trace_sum, psi, AffinePointFp, FpSingularCubic, RealSingularCubic, MAX_LATTICE_PRIME,
};
use std::sync::Arc;

pub(crate) fn render_controls(tool: &mut ModularInvariantsTool, ui: &mut egui::Ui) {
    let mut changed = false;
    ui.label("Analytical Lens:");
    for mode in [
        ModularPlotMode::HasseWeilTraces,
        ModularPlotMode::HurwitzKroneckerSums,
        ModularPlotMode::BreuerValuationBounds,
        ModularPlotMode::ModularQSeriesBars,
    ] {
        changed |= ui
            .radio_value(&mut tool.state.view_mode, mode, mode.label())
            .changed();
    }
    ui.separator();
    let _ = tool.atlas.show_toggle_button(ui);
    render_handoff_buttons(
        ui,
        EllipticHandoffTarget::ModularInvariants,
        EllipticTorusHandoff {
            target: EllipticHandoffTarget::ModularInvariants,
            p: tool.state.inspect_prime as i64,
            a: tool.state.coeff_a,
            b: tool.state.coeff_b,
            point_p: AffinePointFp::Infinity,
            point_q: AffinePointFp::Infinity,
            scalar_k: 1,
        },
    );
    ui.separator();
    changed |= render_parameter_sliders(tool, ui);
    if changed {
        tool.recompute();
    }
    ui.separator();
    render_invariant_readouts(tool, ui);
}

fn render_parameter_sliders(tool: &mut ModularInvariantsTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.coeff_a, -20..=20).text("Coefficient a"))
        .accessible_hover_text("Short-Weierstrass linear coefficient a")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.coeff_b, -20..=20).text("Coefficient b"))
        .accessible_hover_text("Short-Weierstrass constant coefficient b")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tool.state.max_prime, 11..=MAX_LATTICE_PRIME as u64)
                .text("Max Prime Sweep"),
        )
        .changed();
    ui.add_enabled_ui(tool.active_challenge.is_none(), |ui| {
        let mut p_cand = tool.state.inspect_prime;
        if ui
            .add(
                egui::Slider::new(&mut p_cand, 5..=MAX_LATTICE_PRIME as u64)
                    .text("Inspected Prime p"),
            )
            .changed()
        {
            tool.state.inspect_prime = snap_prime_u64(p_cand);
            changed = true;
        }
    });
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.level_n, 2..=30).text("Modular Level N"))
        .changed();
    let max_i = psi(tool.state.level_n).saturating_sub(1);
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.deg_i, 0..=max_i).text("Degree i + j"))
        .changed();
    tool.state.deg_j = 0;
    changed |= ui
        .add(egui::Slider::new(&mut tool.state.q_precision, 4..=24).text("Q-Series Precision"))
        .changed();
    for qk in [
        ModularQSeriesKind::RamanujanDelta,
        ModularQSeriesKind::EisensteinE4,
        ModularQSeriesKind::ScaledJInvariant,
    ] {
        changed |= ui
            .radio_value(&mut tool.state.q_series_kind, qk, qk.label())
            .changed();
    }
    changed
}

fn render_invariant_readouts(tool: &ModularInvariantsTool, ui: &mut egui::Ui) {
    let err_color = crate::app::SemanticPalette::from_visuals(ui.visuals()).error;
    if let Some(cubic) =
        RealSingularCubic::classify(tool.state.coeff_a as f64, tool.state.coeff_b as f64)
    {
        ui.colored_label(
            err_color,
            format!(
                "Singular over Q/R: {} at ({:.2}, 0) — sweeps & Hurwitz sums active.",
                cubic.kind.label(),
                cubic.singular_x
            ),
        );
    } else if let Some(j_re) = tool.real_j {
        ui.label(format!("Real j-invariant j(E): {j_re:.3}"));
    }
    if let Some(frob) = &tool.frobenius_summary {
        ui.label(format!(
            "F_{}: #E = {}, trace a_p = {}, D = {}",
            frob.p, frob.point_count, frob.frobenius_trace, frob.frobenius_discriminant
        ));
        ui.label(format!(
            "Class numbers: h(D) = {}, H(D) = {:.3}",
            frob.imaginary_class_number, frob.hurwitz_kronecker
        ));
        if let Some(bqf) = &frob.principal_form {
            ui.label(format!("Principal BQF: ({}, {}, {})", bqf.a, bqf.b, bqf.c));
        }
    } else if let Ok(Some(sing)) = FpSingularCubic::classify(
        tool.state.coeff_a,
        tool.state.coeff_b,
        tool.state.inspect_prime as i64,
    ) {
        ui.colored_label(
            err_color,
            format!(
                "F_{}: Singular {} at ({}, 0), #E_ns = {}",
                sing.p,
                sing.kind.label(),
                sing.singular_x,
                sing.smooth_group_order()
            ),
        );
    }
    let (h_sum, h_exp, _) = hurwitz_trace_sum(tool.state.inspect_prime);
    ui.label(format!(
        "Hurwitz ∑ H(t² - {}) = {:.3}  (law = {:.3})",
        tool.state.inspect_prime, h_sum, h_exp
    ));
    if let Some(val) = &tool.valuation_summary {
        ui.label(format!(
            "Φ_{} (ψ = {}): j mod p = {}, Thm 1.1 v₂ ≥ {:?}, v₃ ≥ {:?}; Thm 1.2 v₇ ≥ {:?}",
            val.level_n,
            val.psi_n,
            val.j_invariant_mod_p,
            val.theorem_1_1.v2_bound,
            val.theorem_1_1.v3_bound,
            val.theorem_1_2.v7_bound
        ));
    }
}

pub(crate) fn render_modular_atlas_split(tool: &mut ModularInvariantsTool, ui: &mut egui::Ui) {
    if !tool.atlas.is_visible() {
        return;
    }
    let (xk, _) = tool.atlas.axis_keys();
    let p = tool.state.inspect_prime as i64;
    let sig = if xk == "mod_a" { p as u64 } else { 0_u64 };
    let (ax, ay) = (tool.state.coeff_a as f64, tool.state.coeff_b as f64);
    let atlas_h = (ui.available_height() * 0.44_f32).clamp(180.0_f32, 300.0_f32);
    let mut teleport = None;
    ui.allocate_ui(egui::vec2(ui.available_width(), atlas_h), |atlas_ui| {
        teleport = tool
            .atlas
            .show_panel(atlas_ui, "modular_inv", ax, ay, sig, move |keys| {
                Arc::new(WeierstrassRegimeCartographer::new(keys, p, 1.0))
            });
    });
    ui.separator();
    if let Some(ev) = teleport {
        tool.apply_atlas_teleport(ev);
    }
}

pub(crate) fn render_plot(tool: &ModularInvariantsTool, ui: &mut egui::Ui) {
    let plot = Plot::new(scoped_panel_id(ui, "modular_invariants_plot"))
        .legend(Legend::default())
        .allow_zoom(true)
        .allow_drag(true);
    match tool.state.view_mode {
        ModularPlotMode::HasseWeilTraces => {
            plot.show(ui, |plot_ui| {
                plot_ui.line(Line::new("+2√p Hasse Bound", tool.hasse_upper.clone()));
                plot_ui.line(Line::new("-2√p Hasse Bound", tool.hasse_lower.clone()));
                plot_ui.points(
                    Points::new("Frobenius Trace a_p", tool.trace_pts.clone()).radius(4.0_f32),
                );
                plot_ui.line(Line::new(
                    "Class Number h(a_p² - 4p)",
                    tool.class_num_pts.clone(),
                ));
            });
        }
        ModularPlotMode::HurwitzKroneckerSums => {
            plot.show(ui, |plot_ui| {
                plot_ui.line(Line::new("(p - 2) / 3 Law", tool.hurwitz_expected.clone()));
                plot_ui.points(
                    Points::new("∑ H(t² - p)", tool.hurwitz_actual.clone()).radius(4.0_f32),
                );
            });
        }
        ModularPlotMode::BreuerValuationBounds => {
            plot.show(ui, |plot_ui| {
                for (label, series) in [
                    ("Thm 1.1 v₂ Lower Bound", &tool.breuer_bars[0]),
                    ("Thm 1.1 v₃ Lower Bound", &tool.breuer_bars[1]),
                    ("Thm 1.1 v₅ Lower Bound", &tool.breuer_bars[2]),
                    ("Thm 1.2 v₇ Lower Bound", &tool.breuer_bars[3]),
                ] {
                    plot_ui.line(Line::new(label, series.clone()));
                }
            });
        }
        ModularPlotMode::ModularQSeriesBars => {
            let bars: Vec<Bar> = tool
                .q_coeffs
                .iter()
                .enumerate()
                .map(|(n, &c)| Bar::new(n as f64, c as f64).width(0.75))
                .collect();
            plot.show(ui, |plot_ui| {
                plot_ui.bar_chart(BarChart::new(tool.state.q_series_kind.label(), bars));
            });
        }
    }
}

pub(crate) fn build_export_datasets(tool: &ModularInvariantsTool) -> Vec<(String, Vec<[f64; 2]>)> {
    match tool.state.view_mode {
        ModularPlotMode::HasseWeilTraces => vec![
            ("+2√p Hasse Bound".into(), tool.hasse_upper.clone()),
            ("-2√p Hasse Bound".into(), tool.hasse_lower.clone()),
            ("Frobenius Trace a_p".into(), tool.trace_pts.clone()),
            (
                "Class Number h(a_p² - 4p)".into(),
                tool.class_num_pts.clone(),
            ),
        ],
        ModularPlotMode::HurwitzKroneckerSums => vec![
            ("(p - 2) / 3 Law".into(), tool.hurwitz_expected.clone()),
            ("Hurwitz Trace Sum".into(), tool.hurwitz_actual.clone()),
        ],
        ModularPlotMode::BreuerValuationBounds => vec![
            ("Thm 1.1 v2 Bound".into(), tool.breuer_bars[0].clone()),
            ("Thm 1.1 v3 Bound".into(), tool.breuer_bars[1].clone()),
            ("Thm 1.1 v5 Bound".into(), tool.breuer_bars[2].clone()),
            ("Thm 1.2 v7 Bound".into(), tool.breuer_bars[3].clone()),
        ],
        ModularPlotMode::ModularQSeriesBars => vec![(
            tool.state.q_series_kind.label().into(),
            tool.q_coeffs
                .iter()
                .enumerate()
                .map(|(n, &c)| [n as f64, c as f64])
                .collect(),
        )],
    }
}
