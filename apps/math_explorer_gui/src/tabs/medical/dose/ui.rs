//! Side-panel UI for `DoseCalculationTool`: 4-beam IMRT controls, 4D respiratory playback,
//! `SpatialBrushKit` contour sculpting toolbar, clinical verification readouts, and sonification.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{CanvasKeyTarget, DoseCalculationTool, DoseDisplayMode};
use crate::accessibility::{announce_live_status, AccessibleHoverText};
use crate::app::{SemanticPalette, ThemeMode};
use crate::audio::mappings::{imrt_delivery_frame, radar_linac_frame, LinacGatePhase};
use crate::framework::{
    bounded_drag_value, scoped_panel_id, BrushInteractionMode, BrushMode, InteractiveTool,
    PaneSidePanel, PlaybackAction, PlaybackTransport,
};
use crate::tabs::medical::imrt::IMRT_BEAM_COUNT;
use eframe::egui::{self, Color32, RichText};
use math_explorer::physics::medical::radar_gating::BeamState;

/// Renders the `DoseCalculationTool` side panel and central `egui_plot` canvas inside `parent_ui`.
pub fn show_tool_inside(tool: &mut DoseCalculationTool, parent_ui: &mut egui::Ui) {
    let ctx = parent_ui.ctx().clone();
    let is_high_contrast = ThemeMode::from_visuals(parent_ui.visuals()) == ThemeMode::HighContrast;
    let mut changed = false;
    if tool.high_contrast != is_high_contrast {
        tool.high_contrast = is_high_contrast;
        changed = true;
    }

    tick_4d_delivery(tool, &ctx);

    let panel_id = scoped_panel_id(parent_ui, "dose_controls");
    PaneSidePanel::left(panel_id)
        .default_width(310.0)
        .show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                changed |= render_side_panel_contents(tool, ui);
            });
        });

    if changed {
        tool.refresh_model();
    }

    egui::CentralPanel::default().show_inside(parent_ui, |ui| {
        super::canvas::render_dose_canvas(tool, ui, &ctx);
    });
}

fn tick_4d_delivery(tool: &mut DoseCalculationTool, ctx: &egui::Context) {
    if !tool.imrt_mode || !tool.gating_4d_enabled {
        return;
    }
    let bus_steps = tool.poll_link_bus_in_place();
    if tool.playing_4d {
        if bus_steps == 0 {
            tool.linked_to_radar_bus = false;
            let snap = tool.standalone_4d.step_zero_alloc();
            tool.ingest_telemetry_in_place(snap);
        }
        ctx.request_repaint();
    } else if bus_steps > 0 {
        ctx.request_repaint();
    }
}

fn render_side_panel_contents(tool: &mut DoseCalculationTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.heading("Dose & 4D IMRT Workbench");
    ui.add_enabled_ui(tool.active_challenge.is_none(), |ui| {
        if ui
            .checkbox(&mut tool.imrt_mode, "4-Beam Coplanar IMRT Mode")
            .changed()
        {
            changed = true;
        }
        if tool.imrt_mode
            && ui
                .checkbox(&mut tool.gating_4d_enabled, "4D Respiratory-Gated Delivery")
                .changed()
        {
            tool.display_mode = if tool.gating_4d_enabled {
                DoseDisplayMode::Gated4d
            } else {
                DoseDisplayMode::StaticPlan
            };
            tool.texture = None;
        }
    });

    ui.separator();
    render_medical_sonification(tool, ui);
    ui.separator();

    if tool.imrt_mode {
        if tool.gating_4d_enabled {
            changed |= render_4d_gating_panel(tool, ui);
            ui.separator();
        }
        if tool.active_challenge.is_none() {
            changed |= render_contour_brush_toolbar(tool, ui);
            ui.separator();
        }
        changed |= render_imrt_plan_controls(tool, ui);
    } else {
        changed |= render_point_source_controls(tool, ui);
    }
    changed
}

fn render_medical_sonification(tool: &DoseCalculationTool, ui: &mut egui::Ui) {
    let m = &tool.imrt_workspace.motion;
    let s = &tool.imrt_workspace.summary;
    let (gate, ptv_gy, oar_gy, disp_mm, is_paused) = if !tool.imrt_mode {
        (LinacGatePhase::Standby, 0.0, 0.0, 0.0, true)
    } else if tool.gating_4d_enabled {
        let bs = tool.latest_telemetry.beam_state;
        let (ptv, oar) = match tool.display_mode {
            DoseDisplayMode::Ungated4d if m.ungated_steps > 0 => {
                (m.ungated_ptv_mean_gy, m.ungated_oar_max_gy.0)
            }
            DoseDisplayMode::Gated4d if m.gated_steps > 0 => {
                (m.gated_ptv_mean_gy, m.gated_oar_max_gy.0)
            }
            _ => (s.tumor_mean_gy, s.oar_max_gy.0),
        };
        let paused = matches!(bs, BeamState::Standby | BeamState::SafetyInterlock);
        let disp = tool.latest_telemetry.true_disp_mm;
        (LinacGatePhase::from(bs), ptv, oar, disp, paused)
    } else {
        (
            LinacGatePhase::Armed,
            s.tumor_mean_gy,
            s.oar_max_gy.0,
            0.0,
            false,
        )
    };

    let mut frame = imrt_delivery_frame(
        gate,
        disp_mm,
        ptv_gy,
        tool.prescription_gy,
        oar_gy,
        tool.oar_limit_gy,
        tool.imrt_workspace.motion.beam_energy_mev,
    );
    if tool.imrt_mode && tool.gating_4d_enabled {
        let t = tool.latest_telemetry;
        let radar = radar_linac_frame(
            gate,
            t.measured_disp_mm,
            t.velocity_mm_s,
            t.duty_cycle_pct,
            t.sim_time_s,
        );
        if radar.is_audible() {
            frame.voices[4] = radar.voices[0];
            frame.active_voices = 5;
        }
    }
    crate::audio::render_sonification_panel(ui, "dose_calculation", &frame, is_paused);
}

fn render_4d_gating_panel(tool: &mut DoseCalculationTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.label(RichText::new("4D Respiratory Delivery & LINAC").strong());

    let canvas_focused = tool
        .focused_canvas_id()
        .is_some_and(|id| ui.ctx().memory(|m| m.has_focus(id)));
    match PlaybackTransport::in_thread("dose_4d_transport", "4D IMRT Delivery", tool.playing_4d)
        .allow_space_shortcut(!canvas_focused)
        .show(ui)
    {
        Some(PlaybackAction::Play) => tool.playing_4d = true,
        Some(PlaybackAction::Pause) => {
            tool.playing_4d = false;
            crate::audio::notify_playback_paused();
        }
        Some(PlaybackAction::Step) => {
            let _ = tool.step_4d_in_place();
            tool.texture = None;
        }
        Some(PlaybackAction::Reset) => {
            tool.reset_4d_simulation();
            crate::audio::notify_playback_reset();
        }
        Some(PlaybackAction::StepBack | PlaybackAction::ScrubTo(_)) | None => {}
    }

    render_4d_mode_and_fault_buttons(tool, ui);

    ui.horizontal(|ui| {
        ui.label("Gun Current I_b (mA):");
        if ui
            .add(bounded_drag_value(
                &mut tool.beam_current_ma,
                0.0..=300.0,
                2.0,
            ))
            .changed()
        {
            changed = true;
        }
    });

    render_4d_clinical_metrics(tool, ui);
    changed
}

fn render_4d_mode_and_fault_buttons(tool: &mut DoseCalculationTool, ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Heatmap:");
        for mode in [
            DoseDisplayMode::StaticPlan,
            DoseDisplayMode::Gated4d,
            DoseDisplayMode::Ungated4d,
        ] {
            if ui
                .selectable_label(tool.display_mode == mode, mode.label())
                .clicked()
            {
                tool.display_mode = mode;
                tool.texture = None;
            }
        }
    });

    ui.horizontal_wrapped(|ui| {
        if ui
            .button("OAR-Sparing Plan")
            .accessible_hover_text("Load 4-beam coplanar OAR-sparing gantry angles and weights")
            .clicked()
        {
            tool.apply_oar_sparing_4d_preset();
        }
        if ui
            .button("Inject +10 mm Cough")
            .accessible_hover_text(
                "Inject a +10 mm thoracic cough transient into single-pane 4D breathing",
            )
            .clicked()
        {
            tool.standalone_4d.inject_cough();
            let _ = tool.step_4d_in_place();
            tool.texture = None;
        }
        if ui
            .button("Inject NaN Fault")
            .accessible_hover_text("Latch SafetyInterlock and suspend audio within 30 ms")
            .clicked()
        {
            tool.standalone_4d.inject_nan_fault();
            tool.ingest_telemetry_in_place(tool.standalone_4d.latest);
        }
        if ui.button("Reset Interlock").clicked() {
            let _ = tool.standalone_4d.reset_interlock(true);
            tool.latest_telemetry = tool.standalone_4d.latest;
        }
    });
}

fn render_4d_clinical_metrics(tool: &DoseCalculationTool, ui: &mut egui::Ui) {
    let p = SemanticPalette::from_visuals(ui.visuals());
    let m = &tool.imrt_workspace.motion;
    let src = if tool.linked_to_radar_bus {
        "Radar Link Bus"
    } else {
        "Embedded 4D"
    };
    let (state_lbl, state_col) = gate_badge(tool.latest_telemetry.beam_state, &p);
    ui.horizontal_wrapped(|ui| {
        ui.label(format!("{src} | Gate:"));
        ui.label(RichText::new(state_lbl).strong().color(state_col));
        ui.label(format!(
            "({:.1}% duty)",
            tool.latest_telemetry.duty_cycle_pct
        ));
    });
    ui.label(format!(
        "Tracking — r: {:.3} | RMSE: {:.2} mm | Spatial Err: {:+.3}%",
        m.eval_pearson_r, m.eval_rmse_mm, m.eval_spatial_err_pct
    ));
    ui.label(format!(
        "LINAC E_load: {:.3} MeV ({:.1}% fluence) | Shift: {:+.2} cm",
        m.beam_energy_mev,
        m.fluence_scale * 100.0,
        m.latest_shift_cm
    ));
    ui.label(format!(
        "Dice (90% Rx vs PTV) — Static: {:.2} | Gated: {:.2} | Ungated: {:.2}",
        m.static_dice, m.gated_dice, m.ungated_dice
    ));

    let g_col = oar_status_color(m.gated_oar_max_gy.0, tool.oar_limit_gy, &p);
    ui.colored_label(
        g_col,
        format!(
            "Gated ({} steps) — PTV: {:.1} Gy | OAR max: {:.1}/{:.0} Gy",
            m.gated_steps, m.gated_ptv_mean_gy, m.gated_oar_max_gy.0, tool.oar_limit_gy
        ),
    );
    let u_col = oar_status_color(m.ungated_oar_max_gy.0, tool.oar_limit_gy, &p);
    ui.colored_label(
        u_col,
        format!(
            "Ungated ({} steps) — PTV: {:.1} Gy | OAR max: {:.1}/{:.0} Gy",
            m.ungated_steps, m.ungated_ptv_mean_gy, m.ungated_oar_max_gy.0, tool.oar_limit_gy
        ),
    );
}

fn render_contour_brush_toolbar(tool: &mut DoseCalculationTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.label(RichText::new("Contour Sculpting & Gantry Keys").strong());
    ui.horizontal_wrapped(|ui| {
        let sculpt = tool.brush_kit.interaction_mode == BrushInteractionMode::Sculpt;
        if ui.selectable_label(sculpt, "🖌 Sculpt").clicked() {
            tool.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;
        }
        if ui.selectable_label(!sculpt, "✋ Pan/Zoom").clicked() {
            tool.brush_kit.interaction_mode = BrushInteractionMode::Pan;
        }
        if ui.button("Reset Contours").clicked() {
            tool.imrt_workspace.reset_contours_to_canonical();
            changed = true;
        }
    });

    ui.horizontal_wrapped(|ui| {
        for mode in [
            BrushMode::PtvTarget,
            BrushMode::OrganAtRisk,
            BrushMode::Erase,
        ] {
            if ui
                .selectable_label(tool.brush_kit.mode == mode, mode.label())
                .clicked()
            {
                tool.brush_kit.mode = mode;
                tool.brush_kit.interaction_mode = BrushInteractionMode::Sculpt;
            }
        }
    });

    ui.add(
        egui::Slider::new(
            &mut tool.brush_kit.radius,
            tool.brush_kit.min_radius..=tool.brush_kit.max_radius,
        )
        .text("Brush Radius (cm)"),
    );

    ui.horizontal_wrapped(|ui| {
        ui.label("Canvas Arrows:");
        if ui
            .selectable_label(
                tool.canvas_key_target == CanvasKeyTarget::Brush,
                "Brush (B)",
            )
            .clicked()
        {
            tool.canvas_key_target = CanvasKeyTarget::Brush;
        }
        if ui
            .selectable_label(
                tool.canvas_key_target == CanvasKeyTarget::Gantry,
                "Gantry (G / 1-4)",
            )
            .clicked()
        {
            tool.canvas_key_target = CanvasKeyTarget::Gantry;
        }
    });
    changed
}

fn render_imrt_plan_controls(tool: &mut DoseCalculationTool, ui: &mut egui::Ui) -> bool {
    let mut changed = render_imrt_beam_sliders(tool, ui);
    ui.add_enabled_ui(tool.active_challenge.is_none(), |ui| {
        changed |= ui
            .add(egui::Slider::new(&mut tool.beta, 0.01..=1.0).text("Attenuation β (1/cm)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.prescription_gy, 20.0..=80.0).text("PTV D_Rx (Gy)"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut tool.oar_limit_gy, 5.0..=50.0).text("OAR D_lim (Gy)"))
            .changed();
    });

    let s = &tool.imrt_workspace.summary;
    let p = SemanticPalette::from_visuals(ui.visuals());
    let color = if s.passed { p.success } else { p.warning };
    ui.colored_label(
        color,
        format!(
            "Static Cost J = {:.1} (J_base = {:.1}) | PTV = {:.1} Gy (DSC {:.2}) | OAR max = {:.1}/{:.0} Gy",
            s.cost_j,
            s.baseline_cost_j,
            s.tumor_mean_gy,
            tool.imrt_workspace.motion.static_dice,
            s.oar_max_gy.0,
            tool.oar_limit_gy
        ),
    );
    changed
}

fn render_imrt_beam_sliders(tool: &mut DoseCalculationTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.group(|ui| {
        ui.strong("4-Beam Coplanar IMRT Plan");
        for b in 0..IMRT_BEAM_COUNT {
            let angle_lbl = format!("Beam {} Angle θ (deg)", b + 1);
            if ui
                .add(
                    egui::Slider::new(&mut tool.beam_angles_deg[b], 0.0..=360.0)
                        .step_by(1.0)
                        .text(angle_lbl),
                )
                .changed()
            {
                tool.selected_beam = b;
                tool.imrt_workspace.reset_4d_dose();
                changed = true;
            }
            let weight_lbl = format!("Beam {} Weight w", b + 1);
            if ui
                .add(
                    egui::Slider::new(&mut tool.beam_weights[b], 0.0..=6.0)
                        .step_by(0.05)
                        .text(weight_lbl),
                )
                .changed()
            {
                tool.imrt_workspace.reset_4d_dose();
                changed = true;
            }
        }
        let hint_tip = "Take one projected gradient descent step on the 4 beam weights via optimization::update_weights";
        if ui
            .button("✨ 1-Step Weight Hint (update_weights)")
            .accessible_hover_text(hint_tip)
            .clicked()
        {
            tool.apply_gradient_weight_hint();
            let s = &tool.imrt_workspace.summary;
            announce_live_status(
                ui.ctx(),
                format!(
                    "Applied 1 update_weights step: cost J = {:.1}, OAR max = {:.1} Gy",
                    s.cost_j, s.oar_max_gy.0
                ),
            );
        }
    });
    changed
}

fn render_point_source_controls(tool: &mut DoseCalculationTool, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut tool.amplitude, 0.1..=10.0).text("Source Amplitude"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tool.beta, 0.1..=5.0).text("Attenuation (Beta)"))
        .changed();
    ui.separator();
    ui.label("Kernel: Exponential K(r) = (A / r²) · exp(-βr)");
    changed
}

fn oar_status_color(oar_gy: f64, limit_gy: f64, p: &SemanticPalette) -> Color32 {
    if oar_gy <= limit_gy {
        p.success
    } else {
        p.error
    }
}

fn gate_badge(state: BeamState, palette: &SemanticPalette) -> (&'static str, Color32) {
    match state {
        BeamState::Standby => ("STANDBY", palette.muted_text),
        BeamState::Armed => ("ARMED", palette.info),
        BeamState::BeamActive => ("BEAM ACTIVE", palette.success),
        BeamState::BeamHold => ("BEAM HOLD", palette.warning),
        BeamState::SafetyInterlock => ("SAFETY INTERLOCK", palette.error),
    }
}

// [cite:graph_parameters_rust]
