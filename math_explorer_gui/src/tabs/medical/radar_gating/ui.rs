//! Side-panel controls and synchronized range/respiratory `egui_plot` views for `RadarGatingTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::canvas::show_room_canvas;
use super::pipeline::RadarPipeline;
use super::state::{
    RadarGatingState, MAX_SAMPLES_PER_CHIRP, MAX_SNAPSHOTS, MAX_SWEEP_BINS, MIN_SAMPLES_PER_CHIRP,
    MIN_SNAPSHOTS, MIN_SWEEP_BINS,
};
use super::RadarGatingTool;
use crate::app::theme::SemanticPalette;
use crate::framework::{bounded_drag_value, PaneSidePanel, PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Color32, RichText};
use egui_plot::{HLine, Legend, Line, Plot, PlotPoints, VLine};
use math_explorer::physics::medical::radar_gating::BeamState;

/// Renders the complete `RadarGatingTool` side panel and central visualization workspace.
pub fn show_tool_inside(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    let frame_start = crate::web_io::PlatformInstant::now();
    if tool.playing {
        let _ = tool.step_gating();
        ui.ctx().request_repaint();
    }

    PaneSidePanel::left("radar_gating_controls")
        .default_width(305.0)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                render_side_controls(tool, ui);
            });
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        render_central_workspace(tool, ui);
    });
    tool.last_frame_ms = frame_start.elapsed().as_secs_f64() * 1.0e3;
}

fn render_side_controls(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    ui.heading("60 GHz MIMO Radar Gating");
    render_transport_and_status(tool, ui);
    ui.separator();
    let rf_changed = render_rf_and_music_controls(&mut tool.state, ui);
    ui.separator();
    let gating_changed = render_respiratory_and_gating_controls(&mut tool.state, ui);
    ui.separator();
    render_fault_and_interlock_controls(tool, ui);

    if gating_changed {
        tool.pipeline.rebuild_all(&tool.state);
    } else if rf_changed {
        tool.pipeline.refresh_spectra_only(&tool.state);
    }
}

fn render_transport_and_status(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    match PlaybackTransport::in_thread(
        "radar_gating_transport",
        "Radar Respiratory Gating",
        tool.playing,
    )
    .show(ui)
    {
        Some(PlaybackAction::Play) => tool.playing = true,
        Some(PlaybackAction::Pause) => tool.playing = false,
        Some(PlaybackAction::Step) => {
            let _ = tool.step_gating();
        }
        Some(PlaybackAction::Reset) => {
            tool.reset_simulation();
        }
        Some(PlaybackAction::StepBack | PlaybackAction::ScrubTo(_)) | None => {}
    }

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let beam_state = tool.pipeline.gating.state();
    let (badge_text, badge_color) = beam_state_badge(beam_state, &palette);
    ui.horizontal_wrapped(|ui| {
        ui.label("LINAC State:");
        ui.label(RichText::new(badge_text).strong().color(badge_color));
    });
    ui.label(format!(
        "Duty Cycle: {:.1}% | Step: {:.1} us | Frame: {:.2} ms",
        tool.pipeline.duty_cycle_percent(),
        tool.pipeline.last_step_us,
        tool.last_frame_ms
    ));
    ui.label(format!(
        "Peaks — Coarse FFT: {} | MUSIC: {} (c/2B = {:.2} cm)",
        tool.pipeline.spectra.coarse_peak_count,
        tool.pipeline.spectra.music_peak_count,
        tool.pipeline.spectra.rayleigh_limit_m * 100.0
    ));
    ui.label(format!(
        "ULA Gain — Chest: {:.0}% | Secondary: {:.0}%",
        tool.pipeline.spectra.primary_beam_gain * 100.0,
        tool.pipeline.spectra.secondary_beam_gain * 100.0
    ));
}

fn render_rf_and_music_controls(state: &mut RadarGatingState, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.label(RichText::new("RF & MIMO Super-Resolution").strong());
    changed |= drag_f64_row(
        ui,
        "Bandwidth (GHz):",
        &mut state.bandwidth_ghz,
        0.5..=8.0,
        0.1,
    );
    changed |= drag_f64_row(
        ui,
        "Steer Angle (deg):",
        &mut state.steering_deg,
        -60.0..=60.0,
        1.0,
    );
    changed |= drag_usize_row(ui, "ULA Antennas:", &mut state.num_antennas, 2, 16, 1.0);
    changed |= drag_usize_row(
        ui,
        "Samples / Chirp:",
        &mut state.samples_per_chirp,
        MIN_SAMPLES_PER_CHIRP,
        MAX_SAMPLES_PER_CHIRP,
        2.0,
    );
    changed |= drag_usize_row(
        ui,
        "MUSIC Snapshots:",
        &mut state.snapshots,
        MIN_SNAPSHOTS,
        MAX_SNAPSHOTS,
        1.0,
    );
    changed |= drag_usize_row(
        ui,
        "Range Sweep Bins:",
        &mut state.sweep_bins,
        MIN_SWEEP_BINS,
        MAX_SWEEP_BINS,
        8.0,
    );
    changed |= drag_f64_row(
        ui,
        "Thermal IQ Noise:",
        &mut state.noise_sigma,
        0.0..=0.50,
        0.005,
    );
    changed
}

fn drag_f64_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
    speed: f64,
) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(bounded_drag_value(value, range, speed)).changed()
    })
    .inner
}

fn drag_usize_row(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut usize,
    min: usize,
    max: usize,
    speed: f64,
) -> bool {
    let mut val_f = *value as f64;
    let edited = ui
        .horizontal(|ui| {
            ui.label(label);
            ui.add(bounded_drag_value(
                &mut val_f,
                min as f64..=max as f64,
                speed,
            ))
            .changed()
        })
        .inner;
    if edited {
        *value = (val_f.round() as usize).clamp(min, max);
    }
    edited
}

fn render_respiratory_and_gating_controls(state: &mut RadarGatingState, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    ui.label(RichText::new("Respiratory & Kalman Gating").strong());
    changed |= drag_f64_row(
        ui,
        "Breath Amp (mm):",
        &mut state.breath_amplitude_mm,
        1.0..=15.0,
        0.25,
    );
    changed |= drag_f64_row(
        ui,
        "Breath Period (s):",
        &mut state.breath_period_s,
        1.5..=12.0,
        0.2,
    );
    changed |= drag_f64_row(
        ui,
        "Gate Threshold (mm):",
        &mut state.gating_threshold_mm,
        -12.0..=12.0,
        0.2,
    );
    changed |= drag_f64_row(
        ui,
        "Hysteresis (mm):",
        &mut state.gating_hysteresis_mm,
        0.1..=5.0,
        0.1,
    );
    changed |= drag_f64_row(
        ui,
        "LINAC Latency (ms):",
        &mut state.system_latency_ms,
        0.0..=250.0,
        5.0,
    );
    changed |= drag_f64_row(
        ui,
        "Static Clutter:",
        &mut state.static_clutter_gain,
        0.0..=3.0,
        0.05,
    );
    changed
}

fn render_fault_and_interlock_controls(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    ui.label(RichText::new("Clinical Faults & Interlock").strong());
    ui.horizontal_wrapped(|ui| {
        if ui.button("Inject Cough Spike").clicked() {
            tool.inject_cough();
            let _ = tool.step_gating();
        }
        let clutter_label = if tool.pipeline.extra_clutter_active {
            "Clear Couch Reflector"
        } else {
            "Inject Static Reflector"
        };
        if ui.button(clutter_label).clicked() {
            tool.pipeline.extra_clutter_active = !tool.pipeline.extra_clutter_active;
            let _ = tool.step_gating();
        }
        if ui.button("Inject NaN Fault").clicked() {
            tool.inject_nan_fault();
            let _ = tool.step_gating();
        }
    });
    ui.horizontal_wrapped(|ui| {
        if ui.button("Reset Interlock").clicked() {
            let _ = tool.reset_interlock();
        }
        if ui.button("Arm LINAC").clicked() {
            let _ = tool.arm_gating();
        }
        if ui.button("Disarm").clicked() {
            let _ = tool.pipeline.disarm_controller();
        }
    });
}

fn render_central_workspace(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    let interaction = show_room_canvas(
        ui,
        &mut tool.state,
        &tool.pipeline.spectra,
        &mut tool.selected_target,
        &mut tool.dragging_target,
    );
    tool.focused_canvas = interaction.focused_id;
    if interaction.state_changed {
        tool.pipeline.refresh_spectra_only(&tool.state);
    }

    ui.add_space(4.0);
    let remaining_h = ui.available_height().max(260.0);
    let plot_h = (remaining_h * 0.48).max(125.0);
    render_spectra_plot(ui, &tool.state, &tool.pipeline, plot_h);
    ui.add_space(4.0);
    render_gating_timeline_plot(
        ui,
        &tool.state,
        &tool.pipeline,
        ui.available_height().max(125.0),
    );
}

fn render_spectra_plot(
    ui: &mut egui::Ui,
    state: &RadarGatingState,
    pipeline: &RadarPipeline,
    height: f32,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let r1 = state.primary_target.range_m();
    let r2 = state.secondary_target.range_m();
    let center = 0.5 * (r1 + r2);
    let half_rayleigh = 0.5 * pipeline.spectra.rayleigh_limit_m;

    Plot::new("radar_range_super_resolution_plot")
        .height(height)
        .legend(Legend::default())
        .x_axis_label("Slant Range R (m)")
        .y_axis_label("Normalized Power / Pseudospectrum")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(
                    "Coarse FFT",
                    PlotPoints::from(pipeline.spectra.coarse_fft.clone()),
                )
                .color(palette.muted_text)
                .width(1.8_f32),
            );
            plot_ui.line(
                Line::new(
                    "Spatial CZT Zoom",
                    PlotPoints::from(pipeline.spectra.czt_zoom.clone()),
                )
                .color(palette.info)
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    "MUSIC Pseudospectrum",
                    PlotPoints::from(pipeline.spectra.music_pseudospectrum.clone()),
                )
                .color(palette.success)
                .width(2.4_f32),
            );
            plot_ui
                .vline(VLine::new("c / 2B Lower", center - half_rayleigh).color(palette.warning));
            plot_ui
                .vline(VLine::new("c / 2B Upper", center + half_rayleigh).color(palette.warning));
        });
}

fn render_gating_timeline_plot(
    ui: &mut egui::Ui,
    state: &RadarGatingState,
    pipeline: &RadarPipeline,
    height: f32,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    let thresh = state.gating_threshold_mm;
    let hyst = state.gating_hysteresis_mm;
    let active_scaled = scale_gate_strip(&pipeline.beam_active_trace, 4.0, -9.0);
    let hold_scaled = scale_gate_strip(&pipeline.beam_hold_trace, 2.5, 7.0);
    let interlock_scaled = scale_gate_strip(&pipeline.interlock_trace, 4.0, 10.0);

    Plot::new("radar_kalman_gating_timeline_plot")
        .height(height)
        .legend(Legend::default())
        .x_axis_label("Simulation Time (s)")
        .y_axis_label("Displacement (mm) & Gate Windows")
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(
                    "Unwrapped + Surface Chest (mm)",
                    PlotPoints::from(pipeline.measured_trace_mm.clone()),
                )
                .color(palette.series_color(0))
                .width(1.6_f32),
            );
            plot_ui.line(
                Line::new(
                    "Kalman Lookahead (mm)",
                    PlotPoints::from(pipeline.kalman_trace_mm.clone()),
                )
                .color(palette.series_color(4))
                .width(2.2_f32),
            );
            plot_ui.line(
                Line::new("BeamActive Window", PlotPoints::from(active_scaled))
                    .color(palette.success)
                    .width(2.0_f32),
            );
            plot_ui.line(
                Line::new("BeamHold Window", PlotPoints::from(hold_scaled))
                    .color(palette.warning)
                    .width(1.6_f32),
            );
            plot_ui.line(
                Line::new("SafetyInterlock", PlotPoints::from(interlock_scaled))
                    .color(palette.error)
                    .width(2.2_f32),
            );
            plot_ui.hline(HLine::new("Gate ON (T - H)", thresh - hyst).color(palette.success));
            plot_ui.hline(HLine::new("Gate OFF (T + H)", thresh + hyst).color(palette.warning));
        });
}

fn scale_gate_strip(trace: &[[f64; 2]], scale: f64, offset: f64) -> Vec<[f64; 2]> {
    trace
        .iter()
        .map(|&[t, v]| [t, v * scale + offset])
        .collect()
}

fn beam_state_badge(state: BeamState, palette: &SemanticPalette) -> (&'static str, Color32) {
    match state {
        BeamState::Standby => ("STANDBY", palette.muted_text),
        BeamState::Armed => ("ARMED", palette.info),
        BeamState::BeamActive => ("BEAM ACTIVE", palette.success),
        BeamState::BeamHold => ("BEAM HOLD", palette.warning),
        BeamState::SafetyInterlock => ("SAFETY INTERLOCK", palette.error),
    }
}

// [cite:mmwave_radiotherapy_setup]
