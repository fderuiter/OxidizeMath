//! Side-panel controls and synchronized range/respiratory `egui_plot` views for `RadarGatingTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::canvas::{render_gating_timeline_plot, render_spectra_plot, show_room_canvas};
use super::state::{
    RadarGatingState, MAX_SAMPLES_PER_CHIRP, MAX_SNAPSHOTS, MAX_SWEEP_BINS, MIN_SAMPLES_PER_CHIRP,
    MIN_SNAPSHOTS, MIN_SWEEP_BINS,
};
use super::RadarGatingTool;
use crate::app::theme::SemanticPalette;
use crate::framework::{bounded_drag_value, PaneSidePanel, PlaybackAction, PlaybackTransport};
use eframe::egui::{self, Color32, RichText};
use math_explorer::physics::medical::radar_gating::BeamState;

/// Renders the complete `RadarGatingTool` side panel and central visualization workspace.
pub fn show_tool_inside(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    let frame_start = crate::web_io::PlatformInstant::now();
    if tool.playing {
        handle_step_outcome(tool.step_gating());
        ui.ctx().request_repaint();
    }

    let panel_id = crate::framework::scoped_panel_id(ui, "radar_gating_controls");
    PaneSidePanel::left(panel_id)
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

fn handle_step_outcome(state: BeamState) {
    if state == BeamState::SafetyInterlock {
        crate::audio::notify_solver_diverged();
    }
}

fn render_side_controls(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    let locked = tool.is_challenge_locked();
    ui.heading("60 GHz MIMO Radar Gating");
    render_transport_and_status(tool, ui);
    ui.separator();
    let rf_changed = ui
        .add_enabled_ui(!locked, |ui| {
            render_rf_and_music_controls(&mut tool.state, ui)
        })
        .inner;
    ui.separator();
    let gating_changed = render_respiratory_and_gating_controls(&mut tool.state, locked, ui);
    ui.separator();
    render_fault_and_interlock_controls(tool, ui);

    if gating_changed {
        if locked {
            RadarGatingTool::enforce_challenge_locks_on_state(&mut tool.state);
        }
        tool.pipeline.rebuild_all(&tool.state);
        let _ = tool.publish_status_snapshot(false);
    } else if rf_changed {
        tool.pipeline.refresh_spectra_only(&tool.state);
    }

    let is_interlocked = tool.pipeline.gating.state() == BeamState::SafetyInterlock;
    if is_interlocked {
        if !tool.interlock_notified {
            tool.interlock_notified = true;
            crate::audio::notify_solver_diverged();
        }
    } else {
        tool.interlock_notified = false;
    }
    ui.separator();
    let frame = tool.sonification_frame();
    crate::audio::render_sonification_panel(ui, "radar_gating", &frame, is_interlocked);
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
        Some(PlaybackAction::Pause) => {
            tool.playing = false;
            crate::audio::notify_playback_paused();
        }
        Some(PlaybackAction::Step) => {
            handle_step_outcome(tool.step_gating());
        }
        Some(PlaybackAction::Reset) => {
            tool.reset_simulation();
            crate::audio::notify_playback_reset();
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
        "Duty: {:.1}% (Phys {:.1}%) | Misses: {} | Step: {:.1} us",
        tool.pipeline.duty_cycle_percent(),
        tool.pipeline.physical_duty_cycle_percent(),
        tool.pipeline.geographic_miss_frames,
        tool.pipeline.last_step_us,
    ));
    ui.label(format!(
        "Tracking — r: {:.3} | RMSE: {:.2} mm | Err: {:+.3}%",
        tool.pipeline.eval_metrics[0], tool.pipeline.eval_metrics[1], tool.pipeline.eval_metrics[2]
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
    for (label, value, range, speed) in [
        ("Bandwidth (GHz):", &mut state.bandwidth_ghz, 0.5..=8.0, 0.1),
        (
            "Steer Angle (deg):",
            &mut state.steering_deg,
            -60.0..=60.0,
            1.0,
        ),
    ] {
        changed |= drag_f64_row(ui, label, value, range, speed);
    }
    for (label, value, min, max, speed) in [
        ("ULA Antennas:", &mut state.num_antennas, 2, 16, 1.0),
        (
            "Samples / Chirp:",
            &mut state.samples_per_chirp,
            MIN_SAMPLES_PER_CHIRP,
            MAX_SAMPLES_PER_CHIRP,
            2.0,
        ),
        (
            "MUSIC Snapshots:",
            &mut state.snapshots,
            MIN_SNAPSHOTS,
            MAX_SNAPSHOTS,
            1.0,
        ),
        (
            "Range Sweep Bins:",
            &mut state.sweep_bins,
            MIN_SWEEP_BINS,
            MAX_SWEEP_BINS,
            8.0,
        ),
    ] {
        changed |= drag_usize_row(ui, label, value, min, max, speed);
    }
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

fn render_locked_respiratory_rows(state: &mut RadarGatingState, ui: &mut egui::Ui) -> bool {
    let mut changed = false;
    for (label, value, range, speed) in [
        (
            "Breath Amp (mm):",
            &mut state.breath_amplitude_mm,
            1.0..=15.0,
            0.25,
        ),
        (
            "Breath Period (s):",
            &mut state.breath_period_s,
            1.5..=12.0,
            0.2,
        ),
        (
            "LINAC Latency (ms):",
            &mut state.system_latency_ms,
            0.0..=250.0,
            5.0,
        ),
        (
            "Static Clutter:",
            &mut state.static_clutter_gain,
            0.0..=3.0,
            0.05,
        ),
    ] {
        changed |= drag_f64_row(ui, label, value, range, speed);
    }
    changed
}

fn render_respiratory_and_gating_controls(
    state: &mut RadarGatingState,
    locked: bool,
    ui: &mut egui::Ui,
) -> bool {
    let mut changed = false;
    ui.label(RichText::new("Respiratory & Kalman Gating").strong());
    ui.add_enabled_ui(!locked, |ui| {
        changed |= render_locked_respiratory_rows(state, ui);
    });
    for (label, value, range, speed) in [
        (
            "Kalman Q (proc):",
            &mut state.kalman_process_var,
            1.0e-8..=1.0,
            1.0e-4,
        ),
        (
            "Kalman R (meas):",
            &mut state.kalman_meas_var,
            1.0e-9..=1.0,
            1.0e-5,
        ),
        (
            "Gate Threshold (mm):",
            &mut state.gating_threshold_mm,
            -12.0..=12.0,
            0.2,
        ),
        (
            "Hysteresis (mm):",
            &mut state.gating_hysteresis_mm,
            0.1..=5.0,
            0.05,
        ),
    ] {
        changed |= drag_f64_row(ui, label, value, range, speed);
    }
    changed |= drag_usize_row(ui, "Clutter Window:", &mut state.clutter_window, 4, 64, 1.0);
    changed
}

fn render_fault_and_interlock_controls(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    ui.label(RichText::new("Clinical Faults & Interlock").strong());
    ui.horizontal_wrapped(|ui| {
        if ui.button("Inject Cough Spike").clicked() {
            tool.inject_cough();
            handle_step_outcome(tool.step_gating());
        }
        let clutter_label = if tool.pipeline.extra_clutter_active {
            "Clear Couch Reflector"
        } else {
            "Inject Static Reflector"
        };
        if ui
            .add_enabled(
                !tool.is_challenge_locked(),
                egui::Button::new(clutter_label),
            )
            .clicked()
        {
            tool.pipeline.extra_clutter_active = !tool.pipeline.extra_clutter_active;
            handle_step_outcome(tool.step_gating());
        }
        if ui.button("Inject NaN Fault").clicked() {
            tool.inject_nan_fault();
            handle_step_outcome(tool.step_gating());
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
            let _ = tool.publish_status_snapshot(false);
        }
    });
}

fn render_central_workspace(tool: &mut RadarGatingTool, ui: &mut egui::Ui) {
    let locked = tool.is_challenge_locked();
    let interaction = ui
        .add_enabled_ui(!locked, |ui| {
            show_room_canvas(
                ui,
                &mut tool.state,
                &tool.pipeline.spectra,
                &mut tool.selected_target,
                &mut tool.dragging_target,
            )
        })
        .inner;
    tool.focused_canvas = interaction.focused_id;
    if !locked && interaction.state_changed {
        tool.pipeline.refresh_spectra_only(&tool.state);
    }

    tool.sync_radar_callouts();
    ui.add_space(4.0);
    let remaining_h = ui.available_height().max(260.0);
    let plot_h = (remaining_h * 0.48).max(125.0);
    render_spectra_plot(ui, &tool.state, &tool.pipeline, plot_h);
    ui.add_space(4.0);
    render_gating_timeline_plot(
        ui,
        &tool.state,
        &tool.pipeline,
        &mut tool.callout_manager,
        ui.available_height().max(125.0),
    );
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
