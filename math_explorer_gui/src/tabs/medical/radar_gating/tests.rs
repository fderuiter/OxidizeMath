//! Unit, zero-allocation, accessibility, bounds, and export tests for `RadarGatingTool`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::pipeline::{FastCvKalman, SIM_DT};
use super::state::{RadarGatingState, MAX_SAMPLES_PER_CHIRP, MAX_SNAPSHOTS, MAX_SWEEP_BINS};
use super::{RadarGatingTool, STARTER_PRESET_JSON};
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::tabs::medical::MedicalTab;
use crate::tabs::ExplorerTab;
use crate::url_state::{MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES};
use eframe::egui::{self, Event, Key, Modifiers, Pos2, RawInput, Rect, vec2};
use math_explorer::physics::medical::radar_gating::{
    BeamState, ConstantVelocityModel, TrackingFilter,
};

#[test]
fn test_medical_tab_registers_radar_gating_and_starter_resolves_sub_rayleigh() {
    let mut tab = MedicalTab::default();
    let fw = tab.framework().expect("MedicalTab uses SimulationFramework");
    let ids: Vec<&str> = fw.available_tools.iter().map(|m| m.id).collect();
    assert!(ids.contains(&"BeamProfilingTool"));
    assert!(ids.contains(&"DoseCalculationTool"));
    assert!(ids.contains(&"RadarGatingTool"));
    assert!(tab.select_tool_by_name("RadarGatingTool"));

    let mut tool = RadarGatingTool::default();
    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset must load cleanly");

    let sep_m = tool.state.target_separation_m();
    let rayleigh_m = tool.pipeline.spectra.rayleigh_limit_m;
    assert!(
        sep_m < 0.0375 && sep_m < rayleigh_m,
        "Starter preset separation {sep_m} m must beat 3.75 cm Rayleigh limit {rayleigh_m} m"
    );
    assert_eq!(
        tool.pipeline.spectra.coarse_peak_count, 1,
        "Coarse FFT must merge sub-Rayleigh targets into 1 peak"
    );
    assert_eq!(
        tool.pipeline.spectra.music_peak_count, 2,
        "MUSICEstimator must split sub-Rayleigh targets into 2 distinct peaks"
    );
}

#[test]
fn test_dragging_scatterers_across_rayleigh_limit_and_steering_array() {
    let mut tool = RadarGatingTool::default();
    tool.state.primary_target.x_m = 0.0;
    tool.state.primary_target.y_m = 0.650;
    tool.state.secondary_target.x_m = 0.0;
    tool.state.secondary_target.y_m = 0.760;
    tool.state.steering_deg = 0.0;
    tool.pipeline.refresh_spectra_only(&tool.state);

    assert!(tool.state.target_separation_m() > tool.pipeline.spectra.rayleigh_limit_m * 2.0);
    assert!(
        tool.pipeline.spectra.coarse_peak_count >= 2,
        "Well-separated targets must produce >= 2 coarse FFT peaks"
    );
    assert_eq!(tool.pipeline.spectra.music_peak_count, 2);

    let broadside_gain = tool.pipeline.spectra.primary_beam_gain;
    let broadside_res = tool.pipeline.spectra.angle_resolution_deg;

    // Drag second target within 1.4 cm (< c / 2B = 3.75 cm)
    tool.state.secondary_target.y_m = 0.664;
    tool.pipeline.refresh_spectra_only(&tool.state);
    assert!(tool.state.target_separation_m() < tool.pipeline.spectra.rayleigh_limit_m);
    assert_eq!(tool.pipeline.spectra.coarse_peak_count, 1);
    assert_eq!(tool.pipeline.spectra.music_peak_count, 2);

    // Steer ULA off-boresight and verify beamwidth broadening + broadside target attenuation
    tool.state.steering_deg = 32.0;
    tool.pipeline.refresh_spectra_only(&tool.state);
    assert!(tool.pipeline.spectra.angle_resolution_deg > broadside_res);
    assert!(tool.pipeline.spectra.primary_beam_gain < broadside_gain);
}

#[test]
fn test_cough_clutter_and_nan_faults_drive_beam_hold_interlock_and_reset() {
    let mut tool = RadarGatingTool::default();
    let mut reached_active = false;
    for _ in 0..180 {
        if tool.step_gating() == BeamState::BeamActive {
            reached_active = true;
            break;
        }
    }
    assert!(reached_active, "End-exhale phase must enter BeamActive");
    assert!(tool.pipeline.duty_cycle_percent() > 0.0);

    // Inject cough spike during active beam and verify protective BeamHold transition
    tool.inject_cough();
    let mut entered_hold = false;
    for _ in 0..16 {
        if tool.step_gating() == BeamState::BeamHold {
            entered_hold = true;
            break;
        }
    }
    assert!(entered_hold, "Cough excursion must move GatingLogic to BeamHold");

    // Toggle extra static couch reflector and verify finite unwrapped displacement
    tool.pipeline.extra_clutter_active = true;
    let _ = tool.step_gating();
    assert!(
        tool.pipeline
            .measured_trace_mm
            .iter()
            .all(|pt| pt[0].is_finite() && pt[1].is_finite())
    );

    // Inject NaN fault -> SafetyInterlock -> Reset to Standby -> Arm to Armed
    tool.inject_nan_fault();
    assert_eq!(tool.step_gating(), BeamState::SafetyInterlock);
    assert!(
        tool.export_datasets()
            .iter()
            .all(|(_, pts)| pts.iter().all(|p| p[0].is_finite() && p[1].is_finite())),
        "Exported plot points must stay finite during NaN fault simulation"
    );

    assert_eq!(tool.reset_interlock(), BeamState::Standby);
    assert_eq!(tool.arm_gating(), BeamState::Armed);
}

#[test]
fn test_kalman_parity_and_zero_heap_allocations_under_verified_allocator() {
    let q = 2.0e-4;
    let r = 2.0e-5;
    let mut tf = TrackingFilter::new(0.001, SIM_DT, ConstantVelocityModel::new(q, r));
    let mut fast = FastCvKalman::from_tracking_filter(&tf, SIM_DT, q, r);

    for step in 0..64 {
        let z = 0.005 * (step as f64 * 0.11).sin() + 0.0003 * (step as f64 * 0.7).cos();
        tf.predict();
        tf.update(z);
        fast.predict();
        fast.update(z);
        assert!((tf.amplitude() - fast.state[0]).abs() < 1.0e-13);
        assert!((tf.velocity() - fast.state[1]).abs() < 1.0e-13);
    }

    let mut tool = RadarGatingTool::default();
    let _ = tool.step_gating();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for idx in 0..48 {
                if idx == 12 {
                    tool.inject_cough();
                }
                if idx == 28 {
                    tool.pipeline.extra_clutter_active = true;
                }
                if idx == 38 {
                    tool.inject_nan_fault();
                }
                if idx == 42 {
                    let _ = tool.reset_interlock();
                    let _ = tool.arm_gating();
                }
                let _ = tool.step_gating();
            }
        });
    }
    assert_ne!(tool.pipeline.gating.state(), BeamState::SafetyInterlock);
}

#[test]
fn test_state_bounds_share_url_and_csv_svg_exports() {
    let bad = RadarGatingState {
        samples_per_chirp: MAX_SAMPLES_PER_CHIRP + 1,
        ..Default::default()
    };
    assert!(bad.validate_bounds().is_err());

    let bad_snaps = RadarGatingState {
        snapshots: MAX_SNAPSHOTS + 1,
        ..Default::default()
    };
    assert!(bad_snaps.validate_bounds().is_err());

    let bad_bins = RadarGatingState {
        sweep_bins: MAX_SWEEP_BINS + 1,
        ..Default::default()
    };
    assert!(bad_bins.validate_bounds().is_err());

    let mut tool = RadarGatingTool::default();
    assert!(tool.try_load_state(r#"{"unknown":1}"#).is_err());
    let oversized = " ".repeat(5000);
    assert!(tool.state_bounds().validate_json(&oversized).is_err());

    let mut fw = SimulationFramework::new("medical");
    assert!(fw.select_tool_by_name("RadarGatingTool"));
    fw.apply_starter_card(
        "Starter",
        Some("RadarGatingTool"),
        Some(STARTER_PRESET_JSON),
    )
    .expect("starter card applies cleanly");

    let preset_json = fw
        .export_url_preset_json()
        .expect("url preset ok")
        .expect("some preset");
    assert!(preset_json.len() <= MAX_DECODED_URL_STATE_BYTES);

    let url = fw.encode_share_url().expect("share url encodes");
    assert!(url.len() <= MAX_URL_FRAGMENT_BYTES + 64);

    let active = fw.active_tool.as_ref().expect("active tool");
    let datasets = active.export_datasets();
    let opts = active.export_figure_options().expect("figure options");
    assert!(datasets.iter().any(|(n, _)| n.contains("MUSIC")));
    assert!(datasets.iter().any(|(n, _)| n.contains("Kalman")));
    assert!(datasets.iter().any(|(n, _)| n.contains("Duty Cycle")));

    let csv = egui_plot::export::format_csv(&datasets);
    assert!(csv.contains("MUSIC Super-Resolution Pseudospectrum"));
    assert!(csv.contains("Kalman Lookahead Chest Trace (mm)"));
    assert!(csv.contains("LINAC Duty Cycle (%)"));

    let svg = egui_plot::export::export_custom_svg(&datasets, &opts).expect("svg export ok");
    assert!(svg.contains("<svg") && svg.contains("</svg>"));
}

#[test]
fn test_canvas_arrow_nudge_and_non_trapping_escape_and_tab() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = RadarGatingTool::default();
    let screen = Some(Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0)));

    let _ = ctx.run(
        RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );

    // Tab until the 2D room canvas holds keyboard focus
    for _ in 0..80 {
        let mut raw = RawInput {
            screen_rect: screen,
            ..Default::default()
        };
        raw.events.push(Event::Key {
            key: Key::Tab,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Modifiers::NONE,
        });
        let _ = ctx.run(raw, |ctx| tool.show(ctx));
        if tool.focused_canvas_id().is_some() {
            break;
        }
    }
    let canvas_id = tool.focused_canvas_id().expect("canvas must receive Tab focus");
    let y_before = tool.state.primary_target.y_m;

    let mut up_raw = RawInput {
        screen_rect: screen,
        ..Default::default()
    };
    up_raw.events.push(Event::Key {
        key: Key::ArrowUp,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    let _ = ctx.run(up_raw, |ctx| tool.show(ctx));
    assert!(tool.state.primary_target.y_m > y_before);

    // Pressing Tab again must advance focus away from the canvas (never trapped)
    let mut next_tab = RawInput {
        screen_rect: screen,
        ..Default::default()
    };
    next_tab.events.push(Event::Key {
        key: Key::Tab,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    let _ = ctx.run(next_tab, |ctx| tool.show(ctx));
    assert_ne!(tool.focused_canvas_id(), Some(canvas_id));
}

// [cite:mmwave_radiotherapy_setup]
