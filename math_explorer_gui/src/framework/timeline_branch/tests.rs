//! Unit and zero-allocation verification tests for [`super::TimelineRingBuffer`] and [`super::BranchDivergenceHud`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{PlaybackTransport, TimelineBranchAction, TimelineTransportState};
use eframe::egui;

#[test]
fn test_timeline_ring_buffer_caps_at_2_mib_and_wraps_without_realloc() {
    // Request 10,000 slots of 1 KiB each (~10 MiB) -> must clamp strictly to <= 2 MiB (2048 slots).
    let mut ring = TimelineRingBuffer::new_in_place_with(1024, 10_000, 1, || vec![0.0_f64; 128]);
    assert!(ring.memory_budget_bytes() <= MAX_TIMELINE_RING_BYTES);
    assert_eq!(ring.capacity(), 2048);

    #[cfg(not(target_arch = "wasm32"))]
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    #[cfg(not(target_arch = "wasm32"))]
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for step in 0..3000 {
            ring.write_checkpoint_in_place(step, step as f64 * 0.01, |buf| {
                buf[0] = step as f64;
                buf[127] = (step * 2) as f64;
            });
        }
    });
    #[cfg(target_arch = "wasm32")]
    for step in 0..3000 {
        ring.write_checkpoint_in_place(step, step as f64 * 0.01, |buf| {
            buf[0] = step as f64;
            buf[127] = (step * 2) as f64;
        });
    }

    assert_eq!(ring.len(), 2048);
    assert_eq!(ring.step_bounds(), Some((952, 2999)));
    let cp = ring.checkpoint_exact(1500).expect("step 1500 retained");
    assert_eq!(cp.state[0], 1500.0);
    assert_eq!(cp.state[127], 3000.0);

    // Truncate after 1500 preserves 952..=1500 without resetting to 0.
    ring.truncate_after(1500);
    assert_eq!(ring.step_bounds(), Some((952, 1500)));
    assert!(ring.checkpoint_exact(1501).is_none());
}

#[test]
fn test_divergence_accumulator_zero_perturbation_and_exponential_slope_zero_alloc() {
    let mut zero_acc = BranchDivergenceAccumulator::default();
    let mut exp_acc = BranchDivergenceAccumulator::default();
    let primary = [1.0_f64, 2.0, 3.0, 4.0];
    let shadow = [1.0_f64, 2.0, 3.0, 4.0];

    #[cfg(not(target_arch = "wasm32"))]
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    #[cfg(not(target_arch = "wasm32"))]
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        for step in 1..=100 {
            let (l2, peak) = compute_slice_divergence(&primary, &shadow);
            zero_acc.record_step(0.05, l2, peak);

            let t = step as f64 * 0.05;
            let synth_l2 = 1.0e-5 * (0.8 * t).exp();
            exp_acc.record_step(0.05, synth_l2, synth_l2);
        }
    });
    #[cfg(target_arch = "wasm32")]
    for step in 1..=100 {
        let (l2, peak) = compute_slice_divergence(&primary, &shadow);
        zero_acc.record_step(0.05, l2, peak);
        let t = step as f64 * 0.05;
        let synth_l2 = 1.0e-5 * (0.8 * t).exp();
        exp_acc.record_step(0.05, synth_l2, synth_l2);
    }

    let z = zero_acc.metrics();
    assert_eq!(z.steps_since_fork, 100);
    assert_eq!(z.instantaneous_l2, 0.0);
    assert_eq!(z.peak_abs_delta, 0.0);
    assert_eq!(z.cumulative_area_delta, 0.0);
    assert_eq!(z.log_divergence_slope, 0.0);

    let e = exp_acc.metrics();
    assert!(e.instantaneous_l2 > 1.0e-4);
    assert!(e.cumulative_area_delta > 0.0);
    assert!((e.log_divergence_slope - 0.8).abs() < 1.0e-6);
}

#[test]
fn test_perceptual_difference_rgb_zero_alloc_and_contrast() {
    let a = [0.0_f64, 0.25, 0.5, 1.0];
    let b = [0.0_f64, 0.25, 0.2, 0.2];
    let mut rgb = [0_u8; 12];

    #[cfg(not(target_arch = "wasm32"))]
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    #[cfg(not(target_arch = "wasm32"))]
    let ((l2, peak), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        render_perceptual_difference_rgb(&a, &b, &mut rgb)
    });
    #[cfg(target_arch = "wasm32")]
    let (l2, peak) = render_perceptual_difference_rgb(&a, &b, &mut rgb);

    assert!((peak - 0.8).abs() < 1.0e-12);
    assert!(l2 > 0.8);
    // Unchanged cells use dark neutral background [18, 22, 28].
    assert_eq!(&rgb[0..3], &[18, 22, 28]);
    // Perturbed cells lift luminance above dark background.
    assert!(rgb[9..12].iter().any(|&c| c >= 120));
}

#[test]
fn test_playback_timeline_step_back_shortcut_and_hud_clear_announcements() {
    let ctx = egui::Context::default();
    let ts = TimelineTransportState {
        current_step: 42,
        min_step: 0,
        max_step: 100,
        current_time: 4.2,
        time_unit: "s",
        fork_active: false,
        fork_step: None,
        fork_time: None,
    };

    let mut comma_input = egui::RawInput::default();
    comma_input.events.push(egui::Event::Key {
        key: egui::Key::Comma,
        physical_key: Some(egui::Key::Comma),
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });

    let mut got_branch = None;
    let _ = ctx.run(comma_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (_, b_act) = PlaybackTransport::in_thread("tl_test", "TestModel", false)
                .show_with_timeline(ui, ts);
            if b_act.is_some() {
                got_branch = b_act;
            }
            let _ = BranchDivergenceHud::new(
                "hud_test",
                "TestModel",
                20,
                2.0,
                "s",
                BranchDivergenceMetrics::default(),
            )
            .show(ui);
        });
    });

    assert_eq!(got_branch, Some(TimelineBranchAction::StepBack));
    announce_branch_forked(&ctx, "TestModel", 42, 4.2, "s");
    announce_branch_cleared(&ctx, "TestModel");
    let log = crate::accessibility::all_announcements();
    assert!(log
        .iter()
        .any(|(msg, pri)| msg.contains("Stepped TestModel backward") && pri == "polite"));
    assert!(log.iter().any(
        |(msg, pri)| msg.contains("Forked shadow twin branch in TestModel") && pri == "polite"
    ));
    assert!(log.iter().any(|(msg, pri)| msg
        .contains("Cleared shadow twin branch overlay in TestModel")
        && pri == "polite"));
}
