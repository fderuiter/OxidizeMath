#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::{
    SimWorkerState, SnapshotHistory, StateSnapshot, TimelineMetadata, MAX_SNAPSHOT_HISTORY_BYTES,
};
use crate::framework::PlaybackTransport;
use std::sync::{Arc, RwLock};

fn key_press_input(key: egui::Key) -> egui::RawInput {
    egui::RawInput {
        events: vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    }
}

fn sample_timeline_meta() -> TimelineMetadata {
    TimelineMetadata {
        current_step: 40,
        min_step: 0,
        max_step: 64,
        stride: 16,
    }
}

#[test]
fn test_multi_pane_sync_playback_propagates_step_back_and_keyframe_keys_in_lockstep() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.compare_sync.sync_playback = true;
    let meta = sample_timeline_meta();

    let mut p0_action = None;
    let mut p1_action = None;
    let _ = ctx.run(key_press_input(egui::Key::Comma), |ctx| {
        publish_multi_pane_sync_context(&app, ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            p0_action = PlaybackTransport::from_worker_state(
                "pane0_attr",
                "Attractor Plotter",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
            p1_action = PlaybackTransport::from_worker_state(
                "pane1_attr",
                "Attractor Plotter",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
        });
    });
    assert_eq!(p0_action, Some(PlaybackAction::StepBack));
    assert_eq!(p1_action, Some(PlaybackAction::StepBack));

    let mut p0_kf = None;
    let mut p1_kf = None;
    let _ = ctx.run(key_press_input(egui::Key::OpenBracket), |ctx| {
        publish_multi_pane_sync_context(&app, ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            p0_kf = PlaybackTransport::from_worker_state(
                "pane0_attr",
                "Attractor Plotter",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
            p1_kf = PlaybackTransport::from_worker_state(
                "pane1_attr",
                "Attractor Plotter",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
        });
    });
    assert_eq!(p0_kf, Some(PlaybackAction::ScrubTo(32)));
    assert_eq!(p1_kf, Some(PlaybackAction::ScrubTo(32)));
}

#[test]
fn test_multi_pane_sync_playback_broadcasts_scrub_to_across_panes() {
    let ctx = egui::Context::default();
    let mut app = MathExplorerApp::default();
    app.compare_sync.sync_playback = true;
    let meta = sample_timeline_meta();

    let mut p1_scrub = None;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        publish_multi_pane_sync_context(&app, ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            let p0_id = ui.make_persistent_id("pane0_attr");
            assert!(emit_playback_broadcast(
                ctx,
                p0_id,
                PlaybackAction::ScrubTo(18),
                None,
            ));
            p1_scrub = PlaybackTransport::from_worker_state(
                "pane1_attr",
                "Attractor Plotter",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
        });
    });
    assert_eq!(p1_scrub, Some(PlaybackAction::ScrubTo(18)));
}

#[test]
fn test_snapshot_history_ring_enforces_2mib_cap_and_truncates_on_resume() {
    let mut hist = SnapshotHistory::new(64, MAX_SNAPSHOT_HISTORY_BYTES);
    let make_snap = |step: usize, vec_len: usize| StateSnapshot {
        width: 1,
        height: 1,
        pixels: Arc::new(RwLock::new(Vec::new())),
        custom_data: vec![step as f64; vec_len],
        structured_data: None,
        sonification: crate::audio::SonificationFrame::silent(),
    };

    for step in 0..=20 {
        hist.record(step, &make_snap(step, 8));
    }
    assert_eq!(hist.len(), 21);
    assert_eq!(hist.snapshot_at_step(12).unwrap().custom_data[0], 12.0);

    // Truncating after step 8 drops 9..=20
    hist.truncate_after(8);
    assert_eq!(hist.len(), 9);
    assert!(hist.snapshot_at_step(9).is_none());
    assert_eq!(hist.snapshot_at_step(8).unwrap().custom_data[0], 8.0);

    // Exceeding 2 MiB budget evicts oldest frames and never breaches MAX_SNAPSHOT_HISTORY_BYTES
    let mut bounded = SnapshotHistory::new(500, MAX_SNAPSHOT_HISTORY_BYTES);
    for step in 0..100 {
        // 8,192 f64s = 65,536 bytes per frame -> 100 frames (~6.5 MiB) must evict to <= 2 MiB
        bounded.record(step, &make_snap(step, 8_192));
        assert!(bounded.total_bytes() <= MAX_SNAPSHOT_HISTORY_BYTES);
    }
    assert!(bounded.len() < 40);
    assert!(bounded.snapshot_at_step(99).is_some());
    assert!(bounded.snapshot_at_step(0).is_none());

    bounded.clear();
    assert!(bounded.is_empty());
    assert_eq!(bounded.total_bytes(), 0);
}
