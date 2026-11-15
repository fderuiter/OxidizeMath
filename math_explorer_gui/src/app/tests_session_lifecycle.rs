use super::session::SessionLifecycleState;
use super::session::WorkspaceSessionManager;
use super::AppState;
use crate::app::ThemeMode;
use eframe::egui;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::channel;

fn make_dummy_state(tab: usize, theme: ThemeMode) -> AppState {
    AppState {
        selected_tab: tab,
        selected_tab_name: None,
        show_info: true,
        show_warnings: true,
        show_errors: true,
        show_diagnostics_panel: true,
        theme,
        tab_states: HashMap::new(),
        ..Default::default()
    }
}

#[test]
fn test_lifecycle_state_enum_and_display() {
    assert_eq!(
        SessionLifecycleState::default(),
        SessionLifecycleState::Idle
    );
    assert_eq!(SessionLifecycleState::Idle.to_string(), "Idle");
    assert_eq!(
        SessionLifecycleState::DragEditing.to_string(),
        "DragEditing"
    );
    assert_eq!(
        SessionLifecycleState::RestoringHistory.to_string(),
        "RestoringHistory"
    );
    assert_eq!(
        SessionLifecycleState::AwaitingFileDialog.to_string(),
        "AwaitingFileDialog"
    );
    assert_eq!(SessionLifecycleState::Saving.to_string(), "Saving");
    assert_eq!(SessionLifecycleState::Loading.to_string(), "Loading");
    assert_eq!(SessionLifecycleState::Error.to_string(), "Error");
}

#[test]
fn test_trigger_open_rejection_while_awaiting() {
    let mut mgr = WorkspaceSessionManager::default();
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);

    mgr.trigger_open();
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::AwaitingFileDialog
    );

    // Repeated call while awaiting dialog must be rejected without altering receiver
    mgr.trigger_open();
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::AwaitingFileDialog
    );
}

#[test]
fn test_trigger_save_as_rejection_while_awaiting() {
    let mut mgr = WorkspaceSessionManager::default();
    let env_json = mgr.create_envelope(make_dummy_state(0, ThemeMode::Dark));

    mgr.trigger_save_as(env_json.clone());
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::AwaitingFileDialog
    );

    // Repeated call while awaiting dialog must be rejected without altering receiver
    mgr.trigger_save_as(env_json);
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::AwaitingFileDialog
    );
}

#[test]
fn test_state_restoration_two_consecutive_quiescent_frames() {
    let ctx = egui::Context::default();
    let mut mgr = WorkspaceSessionManager::default();

    let state0 = make_dummy_state(0, ThemeMode::Dark);
    let state1 = make_dummy_state(1, ThemeMode::Light);
    let state2 = make_dummy_state(2, ThemeMode::Dark);

    mgr.track_state_changes(&ctx, state0.clone());
    mgr.set_is_restoring(true);
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::RestoringHistory
    );
    assert!(mgr.is_restoring());

    // Frame 1: state changes to state1 (UI still adapting) -> resets quiescent streak
    mgr.track_state_changes(&ctx, state1.clone());
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::RestoringHistory
    );

    // Frame 2: state remains state1 (1st quiescent frame) -> still restoring
    mgr.track_state_changes(&ctx, state1.clone());
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::RestoringHistory
    );

    // Frame 3: another state change to state2 -> breaks streak, resets quiescent count
    mgr.track_state_changes(&ctx, state2.clone());
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::RestoringHistory
    );

    // Frame 4: state remains state2 (1st consecutive quiescent frame) -> still restoring
    mgr.track_state_changes(&ctx, state2.clone());
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::RestoringHistory
    );

    // Frame 5: state remains state2 (2nd consecutive quiescent frame) -> transitions to Idle!
    mgr.track_state_changes(&ctx, state2);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
    assert!(!mgr.is_restoring());
}

#[test]
fn test_state_restoration_preserves_redo_and_no_spurious_undo() {
    let ctx = egui::Context::default();
    let mut mgr = WorkspaceSessionManager::default();

    let s0 = make_dummy_state(0, ThemeMode::Dark);
    let s1 = make_dummy_state(1, ThemeMode::Dark);
    let s2 = make_dummy_state(2, ThemeMode::Dark);

    mgr.push_undo(s0.clone());
    mgr.push_redo(s2.clone());

    mgr.set_is_restoring(true);

    // Attempting push_undo while restoring history must be ignored
    mgr.push_undo(s1.clone());
    assert!(mgr.can_redo());
    assert_eq!(mgr.pop_redo(), Some(s2.clone()));

    // Multi-frame state updates during restoration do not push undo snapshots
    mgr.push_redo(s2.clone());
    mgr.track_state_changes(&ctx, s1.clone());
    mgr.track_state_changes(&ctx, s2.clone());
    assert_eq!(mgr.pop_undo(), Some(s0));
    assert!(mgr.pop_undo().is_none());
    assert_eq!(mgr.pop_redo(), Some(s2));
}

#[test]
fn test_push_undo_preserving_redo_for_multi_step_redo() {
    let mut mgr = WorkspaceSessionManager::default();

    let s0 = make_dummy_state(0, ThemeMode::Dark);
    let s1 = make_dummy_state(1, ThemeMode::Dark);
    let s2 = make_dummy_state(2, ThemeMode::Dark);

    mgr.push_redo(s2.clone());
    mgr.push_redo(s1.clone());

    // Redo step 1
    let next1 = mgr.pop_redo().unwrap();
    assert_eq!(next1, s1);
    mgr.push_undo_preserving_redo(s0);

    // Redo stack must still contain s2
    assert!(mgr.can_redo());
    assert_eq!(mgr.pop_redo(), Some(s2));
}

#[test]
fn test_slider_drag_coalescing_lifecycle_transitions() {
    let ctx = egui::Context::default();
    let mut mgr = WorkspaceSessionManager::default();

    let s0 = make_dummy_state(0, ThemeMode::Dark);
    mgr.track_state_changes(&ctx, s0.clone());
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);

    // Simulate pointer press
    let raw_down = egui::RawInput {
        events: vec![egui::Event::PointerButton {
            pos: egui::pos2(50.0, 50.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let _ = ctx.run(raw_down, |_| {});

    for i in 1..=10 {
        let mut s_i = s0.clone();
        s_i.selected_tab = i;
        mgr.track_state_changes(&ctx, s_i);
        assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::DragEditing);
    }

    // Simulate pointer release
    let raw_up = egui::RawInput {
        events: vec![egui::Event::PointerButton {
            pos: egui::pos2(50.0, 50.0),
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };
    let _ = ctx.run(raw_up, |_| {});

    let mut final_state = s0.clone();
    final_state.selected_tab = 10;
    mgr.track_state_changes(&ctx, final_state);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);

    // Exactly one undo snapshot was created
    assert_eq!(mgr.pop_undo(), Some(s0));
    assert!(mgr.pop_undo().is_none());
}

#[test]
fn test_steady_state_polling_leaves_state_idle() {
    let ctx = egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = WorkspaceSessionManager::default();

    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
    mgr.poll(&mut app, &ctx);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
}

#[test]
fn test_poll_open_success_and_failure() {
    let ctx = egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = WorkspaceSessionManager::default();

    let (tx, rx) = channel();
    mgr.open_rx = Some(rx);
    mgr.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;

    let valid_state = make_dummy_state(1, ThemeMode::Light);
    let env_json = mgr.create_envelope(valid_state);
    tx.send(Ok((PathBuf::from("/tmp/test.mws"), env_json)))
        .unwrap();

    mgr.poll(&mut app, &ctx);
    assert_eq!(
        mgr.lifecycle_state(),
        SessionLifecycleState::RestoringHistory
    );
    assert_eq!(mgr.toast().map(|t| t.1), Some(false));
    let aria = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(aria.is_some());

    // Test open channel error
    let (tx2, rx2) = channel();
    mgr.open_rx = Some(rx2);
    mgr.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;
    tx2.send(Err("Disk read error".to_string())).unwrap();

    mgr.poll(&mut app, &ctx);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Error);
    assert_eq!(mgr.toast().map(|t| t.1), Some(true));
    let aria_err = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_assertive_message")));
    assert_eq!(aria_err, Some("Disk read error".to_string()));

    // Dismiss toast clears error back to Idle
    mgr.clear_toast();
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
}

#[test]
fn test_poll_dialog_cancellation_returns_to_idle() {
    let ctx = egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = WorkspaceSessionManager::default();

    let (tx, rx) = channel::<Result<(PathBuf, String), String>>();
    mgr.open_rx = Some(rx);
    mgr.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;

    drop(tx); // User clicked Cancel
    mgr.poll(&mut app, &ctx);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
    assert!(mgr.open_rx.is_none());

    // Save as cancellation
    let (tx_save, rx_save) = channel::<Result<PathBuf, String>>();
    mgr.save_as_rx = Some(rx_save);
    mgr.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;

    drop(tx_save);
    mgr.poll(&mut app, &ctx);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
    assert!(mgr.save_as_rx.is_none());
}

#[test]
fn test_poll_save_as_and_direct_save_transitions() {
    let ctx = egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = WorkspaceSessionManager::default();

    let (tx, rx) = channel();
    mgr.save_as_rx = Some(rx);
    mgr.lifecycle_state = SessionLifecycleState::AwaitingFileDialog;

    tx.send(Ok(PathBuf::from("/tmp/saved_workspace.mws")))
        .unwrap();
    mgr.poll(&mut app, &ctx);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
    assert_eq!(mgr.toast().map(|t| t.1), Some(false));
    let aria = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(aria.unwrap().contains("Exported workspace session"));

    // Direct save success
    let (tx_direct, rx_direct) = channel();
    mgr.save_rx = Some(rx_direct);
    mgr.lifecycle_state = SessionLifecycleState::Saving;

    tx_direct
        .send(Ok(PathBuf::from("/tmp/saved_workspace.mws")))
        .unwrap();
    mgr.poll(&mut app, &ctx);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Idle);
    assert_eq!(mgr.toast().map(|t| t.1), Some(false));
    let aria_save = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert!(aria_save.unwrap().contains("Workspace session saved"));
}

#[test]
fn test_invalid_state_transition_logging() {
    let ctx = egui::Context::default();
    let mut app = crate::app::MathExplorerApp::default();
    let mut mgr = WorkspaceSessionManager::default();

    let (tx, rx) = channel();
    mgr.open_rx = Some(rx);
    // Invalid state: receiving while Idle instead of AwaitingFileDialog
    mgr.lifecycle_state = SessionLifecycleState::Idle;
    tx.send(Ok((PathBuf::from("/tmp/file.mws"), "{}".to_string())))
        .unwrap();

    mgr.poll(&mut app, &ctx);
    assert_eq!(mgr.lifecycle_state(), SessionLifecycleState::Error);
    assert_eq!(mgr.toast().map(|t| t.1), Some(true));
    let aria = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_assertive_message")));
    assert!(aria.unwrap().contains("Invalid state transition"));
}

#[test]
fn test_session_lifecycle_state_chart_screenshot_generation() {
    let lifecycle_datasets = vec![(
        "Session Lifecycle State Progression".to_string(),
        vec![
            [0.0, 0.0], // Idle
            [1.0, 1.0], // DragEditing
            [2.0, 0.0], // Idle
            [3.0, 2.0], // RestoringHistory
            [4.0, 2.0], // RestoringHistory (frame 1)
            [5.0, 0.0], // Idle (quiescent frame 2)
            [6.0, 3.0], // AwaitingFileDialog
            [7.0, 5.0], // Loading
            [8.0, 2.0], // RestoringHistory
            [9.0, 0.0], // Idle
        ],
    )];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&lifecycle_datasets, &options) {
        let _ = std::fs::write("/tmp/session_lifecycle_transitions.png", png_data);
    }
}
