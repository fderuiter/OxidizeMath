use std::path::PathBuf;
use std::sync::mpsc::Sender;

/// Spawns a file picker dialog (or browser `<input type="file">`) to open a workspace session.
pub fn spawn_open_dialog(tx: Sender<Result<(PathBuf, String), String>>) {
    crate::web_io::spawn_open_file_dialog(
        tx,
        crate::web_io::OpenDialogConfig {
            filter_name: "Workspace Session (*.mws, *.json)",
            extensions: &["mws", "json"],
            accept: ".mws,.json,application/json",
            read_error_prefix: "Failed to read workspace file",
        },
        std::convert::identity,
    );
}

/// Spawns a save file dialog (or triggers browser download) to export a workspace session.
pub fn spawn_save_as_dialog(tx: Sender<Result<PathBuf, String>>, envelope_json: String) {
    crate::web_io::spawn_save_as_file_dialog(
        tx,
        crate::web_io::SaveAsDialogConfig {
            filter_name: "Workspace Session (*.mws, *.json)",
            extensions: &["mws", "json"],
            fallback_filename: "workspace.mws",
            write_error_prefix: "Failed to write workspace file",
        },
        "workspace.mws",
        envelope_json,
        |path, _default_name| path,
    );
}

/// Persists a workspace session directly to disk on desktop or triggers a browser download on WebAssembly.
pub fn spawn_save_direct(
    tx: Sender<Result<PathBuf, String>>,
    path: PathBuf,
    envelope_json: String,
) {
    crate::web_io::spawn_save_direct_file(
        tx,
        crate::web_io::DirectSaveConfig {
            fallback_filename: "workspace.mws",
            write_error_prefix: "Failed to write workspace file",
        },
        path,
        envelope_json,
    );
}

pub fn file_name_str(path: &std::path::Path) -> &str {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace.mws")
}

pub fn handle_transition_error(
    mgr: &mut super::WorkspaceSessionManager,
    ctx: &eframe::egui::Context,
    op: &str,
) {
    let err = format!(
        "Invalid state transition: received {op} result while {}",
        mgr.lifecycle_state.user_label()
    );
    notify(mgr, ctx, err, true);
    mgr.lifecycle_state = super::SessionLifecycleState::Error;
}

pub fn notify(
    mgr: &mut super::WorkspaceSessionManager,
    ctx: &eframe::egui::Context,
    msg: String,
    is_err: bool,
) {
    mgr.set_toast(&msg, is_err);
    let (sev, aria) = if is_err {
        (
            federated_registry::Severity::Error,
            "aria_live_assertive_message",
        )
    } else {
        (federated_registry::Severity::Info, "aria_live_message")
    };
    log_telemetry(msg, sev, ctx, aria);
}

pub fn log_telemetry(
    msg: String,
    severity: federated_registry::Severity,
    ctx: &eframe::egui::Context,
    aria_key: &str,
) {
    crate::telemetry::emit_telemetry(
        "WorkspaceSessionManager",
        severity,
        msg.clone(),
        std::collections::HashMap::new(),
    );
    if aria_key == "aria_live_assertive_message" {
        crate::accessibility::announce_live_assertive_status(ctx, msg);
    } else {
        crate::accessibility::announce_live_status(ctx, msg);
    }
}
