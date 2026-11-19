use std::path::PathBuf;
use std::sync::mpsc::Sender;

// theory_verification!

/// Spawns a file picker dialog (or browser `<input type="file">`) to open a workspace session.
pub fn spawn_open_dialog(tx: Sender<Result<(PathBuf, String), String>>) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Workspace Session (*.mws, *.json)", &["mws", "json"])
                .pick_file()
            {
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        let _ = tx.send(Ok((path, content)));
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("Failed to read workspace file: {}", e)));
                    }
                }
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        egui_plot::export::open_browser_file_picker(
            ".mws,.json,application/json",
            move |outcome| {
                if let Some(res) = outcome {
                    let _ = tx.send(res.map(|(name, content)| (PathBuf::from(name), content)));
                }
            },
        );
    }
}

/// Spawns a save file dialog (or triggers browser download) to export a workspace session.
pub fn spawn_save_as_dialog(tx: Sender<Result<PathBuf, String>>, envelope_json: String) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let default_name = "workspace.mws".to_string();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name(&default_name)
                .add_filter("Workspace Session (*.mws, *.json)", &["mws", "json"])
                .save_file()
            {
                match std::fs::write(&path, envelope_json) {
                    Ok(_) => {
                        let _ = tx.send(Ok(path));
                    }
                    Err(e) => {
                        let _ = tx.send(Err(format!("Failed to write workspace file: {}", e)));
                    }
                }
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        let filename =
            egui_plot::export::sanitize_download_filename("workspace.mws", "workspace.mws");
        let res = egui_plot::export::download_browser_file(&filename, envelope_json.as_bytes())
            .map(|()| PathBuf::from(filename));
        let _ = tx.send(res);
        egui_plot::export::request_browser_repaint();
    }
}

/// Persists a workspace session directly to disk on desktop or triggers a browser download on WebAssembly.
pub fn spawn_save_direct(
    tx: Sender<Result<PathBuf, String>>,
    path: PathBuf,
    envelope_json: String,
) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || match std::fs::write(&path, envelope_json) {
            Ok(_) => {
                let _ = tx.send(Ok(path));
            }
            Err(e) => {
                let _ = tx.send(Err(format!("Failed to write workspace file: {}", e)));
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        let filename =
            egui_plot::export::sanitize_download_filename(file_name_str(&path), "workspace.mws");
        let res = egui_plot::export::download_browser_file(&filename, envelope_json.as_bytes())
            .map(|()| PathBuf::from(filename));
        let _ = tx.send(res);
        egui_plot::export::request_browser_repaint();
    }
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
