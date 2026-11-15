use std::path::PathBuf;
use std::sync::mpsc::Sender;

/// Spawns a background thread showing the file picker dialog to open a workspace session.
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
}

/// Spawns a background thread showing the save file dialog to export a workspace session.
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
}

/// Spawns a background thread to persist a workspace session directly to disk.
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
}
