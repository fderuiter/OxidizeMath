//! Non-blocking binary file save dialogs and browser `Blob` download helpers.
//!
//! Complements [`crate::web_io`] (UTF-8 text payloads) with `Vec<u8>` binary payload
//! delivery for `.stl` solids and other binary takeaway artifacts across native (`rfd`)
//! and `wasm32-unknown-unknown` (`egui_plot::export::download_browser_file`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::path::PathBuf;
use std::sync::mpsc::Sender;

use crate::web_io::sanitize_download_filename;

#[cfg(target_arch = "wasm32")]
use crate::web_io::{download_browser_file, request_browser_repaint};

/// Filter, fallback filename, and error-formatting configuration for [`spawn_save_as_bytes_dialog`].
#[derive(Clone, Copy, Debug)]
pub struct SaveAsBytesDialogConfig {
    /// Human-readable filter label shown in native save dialogs (e.g., `"Binary STL Mesh (*.stl)"`).
    pub filter_name: &'static str,
    /// Allowed file extensions without leading dots for native `rfd` dialogs (e.g., `&["stl"]`).
    pub extensions: &'static [&'static str],
    /// Fallback leaf filename used when sanitizing `default_filename` (e.g., `"isosurface.stl"`).
    pub fallback_filename: &'static str,
    /// Prefix prepended when native `std::fs::write` returns an error.
    pub write_error_prefix: &'static str,
}

/// Fallback filename and error-formatting configuration for [`spawn_save_direct_bytes`].
#[derive(Clone, Copy, Debug)]
pub struct DirectBytesSaveConfig {
    /// Fallback leaf filename used when sanitizing `path` on WebAssembly downloads.
    pub fallback_filename: &'static str,
    /// Prefix prepended when native `std::fs::write` returns an error.
    pub write_error_prefix: &'static str,
}

/// Sanitizes `default_filename` on both targets and spawns a native `rfd` save-file thread
/// or triggers a WebAssembly browser `Blob` download, delivering `map_path(saved_path, &default_name)` over `tx`.
pub fn spawn_save_as_bytes_dialog<T, F>(
    tx: Sender<Result<T, String>>,
    config: SaveAsBytesDialogConfig,
    default_filename: &str,
    bytes: Vec<u8>,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf, &str) -> T + Send + 'static,
{
    let default_name = sanitize_download_filename(default_filename, config.fallback_filename);
    spawn_save_as_bytes_dialog_impl(tx, config, default_name, bytes, map_path);
}

/// Persists raw `bytes` directly to `path` on a native background thread or triggers a
/// sanitized browser `Blob` download on `wasm32`, delivering the resulting [`PathBuf`] over `tx`.
pub fn spawn_save_direct_bytes(
    tx: Sender<Result<PathBuf, String>>,
    config: DirectBytesSaveConfig,
    path: PathBuf,
    bytes: Vec<u8>,
) {
    spawn_save_direct_bytes_impl(tx, config, path, bytes);
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_save_as_bytes_dialog_impl<T, F>(
    tx: Sender<Result<T, String>>,
    config: SaveAsBytesDialogConfig,
    default_name: String,
    bytes: Vec<u8>,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf, &str) -> T + Send + 'static,
{
    std::thread::spawn(move || {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(&default_name)
            .add_filter(config.filter_name, config.extensions)
            .save_file()
        {
            match std::fs::write(&path, bytes) {
                Ok(()) => {
                    let _ = tx.send(Ok(map_path(path, &default_name)));
                }
                Err(err) => {
                    let _ = tx.send(Err(format!("{}: {err}", config.write_error_prefix)));
                }
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn send_browser_bytes_download<T, F>(
    tx: Sender<Result<T, String>>,
    filename: String,
    bytes: &[u8],
    map_filename: F,
) where
    F: FnOnce(String) -> T,
{
    let res = download_browser_file(&filename, bytes).map(|()| map_filename(filename));
    let _ = tx.send(res);
    request_browser_repaint();
}

#[cfg(target_arch = "wasm32")]
fn spawn_save_as_bytes_dialog_impl<T, F>(
    tx: Sender<Result<T, String>>,
    _config: SaveAsBytesDialogConfig,
    default_name: String,
    bytes: Vec<u8>,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf, &str) -> T + Send + 'static,
{
    send_browser_bytes_download(tx, default_name, &bytes, |name| {
        map_path(PathBuf::from(&name), &name)
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_save_direct_bytes_impl(
    tx: Sender<Result<PathBuf, String>>,
    config: DirectBytesSaveConfig,
    path: PathBuf,
    bytes: Vec<u8>,
) {
    std::thread::spawn(move || match std::fs::write(&path, bytes) {
        Ok(()) => {
            let _ = tx.send(Ok(path));
        }
        Err(err) => {
            let _ = tx.send(Err(format!("{}: {err}", config.write_error_prefix)));
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn spawn_save_direct_bytes_impl(
    tx: Sender<Result<PathBuf, String>>,
    config: DirectBytesSaveConfig,
    path: PathBuf,
    bytes: Vec<u8>,
) {
    let filename = sanitize_download_filename(&path.to_string_lossy(), config.fallback_filename);
    send_browser_bytes_download(tx, filename, &bytes, PathBuf::from);
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;

    #[test]
    fn test_bytes_dialog_config_sanitizes_stl_filenames() {
        let cfg = SaveAsBytesDialogConfig {
            filter_name: "Binary STL (*.stl)",
            extensions: &["stl"],
            fallback_filename: "isosurface.stl",
            write_error_prefix: "Failed to write STL",
        };
        assert_eq!(cfg.extensions, &["stl"]);
        assert_eq!(
            sanitize_download_filename("nested/gyroid_isosurface.stl", cfg.fallback_filename),
            "gyroid_isosurface.stl"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_spawn_save_direct_bytes_writes_exact_binary_buffer() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        let temp_dir = std::env::temp_dir();
        let target = temp_dir.join(format!("web_io_bytes_stl_{}.stl", std::process::id()));
        let _ = std::fs::remove_file(&target);

        let cfg = DirectBytesSaveConfig {
            fallback_filename: "isosurface.stl",
            write_error_prefix: "Failed to write STL",
        };
        let payload = vec![0x4F, 0x78, 0x00, 0xFF, 0x01, 0x02, 0x03, 0x04];
        let (tx, rx) = channel();
        spawn_save_direct_bytes(tx, cfg, target.clone(), payload.clone());

        let written_path = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("bytes save channel delivery")
            .expect("bytes save succeeded");
        assert_eq!(written_path, target);
        assert_eq!(std::fs::read(&target).expect("read saved bytes"), payload);
        let _ = std::fs::remove_file(&target);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;
    use std::sync::mpsc::channel;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_spawn_save_bytes_dialogs_download_binary_stl_blobs() {
        let save_cfg = SaveAsBytesDialogConfig {
            filter_name: "Binary STL (*.stl)",
            extensions: &["stl"],
            fallback_filename: "isosurface.stl",
            write_error_prefix: "Failed to write STL",
        };
        let (tx, rx) = channel();
        let stl_stub = vec![0_u8; 134];
        spawn_save_as_bytes_dialog(
            tx,
            save_cfg,
            "nested/gyroid_isosurface.stl",
            stl_stub,
            |path, leaf| (path, leaf.to_string()),
        );
        let (path, leaf) = rx
            .try_recv()
            .expect("wasm bytes download sends synchronously")
            .expect("wasm bytes download succeeds");
        assert_eq!(path, PathBuf::from("gyroid_isosurface.stl"));
        assert_eq!(leaf, "gyroid_isosurface.stl");
    }
}
