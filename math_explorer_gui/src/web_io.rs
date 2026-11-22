//! Cross-platform timing and non-blocking file dialog I/O helpers.
//!
//! Provides [`PlatformInstant`] for safe elapsed-time measurement across desktop and
//! `wasm32-unknown-unknown`, re-exports the canonical browser download and hidden
//! file-picker helpers from [`egui_plot::export`], and centralizes `mpsc`-backed
//! open, save-as, and direct-save file operations for native (`rfd`) and WebAssembly targets.

// theory_verification!

use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::time::Duration;

pub use egui_plot::export::{infer_mime_type, sanitize_download_filename, strip_utf8_bom};

#[cfg(target_arch = "wasm32")]
pub use egui_plot::export::{
    download_browser_file, open_browser_file_picker, request_browser_repaint, PickerOutcome,
};

/// Cross-platform timestamp abstraction safe on both desktop (`std::time::Instant`) and
/// `wasm32-unknown-unknown` (`js_sys::Date::now()`), avoiding runtime panics on WebAssembly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlatformInstant {
    #[cfg(not(target_arch = "wasm32"))]
    inner: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    millis: f64,
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::disallowed_methods)]
fn capture_platform_instant() -> PlatformInstant {
    PlatformInstant {
        inner: std::time::Instant::now(),
    }
}

#[cfg(target_arch = "wasm32")]
fn capture_platform_instant() -> PlatformInstant {
    PlatformInstant {
        millis: js_sys::Date::now(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn platform_duration_since(later: PlatformInstant, earlier: PlatformInstant) -> Duration {
    later.inner.saturating_duration_since(earlier.inner)
}

#[cfg(target_arch = "wasm32")]
fn platform_duration_since(later: PlatformInstant, earlier: PlatformInstant) -> Duration {
    let delta_ms = (later.millis - earlier.millis).max(0.0);
    if delta_ms.is_finite() {
        Duration::from_secs_f64(delta_ms / 1000.0)
    } else {
        Duration::ZERO
    }
}

impl PlatformInstant {
    /// Captures the current platform timestamp.
    #[must_use]
    pub fn now() -> Self {
        capture_platform_instant()
    }

    /// Returns the non-negative duration elapsed between `earlier` and `self`.
    #[must_use]
    pub fn duration_since(&self, earlier: Self) -> Duration {
        platform_duration_since(*self, earlier)
    }

    /// Returns the duration elapsed since this timestamp was captured.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        Self::now().duration_since(*self)
    }
}

/// Filter and error-formatting configuration for [`spawn_open_file_dialog`].
#[derive(Clone, Copy, Debug)]
pub struct OpenDialogConfig {
    /// Human-readable filter label shown in native file dialogs (e.g., `"JSON Preset"`).
    pub filter_name: &'static str,
    /// Allowed file extensions without leading dots for native `rfd` dialogs (e.g., `&["json"]`).
    pub extensions: &'static [&'static str],
    /// HTML `<input type="file" accept="...">` value for WebAssembly pickers.
    pub accept: &'static str,
    /// Prefix prepended when native `std::fs::read_to_string` returns an error.
    pub read_error_prefix: &'static str,
}

/// Filter, fallback filename, and error-formatting configuration for [`spawn_save_as_file_dialog`].
#[derive(Clone, Copy, Debug)]
pub struct SaveAsDialogConfig {
    /// Human-readable filter label shown in native save dialogs (e.g., `"JSON Preset"`).
    pub filter_name: &'static str,
    /// Allowed file extensions without leading dots for native `rfd` dialogs (e.g., `&["json"]`).
    pub extensions: &'static [&'static str],
    /// Fallback leaf filename used when sanitizing `default_filename` (e.g., `"preset.json"`).
    pub fallback_filename: &'static str,
    /// Prefix prepended when native `std::fs::write` returns an error.
    pub write_error_prefix: &'static str,
}

/// Fallback filename and error-formatting configuration for [`spawn_save_direct_file`].
#[derive(Clone, Copy, Debug)]
pub struct DirectSaveConfig {
    /// Fallback leaf filename used when sanitizing `path` on WebAssembly downloads.
    pub fallback_filename: &'static str,
    /// Prefix prepended when native `std::fs::write` returns an error.
    pub write_error_prefix: &'static str,
}

/// Spawns a non-blocking native `rfd` file picker thread or WebAssembly `<input type="file">`
/// picker, delivering `(map_path(selected_path), text_content)` over `tx` or dropping `tx` on cancel.
pub fn spawn_open_file_dialog<T, F>(
    tx: Sender<Result<(T, String), String>>,
    config: OpenDialogConfig,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf) -> T + Send + 'static,
{
    spawn_open_file_dialog_impl(tx, config, map_path);
}

/// Sanitizes `default_filename` on both targets and spawns a native `rfd` save-file thread or
/// triggers a WebAssembly browser download, delivering `map_path(saved_path, &default_name)` over `tx`.
pub fn spawn_save_as_file_dialog<T, F>(
    tx: Sender<Result<T, String>>,
    config: SaveAsDialogConfig,
    default_filename: &str,
    content: String,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf, &str) -> T + Send + 'static,
{
    spawn_save_as_bytes_dialog(tx, config, default_filename, content.into_bytes(), map_path);
}

/// Sanitizes `default_filename` on both targets and writes raw binary `bytes` via a native `rfd`
/// save-file thread or WebAssembly `Blob` download (`download_browser_file`).
pub fn spawn_save_as_bytes_dialog<T, F>(
    tx: Sender<Result<T, String>>,
    config: SaveAsDialogConfig,
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

/// Triggers a non-blocking native save dialog (`rfd`) or WebAssembly browser download for `.stl` / `.obj` bytes.
pub fn download_mesh_bytes(config: SaveAsDialogConfig, default_filename: &str, bytes: Vec<u8>) {
    let (tx, _rx) = std::sync::mpsc::channel();
    spawn_save_as_bytes_dialog(tx, config, default_filename, bytes, |path, _| path);
}

/// Persists `content` directly to `path` on a native background thread (retaining the full [`PathBuf`])
/// or triggers a sanitized browser download on `wasm32`, delivering the resulting [`PathBuf`] over `tx`.
pub fn spawn_save_direct_file(
    tx: Sender<Result<PathBuf, String>>,
    config: DirectSaveConfig,
    path: PathBuf,
    content: String,
) {
    spawn_save_direct_file_impl(tx, config, path, content);
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_open_file_dialog_impl<T, F>(
    tx: Sender<Result<(T, String), String>>,
    config: OpenDialogConfig,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf) -> T + Send + 'static,
{
    std::thread::spawn(move || {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(config.filter_name, config.extensions)
            .pick_file()
        {
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    let _ = tx.send(Ok((map_path(path), content)));
                }
                Err(e) => {
                    let _ = tx.send(Err(format!("{}: {e}", config.read_error_prefix)));
                }
            }
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn spawn_open_file_dialog_impl<T, F>(
    tx: Sender<Result<(T, String), String>>,
    config: OpenDialogConfig,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf) -> T + Send + 'static,
{
    open_browser_file_picker(config.accept, move |outcome| {
        if let Some(res) = outcome {
            let _ = tx.send(res.map(|(name, content)| (map_path(PathBuf::from(name)), content)));
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_save_as_bytes_dialog_impl<T, F>(
    tx: Sender<Result<T, String>>,
    config: SaveAsDialogConfig,
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
                Err(e) => {
                    let _ = tx.send(Err(format!("{}: {e}", config.write_error_prefix)));
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
    _config: SaveAsDialogConfig,
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
fn spawn_save_direct_file_impl(
    tx: Sender<Result<PathBuf, String>>,
    config: DirectSaveConfig,
    path: PathBuf,
    content: String,
) {
    std::thread::spawn(move || match std::fs::write(&path, content) {
        Ok(()) => {
            let _ = tx.send(Ok(path));
        }
        Err(e) => {
            let _ = tx.send(Err(format!("{}: {e}", config.write_error_prefix)));
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn spawn_save_direct_file_impl(
    tx: Sender<Result<PathBuf, String>>,
    config: DirectSaveConfig,
    path: PathBuf,
    content: String,
) {
    let filename = sanitize_download_filename(&path.to_string_lossy(), config.fallback_filename);
    send_browser_bytes_download(tx, filename, content.as_bytes(), PathBuf::from);
}

#[cfg(test)]
#[path = "web_io_tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "web_io_wasm_tests.rs"]
mod wasm_tests;
