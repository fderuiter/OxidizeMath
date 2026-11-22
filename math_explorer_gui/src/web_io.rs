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
    let default_name = sanitize_download_filename(default_filename, config.fallback_filename);
    spawn_save_as_file_dialog_impl(tx, config, default_name, content, map_path);
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
fn spawn_save_as_file_dialog_impl<T, F>(
    tx: Sender<Result<T, String>>,
    config: SaveAsDialogConfig,
    default_name: String,
    content: String,
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
            match std::fs::write(&path, content) {
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
fn send_browser_download<T, F>(
    tx: Sender<Result<T, String>>,
    filename: String,
    content: &str,
    map_filename: F,
) where
    F: FnOnce(String) -> T,
{
    let res = download_browser_file(&filename, content.as_bytes()).map(|()| map_filename(filename));
    let _ = tx.send(res);
    request_browser_repaint();
}

#[cfg(target_arch = "wasm32")]
fn spawn_save_as_file_dialog_impl<T, F>(
    tx: Sender<Result<T, String>>,
    _config: SaveAsDialogConfig,
    default_name: String,
    content: String,
    map_path: F,
) where
    T: Send + 'static,
    F: FnOnce(PathBuf, &str) -> T + Send + 'static,
{
    send_browser_download(tx, default_name, &content, |name| {
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
    send_browser_download(tx, filename, &content, PathBuf::from);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn test_infer_mime_type_json_and_mws() {
        assert_eq!(infer_mime_type("analysis_preset.json"), "application/json");
        assert_eq!(infer_mime_type("PRESET.JSON"), "application/json");
        assert_eq!(infer_mime_type("workspace.mws"), "application/json");
        assert_eq!(infer_mime_type("Session_Backup.MWS"), "application/json");
    }

    #[test]
    fn test_infer_mime_type_other_formats() {
        assert_eq!(infer_mime_type("points.csv"), "text/csv");
        assert_eq!(infer_mime_type("plot.svg"), "image/svg+xml");
        assert_eq!(infer_mime_type("figure.png"), "image/png");
        assert_eq!(infer_mime_type("anim.gif"), "image/gif");
        assert_eq!(infer_mime_type("clip_audio.wav"), "audio/wav");
        assert_eq!(infer_mime_type("CLIP.WAV"), "audio/wav");
        assert_eq!(infer_mime_type("surface.obj"), "application/octet-stream");
    }

    #[test]
    fn test_sanitize_download_filename_strips_paths_and_falls_back() {
        assert_eq!(
            sanitize_download_filename("workspace.mws", "fallback.mws"),
            "workspace.mws"
        );
        assert_eq!(
            sanitize_download_filename("/tmp/nested/session.mws", "workspace.mws"),
            "session.mws"
        );
        assert_eq!(
            sanitize_download_filename("C:\\fakepath\\preset.json", "fallback.json"),
            "preset.json"
        );
        assert_eq!(
            sanitize_download_filename("   ", "workspace.mws"),
            "workspace.mws"
        );
        assert_eq!(
            sanitize_download_filename("..", "preset.json"),
            "preset.json"
        );
    }

    #[test]
    fn test_platform_instant_monotonic_duration() {
        let earlier = PlatformInstant::now();
        let later = PlatformInstant::now();
        assert!(later.duration_since(earlier) >= Duration::ZERO);
        assert_eq!(earlier.duration_since(later), Duration::ZERO);
        assert!(earlier.elapsed() >= Duration::ZERO);
    }

    #[test]
    fn test_dialog_configs_preserve_metadata() {
        let open_cfg = OpenDialogConfig {
            filter_name: "JSON Preset",
            extensions: &["json"],
            accept: ".json,application/json",
            read_error_prefix: "Failed to read file",
        };
        assert_eq!(open_cfg.extensions, &["json"]);
        assert_eq!(open_cfg.read_error_prefix, "Failed to read file");

        let save_cfg = SaveAsDialogConfig {
            filter_name: "Workspace Session (*.mws, *.json)",
            extensions: &["mws", "json"],
            fallback_filename: "workspace.mws",
            write_error_prefix: "Failed to write workspace file",
        };
        assert_eq!(
            sanitize_download_filename("nested/dir/custom.mws", save_cfg.fallback_filename),
            "custom.mws"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_spawn_save_direct_file_retains_full_path_and_formats_write_errors() {
        let temp_dir = std::env::temp_dir();
        let target_path = temp_dir.join(format!("web_io_direct_save_{}.mws", std::process::id()));
        let _ = std::fs::remove_file(&target_path);

        let config = DirectSaveConfig {
            fallback_filename: "workspace.mws",
            write_error_prefix: "Failed to write workspace file",
        };
        let (tx, rx) = channel();
        let payload = r#"{"schema_version":"1.0"}"#.to_string();
        spawn_save_direct_file(tx, config, target_path.clone(), payload.clone());

        let saved_path = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("direct save channel should deliver result")
            .expect("direct save to temp dir should succeed");
        assert_eq!(saved_path, target_path);
        assert_eq!(
            std::fs::read_to_string(&target_path).expect("saved file should exist"),
            payload
        );
        let _ = std::fs::remove_file(&target_path);

        let missing_parent = temp_dir
            .join(format!("nonexistent_web_io_dir_{}", std::process::id()))
            .join("session.mws");
        let (err_tx, err_rx) = channel();
        spawn_save_direct_file(err_tx, config, missing_parent, payload);
        let err = err_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("error channel should deliver result")
            .expect_err("writing to missing parent directory should fail");
        assert!(err.starts_with("Failed to write workspace file: "));
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_spawn_save_direct_file_wasm_typecheck_and_channel_delivery() {
        let config = DirectSaveConfig {
            fallback_filename: "workspace.mws",
            write_error_prefix: "Failed to write workspace file",
        };
        let (tx, rx) = channel();
        spawn_save_direct_file(
            tx,
            config,
            PathBuf::from("browser_session.mws"),
            "{}".to_string(),
        );
        assert!(rx.try_recv().is_ok());
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "web_io_wasm_tests.rs"]
mod wasm_tests;
