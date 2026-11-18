//! Cross-platform timing and re-exported WebAssembly file I/O helpers.
//!
//! Provides [`PlatformInstant`] for safe elapsed-time measurement across desktop and
//! `wasm32-unknown-unknown`, and re-exports the canonical browser download and hidden
//! file-picker helpers from [`egui_plot::export`].

// theory_verification!

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

impl PlatformInstant {
    /// Captures the current platform timestamp.
    #[must_use]
    pub fn now() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self {
                inner: std::time::Instant::now(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self {
                millis: js_sys::Date::now(),
            }
        }
    }

    /// Returns the non-negative duration elapsed between `earlier` and `self`.
    #[must_use]
    pub fn duration_since(&self, earlier: Self) -> Duration {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner.saturating_duration_since(earlier.inner)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let delta_ms = (self.millis - earlier.millis).max(0.0);
            if delta_ms.is_finite() {
                Duration::from_secs_f64(delta_ms / 1000.0)
            } else {
                Duration::ZERO
            }
        }
    }

    /// Returns the duration elapsed since this timestamp was captured.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        Self::now().duration_since(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
