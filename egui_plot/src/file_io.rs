//! Cross-platform and WebAssembly browser file I/O helpers for downloads and text file imports.
//!
//! Centralizes MIME inference, download filename sanitization, UTF-8 BOM stripping,
//! hidden `<input type="file">` + `FileReader` browser pickers, `Blob` + `<a download>`
//! browser file downloads, and plot-scoped asynchronous CSV dataset import queues.

#[path = "file_io_queue.rs"]
mod file_io_queue;

#[cfg(test)]
use std::sync::PoisonError;

pub(crate) use file_io_queue::process_plot_csv_import;
#[cfg(test)]
use file_io_queue::{MAX_PENDING_PLOT_IMPORTS, pending_imports_queue, pending_status_queue};
pub use file_io_queue::{drain_import_status, drain_imported_datasets};

/// Strips a leading UTF-8 byte-order mark (`U+FEFF`) if present.
#[inline]
#[must_use]
pub fn strip_utf8_bom(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text)
}

/// Infers a standard IANA HTTP MIME media type from a target filename extension.
#[must_use]
pub fn infer_mime_type(filename: &str) -> &'static str {
    let lower = filename.trim().to_ascii_lowercase();
    if lower.ends_with(".json") || lower.ends_with(".mws") {
        "application/json"
    } else if lower.ends_with(".csv") {
        "text/csv"
    } else if lower.ends_with(".tsv") {
        "text/tab-separated-values"
    } else if lower.ends_with(".txt") || lower.ends_with(".frag") || lower.ends_with(".glsl") {
        "text/plain"
    } else if lower.ends_with(".svg") {
        "image/svg+xml"
    } else if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".gif") {
        "image/gif"
    } else if lower.ends_with(".stl") {
        "model/stl"
    } else if lower.ends_with(".ply") {
        "model/ply"
    } else if lower.ends_with(".obj") {
        "model/obj"
    } else {
        "application/octet-stream"
    }
}

fn clean_leaf_filename(raw: &str) -> Option<String> {
    let stripped: String = raw.chars().filter(|c| !c.is_control()).collect();
    let trimmed = stripped.trim().trim_matches(['/', '\\']);
    let leaf = trimmed.rsplit(['/', '\\']).next()?.trim();
    if leaf.is_empty() || leaf == "." || leaf == ".." || leaf.ends_with(':') {
        return None;
    }
    std::path::Path::new(leaf)
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "." && *s != "..")
        .map(ToOwned::to_owned)
}

/// Normalizes a candidate download filename to a safe, non-empty leaf name without path components.
#[must_use]
pub fn sanitize_download_filename(filename: &str, fallback: &str) -> String {
    clean_leaf_filename(filename)
        .or_else(|| clean_leaf_filename(fallback))
        .unwrap_or_else(|| "download.bin".to_string())
}

#[cfg(target_arch = "wasm32")]
#[path = "wasm_file_io.rs"]
mod wasm_file_io;

#[cfg(target_arch = "wasm32")]
pub use wasm_file_io::{
    PickerOutcome, download_browser_file, open_browser_file_picker, request_browser_repaint,
};

/// Triggers a CSV/TSV file import dialog scoped to a specific target `plot_id`.
pub fn trigger_import_csv_for_plot(ctx: &egui::Context, plot_id: egui::Id) {
    let ctx_clone = ctx.clone();

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("CSV / TSV File", &["csv", "tsv", "txt"])
                .pick_file()
            {
                let name = sanitize_download_filename(&path.to_string_lossy(), "dataset.csv");
                let res = std::fs::read_to_string(&path)
                    .map(|content| (name, content))
                    .map_err(|e| format!("Failed to read CSV file: {e}"));
                process_plot_csv_import(&ctx_clone, plot_id, res);
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        open_browser_file_picker(
            ".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain",
            move |outcome| match outcome {
                Some(res) => {
                    process_plot_csv_import(&ctx_clone, plot_id, res);
                }
                None => {
                    ctx_clone.request_repaint();
                }
            },
        );
    }
}

/// Triggers a cross-platform file save dialog on desktop or immediate browser download on WebAssembly.
pub fn save_file_dialog(filename: &str, content: &[u8], _filter_name: &str, _filter_ext: &[&str]) {
    #[cfg(test)]
    {
        let _ = (filename, content, _filter_name, _filter_ext);
    }

    #[cfg(all(not(test), not(target_arch = "wasm32")))]
    {
        if std::env::var_os("CI").is_some() || std::env::var_os("RUST_TEST_THREADS").is_some() {
            return;
        }

        let safe_name = sanitize_download_filename(filename, "export.bin");
        let builder = rfd::FileDialog::new()
            .set_file_name(&safe_name)
            .add_filter(_filter_name, _filter_ext);
        if let Some(path) = builder.save_file() {
            let _ = std::fs::write(path, content);
        }
    }

    #[cfg(all(not(test), target_arch = "wasm32"))]
    {
        let _ = download_browser_file(filename, content);
    }
}

#[cfg(test)]
#[path = "file_io_tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "file_io_wasm_tests.rs"]
mod wasm_tests;
