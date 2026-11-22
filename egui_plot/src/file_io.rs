//! Cross-platform and WebAssembly browser file I/O helpers for downloads and text file imports.
//!
//! Centralizes MIME inference, download filename sanitization, UTF-8 BOM stripping,
//! hidden `<input type="file">` + `FileReader` browser pickers, `Blob` + `<a download>`
//! browser file downloads, and plot-scoped asynchronous CSV dataset import queues.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock, PoisonError};

use crate::export::{ImportedDataset, parse_csv};

/// Maximum number of unconsumed plot CSV import payloads or status messages retained globally.
const MAX_PENDING_PLOT_IMPORTS: usize = 64;

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
    } else if lower.ends_with(".txt") {
        "text/plain"
    } else if lower.ends_with(".tex") {
        "text/x-tex"
    } else if lower.ends_with(".rs") {
        "text/x-rust"
    } else if lower.ends_with(".py") {
        "text/x-python"
    } else if lower.ends_with(".svg") {
        "image/svg+xml"
    } else if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".gif") {
        "image/gif"
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
mod wasm_file_io {
    use super::{infer_mime_type, sanitize_download_filename, strip_utf8_bom};
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::JsCast as _;
    use wasm_bindgen::closure::Closure;
    use web_sys::{
        Blob, BlobPropertyBag, Document, Event, File, FileReader, HtmlAnchorElement, HtmlElement,
        HtmlInputElement, Url,
    };

    /// Outcome delivered by [`open_browser_file_picker`]: `None` on dialog dismissal, or
    /// `Some(Ok((file_name, text)))` / `Some(Err(reason))` on selection.
    pub type PickerOutcome = Option<Result<(String, String), String>>;

    struct PickerCleanup {
        input: HtmlInputElement,
        on_change: Closure<dyn FnMut(Event)>,
        on_cancel: Closure<dyn FnMut(Event)>,
    }

    impl PickerCleanup {
        fn detach_and_remove(self) {
            let _ = self.input.remove_event_listener_with_callback(
                "change",
                self.on_change.as_ref().unchecked_ref(),
            );
            let _ = self.input.remove_event_listener_with_callback(
                "cancel",
                self.on_cancel.as_ref().unchecked_ref(),
            );
            self.input.remove();
        }
    }

    fn browser_document() -> Result<Document, String> {
        let window =
            web_sys::window().ok_or_else(|| "Browser window is unavailable".to_string())?;
        window
            .document()
            .ok_or_else(|| "Browser document is unavailable".to_string())
    }

    fn focus_canvas(document: &Document) {
        if let Some(canvas) = document
            .get_element_by_id("the_canvas_id")
            .and_then(|el| el.dyn_into::<HtmlElement>().ok())
        {
            let _ = canvas.focus();
        }
    }

    /// Requests an immediate browser frame repaint and restores canvas focus after DOM dialogs.
    pub fn request_browser_repaint() {
        if let Some(window) = web_sys::window() {
            if let Some(doc) = window.document() {
                focus_canvas(&doc);
            }
            if let Ok(evt) = Event::new("pageshow") {
                let _ = window.dispatch_event(&evt);
            }
        }
    }

    fn create_blob_object_url(content: &[u8], mime_type: &str) -> Result<String, String> {
        let uint8_array = js_sys::Uint8Array::from(content);
        let parts = js_sys::Array::new();
        parts.push(&uint8_array);

        let blob_props = BlobPropertyBag::new();
        blob_props.set_type(mime_type);

        let blob = Blob::new_with_u8_array_sequence_and_options(&parts, &blob_props)
            .map_err(|e| format!("Failed to construct download Blob: {e:?}"))?;
        Url::create_object_url_with_blob(&blob)
            .map_err(|e| format!("Failed to create object URL: {e:?}"))
    }

    type TimerClosure = Closure<dyn FnMut()>;
    type TimerClosureSlot = Rc<RefCell<Option<TimerClosure>>>;

    fn schedule_url_revocation(url: String) {
        let Some(window) = web_sys::window() else {
            let _ = Url::revoke_object_url(&url);
            return;
        };
        let slot: TimerClosureSlot = Rc::new(RefCell::new(None));
        let slot_for_cb = Rc::clone(&slot);
        let url_for_cb = url.clone();
        let cb = TimerClosure::new(move || {
            let _ = Url::revoke_object_url(&url_for_cb);
            let _ = slot_for_cb.borrow_mut().take();
        });
        if window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                1_000,
            )
            .is_ok()
        {
            *slot.borrow_mut() = Some(cb);
        } else {
            let _ = Url::revoke_object_url(&url);
        }
    }

    /// Triggers a browser file download using a `Blob`, object `Url`, and hidden `<a download>`.
    ///
    /// Defers `Url::revoke_object_url` via a self-dropping timer callback so asynchronous browser
    /// download managers reliably read the `Blob` before object URL cleanup.
    pub fn download_browser_file(filename: &str, content: &[u8]) -> Result<(), String> {
        let safe_name = sanitize_download_filename(filename, "download.bin");
        let mime_type = infer_mime_type(&safe_name);
        let document = browser_document()?;
        let url = create_blob_object_url(content, mime_type)?;

        let anchor = match document
            .create_element("a")
            .map_err(|e| format!("Failed to create anchor element: {e:?}"))
            .and_then(|el| {
                el.dyn_into::<HtmlAnchorElement>()
                    .map_err(|_| "Created element is not an HtmlAnchorElement".to_string())
            }) {
            Ok(a) => a,
            Err(err) => {
                let _ = Url::revoke_object_url(&url);
                return Err(err);
            }
        };

        let _ = anchor.set_attribute("style", "display: none;");
        anchor.set_href(&url);
        anchor.set_download(&safe_name);
        if let Some(body) = document.body() {
            let _ = body.append_child(&anchor);
        }
        anchor.click();
        anchor.remove();
        schedule_url_revocation(url);
        focus_canvas(&document);
        Ok(())
    }

    fn create_hidden_file_input(
        document: &Document,
        accept: &str,
    ) -> Result<HtmlInputElement, String> {
        let body = document
            .body()
            .ok_or_else(|| "Browser document body is unavailable".to_string())?;
        let input = document
            .create_element("input")
            .map_err(|e| format!("Failed to create file input element: {e:?}"))?
            .dyn_into::<HtmlInputElement>()
            .map_err(|_| "Created element is not an HtmlInputElement".to_string())?;

        input.set_type("file");
        if !accept.trim().is_empty() {
            input.set_accept(accept);
        }
        let _ = input.set_attribute("style", "display: none;");
        let _ = input.set_attribute("aria-hidden", "true");
        let _ = input.set_attribute("tabindex", "-1");
        body.append_child(&input)
            .map_err(|e| format!("Failed to append file input to document body: {e:?}"))?;
        Ok(input)
    }

    async fn read_file_text(file: File) -> Result<String, String> {
        let reader =
            FileReader::new().map_err(|e| format!("Failed to initialize FileReader: {e:?}"))?;
        let mut onload_slot: Option<Closure<dyn FnMut(Event)>> = None;
        let mut onerror_slot: Option<Closure<dyn FnMut(Event)>> = None;
        let reader_for_promise = reader.clone();
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            let load_reader = reader_for_promise.clone();
            let load_reject = reject.clone();
            let onload =
                Closure::<dyn FnMut(Event)>::new(move |_evt: Event| match load_reader.result() {
                    Ok(val) => {
                        let _ = resolve.call1(&wasm_bindgen::JsValue::NULL, &val);
                    }
                    Err(err) => {
                        let _ = load_reject.call1(&wasm_bindgen::JsValue::NULL, &err);
                    }
                });
            let onerror = Closure::<dyn FnMut(Event)>::new(move |_evt: Event| {
                let _ = reject.call1(
                    &wasm_bindgen::JsValue::NULL,
                    &wasm_bindgen::JsValue::from_str("Browser failed to read selected file"),
                );
            });
            reader_for_promise.set_onload(Some(onload.as_ref().unchecked_ref()));
            reader_for_promise.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            onload_slot = Some(onload);
            onerror_slot = Some(onerror);
        });

        if let Err(e) = reader.read_as_text(&file) {
            reader.set_onload(None);
            reader.set_onerror(None);
            return Err(format!("Failed to start reading selected file: {e:?}"));
        }

        let await_res = wasm_bindgen_futures::JsFuture::from(promise).await;
        reader.set_onload(None);
        reader.set_onerror(None);
        drop(onload_slot);
        drop(onerror_slot);

        let js_val = await_res.map_err(|e| {
            e.as_string()
                .unwrap_or_else(|| format!("Failed to read selected file: {e:?}"))
        })?;
        let raw_text = js_val
            .as_string()
            .ok_or_else(|| "Selected file did not decode as UTF-8 text".to_string())?;
        Ok(strip_utf8_bom(&raw_text).to_string())
    }

    fn spawn_file_read<F>(file: File, done: F)
    where
        F: FnOnce(PickerOutcome) + 'static,
    {
        let name = sanitize_download_filename(&file.name(), "selected_file.txt");
        wasm_bindgen_futures::spawn_local(async move {
            let res = read_file_text(file).await.map(|text| (name, text));
            done(Some(res));
            request_browser_repaint();
        });
    }

    /// Opens a native browser file selection dialog via a temporary hidden `<input type="file">`.
    ///
    /// Removes the temporary `<input>` node and drops event listener closures after `change` or
    /// `cancel`, delivering `None` on dismissal or `Some(Result<(name, text), err>)` on selection.
    pub fn open_browser_file_picker<F>(accept: &str, on_complete: F)
    where
        F: FnOnce(PickerOutcome) + 'static,
    {
        let input = match browser_document().and_then(|doc| create_hidden_file_input(&doc, accept))
        {
            Ok(el) => el,
            Err(err) => {
                on_complete(Some(Err(err)));
                request_browser_repaint();
                return;
            }
        };

        let callback_slot: Rc<RefCell<Option<F>>> = Rc::new(RefCell::new(Some(on_complete)));
        let cleanup_slot: Rc<RefCell<Option<PickerCleanup>>> = Rc::new(RefCell::new(None));

        let input_for_change = input.clone();
        let cb_for_change = Rc::clone(&callback_slot);
        let cl_for_change = Rc::clone(&cleanup_slot);
        let on_change = Closure::<dyn FnMut(Event)>::new(move |_evt: Event| {
            let selected = input_for_change.files().and_then(|list| list.get(0));
            if let Some(cleanup) = cl_for_change.borrow_mut().take() {
                cleanup.detach_and_remove();
            }
            if let Some(done) = cb_for_change.borrow_mut().take() {
                if let Some(file) = selected {
                    spawn_file_read(file, done);
                } else {
                    done(None);
                    request_browser_repaint();
                }
            }
        });

        let cb_for_cancel = Rc::clone(&callback_slot);
        let cl_for_cancel = Rc::clone(&cleanup_slot);
        let on_cancel = Closure::<dyn FnMut(Event)>::new(move |_evt: Event| {
            if let Some(cleanup) = cl_for_cancel.borrow_mut().take() {
                cleanup.detach_and_remove();
            }
            if let Some(done) = cb_for_cancel.borrow_mut().take() {
                done(None);
            }
            request_browser_repaint();
        });

        let _ =
            input.add_event_listener_with_callback("change", on_change.as_ref().unchecked_ref());
        let _ =
            input.add_event_listener_with_callback("cancel", on_cancel.as_ref().unchecked_ref());

        *cleanup_slot.borrow_mut() = Some(PickerCleanup {
            input: input.clone(),
            on_change,
            on_cancel,
        });

        input.click();
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm_file_io::{
    PickerOutcome, download_browser_file, open_browser_file_picker, request_browser_repaint,
};

type PendingImport = (egui::Id, Vec<ImportedDataset>);
type PendingImportStatus = (egui::Id, String, bool);

fn pending_imports_queue() -> &'static Mutex<VecDeque<PendingImport>> {
    static QUEUE: OnceLock<Mutex<VecDeque<PendingImport>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn pending_status_queue() -> &'static Mutex<VecDeque<PendingImportStatus>> {
    static QUEUE: OnceLock<Mutex<VecDeque<PendingImportStatus>>> = OnceLock::new();
    QUEUE.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn push_bounded<T>(queue: &Mutex<VecDeque<T>>, item: T) {
    let mut guard = queue.lock().unwrap_or_else(PoisonError::into_inner);
    if guard.len() >= MAX_PENDING_PLOT_IMPORTS {
        guard.pop_front();
    }
    guard.push_back(item);
}

/// Drains any asynchronously imported datasets queued for `plot_id` (preserving sibling plots' entries).
#[must_use]
pub fn drain_imported_datasets(plot_id: egui::Id) -> Vec<ImportedDataset> {
    let mut datasets = Vec::new();
    let mut guard = pending_imports_queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let mut remaining = VecDeque::with_capacity(guard.len());
    for (id, ds_list) in guard.drain(..) {
        if id == plot_id || id == egui::Id::NULL {
            datasets.extend(ds_list);
        } else {
            remaining.push_back((id, ds_list));
        }
    }
    *guard = remaining;
    datasets
}

/// Drains the latest asynchronous CSV import status `(message, is_error)` for `plot_id`.
#[must_use]
pub fn drain_import_status(plot_id: egui::Id) -> Option<(String, bool)> {
    let mut latest = None;
    let mut guard = pending_status_queue()
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    let mut remaining = VecDeque::with_capacity(guard.len());
    for (id, msg, is_error) in guard.drain(..) {
        if id == plot_id || id == egui::Id::NULL {
            latest = Some((msg, is_error));
        } else {
            remaining.push_back((id, msg, is_error));
        }
    }
    *guard = remaining;
    latest
}

pub(crate) fn process_plot_csv_import(
    ctx: &egui::Context,
    plot_id: egui::Id,
    outcome: Result<(String, String), String>,
) {
    match outcome {
        Ok((name, content)) => {
            let datasets = parse_csv(&content);
            if datasets.is_empty() {
                let msg =
                    format!("Invalid or empty CSV file: no numeric data points found in {name}");
                push_bounded(pending_status_queue(), (plot_id, msg.clone(), true));
                crate::emit_plot_live_message(ctx, msg, true);
            } else {
                let count = datasets.len();
                push_bounded(pending_imports_queue(), (plot_id, datasets));
                let msg = format!("Imported {count} dataset(s) from {name}");
                push_bounded(pending_status_queue(), (plot_id, msg.clone(), false));
                crate::emit_plot_live_message(ctx, msg, false);
            }
        }
        Err(err) => {
            push_bounded(pending_status_queue(), (plot_id, err.clone(), true));
            crate::emit_plot_live_message(ctx, err, true);
        }
    }
    ctx.request_repaint();
}

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
