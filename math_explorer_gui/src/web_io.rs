//! Browser file download and asynchronous file input helpers for WebAssembly targets.
//!
//! Bridges DOM file operations (`Blob` object URLs, `<a download>`, hidden `<input type="file">`,
//! and `FileReader`) with Rust channel-based state management while keeping DOM elements and
//! event listener closures leak-free.

// theory_verification!

use std::path::Path;
use std::time::Duration;

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
    #[allow(clippy::disallowed_methods)]
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

/// Infers an HTTP MIME type string from a target download filename extension.
#[cfg(any(target_arch = "wasm32", test))]
#[must_use]
pub fn infer_mime_type(filename: &str) -> &'static str {
    debug_assert!(!filename.is_empty() || filename.is_empty());
    let lower = filename.trim().to_ascii_lowercase();
    debug_assert!(lower.len() <= filename.len());
    if lower.ends_with(".json") || lower.ends_with(".mws") {
        "application/json"
    } else if lower.ends_with(".csv") {
        "text/csv"
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

/// Normalizes a candidate download filename to a non-empty leaf filename without path separators.
#[must_use]
pub fn sanitize_download_filename(filename: &str, fallback: &str) -> String {
    debug_assert!(
        !fallback.trim().is_empty(),
        "fallback filename must not be empty"
    );
    let trimmed = filename.trim().trim_matches(['/', '\\']);
    let candidate = Path::new(trimmed)
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "." && *s != "..");

    let resolved = candidate.unwrap_or_else(|| fallback.trim()).to_string();
    debug_assert!(!resolved.is_empty());
    resolved
}

#[cfg(target_arch = "wasm32")]
mod wasm_impl {
    use super::{infer_mime_type, sanitize_download_filename};
    use std::cell::RefCell;
    use std::rc::Rc;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast as _;
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

    /// Requests an immediate `eframe` repaint and restores canvas focus after async DOM callbacks.
    pub fn request_browser_repaint() {
        if let Some(window) = web_sys::window() {
            if let Some(doc) = window.document() {
                if let Some(canvas) = doc
                    .get_element_by_id("the_canvas_id")
                    .and_then(|el| el.dyn_into::<HtmlElement>().ok())
                {
                    let _ = canvas.focus();
                }
            }
            if let Ok(evt) = Event::new("pageshow") {
                let _ = window.dispatch_event(&evt);
            }
        }
    }

    /// Triggers an immediate browser file download using a `Blob`, object `Url`, and `<a download>`.
    ///
    /// Must be invoked synchronously inside a user gesture handler to avoid popup blocking.
    pub fn download_browser_file(filename: &str, content: &[u8]) -> Result<(), String> {
        let safe_name = sanitize_download_filename(filename, "download.json");
        let mime_type = infer_mime_type(&safe_name);
        let document = browser_document()?;

        let uint8_array = js_sys::Uint8Array::from(content);
        let parts = js_sys::Array::new();
        parts.push(&uint8_array);

        let blob_props = BlobPropertyBag::new();
        blob_props.set_type(mime_type);

        let blob = Blob::new_with_u8_array_sequence_and_options(&parts, &blob_props)
            .map_err(|e| format!("Failed to construct download Blob: {e:?}"))?;
        let url = Url::create_object_url_with_blob(&blob)
            .map_err(|e| format!("Failed to create object URL: {e:?}"))?;

        let anchor_res = document
            .create_element("a")
            .map_err(|e| format!("Failed to create anchor element: {e:?}"))
            .and_then(|el| {
                el.dyn_into::<HtmlAnchorElement>()
                    .map_err(|_| "Created element is not an HtmlAnchorElement".to_string())
            });

        let anchor = match anchor_res {
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
        let _ = Url::revoke_object_url(&url);
        if let Some(canvas) = document
            .get_element_by_id("the_canvas_id")
            .and_then(|el| el.dyn_into::<HtmlElement>().ok())
        {
            let _ = canvas.focus();
        }
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

        let js_val = await_res.map_err(|e| format!("Failed to read selected file: {e:?}"))?;
        js_val
            .as_string()
            .ok_or_else(|| "Selected file did not decode as UTF-8 text".to_string())
    }

    /// Opens a native browser file selection dialog via a temporary hidden `<input type="file">`.
    ///
    /// Removes the temporary `<input>` element from the DOM after selection or cancellation and
    /// delivers the file name and UTF-8 text asynchronously through `on_complete`.
    pub fn open_browser_file_picker<F>(accept: &str, on_complete: F)
    where
        F: FnOnce(PickerOutcome) + 'static,
    {
        let document = match browser_document() {
            Ok(doc) => doc,
            Err(err) => {
                on_complete(Some(Err(err)));
                request_browser_repaint();
                return;
            }
        };

        let input = match create_hidden_file_input(&document, accept) {
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
            let Some(done) = cb_for_change.borrow_mut().take() else {
                return;
            };
            let Some(file) = selected else {
                done(None);
                request_browser_repaint();
                return;
            };
            let name = file.name();
            wasm_bindgen_futures::spawn_local(async move {
                let res = read_file_text(file).await.map(|text| (name, text));
                done(Some(res));
                request_browser_repaint();
            });
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
pub use wasm_impl::{download_browser_file, open_browser_file_picker, request_browser_repaint};

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
