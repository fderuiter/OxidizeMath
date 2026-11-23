//! WebAssembly browser DOM file picker and download helpers for `egui_plot`.

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
        let _ = self
            .input
            .remove_event_listener_with_callback("change", self.on_change.as_ref().unchecked_ref());
        let _ = self
            .input
            .remove_event_listener_with_callback("cancel", self.on_cancel.as_ref().unchecked_ref());
        self.input.remove();
    }
}

fn browser_document() -> Result<Document, String> {
    let window = web_sys::window().ok_or_else(|| "Browser window is unavailable".to_string())?;
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
        .set_timeout_with_callback_and_timeout_and_arguments_0(cb.as_ref().unchecked_ref(), 1_000)
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

fn create_hidden_file_input(document: &Document, accept: &str) -> Result<HtmlInputElement, String> {
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

fn ensure_file_within_size_limit(file: &File) -> Result<(), String> {
    let byte_size = file.size();
    if byte_size > crate::csv_import::MAX_CSV_IMPORT_BYTES as f64 {
        return Err(format!(
            "Selected file ({:.2} MiB) exceeds maximum allowed size of 2.0 MiB.",
            byte_size / (1024.0 * 1024.0)
        ));
    }
    Ok(())
}

async fn read_file_text(file: File) -> Result<String, String> {
    ensure_file_within_size_limit(&file)?;
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
    let input = match browser_document().and_then(|doc| create_hidden_file_input(&doc, accept)) {
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

    let _ = input.add_event_listener_with_callback("change", on_change.as_ref().unchecked_ref());
    let _ = input.add_event_listener_with_callback("cancel", on_cancel.as_ref().unchecked_ref());

    *cleanup_slot.borrow_mut() = Some(PickerCleanup {
        input: input.clone(),
        on_change,
        on_cancel,
    });

    input.click();
}
