use super::{
    PickerOutcome, download_browser_file, drain_import_status, drain_imported_datasets,
    open_browser_file_picker, request_browser_repaint, trigger_import_csv_for_plot,
};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast as _;
use wasm_bindgen::JsValue;
use wasm_bindgen::closure::Closure;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::{Document, Event, HtmlElement, HtmlInputElement};

fn dom_document() -> Document {
    web_sys::window()
        .expect("browser window must exist")
        .document()
        .expect("browser document must exist")
}

fn find_file_input(doc: &Document, accept_substr: &str) -> HtmlInputElement {
    let selector = format!("input[type='file'][accept*='{accept_substr}']");
    doc.query_selector(&selector)
        .expect("query_selector must succeed")
        .expect("hidden file input must exist in document body")
        .dyn_into::<HtmlInputElement>()
        .expect("queried element must be HtmlInputElement")
}

fn attach_synthetic_file(input: &HtmlInputElement, filename: &str, content: &str) {
    let win = web_sys::window().expect("window");
    let bytes = js_sys::Uint8Array::from(content.as_bytes());
    let parts = js_sys::Array::of1(&bytes);
    let file_ctor = js_sys::Reflect::get(&win, &JsValue::from_str("File"))
        .expect("File constructor")
        .dyn_into::<js_sys::Function>()
        .expect("File must be constructor");
    let file_args = js_sys::Array::of2(&parts, &JsValue::from_str(filename));
    let file = js_sys::Reflect::construct(&file_ctor, &file_args).expect("new File(...)");

    let dt_ctor = js_sys::Reflect::get(&win, &JsValue::from_str("DataTransfer"))
        .expect("DataTransfer constructor")
        .dyn_into::<js_sys::Function>()
        .expect("DataTransfer must be constructor");
    let dt = js_sys::Reflect::construct(&dt_ctor, &js_sys::Array::new()).expect("new DataTransfer");
    let items = js_sys::Reflect::get(&dt, &JsValue::from_str("items")).expect("dt.items");
    let add_fn = js_sys::Reflect::get(&items, &JsValue::from_str("add"))
        .expect("dt.items.add")
        .dyn_into::<js_sys::Function>()
        .expect("add function");
    add_fn.call1(&items, &file).expect("dt.items.add(file)");

    let files = js_sys::Reflect::get(&dt, &JsValue::from_str("files")).expect("dt.files");
    js_sys::Reflect::set(input, &JsValue::from_str("files"), &files).expect("set input.files");
}

async fn sleep_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let win = web_sys::window().expect("window");
        let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

#[wasm_bindgen_test]
fn wasm_download_browser_file_and_repaint_restore_canvas_focus() {
    let doc = dom_document();
    let canvas = doc
        .get_element_by_id("the_canvas_id")
        .expect("runner index.html must include #the_canvas_id")
        .dyn_into::<HtmlElement>()
        .expect("#the_canvas_id must be HtmlElement");
    let _ = canvas.blur();

    download_browser_file("../nested/plot_points.csv", b"x,y\n1.0,2.5\n")
        .expect("browser Blob download should succeed");
    assert!(
        doc.query_selector("a[download]")
            .expect("query_selector")
            .is_none(),
        "temporary download anchor must be detached immediately after click"
    );
    assert_eq!(
        doc.active_element().and_then(|el| el.get_attribute("id")),
        Some("the_canvas_id".to_string())
    );

    let _ = canvas.blur();
    let pageshow_count = Rc::new(RefCell::new(0_usize));
    let count_ref = Rc::clone(&pageshow_count);
    let win = web_sys::window().expect("window");
    let listener = Closure::<dyn FnMut(Event)>::new(move |_evt: Event| {
        *count_ref.borrow_mut() += 1;
    });
    win.add_event_listener_with_callback("pageshow", listener.as_ref().unchecked_ref())
        .expect("register pageshow listener");

    request_browser_repaint();
    let _ = win.remove_event_listener_with_callback("pageshow", listener.as_ref().unchecked_ref());

    assert_eq!(*pageshow_count.borrow(), 1);
    assert_eq!(
        doc.active_element().and_then(|el| el.get_attribute("id")),
        Some("the_canvas_id".to_string())
    );
}

#[wasm_bindgen_test]
async fn wasm_file_picker_reads_utf8_bom_csv_and_cleans_up_dom() {
    let doc = dom_document();
    let outcome_slot: Rc<RefCell<Option<PickerOutcome>>> = Rc::new(RefCell::new(None));
    let slot_for_cb = Rc::clone(&outcome_slot);

    open_browser_file_picker(".csv", move |outcome| {
        *slot_for_cb.borrow_mut() = Some(outcome);
    });

    let input = find_file_input(&doc, ".csv");
    assert_eq!(input.get_attribute("aria-hidden").as_deref(), Some("true"));
    assert_eq!(input.get_attribute("tabindex").as_deref(), Some("-1"));

    attach_synthetic_file(
        &input,
        "nested/bom_wave.csv",
        "\u{feff}x,y\n0.0,1.5\n2.0,4.5\n",
    );
    let change_evt = Event::new("change").expect("change event");
    input
        .dispatch_event(&change_evt)
        .expect("dispatch change event");

    for _ in 0..20 {
        if outcome_slot.borrow().is_some() {
            break;
        }
        sleep_ms(10).await;
    }

    assert!(
        doc.query_selector("input[type='file'][accept='.csv']")
            .expect("query_selector")
            .is_none(),
        "hidden file input must be removed after selection"
    );
    let delivered = outcome_slot
        .borrow_mut()
        .take()
        .expect("picker callback must fire")
        .expect("selection must not be None")
        .expect("FileReader should decode UTF-8 text");
    assert_eq!(delivered.0, "bom_wave.csv");
    assert_eq!(delivered.1, "x,y\n0.0,1.5\n2.0,4.5\n");
}

#[wasm_bindgen_test]
async fn wasm_file_picker_cancel_and_plot_csv_import_queue() {
    let doc = dom_document();
    let cancel_slot: Rc<RefCell<Option<PickerOutcome>>> = Rc::new(RefCell::new(None));
    let slot_for_cancel = Rc::clone(&cancel_slot);
    open_browser_file_picker(".tsv", move |outcome| {
        *slot_for_cancel.borrow_mut() = Some(outcome);
    });

    let cancel_input = find_file_input(&doc, ".tsv");
    let cancel_evt = Event::new("cancel").expect("cancel event");
    cancel_input
        .dispatch_event(&cancel_evt)
        .expect("dispatch cancel event");

    assert_eq!(*cancel_slot.borrow(), Some(None));
    assert!(
        doc.query_selector("input[type='file'][accept='.tsv']")
            .expect("query_selector")
            .is_none()
    );

    let ctx = egui::Context::default();
    let plot_id = egui::Id::new("wasm_browser_csv_import_plot");
    let _ = drain_imported_datasets(plot_id);
    let _ = drain_import_status(plot_id);

    trigger_import_csv_for_plot(&ctx, plot_id);
    let plot_input = find_file_input(&doc, "text/csv");
    attach_synthetic_file(&plot_input, "dir/signal.csv", "t,amp\n0.0,1.25\n1.0,2.75\n");
    plot_input
        .dispatch_event(&Event::new("change").expect("change event"))
        .expect("dispatch plot import change");

    let mut imported = Vec::new();
    for _ in 0..20 {
        imported = drain_imported_datasets(plot_id);
        if !imported.is_empty() {
            break;
        }
        sleep_ms(10).await;
    }

    assert_eq!(imported.len(), 1);
    assert_eq!(imported[0].points.len(), 2);
    let (status_msg, is_err) = drain_import_status(plot_id).expect("import status must be queued");
    assert!(!is_err);
    assert!(status_msg.contains("signal.csv"));
}
