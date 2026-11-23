#![doc = include_str!("../README.md")]
#[allow(missing_docs)]
pub mod accessibility;
mod app;
#[allow(missing_docs)]
pub mod async_sim;
#[allow(missing_docs)]
pub mod camera;
#[cfg(test)]
mod camera_3d_optout_tests;
#[cfg(test)]
mod canvas_accessibility_tests;
#[allow(missing_docs)]
pub mod command_palette;
#[allow(missing_docs)]
pub mod framework;
#[cfg(test)]
mod framework_tests;
#[cfg(test)]
mod interactive_tools_tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod main_wasm_tests;
#[cfg(test)]
mod markup_tests;
#[cfg(test)]
mod modal_accessibility_tests;
pub mod presets;
mod reflective_ui;
#[allow(missing_docs)]
pub mod sensitivity;
pub mod starter_gallery;
mod tabs;
mod telemetry;
pub mod url_state;
pub mod web_io;
#[cfg(target_arch = "wasm32")]
mod webgl_recovery;

pub use egui_plot::{ModalFrameContext, ModalLifecycle};

use app::MathExplorerApp;

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(test)]
pub(crate) static VERIFIED_ALLOC_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    diagnostics::init_panic_hook();
    let _ = federated_registry::global_registry();
    accessibility::init_accessibility_bridge();
    let native_options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Math Explorer"),
        ..Default::default()
    };

    eframe::run_native(
        "Math Explorer",
        native_options,
        Box::new(|cc| {
            cc.egui_ctx.options_mut(|o| o.screen_reader = true);
            Ok(Box::new(MathExplorerApp::new(cc)))
        }),
    )
}

#[cfg(target_arch = "wasm32")]
fn mark_boot_initialized(doc: &web_sys::Document) {
    if let Some(root) = doc.document_element() {
        let _ = root.set_attribute("data-boot-initialized", "true");
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn is_error_boundary_active(doc: &web_sys::Document) -> bool {
    if doc
        .document_element()
        .and_then(|root| root.get_attribute("data-boot-failed"))
        .as_deref()
        == Some("true")
    {
        return true;
    }
    let Some(error_container) = doc.get_element_by_id("error_container") else {
        return false;
    };
    if error_container
        .get_attribute("data-error-active")
        .as_deref()
        == Some("true")
    {
        return true;
    }
    error_container
        .get_attribute("style")
        .is_some_and(|style| style.contains("display: flex") || style.contains("display:flex"))
}

#[cfg(target_arch = "wasm32")]
fn clear_global_wasm_bindings(window: Option<&web_sys::Window>) {
    if let Some(win) = window {
        let _ = js_sys::Reflect::delete_property(
            win.as_ref(),
            &wasm_bindgen::JsValue::from_str("wasmBindings"),
        );
    }
}

#[cfg(target_arch = "wasm32")]
fn notify_js_boot_complete(window: Option<&web_sys::Window>) {
    let Some(win) = window else {
        return;
    };
    let Ok(boot) = js_sys::Reflect::get(
        win.as_ref(),
        &wasm_bindgen::JsValue::from_str("__mathExplorerBoot"),
    ) else {
        return;
    };
    if boot.is_undefined() || boot.is_null() {
        return;
    }
    if let Ok(mark_fn) =
        js_sys::Reflect::get(&boot, &wasm_bindgen::JsValue::from_str("markBootComplete"))
    {
        use wasm_bindgen::JsCast;
        if let Ok(func) = mark_fn.dyn_into::<js_sys::Function>() {
            let _ = func.call0(&boot);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn bind_fallback_reload_listener(doc: &web_sys::Document) {
    use wasm_bindgen::JsCast;
    let Some(reload_btn) = doc.get_element_by_id("webgl_reload_button") else {
        return;
    };
    if reload_btn.get_attribute("data-reload-bound").as_deref() == Some("true") {
        return;
    }
    let _ = reload_btn.set_attribute("data-reload-bound", "true");
    let listener = wasm_bindgen::closure::Closure::wrap(Box::new(|event: web_sys::Event| {
        event.prevent_default();
        if let Some(win) = web_sys::window() {
            let _ = win.location().reload();
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    let _ = reload_btn.add_event_listener_with_callback("click", listener.as_ref().unchecked_ref());
    listener.forget();
}

#[cfg(target_arch = "wasm32")]
fn select_webgl_error_category(doc: &web_sys::Document) {
    if let Some(overlay) = doc.get_element_by_id("webgl_recovery_overlay") {
        let _ = overlay.set_attribute("aria-busy", "false");
        let _ = overlay.set_attribute("hidden", "");
        let _ = overlay.set_attribute("style", "display: none;");
    }
    if let Some(error_container) = doc.get_element_by_id("error_container") {
        let _ = error_container.set_attribute("data-error-category", "webgl");
        let _ = error_container.set_attribute("data-error-active", "true");
    }
    if let Some(title_el) = doc.get_element_by_id("error_title") {
        title_el.set_text_content(Some("WebGL Initialization Failed"));
    }
    if let Some(msg_el) = doc.get_element_by_id("error_message") {
        msg_el.set_text_content(Some(
            "Math Explorer requires WebGL and hardware acceleration to render mathematics and interactive simulations. Your browser was unable to initialize a WebGL rendering context.",
        ));
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn handle_webgl_failure(document: Option<&web_sys::Document>, error_info: &str) {
    accessibility::announce_status_with_priority(
        "WebGL initialization failed. Please enable hardware acceleration in your browser settings.",
        "assertive",
    );

    if let Some(doc) = document {
        mark_boot_initialized(doc);
        if let Some(root) = doc.document_element() {
            let _ = root.set_attribute("data-boot-failed", "true");
        }

        if let Some(loading) = doc.get_element_by_id("loading_indicator") {
            let _ = loading.set_attribute("aria-busy", "false");
            loading.remove();
        }

        if let Some(canvas) = doc.get_element_by_id("the_canvas_id") {
            let _ = canvas.set_attribute("style", "display: none;");
        }

        select_webgl_error_category(doc);

        if let Some(error_container) = doc.get_element_by_id("error_container") {
            let _ = error_container.set_attribute("style", "display: flex;");
        }

        if let Some(details_el) = doc.get_element_by_id("error_details") {
            details_el.set_text_content(Some(error_info));
            let _ = details_el.set_attribute("style", "display: block;");
        }

        bind_fallback_reload_listener(doc);

        use wasm_bindgen::JsCast;
        if let Some(reload_btn) = doc
            .get_element_by_id("webgl_reload_button")
            .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
        {
            let _ = reload_btn.focus();
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn finish_wasm_startup(
    doc: &web_sys::Document,
    window: Option<&web_sys::Window>,
    canvas: &web_sys::HtmlCanvasElement,
    runner: eframe::WebRunner,
    start_res: Result<(), wasm_bindgen::JsValue>,
) {
    use wasm_bindgen::JsCast;

    if is_error_boundary_active(doc) {
        runner.destroy();
        clear_global_wasm_bindings(window);
        return;
    }

    match start_res {
        Ok(()) => {
            mark_boot_initialized(doc);
            notify_js_boot_complete(window);

            if let Some(loading) = doc.get_element_by_id("loading_indicator") {
                let _ = loading.set_attribute("aria-busy", "false");
                loading.remove();
            }

            webgl_recovery::install_webgl_recovery(canvas, doc, window, runner);

            if let Some(canvas_el) = doc
                .get_element_by_id("the_canvas_id")
                .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
            {
                let _ = canvas_el.focus();
            }
        }
        Err(err) => {
            runner.destroy();
            let err_msg = format!("{err:?}");
            eprintln!("Failed to start eframe WebRunner: {err_msg}");
            handle_webgl_failure(Some(doc), &err_msg);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {
    let initial_window = web_sys::window();
    if let Some(doc) = initial_window.as_ref().and_then(web_sys::Window::document) {
        mark_boot_initialized(&doc);
        bind_fallback_reload_listener(&doc);
        if is_error_boundary_active(&doc) {
            clear_global_wasm_bindings(initial_window.as_ref());
            return;
        }
    }

    console_error_panic_hook::set_once();
    diagnostics::init_panic_hook();
    let _ = federated_registry::global_registry();
    accessibility::init_accessibility_bridge();
    wasm_bindgen_futures::spawn_local(async {
        use wasm_bindgen::JsCast;

        let window = web_sys::window();
        let document = window.as_ref().and_then(web_sys::Window::document);

        let Some(ref doc) = document else {
            eprintln!("Failed to obtain DOM Document during initialization.");
            return;
        };

        if is_error_boundary_active(doc) {
            clear_global_wasm_bindings(window.as_ref());
            return;
        }

        let canvas_el = doc.get_element_by_id("the_canvas_id");
        let canvas = match canvas_el.and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
        {
            Some(canvas) => canvas,
            None => {
                eprintln!("Canvas element with id 'the_canvas_id' not found or invalid.");
                handle_webgl_failure(
                    Some(doc),
                    "Canvas element 'the_canvas_id' not found in DOM.",
                );
                return;
            }
        };

        let runner = eframe::WebRunner::new();
        let web_options = eframe::WebOptions::default();

        let start_res = runner
            .start(
                canvas.clone(),
                web_options,
                Box::new(|cc| {
                    cc.egui_ctx.options_mut(|o| o.screen_reader = true);
                    Ok(Box::new(MathExplorerApp::new(cc)))
                }),
            )
            .await;

        finish_wasm_startup(doc, window.as_ref(), &canvas, runner, start_res);
    });
}
