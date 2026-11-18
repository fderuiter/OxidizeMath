#![doc = include_str!("../README.md")]
#[allow(missing_docs)]
pub mod accessibility;
mod app;
#[allow(missing_docs)]
pub mod async_sim;
#[allow(missing_docs)]
pub mod camera;
#[allow(missing_docs)]
pub mod command_palette;
#[allow(missing_docs)]
pub mod framework;
#[cfg(test)]
mod framework_tests;
#[cfg(test)]
mod interactive_tools_tests;
#[cfg(test)]
mod markup_tests;
pub mod presets;
#[allow(missing_docs)]
pub mod reflective_ui;
#[allow(missing_docs)]
pub mod sensitivity;
pub mod starter_gallery;
mod tabs;
pub mod web_io;

use app::MathExplorerApp;

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
fn handle_webgl_failure(document: Option<&web_sys::Document>, error_info: &str) {
    accessibility::announce_status_with_priority(
        "WebGL initialization failed. Please enable hardware acceleration in your browser settings.",
        "assertive",
    );

    if let Some(doc) = document {
        if let Some(loading) = doc.get_element_by_id("loading_indicator") {
            loading.remove();
        }

        if let Some(canvas) = doc.get_element_by_id("the_canvas_id") {
            let _ = canvas.set_attribute("style", "display: none;");
        }

        if let Some(error_container) = doc.get_element_by_id("error_container") {
            let _ = error_container.set_attribute("style", "display: flex;");
        }

        if let Some(details_el) = doc.get_element_by_id("error_details") {
            details_el.set_text_content(Some(error_info));
            let _ = details_el.set_attribute("style", "display: block;");
        }

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
// theory_verification!
fn main() {
    console_error_panic_hook::set_once();
    diagnostics::init_panic_hook();
    let _ = federated_registry::global_registry();
    accessibility::init_accessibility_bridge();
    wasm_bindgen_futures::spawn_local(async {
        use wasm_bindgen::JsCast;

        let window = web_sys::window();
        let document = window.as_ref().and_then(|w| w.document());

        let Some(ref doc) = document else {
            eprintln!("Failed to obtain DOM Document during initialization.");
            return;
        };

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
                canvas,
                web_options,
                Box::new(|cc| {
                    cc.egui_ctx.options_mut(|o| o.screen_reader = true);
                    Ok(Box::new(MathExplorerApp::new(cc)))
                }),
            )
            .await;

        match start_res {
            Ok(()) => {
                if let Some(loading) = doc.get_element_by_id("loading_indicator") {
                    loading.remove();
                }

                if let Some(canvas) = doc
                    .get_element_by_id("the_canvas_id")
                    .and_then(|el| el.dyn_into::<web_sys::HtmlElement>().ok())
                {
                    let _ = canvas.focus();
                }
            }
            Err(err) => {
                let err_msg = format!("{err:?}");
                eprintln!("Failed to start eframe WebRunner: {err_msg}");
                handle_webgl_failure(Some(doc), &err_msg);
            }
        }
    });
}
// theory_verification!
