#![doc = include_str!("../README.md")]
#[allow(missing_docs)]
pub mod accessibility;
mod app;
#[allow(missing_docs)]
pub mod async_sim;
#[allow(missing_docs)]
pub mod camera;
#[allow(missing_docs)]
pub mod framework;
#[cfg(test)]
mod framework_tests;
pub mod presets;
#[allow(missing_docs)]
pub mod reflective_ui;
mod tabs;

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
// theory_verification!
fn main() {
    console_error_panic_hook::set_once();
    diagnostics::init_panic_hook();
    let _ = federated_registry::global_registry();
    accessibility::init_accessibility_bridge();
    wasm_bindgen_futures::spawn_local(async {
        use wasm_bindgen::JsCast;
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id("the_canvas_id")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        let runner = eframe::WebRunner::new();
        let web_options = eframe::WebOptions::default();
        runner
            .start(
                canvas,
                web_options,
                Box::new(|cc| {
                    cc.egui_ctx.options_mut(|o| o.screen_reader = true);
                    Ok(Box::new(MathExplorerApp::new(cc)))
                }),
            )
            .await
            .expect("failed to start eframe");
    });
}
// theory_verification!
