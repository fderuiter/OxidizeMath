#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{CUSTOM_POTENTIAL_KEY, WaveData, WaveUnified};
use crate::app::MathExplorerApp;
use crate::async_sim::SimCommand;
use crate::async_sim::unified::UnifiedSimTool;
use crate::url_state::{
    MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES, read_browser_location_parts,
    replace_browser_url_fragment,
};
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

async fn yield_browser_tick() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if let Some(win) = web_sys::window() {
            let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
        } else {
            let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

fn prepare_sculpted_quantum_app() -> (MathExplorerApp, usize, String) {
    let mut app = MathExplorerApp::default();
    let q_idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "quantum")
        .expect("quantum tab must exist");
    app.selected_tab = q_idx;
    assert!(app.tabs[q_idx].select_tool_by_name("WaveSimulator"));

    let sculpted: Vec<f64> = (0..200)
        .map(|i| if (92..=108).contains(&i) { 17.25 } else { -1.5 })
        .collect();
    let custom_json = serde_json::json!({
        "dt": 0.02,
        "k0": 6.5,
        "potential_type": 2.0,
        "sigma": 0.4,
        "x0": -2.5,
        CUSTOM_POTENTIAL_KEY: sculpted,
    })
    .to_string();

    app.tabs[q_idx]
        .framework_mut()
        .unwrap()
        .active_tool
        .as_mut()
        .unwrap()
        .try_load_state(&custom_json)
        .expect("sculpted state must load");
    let envelope = app.tabs[q_idx]
        .export_url_preset_json()
        .unwrap()
        .expect("envelope expected");
    (app, q_idx, envelope)
}

#[wasm_bindgen_test]
async fn wasm_paused_sculpt_stroke_wakes_worker_and_emits_updated_snapshot() {
    let mut tool = UnifiedSimTool::<WaveUnified>::new();
    yield_browser_tick().await;

    tool.controller
        .send_command(SimCommand::SculptStroke1D {
            i0: 95,
            v0: 12.0,
            i1: 105,
            v1: 24.0,
        })
        .expect("SculptStroke1D command should send cleanly");

    for _ in 0..10 {
        yield_browser_tick().await;
        if tool.controller.update().is_some() {
            break;
        }
    }

    let snap = tool
        .controller
        .latest_snapshot()
        .expect("Paused wasm32 worker must emit updated snapshot");
    let wave = snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<WaveData>())
        .expect("WaveData payload expected");
    assert!((wave.potential[95] - 12.0).abs() < 1e-6);
    assert!((wave.potential[100] - 18.0).abs() < 1e-6);
    assert!((wave.potential[105] - 24.0).abs() < 1e-6);
    assert!((wave.p_transmitted + wave.p_reflected - 1.0).abs() < 1e-12);
}

#[wasm_bindgen_test]
fn wasm_custom_potential_share_url_roundtrips_losslessly_in_browser_history() {
    let window = web_sys::window().expect("browser window must exist");
    let history = window.history().expect("browser history must exist");
    let initial_href = window.location().href().unwrap_or_default();

    let (source_app, q_idx, expected_envelope) = prepare_sculpted_quantum_app();
    assert!(expected_envelope.len() <= MAX_DECODED_URL_STATE_BYTES);

    let share_url = source_app
        .encode_current_share_url()
        .expect("share URL must encode");
    let fragment = &share_url[share_url.find('#').expect("hash fragment expected")..];
    assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);

    replace_browser_url_fragment(fragment).expect("replace_browser_url_fragment must succeed");
    let (hash, search) = read_browser_location_parts().expect("location parts readable");

    let mut dest_app = MathExplorerApp::default();
    assert!(dest_app
        .apply_share_location_strings(&hash, &search, None)
        .expect("apply_share_location_strings must succeed"));
    assert_eq!(dest_app.selected_tab, q_idx);
    assert_eq!(
        dest_app.tabs[q_idx].export_url_preset_json().unwrap(),
        Some(expected_envelope)
    );

    let _ = replace_browser_url_fragment("");
    if !initial_href.is_empty() {
        let _ =
            history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(&initial_href));
    }
}
