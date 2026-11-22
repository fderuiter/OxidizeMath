#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;
use math_explorer::biology::phyllotaxis::is_consecutive_fibonacci_pair;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_lenia_and_phyllotaxis_and_evolution_tools_step_and_roundtrip_url() {
    let mut lenia = LeniaTool::default();
    lenia.step_simulation();
    assert!(lenia.telemetry.total_mass > 15.0);
    assert!(lenia.export_raster_image().is_some());

    let mut phyllo = PhyllotaxisLSystemTool::default();
    let (m, n) = phyllo.phyllotaxis.detect_parastichy_pair();
    assert!(is_consecutive_fibonacci_pair(m, n));
    phyllo.step_simulation();
    assert!(!phyllo.lsystem.segments().is_empty());

    let evo = EvolutionKineticsTool::default();
    assert_eq!(evo.export_datasets().len(), 5);

    let mut app = crate::app::MathExplorerApp::default();
    let idx = app
        .tabs
        .iter()
        .position(|t| t.name() == "Morphogenesis")
        .expect("Morphogenesis tab");
    app.selected_tab = idx;
    assert!(app.tabs[idx].select_tool_by_name("LeniaTool"));
    let url = app.encode_current_share_url().expect("wasm share url");
    let hash = &url[url.find('#').expect("share URL fragment")..];

    let mut restored = crate::app::MathExplorerApp::default();
    assert!(restored.apply_share_location_strings(hash, "", None).unwrap());
    assert_eq!(restored.tabs[idx].active_tool_id(), Some("LeniaTool"));
}
