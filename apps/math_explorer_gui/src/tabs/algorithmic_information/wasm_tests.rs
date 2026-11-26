#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::state::{FractranMachineTool, FractranPreset, FractranToolConfig};
use super::AlgorithmicInfoTab;
use crate::tabs::ExplorerTab;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_algorithmic_info_tab_and_floyd_cycle_guard() {
    let tab = AlgorithmicInfoTab::default();
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "algorithmic_information");

    let mut tool = FractranMachineTool::default();
    assert_eq!(tool.emitted_primes, vec![2]);

    tool.apply_config(FractranToolConfig {
        playback_rate: 1.0,
        steps_per_frame: 256,
        warmup_steps: 0,
        initial_value: 2,
        preset: FractranPreset::Custom,
        custom_fractions: "5/2, 2/5".to_string(),
    })
    .expect("valid periodic FRACTRAN config");
    tool.is_playing = true;
    tool.step_frame_batch();

    assert!(!tool.is_playing);
    assert_eq!(tool.live.cycle.map(|c| c.period), Some(2));
}

// [cite:algorithmic_information_rust]
