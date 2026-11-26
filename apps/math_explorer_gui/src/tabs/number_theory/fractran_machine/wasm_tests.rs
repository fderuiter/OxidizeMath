#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{FractranMachineTool, MachineStopReason, PresetChoice};
use crate::framework::InteractiveTool;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_fractran_machine_primegame_and_state_roundtrip() {
    let mut tool = FractranMachineTool::default();
    assert_eq!(
        tool.run_until_power_of_two(),
        MachineStopReason::HitPowerOfTwo(2)
    );
    assert_eq!(tool.step_count, 19);

    tool.select_preset(PresetChoice::Add);
    let saved = tool
        .save_state()
        .expect("FRACTRAN state must serialize on wasm32");
    let mut restored = FractranMachineTool::default();
    restored
        .try_load_state(&saved)
        .expect("FRACTRAN state must deserialize on wasm32");
    assert_eq!(restored.save_state(), Some(saved));
    assert!(!restored.export_datasets().is_empty());
}
