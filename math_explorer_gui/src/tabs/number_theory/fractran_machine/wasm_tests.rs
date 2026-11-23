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

    let wav = restored
        .export_wav_audio(44_100, 1.0)
        .expect("WASM FRACTRAN WAV");
    assert_eq!(&wav[0..4], b"RIFF");
    let midi = restored.export_midi_score(1.0).expect("WASM FRACTRAN MIDI");
    assert_eq!(&midi[0..4], b"MThd");
    let (scl, kbm) = restored.export_scala_tuning().expect("WASM FRACTRAN Scala");
    assert!(scl.contains("2/1") && !kbm.is_empty());

    let fact = crate::tabs::number_theory::factorization::FactorizationTool::default();
    let f_midi = fact
        .export_midi_score(1.0)
        .expect("WASM Factorization MIDI");
    assert_eq!(&f_midi[0..4], b"MThd");
    let (f_scl, f_kbm) = fact
        .export_scala_tuning()
        .expect("WASM Factorization Scala");
    assert!(f_scl.contains("5/4") && !f_kbm.is_empty());
}
