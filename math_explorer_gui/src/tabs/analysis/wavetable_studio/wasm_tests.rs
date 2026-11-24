#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use wasm_bindgen_test::*;

use super::*;
use crate::audio::studio_preview::snapshot_studio_preview;
use crate::audio::wav::inspect_wav_riff;

#[wasm_bindgen_test]
fn wasm_wavetable_studio_formula_and_pde_export_and_periodic_wave_audition() {
    let mut tool = WavetableStudioTool::default();
    tool.state.frame_count = 8;
    let formula_wav = tool.synthesize_and_encode_wav();
    let summary = inspect_wav_riff(&formula_wav).expect("WASM formula wavetable RIFF");
    assert_eq!(summary.channels, 1);
    assert_eq!(summary.data_bytes, 8 * 2048 * 4);
    assert_eq!(summary.extra_chunks.len(), 2);

    tool.state.source_mode = WavetableSourceMode::PdeWaveSlice;
    tool.pde_dirty = true;
    tool.refresh_active_preview();
    let pde_wav = tool.synthesize_and_encode_wav();
    assert_eq!(inspect_wav_riff(&pde_wav).unwrap().data_bytes, 8 * 2048 * 4);

    tool.toggle_audition_from_gesture();
    assert!(snapshot_studio_preview().wavetable_auditioning);
    tool.toggle_audition_from_gesture();
    assert!(!snapshot_studio_preview().wavetable_auditioning);
}
