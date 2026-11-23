//! Unit tests for 4D radar-gated LINAC and IMRT delivery sonification frames.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::mappings::{LinacGatePhase, imrt_delivery_frame, radar_linac_frame};

#[test]
fn test_medical_radar_and_imrt_sonification_frames() {
    let radar_on = radar_linac_frame(LinacGatePhase::BeamActive, -2.5, -1.0, 42.0, 0.02);
    let radar_hold = radar_linac_frame(LinacGatePhase::BeamHold, 8.5, 18.0, 42.0, 0.02);
    let radar_fault = radar_linac_frame(LinacGatePhase::SafetyInterlock, f64::NAN, 0.0, 42.0, 0.1);
    assert!(radar_on.voices[1].gain > 0.5 && radar_hold.voices[1].gain == 0.0);
    assert!(!radar_fault.is_audible());

    let imrt_clean = imrt_delivery_frame(
        LinacGatePhase::BeamActive,
        -2.4,
        59.5,
        60.0,
        19.0,
        26.0,
        5.6,
    );
    let imrt_over = imrt_delivery_frame(LinacGatePhase::BeamHold, 9.2, 50.0, 60.0, 38.0, 26.0, 5.6);
    let imrt_fault = imrt_delivery_frame(
        LinacGatePhase::SafetyInterlock,
        0.0,
        60.0,
        60.0,
        20.0,
        26.0,
        5.6,
    );
    assert!(imrt_clean.voices[0].gain > 0.3 && imrt_clean.noise_gain == 0.0);
    assert!(imrt_over.voices[0].gain == 0.0 && imrt_over.voices[3].gain > 0.2);
    assert!(!imrt_fault.is_audible());
}
