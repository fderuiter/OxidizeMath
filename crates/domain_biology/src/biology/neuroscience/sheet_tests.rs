//! Verification and emergent arrhythmia tests for [`ExcitableSheet2D`] and [`HodgkinHuxleyCable`].

use super::{DEFAULT_SHEET_DIM, ExcitableSheet2D, HodgkinHuxleyCable, SheetPresetKind};

#[test]
#[verified_engine::verified]
fn test_dual_edge_planar_wavefronts_annihilate_on_collision() {
    let mut sheet = ExcitableSheet2D::new_default();
    sheet.trigger_dual_edge_collision();
    assert!(sheet.max_tissue_voltage() > 1.5);

    for _ in 0..250 {
        sheet.step(0.2);
    }
    let mid_u = sheet.max_tissue_voltage();
    assert!(
        mid_u > 1.0,
        "Wavefronts should still be propagating prior to collision, max_u={}",
        mid_u
    );

    for _ in 0..650 {
        sheet.step(0.2);
    }
    let post_collision_max = sheet.max_tissue_voltage();
    assert!(
        post_collision_max < -0.5,
        "Opposing wavefronts must annihilate on collision instead of re-exciting, max_u={}",
        post_collision_max
    );
}

#[test]
#[verified_engine::verified]
fn test_s1_s2_protocol_induces_self_sustaining_spiral_and_defibrillates() {
    let mut sheet = ExcitableSheet2D::new_default();
    sheet.trigger_s1_s2_spiral();
    assert!(sheet.pending_s2().is_some());

    for _ in 0..1400 {
        sheet.step(0.2);
    }
    assert!(sheet.pending_s2().is_none());
    let spiral_max = sheet.max_tissue_voltage();
    assert!(
        spiral_max > 0.8,
        "S1-S2 stimulation should sustain a spiral wave at t=280, got max_u={}",
        spiral_max
    );

    sheet.defibrillate();
    let (u_rest, _) = sheet.system.kinetics.resting_state();
    assert!((sheet.max_tissue_voltage() - u_rest).abs() < 1e-9);
}

#[test]
#[verified_engine::verified]
fn test_scar_pinned_rotor_persists_until_ablation_line_to_boundary() {
    let mut sheet = ExcitableSheet2D::new_default();
    sheet.apply_preset(SheetPresetKind::ScarPinnedRotor);

    for _ in 0..1100 {
        sheet.step(0.2);
    }
    assert!(
        sheet.max_tissue_voltage() > 0.8,
        "Scar-pinned rotor should sustain circulating tachycardia before ablation"
    );

    sheet.ablate_scar_to_boundary();
    for _ in 0..1600 {
        sheet.step(0.2);
    }
    assert!(
        sheet.max_tissue_voltage() < -0.5,
        "Ablation line from scar to boundary must extinguish re-entrant rotor, got max_u={}",
        sheet.max_tissue_voltage()
    );
}

#[test]
#[verified_engine::verified]
fn test_hodgkin_huxley_cable_spike_propagation_and_collision() {
    let mut cable = HodgkinHuxleyCable::new_default();
    cable.pulse_compartment_range(0, 6, 15.0);
    for _ in 0..300 {
        cable.step(0.02);
    }
    let peak = cable
        .voltage()
        .iter()
        .skip(12)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        peak > 0.0,
        "1D Hodgkin-Huxley cable should propagate ionic spike downstream, peak={}",
        peak
    );

    cable.trigger_dual_end_collision();
    for _ in 0..1800 {
        cable.step(0.02);
    }
    let settled_max = cable
        .voltage()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        settled_max < -45.0,
        "Colliding 1D Hodgkin-Huxley spikes should annihilate, settled_max={}",
        settled_max
    );
}

#[test]
#[verified_engine::verified]
fn test_excitable_sheet_brushes_and_voltage_clamp_probes() {
    let mut sheet = ExcitableSheet2D::new_default();
    assert_eq!(sheet.voltage().len(), DEFAULT_SHEET_DIM * DEFAULT_SHEET_DIM);
    sheet.paint_scar_disc(48.0, 48.0, 6.0, true);
    sheet.paint_current_disc(20.0, 20.0, 3.0, 1.4);
    sheet.paint_voltage_clamp_disc(70.0, 70.0, 2.5, 1.25, true);
    for _ in 0..25 {
        sheet.step(0.1);
    }
    let clamp_idx = math_commons::math_kernel::types::flatten_2d_index(70, 70, DEFAULT_SHEET_DIM);
    let scar_idx = math_commons::math_kernel::types::flatten_2d_index(48, 48, DEFAULT_SHEET_DIM);
    let (u_rest, _) = sheet.system.kinetics.resting_state();
    assert!((sheet.voltage()[clamp_idx] - 1.25).abs() < 1e-12);
    assert!((sheet.voltage()[scar_idx] - u_rest).abs() < 1e-12);
}
