//! Zero-allocation `VerifiedAllocator` verification for 3-lead extracellular dipole integration.

use domain_physics::physics::medical::bioelectric::{
    BioelectricPresetKind, DipoleLeadArray, LeadPosition, LeadReadoutMode,
    synthesize_bioelectric_preset_sheet,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_dipole_lead_array_zero_heap_allocations_in_step_loop() {
    let mut leads = DipoleLeadArray::new_default(96, 96);
    let mut voltage = vec![-1.2; 96 * 96];
    let scars = vec![0_u8; 96 * 96];
    let cable_v = vec![-65.0; 96];
    let cable_scars = vec![0_u8; 96];

    let (ecg, metrics) = VerifiedEngine::run_verified(|| {
        leads.set_lead(0, LeadPosition::new(22.0, 24.0, 8.0));
        synthesize_bioelectric_preset_sheet(
            &mut voltage,
            &scars,
            96,
            96,
            BioelectricPresetKind::S1S2SpiralInduction,
            1.5,
        );
        let _lfp_1d = leads.evaluate_cable_1d(
            &cable_v,
            &cable_scars,
            48.0,
            0.5,
            LeadReadoutMode::UnipolarLfp,
        );
        leads.evaluate_sheet(&voltage, &scars, LeadReadoutMode::BipolarPseudoEcg)
    });

    assert!(ecg[0].is_finite() && ecg[1].is_finite() && ecg[2].is_finite());
    assert!(metrics.function_calls > 0);
    assert!(metrics.arithmetic_ops > 0);
}
