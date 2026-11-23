//! Zero-allocation verification for 64-event pharmacokinetic regimen sampling and analytics.

use domain_applied::applied::pharmacokinetics::{
    BatemanModel, DoseEvent, EnantiomerModel, MAX_REGIMEN_EVENTS, PKParameters,
    RegimenSuperposition, ReleaseFormulation, TherapeuticWindow, summarize_exposure,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_regimen_and_exposure_zero_heap_allocations_for_64_events() {
    let params = PKParameters::new(1.0, 100.0, 1.2, 0.1, 20.0).unwrap();
    let d_params = PKParameters::new(1.0, 20.0, 1.1, 0.0693, 1.0).unwrap();
    let l_params = PKParameters::new(1.0, 20.0, 1.1, 0.0533, 1.0).unwrap();
    let chiral = EnantiomerModel::new(d_params, l_params, 0.75, 0.25).unwrap();
    let window = TherapeuticWindow::new(3.0, 8.5).unwrap();

    let mut events = [DoseEvent {
        time_h: 0.0,
        dose_mg: 100.0,
        formulation: ReleaseFormulation::ImmediateRelease,
    }; MAX_REGIMEN_EVENTS];
    for (idx, ev) in events.iter_mut().enumerate() {
        let dose = if idx == 0 { 200.0 } else { 100.0 };
        *ev = if idx % 2 == 0 {
            DoseEvent::ir(idx as f64 * 12.0, dose).unwrap()
        } else {
            DoseEvent::xr(idx as f64 * 12.0, dose, 4.0, 0.5, 0.5).unwrap()
        };
    }

    let mut times = [0.0; 257];
    for (idx, t) in times.iter_mut().enumerate() {
        *t = idx as f64 * 3.0;
    }
    let mut conc = [0.0; 257];
    let mut d_conc = [0.0; 257];
    let mut l_conc = [0.0; 257];
    let mut tot_conc = [0.0; 257];

    // Keep all VerifiedEngine::run_verified assertions in one test to prevent MEMORY_LOCK races.
    let (summary, metrics) = VerifiedEngine::run_verified(|| {
        let regimen = RegimenSuperposition::new(BatemanModel::new(params), &events).unwrap();
        regimen.sample_into(&times, &mut conc).unwrap();

        let chiral_regimen = RegimenSuperposition::new(chiral, &events).unwrap();
        chiral_regimen
            .sample_isomers_into(&times, &mut d_conc, &mut l_conc, &mut tot_conc)
            .unwrap();

        summarize_exposure(&times, &conc, window, 12.0, params.ke()).unwrap()
    });

    assert!(summary.c_max > 0.0);
    assert!(summary.auc_total > 0.0);
    assert!(summary.accumulation_ratio > 1.0);
    assert!(metrics.function_calls > 0);
}
