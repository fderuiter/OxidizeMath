//! Zero-allocation `VerifiedAllocator` verification for 2D FitzHugh-Nagumo sheet and 1D Hodgkin-Huxley cable.

use domain_biology::biology::neuroscience::{ExcitableSheet2D, HodgkinHuxleyCable};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_excitable_sheet_and_hh_cable_zero_heap_allocations_in_step_loop() {
    let mut sheet = ExcitableSheet2D::new_default();
    sheet.trigger_s1_s2_spiral();
    sheet.paint_scar_disc(48.0, 48.0, 6.0, true);
    sheet.paint_current_disc(20.0, 20.0, 3.0, 0.8);

    let mut cable = HodgkinHuxleyCable::new_default();
    cable.pulse_compartment_range(0, 6, 15.0);

    // Keep all VerifiedEngine::run_verified assertions in one test to prevent MEMORY_LOCK races.
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..10 {
            sheet.step(0.2);
            cable.step(0.02);
        }
        sheet.defibrillate();
    });

    assert!(metrics.function_calls > 0);
    assert!(metrics.arithmetic_ops > 0);
}
