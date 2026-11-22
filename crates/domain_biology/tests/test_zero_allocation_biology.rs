#![allow(missing_docs)]

use domain_biology::biology::evolution::HawkDovePopulation;
use domain_biology::biology::lenia::{LeniaPreset, LeniaSystem};
use domain_biology::biology::morphogenesis::{GrayScottKinetics, ReactionKinetics};
use domain_biology::biology::phyllotaxis::{
    LSystemConfig, LSystemEngine, NutrientCoupling, NutrientKineticsKind, PhyllotaxisConfig,
    PhyllotaxisState,
};
use pure_math::pure_math::analysis::ode::{ArrayState, OdeSystem};
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static GLOBAL_ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static ZERO_ALLOC_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[verified_engine::verified]
fn test_lenia_step_128x128_zero_heap_allocations() {
    let _guard = ZERO_ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut system = LeniaSystem::from_preset(128, 128, LeniaPreset::Orbium)
        .expect("128x128 Lenia initialization must succeed");

    let (tel, metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..5 {
            system.step();
        }
        system.telemetry()
    });
    assert!(metrics.function_calls > 0);
    assert!(tel.total_mass > 20.0);
    assert!(tel.speed.is_finite());
}

#[test]
#[verified_engine::verified]
fn test_phyllotaxis_and_lsystem_zero_heap_allocations() {
    let _guard = ZERO_ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let nutrient = NutrientCoupling::new(NutrientKineticsKind::Hill, 2.0, 1.5, 0.8, 2.5)
        .expect("valid nutrient coupling");
    let mut state = PhyllotaxisState::new(512, 60, PhyllotaxisConfig::default(), nutrient);
    let mut lsys = LSystemEngine::new(16_384, 16_384, 128);
    let lsys_cfg = LSystemConfig::default();

    let ((m, n), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..8 {
            state.step(0.08);
        }
        lsys.generate(lsys_cfg, &nutrient);
        state.detect_parastichy_pair()
    });
    assert!(metrics.function_calls > 0);
    assert!(state.active_count() > 60);
    assert!(!lsys.segments().is_empty());
    assert!(m >= 3 && n >= m);
}

#[test]
#[verified_engine::verified]
fn test_gray_scott_and_hawk_dove_zero_heap_allocations() {
    let _guard = ZERO_ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let gs = GrayScottKinetics::new(0.037, 0.060);
    let hd = HawkDovePopulation::new(2.0, 4.0);
    let hd_state = ArrayState([0.25]);

    let ((rates, deriv), metrics) = VerifiedEngine::run_verified(|| {
        let r = <GrayScottKinetics as ReactionKinetics<2>>::reaction(&gs, [0.75, 0.25]);
        let d = hd.derivative(0.0, &hd_state);
        (r, d)
    });
    assert!(metrics.function_calls > 0);
    assert!(rates[0].is_finite() && rates[1].is_finite());
    assert!(
        deriv.0[0] > 0.0,
        "Hawks below ESS=0.5 must grow, got {}",
        deriv.0[0]
    );
}
