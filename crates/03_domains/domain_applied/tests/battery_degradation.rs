//! Integration tests for battery degradation.
use domain_applied::applied::battery_degradation::{
    Capacity, ChemistryPreset, Cycles, DegradationModel, DepthOfDischarge, DutySegment,
    ElectrochemicalAgingModel, MAX_C_RATE, MAX_EXPONENT_Z, MAX_TEMPERATURE_C, MIN_C_RATE,
    MIN_EXPONENT_Z, MIN_TEMPERATURE_C, PowerLawModel, calibrate_electrochemical_model,
    estimate_duty_lifetime, evaluate_duty_point,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

const TOLERANCE: f64 = 0.3;

#[test]
#[verified_engine::verified]
fn test_n70() {
    let model = PowerLawModel::standard();
    let n70_60 = model.n70(DepthOfDischarge::new_clamped(60.0)).as_f64();
    assert!(
        (n70_60 - 576.7).abs() < TOLERANCE,
        "Expected ~576.7, got {}",
        n70_60
    );

    let n70_80 = model.n70(DepthOfDischarge::new_clamped(80.0)).as_f64();
    assert!(
        (n70_80 - 400.9).abs() < TOLERANCE,
        "Expected ~400.9, got {}",
        n70_80
    );

    let n70_100 = model.n70(DepthOfDischarge::new_clamped(100.0)).as_f64();
    assert!(
        (n70_100 - 302.4).abs() < TOLERANCE,
        "Expected ~302.4, got {}",
        n70_100
    );
}

#[test]
#[verified_engine::verified]
fn test_cycles_to_capacity_dod_60() {
    let model = PowerLawModel::standard();
    let dod = DepthOfDischarge::new_clamped(60.0);
    let to_90 = model
        .cycles_to_capacity(Capacity::new_clamped(0.90), dod)
        .as_f64();
    assert!(
        (to_90 - 170.4).abs() < TOLERANCE,
        "DoD 60 to 90%: Expected ~170.4, got {}",
        to_90
    );

    let to_85 = model
        .cycles_to_capacity(Capacity::new_clamped(0.85), dod)
        .as_f64();
    assert!(
        (to_85 - 262.8).abs() < TOLERANCE,
        "DoD 60 to 85%: Expected ~262.8, got {}",
        to_85
    );

    let to_80 = model
        .cycles_to_capacity(Capacity::new_clamped(0.80), dod)
        .as_f64();
    assert!(
        (to_80 - 360.8).abs() < TOLERANCE,
        "DoD 60 to 80%: Expected ~360.8, got {}",
        to_80
    );
}

#[test]
#[verified_engine::verified]
fn test_cycles_to_capacity_dod_80() {
    let model = PowerLawModel::standard();
    let dod = DepthOfDischarge::new_clamped(80.0);
    let to_90 = model
        .cycles_to_capacity(Capacity::new_clamped(0.90), dod)
        .as_f64();
    assert!(
        (to_90 - 118.4).abs() < TOLERANCE,
        "DoD 80 to 90%: Expected ~118.4, got {}",
        to_90
    );

    let to_85 = model
        .cycles_to_capacity(Capacity::new_clamped(0.85), dod)
        .as_f64();
    assert!(
        (to_85 - 182.7).abs() < TOLERANCE,
        "DoD 80 to 85%: Expected ~182.7, got {}",
        to_85
    );

    let to_80 = model
        .cycles_to_capacity(Capacity::new_clamped(0.80), dod)
        .as_f64();
    assert!(
        (to_80 - 250.8).abs() < TOLERANCE,
        "DoD 80 to 80%: Expected ~250.8, got {}",
        to_80
    );
}

#[test]
#[verified_engine::verified]
fn test_cycles_to_capacity_dod_100() {
    let model = PowerLawModel::standard();
    let dod = DepthOfDischarge::new_clamped(100.0);
    let to_90 = model
        .cycles_to_capacity(Capacity::new_clamped(0.90), dod)
        .as_f64();
    assert!(
        (to_90 - 89.3).abs() < TOLERANCE,
        "DoD 100 to 90%: Expected ~89.3, got {}",
        to_90
    );

    let to_85 = model
        .cycles_to_capacity(Capacity::new_clamped(0.85), dod)
        .as_f64();
    assert!(
        (to_85 - 137.8).abs() < TOLERANCE,
        "DoD 100 to 85%: Expected ~137.8, got {}",
        to_85
    );

    let to_80 = model
        .cycles_to_capacity(Capacity::new_clamped(0.80), dod)
        .as_f64();
    assert!(
        (to_80 - 189.2).abs() < TOLERANCE,
        "DoD 100 to 80%: Expected ~189.2, got {}",
        to_80
    );
}

#[test]
fn test_electrochemical_arrhenius_and_stretched_exponent_physics() {
    let dod = DepthOfDischarge::new_clamped(60.0);
    let nmc_25 = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811);
    assert!((nmc_25.arrhenius_factor() - 1.0).abs() < 1e-12);
    assert!((nmc_25.temperature_k() - 298.15).abs() < 1e-12);

    // At 45 °C (318.15 K) with E_a = 31.5 kJ/mol, Arrhenius acceleration is ~2.2229x (+122.29%).
    let nmc_45 = nmc_25.with_temperature_c(45.0);
    let ratio = nmc_25.n70(dod).as_f64() / nmc_45.n70(dod).as_f64();
    assert!(
        (ratio - 2.2229).abs() < 5e-3,
        "Expected ~2.2229 thermal acceleration at 45 °C, got {ratio}"
    );

    // Diffusion-limited SEI cycling exponent z = 0.50 reaches 80% SOH at (2/3)^2 = 4/9 of N_70.
    let sei_z05 = nmc_25.with_exponent_z(0.50);
    let n70 = sei_z05.n70(dod).as_f64();
    let n80 = sei_z05
        .cycles_to_capacity(Capacity::new_clamped(0.80), dod)
        .as_f64();
    assert!(((n80 / n70) - (4.0 / 9.0)).abs() < 1e-9);
    let cap_at_n80 = sei_z05.capacity(Cycles::new_clamped(n80), dod).as_f64();
    assert!((cap_at_n80 - 0.80).abs() < 1e-9);

    // Verify preset ordering and parameter clamping.
    for preset in ChemistryPreset::ALL {
        let m = ElectrochemicalAgingModel::from_preset(preset);
        assert!(!preset.label().is_empty());
        assert!(!preset.description().is_empty());
        assert!(DegradationModel::n70(&m, dod).as_f64() > 0.0);
    }
    let clamped =
        ElectrochemicalAgingModel::new(ChemistryPreset::Lfp, f64::NAN, -1.15, 9.0, 120.0, 99.0)
            .with_temperature_c(-90.0)
            .with_c_rate(0.001)
            .with_exponent_z(0.01);
    assert_eq!(clamped.temperature_c(), MIN_TEMPERATURE_C);
    assert_eq!(clamped.c_rate(), MIN_C_RATE);
    assert_eq!(clamped.exponent_z(), MIN_EXPONENT_Z);
    assert_eq!(
        clamped.with_temperature_c(99.0).temperature_c(),
        MAX_TEMPERATURE_C
    );
    assert_eq!(clamped.with_c_rate(50.0).c_rate(), MAX_C_RATE);
    assert_eq!(clamped.with_exponent_z(5.0).exponent_z(), MAX_EXPONENT_Z);
}

#[test]
fn test_calibration_and_storage_duty_edge_cases() {
    let dod = DepthOfDischarge::new_clamped(70.0);
    let truth = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Lfp)
        .with_temperature_c(35.0)
        .with_c_rate(1.2)
        .with_exponent_z(0.52);

    let mut samples = [[0.0_f64; 2]; 16];
    for (i, slot) in samples.iter_mut().enumerate() {
        let cyc = (i as f64) * 180.0;
        let cap = truth.capacity(Cycles::new_clamped(cyc), dod).as_f64();
        *slot = [cyc, cap];
    }

    let seed = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811)
        .with_temperature_c(35.0)
        .with_c_rate(1.2);
    let fit = seed.calibrate_from_cycles(&samples, dod).expect("LM fit");
    assert!(fit.rmse < 1e-5, "Expected tight fit RMSE, got {}", fit.rmse);
    assert!(fit.r_squared > 0.9999);
    assert!((fit.fitted_exponent_z - 0.52).abs() < 1e-2);
    assert!((fit.fitted_n70_cycles - truth.n70(dod).as_f64()).abs() < 5.0);

    // Reject insufficient or out-of-domain data without panicking.
    assert!(seed.calibrate_from_cycles(&samples[..2], dod).is_err());
    let bad = [[0.0, 1.0], [100.0, 1.5], [200.0, 0.8]];
    assert!(calibrate_electrochemical_model(&seed, &bad, dod).is_err());

    // Pure storage segment (daily_cycles == 0.0) integrates sqrt(t) SEI fade with zero division-by-zero.
    let storage_only = [DutySegment::storage(30.0, 30.0)];
    let summary = estimate_duty_lifetime(&truth, &storage_only, Capacity::new_clamped(0.80));
    assert!(summary.years_to_80.is_finite() && summary.years_to_80 > 0.0);
    assert!(summary.years_to_70 > summary.years_to_80);
    assert_eq!(summary.cycles_to_80, 0.0);
    assert_eq!(summary.effective_daily_cycles, 0.0);

    let pt = evaluate_duty_point(&truth, &storage_only, summary.days_to_80);
    assert!((pt.soh - 0.80).abs() < 1e-4);
    assert_eq!(pt.cycling_loss, 0.0);
    assert!((pt.calendar_sei_loss - 0.20).abs() < 1e-4);
}

#[test]
fn test_verified_zero_heap_allocations_for_coupled_aging_calibration_and_duty() {
    let dod = DepthOfDischarge::new_clamped(60.0);
    let target = Capacity::new_clamped(0.80);
    let base = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811)
        .with_temperature_c(32.0)
        .with_c_rate(1.5);
    let duty = [
        DutySegment::commute(),
        DutySegment::fast_charge(),
        DutySegment::storage(1.0, 20.0),
    ];

    let mut lab_samples = [[0.0_f64; 2]; 12];
    for (idx, row) in lab_samples.iter_mut().enumerate() {
        let cyc = (idx as f64) * 90.0;
        let cap = base.capacity(Cycles::new_clamped(cyc), dod).as_f64();
        *row = [cyc, cap];
    }

    // Keep all VerifiedEngine::run_verified assertions inside one test to avoid MEMORY_LOCK contention.
    let ((coupled_soh, rmse, years_80), metrics) = VerifiedEngine::run_verified(|| {
        let hot = base.with_temperature_c(45.0).with_c_rate(2.0);
        let soh = hot
            .coupled_capacity(Cycles::new_clamped(400.0), dod, 365.25)
            .as_f64();
        let fit = hot
            .calibrate_from_cycles(&lab_samples, dod)
            .expect("zero-alloc LM calibration");
        let life = estimate_duty_lifetime(&fit.model, &duty, target);
        let pt = evaluate_duty_point(&fit.model, &duty, life.days_to_80);
        assert!((pt.soh - 0.80).abs() < 1e-3);
        (soh, fit.rmse, life.years_to_80)
    });

    assert!(coupled_soh > 0.0 && coupled_soh < 1.0);
    assert!(rmse < 1e-4);
    assert!(years_80.is_finite() && years_80 > 0.0);
    assert!(metrics.function_calls > 0);
}
