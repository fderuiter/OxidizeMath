// [cite:battery_degradation]

use super::*;

#[test]
#[verified_engine::verified]
fn test_standard_model_n70() {
    let model = PowerLawModel::standard();

    let n70_100 = model.n70(DepthOfDischarge::new_clamped(100.0)) / Cycles::new_clamped(1.0);
    assert!(
        (n70_100 - 300.0).abs() < 50.0,
        "Expected ~300, got {}",
        n70_100
    );

    let n70_10 = model.n70(DepthOfDischarge::new_clamped(10.0)) / Cycles::new_clamped(1.0);
    assert!(
        (n70_10 - 6000.0).abs() < 500.0,
        "Expected ~6000, got {}",
        n70_10
    );
}

#[test]
#[verified_engine::verified]
fn test_capacity_decay() {
    let model = PowerLawModel::standard();
    let dod = DepthOfDischarge::new_clamped(60.0);
    let n70 = model.n70(dod) / Cycles::new_clamped(1.0);

    let cap_0 = model.capacity(Cycles::new_clamped(0.0), dod);
    assert!(
        ((cap_0 / Capacity::new_clamped(1.0)) - 1.0).abs() < math_commons::registry::TOLERANCE_FAST
    );

    let cap_n70 = model.capacity(Cycles::new_clamped(n70), dod);
    assert!(
        ((cap_n70 / Capacity::new_clamped(1.0)) - 0.7).abs()
            < math_commons::registry::TOLERANCE_FAST
    );
}

#[test]
#[verified_engine::verified]
fn test_cycles_to_capacity() {
    let model = PowerLawModel::standard();
    let dod = DepthOfDischarge::new_clamped(50.0);

    let cycles = model.cycles_to_capacity(Capacity::new_clamped(0.7), dod);
    let n70 = model.n70(dod);
    assert!(
        ((cycles / Cycles::new_clamped(1.0)) - (n70 / Cycles::new_clamped(1.0))).abs()
            < math_commons::registry::TOLERANCE_FAST
    );

    let cycles_0 = model.cycles_to_capacity(Capacity::new_clamped(1.0), dod);
    assert!((cycles_0 / Cycles::new_clamped(1.0)) < math_commons::registry::TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_trait_implementation() {
    #[verified_engine::verified]
    fn evaluate_model<M: DegradationModel>(model: &M) -> f64 {
        let d = DepthOfDischarge::new_clamped(50.0);
        model.n70(d) / Cycles::new_clamped(1.0)
    }

    let model = PowerLawModel::standard();
    let result = evaluate_model(&model);
    assert!(result > 0.0);

    let coupled = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Lfp);
    assert!(evaluate_model(&coupled) > result);
}

#[test]
#[verified_engine::verified]
fn test_celsius_kelvin_offset_and_dimensionless_sei_calendar_loss() {
    let cold_c = math_scalars::Celsius::<f64>::new(-20.0);
    assert!((cold_c.to_kelvin().get() - 253.15).abs() < 1.0e-12);

    let baseline = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811);
    let cold_model = baseline.with_temperature(cold_c);
    assert_eq!(cold_model.temperature(), cold_c);
    assert!((cold_model.temperature_c() - (-20.0)).abs() < 1.0e-12);
    assert!(cold_model.sei_calendar_loss(36.0) < baseline.sei_calendar_loss(36.0));

    // Dimensionless ratio (t / t_ref).sqrt() quadruples time to double SEI loss.
    let loss_9d = baseline.sei_calendar_loss(9.0);
    let loss_36d = baseline.sei_calendar_loss(36.0);
    assert!(((loss_36d / loss_9d) - 2.0).abs() < 1.0e-12);
}
