//! Example target for coupled electrochemical battery degradation, CSV calibration, and duty lifetimes.
// [cite:battery_degradation]

use oxidizemath::applied::battery_degradation::{
    Capacity, ChemistryPreset, Cycles, DepthOfDischarge, DutySegment, ElectrochemicalAgingModel,
    PowerLawModel, estimate_duty_lifetime,
};

fn main() {
    let dod = DepthOfDischarge::new_clamped(60.0);

    // 1. Standard isothermal power-law baseline
    let baseline = PowerLawModel::standard();
    println!(
        "Baseline PowerLaw N70 @ 60% DoD: {:.0} cycles (SOH @ 1000 cyc = {:.1}%)",
        baseline.n70(dod).as_f64(),
        baseline.capacity(Cycles::new_clamped(1000.0), dod).as_f64() * 100.0
    );

    // 2. Coupled NMC811 aging: 25 °C vs 45 °C Arrhenius thermal acceleration
    let nmc_25c = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Nmc811);
    let nmc_45c = nmc_25c.with_temperature_c(45.0).with_c_rate(1.5);
    println!(
        "NMC811 Coupled N70 @ 25 °C / 1.0C: {:.0} cycles | @ 45 °C / 1.5C: {:.0} cycles",
        nmc_25c.n70(dod).as_f64(),
        nmc_45c.n70(dod).as_f64(),
    );

    // 3. Stack-allocated Levenberg-Marquardt calibration on synthetic lab checkups
    let samples = [
        [0.0, 1.0],
        [250.0, 0.94],
        [500.0, 0.89],
        [1000.0, 0.82],
        [1500.0, 0.76],
    ];
    if let Ok(fit) = nmc_25c.calibrate_from_cycles(&samples, dod) {
        println!(
            "Calibrated Lab Cell: N70 = {:.0} cycles, z = {:.3}, RMSE = {:.4}",
            fit.fitted_n70_cycles, fit.fitted_exponent_z, fit.rmse
        );
    }

    // 4. Multi-segment duty lifetime (commute + fast-charge + storage)
    let duty = [
        DutySegment::commute(),
        DutySegment::fast_charge(),
        DutySegment::storage(1.0, 22.0),
    ];
    let summary = estimate_duty_lifetime(&nmc_25c, &duty, Capacity::new_clamped(0.80));
    println!(
        "Duty Warranty Milestones: 80% SOH = {:.2} yr ({:.0} cyc), 70% SOH = {:.2} yr ({:.0} cyc)",
        summary.years_to_80, summary.cycles_to_80, summary.years_to_70, summary.cycles_to_70
    );
}
