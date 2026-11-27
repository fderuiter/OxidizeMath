use oxidizemath::diagnostics::{Diagnostic, Severity};
use oxidizemath::engineering::battery_degradation::{
    BatteryError as SubBatteryError, Capacity, ChemistryPreset, Cycles, DepthOfDischarge,
    ElectrochemicalAgingModel, PowerLawModel,
};
use oxidizemath::engineering::radar::FmcwConfig;
use oxidizemath::engineering::{
    BaudRate, CuttingSpeed, DataBits, MaterialConstant, ToolExponent, TotalBits,
    calculate_taylor_tool_life, calculate_uart_throughput,
};
use oxidizemath::error::{BatteryError, EngineeringDomainError, Error, HardwareError, RadarError};

pub(super) fn assert_engineering_facade_exports() {
    let model = PowerLawModel::standard();
    let cap = model.capacity(
        Cycles::new_clamped(100.0),
        DepthOfDischarge::new_clamped(0.8),
    );
    assert!(cap / Capacity::new_clamped(1.0) > 0.0);
    let coupled = ElectrochemicalAgingModel::from_preset(ChemistryPreset::Lfp);
    assert!(coupled.n70(DepthOfDischarge::new_clamped(60.0)).as_f64() > 0.0);
    let bad_err: SubBatteryError = Cycles::new(-1.0).expect_err("negative cycles must fail");
    let root_err: BatteryError = bad_err;
    assert_eq!(root_err.severity(), Severity::Error);

    let radar = FmcwConfig::iwr6843_default();
    assert!((radar.range_resolution() - 0.0375).abs() < 1e-4);

    let life = calculate_taylor_tool_life(
        CuttingSpeed::new(50.0).unwrap(),
        MaterialConstant::new(100.0).unwrap(),
        ToolExponent::new(0.25).unwrap(),
    );
    assert!((life.value() - 16.0).abs() < 1e-12);

    let bps = calculate_uart_throughput(
        BaudRate::new(115_200),
        DataBits::new(8),
        TotalBits::new(10).unwrap(),
    );
    assert!((bps - 92_160.0).abs() < 1e-9);

    let umbrella =
        EngineeringDomainError::from(RadarError::InvalidConfiguration("antennas".into()));
    let _facade: Error = umbrella.into();
    let severities = [
        BatteryError::InvalidCapacity(1.5).severity(),
        HardwareError::InvalidParameter {
            name: "bits".to_string(),
            value: 0.0,
        }
        .severity(),
        RadarError::InvalidConfiguration("antennas".to_string()).severity(),
    ];
    assert!(severities.iter().all(|s| *s == Severity::Error));
}
