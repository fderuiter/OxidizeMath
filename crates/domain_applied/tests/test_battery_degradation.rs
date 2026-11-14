#![allow(missing_docs)]
#[cfg(test)]
mod tests {
    use domain_applied::applied::battery_degradation::{
        Capacity, Cycles, DepthOfDischarge, PowerLawModel,
    };

    #[test]
    #[verified_engine::verified]
    fn test_strong_types() {
        let dod = DepthOfDischarge::new(60.0).unwrap();
        let model = PowerLawModel::standard();

        let cycles = model.n70(dod);
        assert!(((cycles / Cycles::new_clamped(1.0)) - 600.0).abs() < 50.0);

        let cap = model.capacity(cycles, dod);
        assert!(
            ((cap / Capacity::new_clamped(1.0)) - 0.7).abs()
                < math_commons::registry::TOLERANCE_FAST
        );

        let calculated_cycles = model.cycles_to_capacity(Capacity::new(0.7).unwrap(), dod);
        assert!(
            ((calculated_cycles / Cycles::new_clamped(1.0)) - (cycles / Cycles::new_clamped(1.0)))
                .abs()
                < math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_dod_bounds_upper() {
        assert!(DepthOfDischarge::new(100.1).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_dod_bounds_lower() {
        assert!(DepthOfDischarge::new(-0.1).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_custom_model() {
        let standard = PowerLawModel::standard();
        let _improved = PowerLawModel::new(
            (standard.n70(DepthOfDischarge::new(100.0).unwrap()) / Cycles::new_clamped(1.0)) * 2.0
                / 100.0_f64.powf(-1.2639),
            -1.2639,
        );

        let improved2 = PowerLawModel::new(2.0e5, -1.2);
        let dod = DepthOfDischarge::new(50.0);

        let cycles_std = standard.n70(dod.unwrap()) / Cycles::new_clamped(1.0);
        let dod2 = DepthOfDischarge::new(50.0).unwrap();
        let cycles_imp = improved2.n70(dod2) / Cycles::new_clamped(1.0);

        assert!(cycles_imp != cycles_std);
    }
}
