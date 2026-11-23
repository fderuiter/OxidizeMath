//! Battery degradation models.

use super::types::{Capacity, Cycles, DepthOfDischarge};
use verified_engine::Theory;

#[allow(missing_docs)]
pub trait DegradationModel {
    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn n70(&self, d: DepthOfDischarge) -> Cycles;

    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity;

    #[verified_engine::verified]
    #[allow(missing_docs)]
    fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles;
}

#[derive(Debug, Clone, Copy, Theory, PartialEq)]
#[theory(
    description = "Empirical power-law model characterizing the cycle life of lithium-ion batteries as a function of depth of discharge.",
    citation = "Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)"
)]
#[allow(missing_docs)]
pub struct PowerLawModel {
    alpha: f64,
    beta: f64,
}

impl PowerLawModel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn standard() -> Self {
        Self {
            alpha: 1.019e5,
            beta: -1.2639,
        }
    }
}

impl DegradationModel for PowerLawModel {
    #[verified_engine::verified]
    fn n70(&self, d: DepthOfDischarge) -> Cycles {
        // use raw f64 extracted via division to avoid failing bounds during exponentiation
        let raw_d = d / DepthOfDischarge::new_clamped(1.0);
        let val = self.alpha * raw_d.powf(self.beta);
        Cycles::new_clamped(val)
    }

    #[verified_engine::verified]
    fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        let n70_val = self.n70(d) / Cycles::new_clamped(1.0);
        if n70_val == 0.0 {
            return Capacity::new_clamped(0.0);
        }
        let n_val = n / Cycles::new_clamped(1.0);
        let exponent = n_val / n70_val;
        let cap = 0.7_f64.powf(exponent);
        Capacity::new_clamped(cap)
    }

    #[verified_engine::verified]
    fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles {
        let n70_val = self.n70(d) / Cycles::new_clamped(1.0);
        let target_raw = target / Capacity::new_clamped(1.0);
        let ln_raw = target_raw.ln();

        let val = (ln_raw / math_commons::constants::LN_0_7) * n70_val;
        Cycles::new_clamped(val)
    }
}

impl PowerLawModel {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn n70(&self, d: DepthOfDischarge) -> Cycles {
        DegradationModel::n70(self, d)
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn capacity(&self, n: Cycles, d: DepthOfDischarge) -> Capacity {
        DegradationModel::capacity(self, n, d)
    }

    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn cycles_to_capacity(&self, target: Capacity, d: DepthOfDischarge) -> Cycles {
        DegradationModel::cycles_to_capacity(self, target, d)
    }
}

#[cfg(test)]
mod tests {
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
            ((cap_0 / Capacity::new_clamped(1.0)) - 1.0).abs()
                < math_commons::registry::TOLERANCE_FAST
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
    }
}
