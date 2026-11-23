pub use engine_diffeq::ode::OdeModel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::analysis::ode::{ArrayState, OdeSystem, RungeKutta4, TimeStepper};

    #[derive(Clone)]
    struct TimeRampOde;

    impl OdeSystem<ArrayState<1>> for TimeRampOde {
        #[verified_engine::verified]
        fn derivative(&self, t: f64, _state: &ArrayState<1>) -> ArrayState<1> {
            ArrayState([2.0 * t])
        }

        #[verified_engine::verified]
        fn derivative_in_place(&self, t: f64, _state: &ArrayState<1>, out: &mut ArrayState<1>) {
            out.0[0] = 2.0 * t;
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_ode_model_advances_explicit_time_t() {
        let init = ArrayState([0.0]);
        let mut model = OdeModel::new(init, TimeRampOde, RungeKutta4::new(&init));
        for _ in 0..20 {
            model.step(0.1);
        }
        assert!((model.get_state().0[0] - 4.0).abs() < 1e-10);
        assert!((model.get_time() - 2.0).abs() < 1e-12);
    }
}
