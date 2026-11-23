//! Integration tests for test neuroscience solver.
use domain_biology::biology::neuroscience::neuron::HodgkinHuxleyNeuron;
use domain_biology::biology::neuroscience::types::HodgkinHuxleyState;
use pure_math::pure_math::analysis::ode::RungeKutta4;

#[test]
#[verified_engine::verified]
fn test_neuron_rk4() {
    let mut neuron = HodgkinHuxleyNeuron::new(-65.0);

    let state = HodgkinHuxleyState {
        v: neuron.v(),
        n: neuron.n(),
        m: neuron.m(),
        h: neuron.h(),
    };

    // Step with RK4
    neuron.update_with(0.01, 10.0, &mut RungeKutta4::new(&state));

    // Check if values changed (simple smoke test)
    assert_ne!(neuron.v(), -65.0);
}
