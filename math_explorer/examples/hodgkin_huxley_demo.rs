//! Example target for hodgkin huxley demo.
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;

fn main() {
    // Initialize a neuron at resting potential (-65.0 mV)
    let mut neuron = HodgkinHuxleyNeuron::new(-65.0);
    let dt = 0.01; // 0.01 ms time step

    // Simulate for 10ms with 10 uA/cm^2 current injection
    for _ in 0..1000 {
        neuron.update(dt, 10.0);
        if neuron.v() > 0.0 {
            println!("Action Potential Generated!");
            break;
        }
    }
}
