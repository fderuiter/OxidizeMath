//! Example target for demo neuron builder.
use math_explorer::biology::neuroscience::neuron::HodgkinHuxleyNeuron;

fn main() {
    // 1. Safe construction via Builder
    let neuron_result = HodgkinHuxleyNeuron::builder()
        .with_initial_v(-65.0)
        .with_n(0.5) // Valid
        .with_m(0.05)
        .with_h(0.6)
        .build();

    match neuron_result {
        Ok(neuron) => println!("Successfully created neuron with v = {}", neuron.v()),
        Err(e) => println!("Failed to create neuron: {}", e),
    }

    // 2. Demonstration of safety (preventing invalid state)
    let invalid_neuron = HodgkinHuxleyNeuron::builder()
        .with_n(1.5) // Invalid: > 1.0
        .build();

    match invalid_neuron {
        Ok(_) => println!("Error: Should not have created invalid neuron!"),
        Err(e) => println!("Correctly rejected invalid neuron: {}", e),
    }

    // 3. Modifying state safely
    let mut neuron = HodgkinHuxleyNeuron::new(-65.0);
    match neuron.set_n(0.8) {
        Ok(_) => println!("Set n to 0.8"),
        Err(e) => println!("Failed to set n: {}", e),
    }

    match neuron.set_n(1.2) {
        Ok(_) => println!("Error: Should not allow setting n > 1.0"),
        Err(e) => println!("Correctly prevented setting n > 1.0: {}", e),
    }
}
