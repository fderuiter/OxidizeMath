//! Example target for battery degradation demo.
use math_explorer::applied::battery_degradation::{Cycles, DepthOfDischarge, PowerLawModel};

fn main() {
    // 1. Initialize the standard model
    let model = PowerLawModel::standard();

    // 2. Define a scenario: 80% to 20% charge window = 60% DoD
    let dod = DepthOfDischarge::new_clamped(60.0);

    // 3. Estimate Life Expectancy (Cycles to 70% SOH)
    let life_cycles = model.n70(dod);
    println!("Expected Life: {:.0} cycles", life_cycles.as_f64());

    // 4. Predict Capacity after 1000 cycles
    let current_cycles = Cycles::new_clamped(1000.0);
    let remaining_capacity = model.capacity(current_cycles, dod);
    println!(
        "Capacity after 1000 cycles: {:.1}%",
        remaining_capacity.as_f64() * 100.0
    );
}
