//! Integration tests for test evolutionary biology.
use domain_biology::biology::evolution::HawkDovePopulation;

#[test]
#[verified_engine::verified]
fn test_hawk_dove_update() {
    let pop = HawkDovePopulation::new(2.0, 4.0); // v=2, c=4
    let p_h = math_commons::primitives::UnitInterval::new(0.8).unwrap();
    let dt = 0.1;

    let next_p_h = pop.update_frequencies(p_h, dt).unwrap();
    println!("Next p_h: {:.10}", next_p_h.value());

    // Captured value from manual inspection (conceptually)
    // Payoff:
    // H-H: (2-4)/2 = -1
    // H-D: 2
    // D-H: 0
    // D-D: 1
    // Matrix: [[-1, 2], [0, 1]]
    // p = [0.8, 0.2]
    // fitness = A * p = [-1*0.8 + 2*0.2, 0*0.8 + 1*0.2] = [-0.8 + 0.4, 0.2] = [-0.4, 0.2]
    // avg_fitness = p . fitness = 0.8 * -0.4 + 0.2 * 0.2 = -0.32 + 0.04 = -0.28
    // dp_H = p_H * (f_H - avg_f) = 0.8 * (-0.4 - (-0.28)) = 0.8 * (-0.12) = -0.096
    // new_p_H = 0.8 + (-0.096) * 0.1 = 0.8 - 0.0096 = 0.7904

    assert!(
        (next_p_h.value() - 0.7904).abs() < math_commons::registry::TOLERANCE_FAST,
        "Expected 0.7904, got {}",
        next_p_h.value()
    );
}
