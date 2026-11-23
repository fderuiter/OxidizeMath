//! Integration tests for test hawk dove.
#[cfg(test)]
mod tests {
    use domain_biology::biology::evolution::HawkDovePopulation;
    use nalgebra::DVector;
    use pure_math::pure_math::analysis::ode::RungeKutta4;

    #[test]
    #[verified_engine::verified]
    fn test_hawk_dove_evolution() {
        // 1. Define the environment
        // Value = 2.0, Cost = 10.0 (High cost of fighting)
        let population = HawkDovePopulation::new(2.0, 10.0);

        // 2. Initial State: Mostly Hawks (90%)
        let mut hawk_freq = math_commons::primitives::UnitInterval::new(0.9).unwrap();
        let dt = 0.1;

        // 3. Evolve over time
        for _ in 0..100 {
            hawk_freq = population.update_frequencies(hawk_freq, dt).unwrap();
        }

        // 4. Check convergence
        // Theoretical Equilibrium: p = V/C = 2/10 = 0.2
        println!("Final Hawk Frequency: {:.3}", hawk_freq.value());
        assert!((hawk_freq.value() - 0.2).abs() < 0.05);
    }

    #[test]
    #[verified_engine::verified]
    fn test_hawk_dove_with_rk4() {
        // 1. Define the environment
        let population = HawkDovePopulation::new(2.0, 10.0);

        // 2. Initial State
        let mut hawk_freq = math_commons::primitives::UnitInterval::new(0.9).unwrap();
        let dt = 0.1;

        // 3. Evolve using RK4
        // We need to maintain state vector for RK4 buffer consistency if we were reusing it,
        // but here we just re-create it per step for simplicity or create one solver.
        // Ideally, for RK4, we should reuse the solver to avoid allocation.

        let initial_state = DVector::from_vec(vec![hawk_freq.value(), hawk_freq.complement()]);
        let mut solver = RungeKutta4::new(&initial_state);

        for _ in 0..100 {
            hawk_freq = population
                .update_frequencies_with_solver(hawk_freq, dt, &mut solver)
                .unwrap();
        }

        // 4. Check convergence
        println!("Final Hawk Frequency (RK4): {:.3}", hawk_freq.value());
        assert!((hawk_freq.value() - 0.2).abs() < 0.05);
    }
}
