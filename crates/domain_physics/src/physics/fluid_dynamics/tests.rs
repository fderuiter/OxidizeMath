#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::physics::fluid_dynamics::{
        analysis::{bernoulli_constant, reynolds_number, shear_stress},
        conservation::{
            Euler as FluidEuler, MomentumEquation, NavierStokes, continuity_divergence,
            material_derivative_scalar,
        },
        regimes::{FlatPlateClassifier, FlowClassifier, FlowRegime, PipeFlowClassifier},
        solver::FluidParticleSystem,
        types::{FlowState, FluidProperties, SpatialGradients},
    };
    use nalgebra::{Matrix3, Vector3};
    use pure_math::pure_math::analysis::ode::{Euler as OdeEuler, SolverExt};

    #[test]
    #[verified_engine::verified]
    fn test_fluid_properties() {
        let water = FluidProperties::water();
        assert!((water.density() - 998.2).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!(
            (water.dynamic_viscosity() - 1.002e-3).abs() < math_commons::registry::TOLERANCE_FAST
        );

        let nu = water.kinematic_viscosity();
        assert!((nu - 1.002e-3 / 998.2).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_fluid_properties_validation() {
        assert!(FluidProperties::new(-10.0, 1.0).is_err()); // Negative density
        assert!(FluidProperties::new(10.0, -1.0).is_err()); // Negative viscosity
        assert!(FluidProperties::new(0.0, 1.0).is_err()); // Zero density
    }

    #[test]
    #[verified_engine::verified]
    fn test_material_derivative() {
        let velocity = Vector3::new(1.0, 2.0, 3.0);
        let gradient = Vector3::new(0.1, 0.2, 0.3);
        let local_change = 0.5;

        // D/Dt = 0.5 + 1*0.1 + 2*0.2 + 3*0.3 = 0.5 + 0.1 + 0.4 + 0.9 = 1.9
        let result = material_derivative_scalar(local_change, velocity, gradient);
        assert!((result - 1.9).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_reynolds_number_and_strategies() {
        let props = FluidProperties::new(1000.0, 0.001).unwrap(); // Water-like
        let u = 2.0;
        let l = 0.5;

        // Re = 1000 * 2 * 0.5 / 0.001 = 1_000_000
        let re = reynolds_number(&props, u, l);
        assert!((re - 1_000_000.0).abs() < math_commons::registry::TOLERANCE_STANDARD);

        // Pipe Flow (default)
        let pipe_classifier = PipeFlowClassifier;
        assert_eq!(pipe_classifier.classify(re), FlowRegime::Turbulent);

        let re_laminar = reynolds_number(&props, 0.001, 0.1); // Re = 100
        assert_eq!(pipe_classifier.classify(re_laminar), FlowRegime::Laminar);

        // Flat Plate
        let plate_classifier = FlatPlateClassifier;
        // Re = 1,000,000 > 500,000 -> Turbulent
        assert_eq!(plate_classifier.classify(re), FlowRegime::Turbulent);

        // Re = 400,000 -> Laminar for Plate, but Turbulent for Pipe
        let re_intermediate = 400_000.0;
        assert_eq!(
            plate_classifier.classify(re_intermediate),
            FlowRegime::Laminar
        );
        assert_eq!(
            pipe_classifier.classify(re_intermediate),
            FlowRegime::Turbulent
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_bernoulli() {
        let props = FluidProperties::new(1000.0, 1.0).unwrap(); // rho=1000
        let g = 9.81;
        let state = FlowState::new(Vector3::new(10.0, 0.0, 0.0), 101325.0);
        let h = 5.0;

        // P + 0.5 rho v^2 + rho g h
        // 101325 + 500 * 100 + 1000 * 9.81 * 5
        // 101325 + 50000 + 49050 = 200375
        let constant = bernoulli_constant(&state, &props, h, g);
        assert!((constant - 200375.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_shear_stress() {
        let props = FluidProperties::new(1000.0, 0.001).unwrap(); // mu = 0.001
        let grad_u = 500.0; // 1/s

        // tau = mu * du/dy = 0.001 * 500 = 0.5 Pa
        let tau = shear_stress(&props, grad_u);
        assert!((tau - 0.5).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_navier_stokes_simple_couette() {
        let props = FluidProperties::new(1.0, 1.0).unwrap(); // rho=1, mu=1 -> nu=1
        let state = FlowState::new(Vector3::new(1.0, 0.0, 0.0), 0.0);

        let vel_grad = Matrix3::zeros();
        let p_grad = Vector3::zeros();
        let lap_vel = Vector3::zeros();
        let g = Vector3::zeros();

        let gradients = SpatialGradients::new(vel_grad, p_grad, lap_vel);
        let accel = NavierStokes.acceleration(&props, &state, &gradients, g);
        assert_eq!(accel, Vector3::zeros());

        let p_grad_x = Vector3::new(2.0, 0.0, 0.0);
        let gradients_p = SpatialGradients::new(vel_grad, p_grad_x, lap_vel);
        let accel_p = NavierStokes.acceleration(&props, &state, &gradients_p, g);
        assert!((accel_p.x - (-2.0)).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_continuity() {
        assert_eq!(continuity_divergence(0.0), 0.0);
        assert_eq!(continuity_divergence(0.5), 0.5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_strategy_pattern_composability() {
        let props = FluidProperties::new(1000.0, 0.001).unwrap();
        let state = FlowState::new(Vector3::zeros(), 101325.0);
        let gradients = SpatialGradients::new(
            Matrix3::zeros(),
            Vector3::new(100.0, 0.0, 0.0), // Pressure gradient
            Vector3::zeros(),
        );
        let g = Vector3::zeros();

        // 1. Test Navier-Stokes Strategy directly
        let ns = NavierStokes;
        let accel_ns = ns.acceleration(&props, &state, &gradients, g);
        // a = -grad(p)/rho = -100 / 1000 = -0.1
        assert!((accel_ns.x - (-0.1)).abs() < math_commons::registry::TOLERANCE_STANDARD);

        // 2. Test Euler Strategy directly
        let euler = FluidEuler;
        let accel_euler = euler.acceleration(&props, &state, &gradients, g);
        assert!((accel_euler.x - (-0.1)).abs() < math_commons::registry::TOLERANCE_STANDARD);

        // 3. Test dynamic dispatch (simulated)
        let strategies: Vec<Box<dyn MomentumEquation>> =
            vec![Box::new(NavierStokes), Box::new(FluidEuler)];
        for strategy in strategies {
            let acc = strategy.acceleration(&props, &state, &gradients, g);
            assert!((acc.x - (-0.1)).abs() < math_commons::registry::TOLERANCE_STANDARD);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_fluid_particle_integration() {
        // Setup: Water particle driven by pressure gradient
        let props = FluidProperties::water();
        let initial_velocity = Vector3::zeros();
        let state = FlowState::new(initial_velocity, 101325.0);

        // Gradient: dp/dx = -100 Pa/m (pressure drops in +x direction)
        // This should cause acceleration a = -(-100)/998.2 = +0.10018 m/s^2
        let gradients = SpatialGradients::new(
            Matrix3::zeros(),
            Vector3::new(-100.0, 0.0, 0.0),
            Vector3::zeros(),
        );

        let body_force = Vector3::zeros();
        let strategy = NavierStokes;

        let system = FluidParticleSystem::new(&props, strategy, gradients, body_force, state);

        // Integrate for 1 second
        let dt = 1.0;
        let mut solver = OdeEuler::new(&state);
        let next_state = solver.solve(&system, 0.0, &state, dt);

        let expected_accel = 100.0 / 998.2;
        assert!((next_state.velocity.x - expected_accel).abs() < 1e-4);
        assert!(
            (next_state.pressure - 101325.0).abs() < math_commons::registry::TOLERANCE_STANDARD
        ); // Pressure should be constant
    }
}
