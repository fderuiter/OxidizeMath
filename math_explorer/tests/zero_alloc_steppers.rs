//! Runtime verification tests confirming zero post-initialization heap allocations
//! across ODE, physics, and reaction-diffusion steppers under `VerifiedAllocator`.

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOCATOR: math_explorer::verified_engine::allocator::VerifiedAllocator =
    math_explorer::verified_engine::allocator::VerifiedAllocator;

#[cfg(any(feature = "pure_math", feature = "physics", feature = "biology"))]
static ALLOC_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "pure_math")]
mod pure_math_fixtures {
    use math_explorer::nalgebra::DVector;
    use math_explorer::pure_math::analysis::ode::state::VecState;
    use math_explorer::pure_math::analysis::ode::{OdeSystem, TimeStepper};

    pub struct VecDecay {
        pub state: VecState,
        pub t: f64,
    }

    impl OdeSystem<VecState> for VecDecay {
        fn derivative(&self, t: f64, s: &VecState) -> VecState {
            VecState(s.0.iter().map(|&v| -v + 0.1 * t).collect())
        }

        fn derivative_in_place(&self, t: f64, s: &VecState, out: &mut VecState) {
            for (o, &v) in out.0.iter_mut().zip(&s.0) {
                *o = -v + 0.1 * t;
            }
        }
    }

    impl TimeStepper<VecState> for VecDecay {
        fn get_state(&self) -> &VecState {
            &self.state
        }

        fn get_state_mut(&mut self) -> &mut VecState {
            &mut self.state
        }

        fn current_time(&self) -> f64 {
            self.t
        }

        fn advance_time(&mut self, dt: f64) {
            self.t += dt;
        }
    }

    pub struct DVecDecay {
        pub state: DVector<f64>,
        pub t: f64,
    }

    impl OdeSystem<DVector<f64>> for DVecDecay {
        fn derivative(&self, t: f64, s: &DVector<f64>) -> DVector<f64> {
            s.map(|v| -v + 0.1 * t)
        }

        fn derivative_in_place(&self, t: f64, s: &DVector<f64>, out: &mut DVector<f64>) {
            for (o, &v) in out.iter_mut().zip(s.iter()) {
                *o = -v + 0.1 * t;
            }
        }
    }

    impl TimeStepper<DVector<f64>> for DVecDecay {
        fn get_state(&self) -> &DVector<f64> {
            &self.state
        }

        fn get_state_mut(&mut self) -> &mut DVector<f64> {
            &mut self.state
        }

        fn current_time(&self) -> f64 {
            self.t
        }

        fn advance_time(&mut self, dt: f64) {
            self.t += dt;
        }
    }
}

#[cfg(feature = "pure_math")]
#[test]
fn test_time_stepper_step_with_zero_alloc_on_heap_backed_states() {
    use math_explorer::nalgebra::DVector;
    use math_explorer::pure_math::analysis::ode::state::VecState;
    use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
    use math_explorer::verified_engine::engine::VerifiedEngine;
    use pure_math_fixtures::{DVecDecay, VecDecay};

    let _guard = ALLOC_LOCK.lock().unwrap();
    let mut vec_sys = VecDecay {
        state: VecState(vec![1.0, 2.0, 3.0, 4.0]),
        t: 0.0,
    };
    let mut vec_rk4 = RungeKutta4::new(&vec_sys.state);

    let mut dvec_sys = DVecDecay {
        state: DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
        t: 0.0,
    };
    let mut dvec_rk4 = RungeKutta4::new(&dvec_sys.state);

    let ((), _) = VerifiedEngine::run_verified(|| {
        for _ in 0..16 {
            vec_sys.step_with(&mut vec_rk4, 0.05);
            dvec_sys.step_with(&mut dvec_rk4, 0.05);
        }
    });

    assert!((vec_sys.current_time() - 0.8).abs() < 1e-12);
    assert!((dvec_sys.current_time() - 0.8).abs() < 1e-12);
    assert_eq!(vec_sys.state.0[0].to_bits(), dvec_sys.state[0].to_bits());
}

#[cfg(feature = "physics")]
#[test]
fn test_lorenz_bloch_and_fluid_particle_step_zero_alloc() {
    use math_explorer::nalgebra::{Matrix3, Vector3};
    use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
    use math_explorer::physics::fluid_dynamics::conservation::NavierStokes;
    use math_explorer::physics::fluid_dynamics::solver::FluidParticleSystem;
    use math_explorer::physics::fluid_dynamics::types::{
        FlowState, FluidProperties, SpatialGradients,
    };
    use math_explorer::physics::mri::bloch::BlochSimulator;
    use math_explorer::pure_math::analysis::ode::{IntegrationMethod, TimeStepper};
    use math_explorer::verified_engine::engine::VerifiedEngine;

    let _guard = ALLOC_LOCK.lock().unwrap();
    let init = LorenzState::new(1.0, 1.0, 1.0);
    let mut lorenz_euler = LorenzBuilder::new()
        .integration_method(IntegrationMethod::Euler)
        .build(init);
    let mut lorenz_heun = LorenzBuilder::new()
        .integration_method(IntegrationMethod::Heun)
        .build(init);
    let mut lorenz_rk4 = LorenzBuilder::new()
        .integration_method(IntegrationMethod::RungeKutta4)
        .build(init);

    let m_init = Vector3::new(0.0, 0.0, 1.0);
    let mut bloch_euler =
        BlochSimulator::new(m_init, 1.0).with_integration_method(IntegrationMethod::Euler);
    let mut bloch_heun =
        BlochSimulator::new(m_init, 1.0).with_integration_method(IntegrationMethod::Heun);
    let mut bloch_rk4 =
        BlochSimulator::new(m_init, 1.0).with_integration_method(IntegrationMethod::RungeKutta4);
    let b = Vector3::new(1e-5, 0.0, 2e-5);
    bloch_euler.set_b_field(b);
    bloch_heun.set_b_field(b);
    bloch_rk4.set_b_field(b);

    let water = FluidProperties::water();
    let grads = SpatialGradients::new(
        Matrix3::zeros(),
        Vector3::new(-10.0, 0.0, 0.0),
        Vector3::zeros(),
    );
    let mut fluid = FluidParticleSystem::new(
        &water,
        NavierStokes,
        grads,
        Vector3::new(0.0, -9.81, 0.0),
        FlowState::new(Vector3::new(1.0, 0.0, 0.0), 101_325.0),
    );

    let ((), _) = VerifiedEngine::run_verified(|| {
        for _ in 0..16 {
            lorenz_euler.step(0.01);
            TimeStepper::step(&mut lorenz_heun, 0.01);
            lorenz_rk4.step(0.01);
            bloch_euler.step(1e-4);
            TimeStepper::step(&mut bloch_heun, 1e-4);
            bloch_rk4.step(1e-4);
            TimeStepper::step(&mut fluid, 0.01);
        }
    });

    assert!((lorenz_rk4.time() - 0.16).abs() < 1e-12);
    assert!((bloch_rk4.time() - 1.6e-3).abs() < 1e-12);
    assert!((fluid.time() - 0.16).abs() < 1e-12);
}

#[cfg(feature = "biology")]
mod biology_fixtures {
    use math_explorer::biology::reaction_diffusion::{
        ChemicalState, DiffusionModel, ReactionModel,
    };

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct ThreeSpeciesKinetics;

    impl ReactionModel for ThreeSpeciesKinetics {
        fn reaction_at(&self, t: f64, u: &[f64], out: &mut [f64]) {
            out[0] = -0.1 * u[0] + 0.01 * t;
            out[1] = 0.1 * u[0] - 0.05 * u[1];
            out[2] = 0.05 * u[1];
        }

        fn reaction(&self, u: &[f64], out: &mut [f64]) {
            self.reaction_at(0.0, u, out);
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct SimpleLaplacian1D;

    impl DiffusionModel for SimpleLaplacian1D {
        fn apply(&self, state: &ChemicalState, out: &mut ChemicalState, coeffs: &[f64]) {
            let n = state.grid_size();
            for (s, &coeff) in coeffs.iter().enumerate() {
                let u = state.species(s);
                let dst = out.species_mut(s);
                for i in 0..n {
                    let prev = u[(i + n - 1) % n];
                    let next = u[(i + 1) % n];
                    dst[i] = coeff * (prev - 2.0 * u[i] + next);
                }
            }
        }
    }
}

#[cfg(feature = "biology")]
#[test]
fn test_reaction_diffusion_and_turing_step_zero_alloc() {
    use biology_fixtures::{SimpleLaplacian1D, ThreeSpeciesKinetics};
    use math_explorer::biology::diffusion::FiniteDifference1D;
    use math_explorer::biology::morphogenesis::{
        DiffusionCoeff, SchnakenbergKinetics, StandardSolverAdapter, TuringState, TuringSystem,
    };
    use math_explorer::biology::reaction_diffusion::{ChemicalState, ReactionDiffusionSystem};
    use math_explorer::math_kernel::types::{Dimension, StepSize};
    use math_explorer::pure_math::analysis::ode::{RungeKutta4, TimeStepper};
    use math_explorer::verified_engine::engine::VerifiedEngine;

    let _guard = ALLOC_LOCK.lock().unwrap();
    let rd_template = ChemicalState::new(3, 8);
    let mut rd_sys = ReactionDiffusionSystem::builder()
        .num_species(3)
        .grid_size(8)
        .reaction(ThreeSpeciesKinetics)
        .diffusion(SimpleLaplacian1D)
        .diffusion_coeffs(vec![0.1, 0.05, 0.02])
        .solver(RungeKutta4::new(&rd_template))
        .build_with_solver()
        .unwrap();
    rd_sys.state.species_mut(0).fill(1.0);

    let mut turing_fused = TuringSystem::new(
        Dimension(8),
        DiffusionCoeff(0.16),
        DiffusionCoeff(0.08),
        StepSize(1.0),
    );
    let turing_state = TuringState::<2>::new(8);
    let adapter = StandardSolverAdapter::new(RungeKutta4::new(&turing_state));
    let mut turing_rk4 = TuringSystem::new_with_solver(
        Dimension(8),
        [DiffusionCoeff(0.16), DiffusionCoeff(0.08)],
        SchnakenbergKinetics::default(),
        FiniteDifference1D::new(StepSize(1.0)),
        adapter,
    );

    let ((), _) = VerifiedEngine::run_verified(|| {
        for _ in 0..8 {
            rd_sys.step(0.02);
            TimeStepper::step(&mut turing_fused, 0.02);
            TimeStepper::step(&mut turing_rk4, 0.02);
        }
    });

    assert!((rd_sys.time() - 0.16).abs() < 1e-12);
    assert!((turing_rk4.solver.time() - 0.16).abs() < 1e-12);
}
