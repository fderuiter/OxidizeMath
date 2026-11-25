//! Zero-allocation integration tests under `VerifiedAllocator` for `LSystemEngine`,
//! `VogelPhyllotaxisEngine`, and `ReactionDiffusionSystem` (both generic `ReactionModel`
//! default `add_reaction_batch` and `HillSubstrateReaction`).

use domain_biology::biology::diffusion::FiniteDifference1D;
use domain_biology::biology::lsystem::{
    BotanicalLSystemPreset, LSystemEngine, MAX_LSYSTEM_SYMBOLS,
};
use domain_biology::biology::morphogenesis::{
    DiffusionCoeff, ReactionKinetics, StandardSolverAdapter, TuringState, TuringSystem,
};
use domain_biology::biology::phyllotaxis::{
    DivergenceAnglePreset, GOLDEN_ANGLE_RAD, VogelPhyllotaxisEngine,
};
use domain_biology::biology::reaction_diffusion::{
    ChemicalState, HillSubstrateReaction, ReactionDiffusionSystem, ReactionModel,
};
use math_commons::math_kernel::types::{Dimension, StepSize};
use pure_math::pure_math::analysis::ode::{Heun, RungeKutta4, TimeStepper};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

struct GenericTwoSpeciesBrusselator {
    a: f64,
    b: f64,
}

impl ReactionModel for GenericTwoSpeciesBrusselator {
    #[verified_engine::verified]
    fn reaction(&self, concs: &[f64], rates: &mut [f64]) {
        let u = concs[0];
        let v = concs[1];
        let u2v = u * u * v;
        rates[0] = self.a - (self.b + 1.0) * u + u2v;
        rates[1] = self.b * u - u2v;
    }
}

#[test]
#[verified_engine::verified]
fn test_botanical_zero_heap_allocations() {
    let (d0l_grammar, d0l_cfg) = BotanicalLSystemPreset::HerbaceousShoot
        .build_grammar(25.0, 0.82, 0.72)
        .expect("valid D0L grammar");
    let mut d0l_engine =
        LSystemEngine::new(d0l_grammar, d0l_cfg, MAX_LSYSTEM_SYMBOLS).expect("pre-allocated D0L");

    let (param_grammar, param_cfg) = BotanicalLSystemPreset::ParametricDichotomousTree
        .build_grammar(30.0, 0.75, 0.68)
        .expect("valid parametric grammar");
    let mut param_engine = LSystemEngine::new(param_grammar, param_cfg, MAX_LSYSTEM_SYMBOLS)
        .expect("pre-allocated parametric L-system");

    let (_, lsys_metrics) = VerifiedEngine::run_verified(|| {
        d0l_engine.reset_to_axiom().unwrap();
        for _ in 0..5 {
            d0l_engine.step().unwrap();
        }
        param_engine.derive_bounded(8).unwrap();
    });
    assert!(lsys_metrics.function_calls > 0);
    assert!(!d0l_engine.segments().is_empty());
    assert!(!param_engine.segments().is_empty());

    let mut vogel = VogelPhyllotaxisEngine::golden(512, 1.0).expect("pre-allocated phyllotaxis");
    let (_, phyl_metrics) = VerifiedEngine::run_verified(|| {
        vogel.set_parameters(600, GOLDEN_ANGLE_RAD, 1.2).unwrap();
        let _ = vogel.trace_parastichy_arm(0, vogel.analysis().primary_count);
        vogel
            .set_parameters(
                300,
                DivergenceAnglePreset::RationalTwoFifths.angle_rad(),
                1.0,
            )
            .unwrap();
    });
    assert!(phyl_metrics.function_calls > 0);
}

#[test]
#[verified_engine::verified]
fn test_reaction_diffusion_zero_heap_allocations() {
    let mut generic_rd = ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(64)
        .reaction(GenericTwoSpeciesBrusselator { a: 1.0, b: 2.5 })
        .diffusion(FiniteDifference1D::new(StepSize(0.5)))
        .diffusion_coeffs(vec![0.1, 0.4])
        .build()
        .expect("build generic Brusselator RD");
    for i in 0..64 {
        generic_rd.state.species_mut(0)[i] = 1.0 + 0.05 * ((i as f64) * 0.2).sin();
        generic_rd.state.species_mut(1)[i] = 2.5;
    }

    let hill_reaction =
        HillSubstrateReaction::new(1.8, 0.5, 3.0, 2.0, 0.25, 0.4, 0.02).expect("valid Hill RD");
    let mut hill_rd = ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(64)
        .reaction(hill_reaction)
        .diffusion(FiniteDifference1D::new(StepSize(0.5)))
        .diffusion_coeffs(vec![0.08, 0.35])
        .build()
        .expect("build Hill RD");
    for i in 0..64 {
        hill_rd.state.species_mut(0)[i] = if (28..36).contains(&i) { 1.2 } else { 0.1 };
        hill_rd.state.species_mut(1)[i] = 2.0;
    }

    let mut prealloc_heun = Heun::new(&generic_rd.state);
    let (_, rd_metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..10 {
            generic_rd.step(0.01);
            <ReactionDiffusionSystem<_, _, _> as TimeStepper<ChemicalState>>::step_with(
                &mut generic_rd,
                &mut prealloc_heun,
                0.01,
            );
            hill_rd.step(0.01);
        }
    });
    assert!(rd_metrics.function_calls > 0);
    assert!(generic_rd.state.species(0).iter().all(|v| v.is_finite()));
    assert!(hill_rd.state.species(0).iter().all(|v| v.is_finite()));
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct TimeDrivenReaction;

impl ReactionModel for TimeDrivenReaction {
    #[verified_engine::verified]
    fn reaction(&self, _concs: &[f64], rates: &mut [f64]) {
        rates.fill(0.0);
    }

    #[verified_engine::verified]
    fn reaction_at_time(&self, t: f64, _concs: &[f64], rates: &mut [f64]) {
        rates[0] = 2.0 * t;
        if rates.len() > 1 {
            rates[1] = 0.0;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct TimeDrivenKinetics;

impl ReactionKinetics<2> for TimeDrivenKinetics {
    #[verified_engine::verified]
    fn reaction(&self, _concentrations: [f64; 2]) -> [f64; 2] {
        [0.0, 0.0]
    }

    #[verified_engine::verified]
    fn reaction_at_time(&self, t: f64, _index: usize, _concentrations: [f64; 2]) -> [f64; 2] {
        [2.0 * t, 0.0]
    }
}

#[test]
#[verified_engine::verified]
fn test_non_autonomous_reaction_diffusion_time_advancement() {
    let mut rd = ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(8)
        .reaction(TimeDrivenReaction)
        .diffusion(FiniteDifference1D::new(StepSize(1.0)))
        .diffusion_coeffs(vec![0.0, 0.0])
        .build()
        .unwrap();
    assert_eq!(rd, rd.clone());
    assert!(!format!("{rd:?}").is_empty());

    let dt = 0.25;
    for _ in 0..8 {
        rd.step(dt);
    }
    assert!((rd.time() - 2.0).abs() < 1e-14);
    assert!((rd.state.species(0)[0] - 3.5).abs() < 1e-13);

    let mut rd_rk4 = ReactionDiffusionSystem::builder()
        .num_species(2)
        .grid_size(8)
        .reaction(TimeDrivenReaction)
        .diffusion(FiniteDifference1D::new(StepSize(1.0)))
        .diffusion_coeffs(vec![0.0, 0.0])
        .build()
        .unwrap();
    let mut rk4 = RungeKutta4::new(&rd_rk4.state);
    for _ in 0..8 {
        <ReactionDiffusionSystem<_, _, _> as TimeStepper<ChemicalState>>::step_with(
            &mut rd_rk4,
            &mut rk4,
            dt,
        );
    }
    assert!((rd_rk4.state.species(0)[0] - 4.0).abs() < 1e-13);

    let rd_err = ReactionDiffusionSystem::builder()
        .num_species(2)
        .reaction(TimeDrivenReaction)
        .diffusion(FiniteDifference1D::new(StepSize(1.0)))
        .build()
        .unwrap_err();
    assert_eq!(rd_err, rd_err.clone());
    assert!(!format!("{rd_err:?}").is_empty());
}

#[test]
#[verified_engine::verified]
fn test_non_autonomous_turing_standard_solver_adapter_time_advancement() {
    let seed_state = TuringState::<2>::new(8);
    let adapter = StandardSolverAdapter::new(RungeKutta4::new(&seed_state));
    let mut turing = TuringSystem::new_with_solver(
        Dimension(8),
        [DiffusionCoeff(0.0), DiffusionCoeff(0.0)],
        TimeDrivenKinetics,
        FiniteDifference1D::new(StepSize(1.0)),
        adapter,
    );
    assert_eq!(turing, turing.clone());
    assert!(!format!("{turing:?}").is_empty());

    let dt = 0.25;
    for _ in 0..8 {
        turing.step(dt);
    }
    assert!((turing.solver.time() - 2.0).abs() < 1e-14);
    assert!((turing.u()[0] - 4.0).abs() < 1e-13);
}
