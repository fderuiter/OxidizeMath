use super::*;
use crate::biology::diffusion::FiniteDifference1D;
use crate::biology::morphogenesis::SchnakenbergKinetics;
use math_commons::math_kernel::types::StepSize;
use oxidize_core::rng::OxidizeRng;

#[test]
#[verified_engine::verified]
fn test_fused_euler_evolution_forwards_rng_to_engine_pde() {
    let kinetics = SchnakenbergKinetics::default();
    let diffusion = FiniteDifference1D::new(StepSize(1.0));
    let dynamics = TuringDynamics {
        kinetics: &kinetics,
        diffusion: &diffusion,
        diffusion_coeffs: [1.0, 20.0],
    };
    let mut state_a = TuringState::<2>::new(32);
    let mut state_b = TuringState::<2>::new(32);
    let mut aux = TuringState::<2>::new(32);
    state_a.concentrations[0].fill(1.0);
    state_a.concentrations[1].fill(0.5);
    state_b.copy_from(&state_a);

    let mut evo_det = FusedEulerEvolution::new(FusedEulerSolver::new(), &dynamics);
    let mut evo_stoch =
        FusedEulerEvolution::with_stochastic_noise(FusedEulerSolver::new(), &dynamics, 0.05, 1.0);
    let mut rng = OxidizeRng::new(77);
    evo_det
        .step(&mut state_a, &mut aux, &mut rng, 0.01)
        .unwrap();
    evo_stoch
        .step(&mut state_b, &mut aux, &mut rng, 0.01)
        .unwrap();
    let u_min = state_b.concentrations[0]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let u_max = state_b.concentrations[0]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(u_max - u_min > 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_cfl_2d_helpers_guarantee_substep_bound() {
    let cfl = FusedEulerSolver::cfl_dt_2d(1.0, 1.0, 100.0);
    assert!((cfl - 0.002).abs() < 1e-15);

    let (substeps, sub_dt) = FusedEulerSolver::cfl_substeps_2d(0.05, 1.0, 1.0, 100.0, 100);
    assert!((25..=100).contains(&substeps));
    assert!(sub_dt <= cfl + 1e-15);
    assert!(((substeps as f64) * sub_dt - 0.05).abs() < 1e-12);

    let (capped_n, capped_dt) = FusedEulerSolver::cfl_substeps_2d(1.0, 1.0, 1.0, 100.0, 10);
    assert_eq!(capped_n, 10);
    assert!(capped_dt <= cfl + 1e-15);
    assert_eq!(
        FusedEulerSolver::cfl_substeps_2d(0.5, 1.0, 0.16, 0.08, 100),
        (1, 0.5)
    );
}
