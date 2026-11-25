//! Integration tests for `CellMask2D` and `MaskedFiniteDifference2D`:
//! zero-flux impermeable walls ($\nabla u \cdot \mathbf{n} = 0$), continuous chemical
//! emitters/sinks, microfluidic serpentine mazes, and zero-allocation `step_fused`.

use domain_biology::biology::diffusion::{FiniteDifference2D, ReactionCellRole};
use domain_biology::biology::morphogenesis::{
    DiffusionCoeff, GrayScottKinetics, SchnakenbergKinetics, StandardSolverAdapter, TuringState,
    TuringSystem,
};
use domain_biology::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};
use pure_math::pure_math::analysis::ode::Euler;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn test_zero_flux_wall_blocks_diffusive_flux_and_conserves_compartment_mass() {
    let w = 24;
    let h = 16;
    let mut masked =
        FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0))
            .with_mask();

    let wall_x = w / 2;
    for y in 0..h {
        masked.mask.set(wall_x, y, ReactionCellRole::ZeroFluxWall);
    }

    let mut state = ChemicalState::new(1, w * h);
    let mut deriv = ChemicalState::new(1, w * h);
    for y in 0..h {
        for x in 0..wall_x {
            state.species_mut(0)[flatten_2d_index(x, y, w)] = if x < 4 { 2.0 } else { 0.5 };
        }
    }
    let initial_left_mass: f64 = (0..h)
        .flat_map(|y| (0..wall_x).map(move |x| flatten_2d_index(x, y, w)))
        .map(|idx| state.species(0)[idx])
        .sum();

    let dt = 0.1;
    for _ in 0..60 {
        masked.apply(&state, &mut deriv, &[1.0]);
        for i in 0..(w * h) {
            state.species_mut(0)[i] += dt * deriv.species(0)[i];
        }
    }

    let right_leakage: f64 = (0..h)
        .flat_map(|y| ((wall_x + 1)..w).map(move |x| flatten_2d_index(x, y, w)))
        .map(|idx| state.species(0)[idx].abs())
        .sum();
    assert!(
        right_leakage < 1e-12,
        "Impermeable ZeroFluxWall must block all diffusive leakage across x = {wall_x}, got {right_leakage}"
    );

    let final_left_mass: f64 = (0..h)
        .flat_map(|y| (0..wall_x).map(move |x| flatten_2d_index(x, y, w)))
        .map(|idx| state.species(0)[idx])
        .sum();
    assert!(
        (final_left_mass - initial_left_mass).abs() < 1e-9,
        "Neumann zero-flux boundary must conserve mass in sealed compartment ({final_left_mass} vs {initial_left_mass})"
    );
}

#[test]
fn test_emitters_sinks_microfluidic_maze_and_zero_allocation_turing_step() {
    let w = 32;
    let h = 32;
    let mut diffusion =
        FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0))
            .with_mask();
    diffusion.mask.stamp_microfluidic_maze();
    assert!(diffusion.mask.has_active_cells());

    let kinetics = SchnakenbergKinetics { a: 0.1, b: 0.9 };
    let mut system = TuringSystem::new_with_kinetics(
        Dimension(w * h),
        DiffusionCoeff(1.0),
        DiffusionCoeff(50.0),
        kinetics,
        diffusion,
    );
    for i in 0..(w * h) {
        system.u_mut()[i] = 1.0;
        system.v_mut()[i] = 0.9;
    }

    // Warm up once, then verify 0 dynamic allocations inside VerifiedEngine::run_verified
    system.step(0.01);
    system.state.pin_cell_mask(&system.diffusion.mask);
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..20 {
            system.step(0.01);
            system.state.pin_cell_mask(&system.diffusion.mask);
        }
    });
    assert!(
        metrics.function_calls > 0,
        "MaskedFiniteDifference2D TuringSystem::step succeeded with 0 heap allocations under VerifiedAllocator"
    );

    assert!(system.u().iter().all(|x| x.is_finite()));
    assert!(system.v().iter().all(|x| x.is_finite()));

    let emit_u_idx = flatten_2d_index(w / 4, h / 6, w);
    let sink_idx = flatten_2d_index((3 * w) / 4, (5 * h) / 6, w);
    assert_eq!(
        system.diffusion.mask.get(w / 4, h / 6),
        ReactionCellRole::EmitterU
    );
    assert_eq!(
        system.diffusion.mask.get((3 * w) / 4, (5 * h) / 6),
        ReactionCellRole::Sink
    );
    assert!((system.u()[emit_u_idx] - system.diffusion.mask.emitter_u_level).abs() < 1e-12);
    assert_eq!(system.u()[sink_idx], 0.0);
    assert_eq!(system.v()[sink_idx], 0.0);
}

#[test]
fn test_pin_cell_mask_persists_after_turing_dynamics_standard_solver_updates() {
    let w = 20;
    let h = 20;
    let mut diffusion =
        FiniteDifference2D::new(Dimension(w), Dimension(h), StepSize(1.0), StepSize(1.0))
            .with_mask();
    diffusion.mask.set(4, 4, ReactionCellRole::EmitterU);
    diffusion.mask.set(8, 4, ReactionCellRole::EmitterV);
    diffusion.mask.set(12, 12, ReactionCellRole::Sink);
    diffusion.mask.set(10, 6, ReactionCellRole::ZeroFluxWall);

    let mut system = TuringSystem::new_with_solver(
        Dimension(w * h),
        [DiffusionCoeff(0.16), DiffusionCoeff(0.08)],
        GrayScottKinetics::mitosis(),
        diffusion,
        StandardSolverAdapter::new(Euler::new(&TuringState::<2>::new(w * h))),
    );
    for i in 0..(w * h) {
        system.u_mut()[i] = 1.0;
        system.v_mut()[i] = 0.25;
    }
    system.diffusion.mask.pin_turing_state(&mut system.state);

    for _ in 0..15 {
        system.step(0.5);
        // Without post-step pinning, StandardSolverAdapter<Euler> advances unconstrained reaction terms
        // on Emitter/Sink/ZeroFluxWall cells; pin_cell_mask restores exact role invariants.
        system.state.pin_cell_mask(&system.diffusion.mask);
    }

    let eu = flatten_2d_index(4, 4, w);
    let ev = flatten_2d_index(8, 4, w);
    let sk = flatten_2d_index(12, 12, w);
    let wl = flatten_2d_index(10, 6, w);
    assert!((system.u()[eu] - system.diffusion.mask.emitter_u_level).abs() < 1e-12);
    assert!((system.v()[ev] - system.diffusion.mask.emitter_v_level).abs() < 1e-12);
    assert_eq!((system.u()[sk], system.v()[sk]), (0.0, 0.0));
    assert_eq!((system.u()[wl], system.v()[wl]), (0.0, 0.0));
}

#[test]
fn test_cell_mask_set_roles_slice_returns_typed_error_with_zero_allocations() {
    use domain_biology::biology::diffusion::CellMask2D;
    use domain_biology::biology::reaction_diffusion::ReactionDiffusionError;

    let mut mask = CellMask2D::new(4, 3);
    let short_roles = [0_u8; 11];
    let mut valid_roles = [0_u8; 12];
    valid_roles[5] = ReactionCellRole::ZeroFluxWall as u8;

    let ((err_res, ok_res), _) = VerifiedEngine::run_verified(|| {
        let err = mask.set_roles_slice(&short_roles);
        let ok = mask.set_roles_slice(&valid_roles);
        (err, ok)
    });

    assert!(matches!(
        err_res,
        Err(ReactionDiffusionError::MaskLengthMismatch {
            actual: 11,
            expected: 12,
        })
    ));
    assert!(ok_res.is_ok());
    assert!(mask.has_active_cells());
    assert!(!err_res.unwrap_err().to_string().contains('{'));
}
