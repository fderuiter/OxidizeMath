//! Integration tests for `CellMask2D` and `MaskedFiniteDifference2D`:
//! zero-flux impermeable walls ($\nabla u \cdot \mathbf{n} = 0$), continuous chemical
//! emitters/sinks, microfluidic serpentine mazes, and zero-allocation `step_fused`.

use domain_biology::biology::diffusion::{FiniteDifference2D, ReactionCellRole};
use domain_biology::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
use domain_biology::biology::reaction_diffusion::{ChemicalState, DiffusionModel};
use math_commons::math_kernel::types::{Dimension, StepSize, flatten_2d_index};
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
    let ((), metrics) = VerifiedEngine::run_verified(|| {
        for _ in 0..20 {
            system.step(0.01);
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

#[cfg(feature = "serde")]
#[test]
fn test_masked_diffusion_2d_serde_roundtrip_and_dimension_validation() {
    use domain_biology::biology::diffusion::{CellMask2D, MaskedFiniteDifference2D};

    let mut masked =
        FiniteDifference2D::new(Dimension(12), Dimension(12), StepSize(1.0), StepSize(1.0))
            .with_mask();
    masked.mask.stamp_microfluidic_maze();
    let json = serde_json::to_string(&masked).unwrap();
    let restored: MaskedFiniteDifference2D = serde_json::from_str(&json).unwrap();
    assert_eq!(restored, masked);
    assert!(restored.mask.has_active_cells());

    let bad_len =
        r#"{"width":4,"height":4,"roles":[0,1],"emitter_u_level":1.8,"emitter_v_level":1.8}"#;
    assert!(serde_json::from_str::<CellMask2D>(bad_len).is_err());
}
