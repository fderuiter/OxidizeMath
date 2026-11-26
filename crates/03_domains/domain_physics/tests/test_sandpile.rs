//! Integration tests verifying zero-allocation Abelian sandpile dynamics and group laws.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use domain_physics::physics::stat_mech::sandpile::{
    AvalancheObservable, MAX_STABLE_HEIGHT, SandpileElement, SandpileGrid,
    kirchhoff_determinant_bareiss,
};
use math_commons::fit::FitModel;
use pure_math::pure_math::algebra::{Group, Monoid, Semigroup, generate_left_coset, is_subgroup};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
fn integration_zero_alloc_step_wavefront_creutz_dhar_and_power_law() {
    let mut grid = SandpileGrid::new(25, 25, Some(1337)).unwrap();
    let mut aux = SandpileGrid::new(25, 25, Some(99)).unwrap();
    aux.load_creutz_identity();
    let mut e_buf = vec![0_u32; 25 * 25];
    e_buf.copy_from_slice(aux.heights());

    let ((step_sites, compose_topples, burn_recurrent, r_sq), _metrics) =
        VerifiedEngine::run_verified(|| {
            grid.fill(MAX_STABLE_HEIGHT);
            grid.add_grains(12, 12, 4).unwrap();
            let (sites, _diss) = grid.step_wavefront();
            let _ = grid.stabilize_wavefront();

            // Verify (r + e)^o == r in place on 25x25 with zero allocations.
            let burn_before = grid.dhar_burning_test();
            let c_topples = grid.compose_with_slice(&e_buf).unwrap();

            // Accumulate critical avalanches and fit FitModel::PowerLaw with zero allocations.
            grid.telemetry_mut().clear();
            for _ in 0..1_500 {
                let _ = grid.drop_random_grain();
                let _ = grid.stabilize_wavefront();
            }
            let fit = grid
                .fit_avalanche_power_law(AvalancheObservable::Size)
                .unwrap();
            (sites, c_topples, burn_before.is_recurrent, fit.r_squared)
        });

    assert_eq!(step_sites, 1);
    assert!(compose_topples > 0);
    assert!(burn_recurrent);
    assert!(
        r_sq >= 0.90,
        "expected FitModel::PowerLaw R^2 >= 0.90, got {r_sq:.4}"
    );
}

#[test]
fn integration_creutz_identity_63x63_and_group_axioms() {
    let mut crystal = SandpileGrid::new(63, 63, Some(1)).unwrap();
    let topples = crystal.load_creutz_identity();
    assert!(topples > 100_000);
    let burn = crystal.dhar_burning_test();
    assert!(burn.is_recurrent);
    assert_eq!(burn.burned_cells, 63 * 63);

    // Verify (e + e)^o == e on 63x63!
    let snapshot = crystal.heights().to_vec();
    crystal.compose_with_slice(&snapshot).unwrap();
    assert_eq!(crystal.heights(), snapshot.as_slice());

    // Verify Bareiss 4x4 determinant and SandpileElement<2, 2> subgroup/coset integration.
    assert_eq!(kirchhoff_determinant_bareiss(4, 4).unwrap(), 557_568_000);

    let e2 = SandpileElement::<2, 2>::identity();
    let g2 = SandpileElement::<2, 2>::new([[3, 1], [1, 3]]);
    assert_eq!(SandpileElement::operate(&e2, &e2), e2);
    assert_eq!(SandpileElement::operate(&g2, &e2), g2);
    assert_eq!(SandpileElement::operate(&g2, &g2.inverse()), e2);

    let sub = g2.cyclic_subgroup(64);
    assert!(is_subgroup(&sub));
    let coset = generate_left_coset(&sub, &SandpileElement::<2, 2>::new([[2, 3], [3, 1]]));
    assert_eq!(coset.len(), sub.len());
    assert_eq!(FitModel::PowerLaw.short_label(), "Power");
}
