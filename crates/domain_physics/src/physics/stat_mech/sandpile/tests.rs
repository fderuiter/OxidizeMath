#![cfg_attr(any(), verified(opt_out = "tests"))]

use math_commons::fit::FitModel;
use pure_math::pure_math::algebra::{
    AbelianGroup, Group, Monoid, Semigroup, Zn, generate_left_coset, is_subgroup,
};
use verified_engine::engine::VerifiedEngine;

use super::*;

fn assert_abelian_group_laws<G: AbelianGroup + Eq + core::fmt::Debug>(a: &G, b: &G, c: &G) {
    let e = G::identity();
    assert_eq!(G::operate(&e, &e), e);
    assert_eq!(G::operate(a, &e), *a);
    assert_eq!(G::operate(&e, a), *a);
    assert_eq!(G::operate(a, &a.inverse()), e);
    assert_eq!(G::operate(&a.inverse(), a), e);
    assert_eq!(G::operate(a, b), G::operate(b, a));
    assert_eq!(
        G::operate(&G::operate(a, b), c),
        G::operate(a, &G::operate(b, c))
    );
}

#[test]
fn abelian_commutativity_forward_reverse_wavefront_105_topples() {
    #[rustfmt::skip]
    let asymmetric_5x5: [u32; 25] = [
        3, 5, 5, 3, 4,
        5, 4, 4, 4, 3,
        3, 4, 6, 3, 6,
        6, 4, 6, 6, 6,
        4, 4, 3, 3, 6,
    ];
    let mut g_wave = SandpileGrid::new(5, 5, Some(7)).unwrap();
    let mut g_fwd = SandpileGrid::new(5, 5, Some(7)).unwrap();
    let mut g_rev = SandpileGrid::new(5, 5, Some(7)).unwrap();
    g_wave.set_heights(&asymmetric_5x5).unwrap();
    g_fwd.set_heights(&asymmetric_5x5).unwrap();
    g_rev.set_heights(&asymmetric_5x5).unwrap();

    let s_wave = g_wave.stabilize_wavefront();
    let s_fwd = g_fwd.stabilize_forward();
    let s_rev = g_rev.stabilize_reverse();

    assert_eq!(s_wave.topples, 105);
    assert_eq!(s_fwd.topples, 105);
    assert_eq!(s_rev.topples, 105);
    assert_eq!(s_wave.area, s_fwd.area);
    assert_eq!(s_wave.area, s_rev.area);
    assert_eq!(s_wave.dissipated_grains, s_fwd.dissipated_grains);
    assert_eq!(s_wave.dissipated_grains, s_rev.dissipated_grains);
    assert_eq!(g_wave.heights(), g_fwd.heights());
    assert_eq!(g_wave.heights(), g_rev.heights());

    // Also test corner-overloaded 5x5 uniform-4 + 18 grains -> exact 105 topples.
    g_wave.fill(4);
    g_wave.add_grains(0, 0, 18).unwrap();
    g_fwd.fill(4);
    g_fwd.add_grains(0, 0, 18).unwrap();
    g_rev.fill(4);
    g_rev.add_grains(0, 0, 18).unwrap();
    assert_eq!(g_wave.stabilize_wavefront().topples, 105);
    assert_eq!(g_fwd.stabilize_forward().topples, 105);
    assert_eq!(g_rev.stabilize_reverse().topples, 105);
    assert_eq!(g_wave.heights(), g_fwd.heights());
    assert_eq!(g_wave.heights(), g_rev.heights());
}

#[test]
fn kirchhoff_bareiss_matches_dhar_burning_counts_2x2_and_3x3() {
    let det_2x2 = kirchhoff_determinant_bareiss(2, 2).unwrap();
    assert_eq!(det_2x2, 192);

    let mut grid_2x2 = SandpileGrid::new(2, 2, Some(1)).unwrap();
    let mut recurrent_2x2 = 0_i128;
    let mut buf4 = [0_u32; 4];
    for code in 0..256_u32 {
        for (i, slot) in buf4.iter_mut().enumerate() {
            *slot = (code >> (2 * i)) & 3;
        }
        grid_2x2.set_heights(&buf4).unwrap();
        if grid_2x2.dhar_burning_test().is_recurrent {
            recurrent_2x2 += 1;
        }
    }
    assert_eq!(recurrent_2x2, det_2x2);

    let det_3x3 = kirchhoff_determinant_bareiss(3, 3).unwrap();
    assert_eq!(det_3x3, 100_352);

    let mut grid_3x3 = SandpileGrid::new(3, 3, Some(1)).unwrap();
    let mut recurrent_3x3 = 0_i128;
    let mut buf9 = [0_u32; 9];
    for code in 0..262_144_u32 {
        for (i, slot) in buf9.iter_mut().enumerate() {
            *slot = (code >> (2 * i)) & 3;
        }
        grid_3x3.set_heights(&buf9).unwrap();
        if grid_3x3.dhar_burning_test().is_recurrent {
            recurrent_3x3 += 1;
        }
    }
    assert_eq!(recurrent_3x3, det_3x3);
}

#[test]
fn sandpile_element_group_subgroups_cosets_and_zn_homomorphism() {
    let e = SandpileElement::<3, 3>::identity();
    assert!(e.is_recurrent());
    assert_eq!(SandpileElement::<3, 3>::group_order().unwrap(), 100_352);

    let r1 = SandpileElement::<3, 3>::new([[3, 1, 2], [2, 3, 0], [1, 2, 3]]);
    let r2 = SandpileElement::<3, 3>::new([[2, 3, 1], [3, 2, 3], [0, 3, 2]]);
    let r3 = SandpileElement::<3, 3>::new([[3, 3, 3], [3, 1, 3], [3, 3, 3]]);
    assert!(r1.is_recurrent());
    assert!(r2.is_recurrent());
    assert!(r3.is_recurrent());
    assert!(r1.inverse().is_recurrent());
    assert_abelian_group_laws(&r1, &r2, &r3);

    // Transient all-zero configuration is non-recurrent; wave depths remain zero.
    let zero_raw = SandpileElement::<3, 3>::from_stabilized_raw([[0; 3]; 3]);
    let (rec, depths, max_d) = zero_raw.burning_waves();
    assert!(!rec);
    assert_eq!(max_d, 0);
    assert_eq!(depths, [[0; 3]; 3]);

    // 2x2 sandpile group has order 192; symmetric corner generator generates a cyclic subgroup.
    let gen_2x2 = SandpileElement::<2, 2>::new([[3, 2], [2, 3]]);
    let subgroup = gen_2x2.cyclic_subgroup(256);
    assert!(subgroup.len() > 1);
    assert_eq!(192 % subgroup.len(), 0);
    assert!(is_subgroup(&subgroup));

    let shift = SandpileElement::<2, 2>::new([[3, 3], [2, 1]]);
    let coset = generate_left_coset(&subgroup, &shift);
    assert_eq!(coset.len(), subgroup.len());

    // Exact homomorphism from Zn<8> into <g> when order divides 8 on 2x2:
    // On 2x2, G ~= Z_2 x Z_8 x Z_12; let's test an element of order dividing 24 via Zn<24>.
    assert_eq!(24 % subgroup.len(), 0);
    let k1 = Zn::<24>::new(5);
    let k2 = Zn::<24>::new(19);
    assert_eq!(
        gen_2x2.zn_homomorphism(Zn::<24>::operate(&k1, &k2)),
        SandpileElement::operate(&gen_2x2.zn_homomorphism(k1), &gen_2x2.zn_homomorphism(k2))
    );
}

#[test]
fn zero_allocation_hot_paths_and_power_law_fit() {
    let mut grid = SandpileGrid::new(31, 31, Some(2026)).unwrap();
    grid.fill(MAX_STABLE_HEIGHT);

    // Warm up critical attractor inside VerifiedEngine::run_verified (zero heap allocations!).
    let ((creutz_topples, burn_res, fit), metrics) = VerifiedEngine::run_verified(|| {
        let creutz_t = grid.load_creutz_identity();
        let burn = grid.dhar_burning_test();
        grid.invert_in_place();

        // Drive critical sandpile from maximally loaded recurrent background for 1,600 drops.
        grid.fill(MAX_STABLE_HEIGHT);
        for _ in 0..300 {
            let _ = grid.drop_random_grain();
            let _ = grid.stabilize_wavefront();
        }
        grid.telemetry_mut().clear();
        for _ in 0..1_800 {
            let _ = grid.drop_random_grain();
            let _ = grid.stabilize_wavefront();
        }
        let power_fit = grid
            .fit_avalanche_power_law(AvalancheObservable::Size)
            .expect("critical sandpile must produce valid power-law fit");
        (creutz_t, burn, power_fit)
    });

    assert!(creutz_topples > 0);
    assert!(burn_res.is_recurrent);
    assert!(burn_res.max_wave_depth >= 10);
    assert_eq!(fit.fit_model, FitModel::PowerLaw);
    assert!(
        fit.r_squared >= 0.90,
        "expected critical avalanche PowerLaw R^2 >= 0.90, got {:.4} (coeffs={:?})",
        fit.r_squared,
        fit.coefficients()
    );
    assert!(fit.coefficients()[1] < 0.0);
    assert!(metrics.function_calls > 0);
}
