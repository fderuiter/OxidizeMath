//! Integration, topological invariance, mesh manifold, and zero-allocation verification tests
//! for `LaurentPolynomial`, `BraidWord`, and `pure_math::knot_theory`.

use nalgebra::Point3;
use pure_math::pure_math::algebra::traits::Ring;
use pure_math::pure_math::algebra::{BraidWord, LaurentPolynomial};
use pure_math::pure_math::knot_theory::{
    BishopFrame, BurauWorkspace, CrossingSign, KnotError, LissajousKnot, MAX_KAUFFMAN_CROSSINGS,
    PlanarDiagram, TorusKnot, alexander_polynomial_burau, alexander_polynomial_fox,
    borromean_rings, build_link_tubular_mesh, compute_bishop_frames_in_place, figure_eight_knot,
    format_jones_polynomial, gauss_linking_number, gauss_linking_quadrature,
    gauss_writhe_quadrature, hopf_link, is_chiral_by_jones, jones_polynomial,
    pairwise_linking_matrix, pairwise_linking_numbers_in_place,
};
use std::collections::HashMap;
use verified_engine::engine::VerifiedEngine;

#[test]
fn laurent_polynomial_ring_laws_and_trefoil_chirality() {
    let p = LaurentPolynomial::new(-3, vec![2_i64, -1, 0, 4, -5]);
    let q = LaurentPolynomial::new(-1, vec![-3_i64, 2, 7]);
    let r = LaurentPolynomial::new(1, vec![1_i64, -2]);

    assert_eq!((&p + &q) + &r, &p + (&q + &r));
    assert_eq!(&p * (&q + &r), (&p * &q) + (&p * &r));
    assert_eq!(&p + (-&p), LaurentPolynomial::zero());
    assert_eq!(&p * LaurentPolynomial::one(), p);
    assert_eq!(p.invert_variable().invert_variable(), p);

    // Right and left trefoils from both BraidWord and 3D TorusKnot projection
    let right_diag = PlanarDiagram::trefoil(true);
    let left_diag = PlanarDiagram::trefoil(false);
    let expected_right_jones = LaurentPolynomial::from_terms(&[(1, 1_i64), (3, 1), (4, -1)]);
    let expected_left_jones = LaurentPolynomial::from_terms(&[(-4, -1_i64), (-3, 1), (-1, 1)]);
    let expected_alex = LaurentPolynomial::new(-1, vec![1_i64, -1, 1]);

    let jr = jones_polynomial(&right_diag).unwrap();
    let jl = jones_polynomial(&left_diag).unwrap();
    assert_eq!(jr, expected_right_jones);
    assert_eq!(jl, expected_left_jones);
    assert_ne!(jr, jl);
    assert_eq!(jr.invert_variable(), jl);
    assert!(is_chiral_by_jones(&right_diag).unwrap());
    assert!(is_chiral_by_jones(&left_diag).unwrap());

    assert_eq!(
        alexander_polynomial_fox(&right_diag).unwrap(),
        expected_alex
    );
    assert_eq!(alexander_polynomial_fox(&left_diag).unwrap(), expected_alex);

    // Verify 3D TorusKnot(2, +-3) projections produce identical Jones & Alexander invariants
    let torus_r = TorusKnot::right_trefoil().to_link(120).unwrap();
    let torus_l = TorusKnot::left_trefoil().to_link(120).unwrap();
    let proj_r = PlanarDiagram::from_3d_link(&torus_r).unwrap();
    let proj_l = PlanarDiagram::from_3d_link(&torus_l).unwrap();
    assert_eq!(jones_polynomial(&proj_r).unwrap(), expected_right_jones);
    assert_eq!(jones_polynomial(&proj_l).unwrap(), expected_left_jones);
    assert!(gauss_writhe_quadrature(&torus_r.components[0]).unwrap() > 2.5);
    assert!(gauss_writhe_quadrature(&torus_l.components[0]).unwrap() < -2.5);
}

#[test]
fn fox_and_burau_alexander_agree_across_knots_and_links() {
    let cases = [
        (
            BraidWord::new(2, vec![1, 1, 1]).unwrap(),
            LaurentPolynomial::new(-1, vec![1_i64, -1, 1]),
        ),
        (
            BraidWord::new(3, vec![1, -2, 1, -2]).unwrap(),
            LaurentPolynomial::new(-1, vec![-1_i64, 3, -1]),
        ),
        (
            BraidWord::new(2, vec![1, 1]).unwrap(),
            LaurentPolynomial::new(0, vec![-1_i64, 1]),
        ),
        (
            BraidWord::new(3, vec![1, -2, 1, -2, 1, -2]).unwrap(),
            LaurentPolynomial::new(-2, vec![1_i64, -4, 6, -4, 1]),
        ),
        (
            BraidWord::new(2, vec![1, 1, 1, 1, 1]).unwrap(),
            LaurentPolynomial::new(-2, vec![1_i64, -1, 1, -1, 1]),
        ),
    ];

    for (braid, expected) in &cases {
        let diag = PlanarDiagram::from_braid(braid);
        let fox = alexander_polynomial_fox(&diag).unwrap();
        let burau = alexander_polynomial_burau(braid).unwrap();
        assert_eq!(&fox, expected);
        assert_eq!(&burau, expected);
    }

    // Figure-eight 3D curve projection check: amphichiral Jones t^-2 - t^-1 + 1 - t + t^2
    let fig8_3d = figure_eight_knot(1.5, 160).unwrap();
    let fig8_diag = PlanarDiagram::from_3d_link(&fig8_3d).unwrap();
    assert_eq!(
        jones_polynomial(&fig8_diag).unwrap(),
        LaurentPolynomial::new(-2, vec![1_i64, -1, 1, -1, 1])
    );
    assert!(!is_chiral_by_jones(&fig8_diag).unwrap());
}

#[test]
fn reidemeister_i_ii_iii_preserve_alexander_and_jones_while_ri_updates_writhe() {
    let mut diag = PlanarDiagram::figure_eight();
    let base_alex = alexander_polynomial_fox(&diag).unwrap();
    let base_jones = jones_polynomial(&diag).unwrap();
    let base_writhe = diag.writhe();
    assert_eq!(base_writhe, 0);

    // Reidemeister I (+): adds +1 to diagram writhe, preserves Alexander & Jones
    let c_curl = diag
        .apply_reidemeister_i_add(0, CrossingSign::Positive)
        .unwrap();
    diag.validate().unwrap();
    assert_eq!(diag.writhe(), base_writhe + 1);
    assert_eq!(alexander_polynomial_fox(&diag).unwrap(), base_alex);
    assert_eq!(jones_polynomial(&diag).unwrap(), base_jones);

    // Lifted 3D writhe reflects positive vs negative R-I twist
    let lifted_pos = diag.lift_to_3d_link(8).unwrap();
    let writhe_3d_pos = gauss_writhe_quadrature(&lifted_pos.components[0]).unwrap();
    assert!(diag.can_apply_reidemeister_i_remove(c_curl));
    diag.apply_reidemeister_i_remove(c_curl).unwrap();
    diag.validate().unwrap();
    assert_eq!(diag.writhe(), base_writhe);

    let c_neg = diag
        .apply_reidemeister_i_add(0, CrossingSign::Negative)
        .unwrap();
    let lifted_neg = diag.lift_to_3d_link(8).unwrap();
    let writhe_3d_neg = gauss_writhe_quadrature(&lifted_neg.components[0]).unwrap();
    assert!(writhe_3d_pos > writhe_3d_neg);
    diag.apply_reidemeister_i_remove(c_neg).unwrap();

    // Reidemeister II poke & unpoke
    let (c_b1, c_b2) = diag.apply_reidemeister_ii_add(0, 2).unwrap();
    diag.validate().unwrap();
    assert_eq!(diag.writhe(), base_writhe);
    assert_eq!(alexander_polynomial_fox(&diag).unwrap(), base_alex);
    assert_eq!(jones_polynomial(&diag).unwrap(), base_jones);
    assert!(diag.can_apply_reidemeister_ii_remove(c_b1, c_b2));
    diag.apply_reidemeister_ii_remove(c_b1, c_b2).unwrap();
    diag.validate().unwrap();
    assert_eq!(alexander_polynomial_fox(&diag).unwrap(), base_alex);
    assert_eq!(jones_polynomial(&diag).unwrap(), base_jones);

    // Reidemeister III triangle slide on a 3-strand braid s1 * s2 * s1 * s2^-1
    let r3_braid = BraidWord::new(3, vec![1, 2, 1, -2]).unwrap();
    let mut r3_diag = PlanarDiagram::from_braid(&r3_braid);
    let r3_alex = alexander_polynomial_fox(&r3_diag).unwrap();
    let r3_jones = jones_polynomial(&r3_diag).unwrap();
    let r3_writhe = r3_diag.writhe();
    let tris = r3_diag.find_reidemeister_iii_triangles();
    assert!(!tris.is_empty(), "expected at least one R-III triangle");
    let tri = tris[0];
    r3_diag
        .apply_reidemeister_iii(tri.c_top_bot, tri.c_top_mid, tri.c_mid_bot)
        .unwrap();
    r3_diag.validate().unwrap();
    assert_eq!(r3_diag.writhe(), r3_writhe);
    assert_eq!(alexander_polynomial_fox(&r3_diag).unwrap(), r3_alex);
    assert_eq!(jones_polynomial(&r3_diag).unwrap(), r3_jones);
}

#[test]
fn hopf_link_and_borromean_rings_gauss_linking_and_watertight_meshes() {
    let hopf_pos = hopf_link(1.5, true, 96).unwrap();
    let hopf_neg = hopf_link(1.5, false, 96).unwrap();
    assert_eq!(
        gauss_linking_number(&hopf_pos.components[0], &hopf_pos.components[1]).unwrap(),
        1
    );
    assert_eq!(
        gauss_linking_number(&hopf_neg.components[0], &hopf_neg.components[1]).unwrap(),
        -1
    );
    let hopf_diag = PlanarDiagram::from_3d_link(&hopf_pos).unwrap();
    assert_eq!(hopf_diag.combinatorial_linking_matrix()[0][1], 1);
    assert_eq!(
        format_jones_polynomial(&hopf_diag).unwrap(),
        "-t^(1/2) - t^(5/2)"
    );

    let borromean = borromean_rings(1.8, 0.9, 120).unwrap();
    let lk_mat = pairwise_linking_matrix(&borromean).unwrap();
    assert_eq!(lk_mat, vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]);

    let borr_diag = PlanarDiagram::from_3d_link(&borromean).unwrap();
    assert_eq!(borr_diag.crossing_count(), 12);
    assert_eq!(PlanarDiagram::borromean_rings().crossing_count(), 6);
    assert_eq!(borr_diag.component_count(), 3);
    let borr_jones = jones_polynomial(&borr_diag).unwrap();
    assert_eq!(
        borr_jones,
        jones_polynomial(&PlanarDiagram::borromean_rings()).unwrap()
    );
    assert_eq!(
        borr_jones,
        LaurentPolynomial::new(-3, vec![-1_i64, 3, -2, 4, -2, 3, -1])
    );

    // Watertight oriented manifold check: every directed edge (u, v) has exact reverse (v, u)
    let mesh = build_link_tubular_mesh(&borromean, 0.14, 12).unwrap();
    assert_eq!(mesh.indices.len() % 3, 0);
    let mut half_edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
    for &[a, b, c] in mesh.indices.as_chunks::<3>().0 {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *half_edge_counts.entry((u, v)).or_insert(0) += 1;
        }
    }
    for (&(u, v), &cnt) in &half_edge_counts {
        assert_eq!(cnt, 1, "non-manifold duplicate half-edge ({u}, {v})");
        assert_eq!(
            half_edge_counts.get(&(v, u)).copied(),
            Some(1),
            "boundary leak on edge ({u}, {v})"
        );
    }
}

#[test]
fn crossing_cap_gracefully_guards_state_sum_above_sixteen() {
    let gens = vec![1_i32; MAX_KAUFFMAN_CROSSINGS + 1];
    let big_braid = BraidWord::new(2, gens).unwrap();
    let big_diag = PlanarDiagram::from_braid(&big_braid);
    assert_eq!(big_diag.crossing_count(), 17);

    let err = jones_polynomial(&big_diag).unwrap_err();
    assert!(matches!(
        err,
        KnotError::CrossingCapExceeded {
            crossings: 17,
            max_crossings: 16
        }
    ));

    // Polynomial-time writhe and Fox/Burau Alexander polynomials remain active above 16 crossings
    assert_eq!(big_diag.writhe(), 17);
    let alex_fox = alexander_polynomial_fox(&big_diag).unwrap();
    let alex_burau = alexander_polynomial_burau(&big_braid).unwrap();
    assert_eq!(alex_fox, alex_burau);
    assert_eq!(alex_fox.span(), 16);
}

#[test]
fn zero_allocation_verified_engine_kernels() {
    let torus = TorusKnot::right_trefoil();
    let lissajous = LissajousKnot::three_twist();
    let hopf = hopf_link(1.4, true, 64).unwrap();
    let braid = BraidWord::new(3, vec![1, -2, 1, -2, 1, -2]).unwrap();

    let mut curve_buf = vec![Point3::origin(); 64];
    let mut liss_buf = vec![Point3::origin(); 64];
    let mut frame_buf = vec![BishopFrame::default(); 64];
    let mut lk_buf = vec![0_i64; 4];
    let mut burau_ws = BurauWorkspace::new(3).unwrap();

    let ((writhe, lk_val, alex_at_2), _telemetry) = VerifiedEngine::run_verified(|| {
        torus.sample_component_in_place(0, &mut curve_buf).unwrap();
        lissajous.sample_in_place(&mut liss_buf).unwrap();
        compute_bishop_frames_in_place(&curve_buf, &mut frame_buf).unwrap();
        let w = gauss_writhe_quadrature(&curve_buf).unwrap();
        let lk_q = gauss_linking_quadrature(&hopf.components[0], &hopf.components[1]).unwrap();
        pairwise_linking_numbers_in_place(&hopf, &mut lk_buf).unwrap();
        let a2 = burau_ws.eval_alexander_in_place(&braid, 2.0).unwrap();
        (w, lk_q, a2)
    });

    assert!(writhe > 2.5);
    assert!((lk_val - 1.0).abs() < 1e-2);
    assert_eq!(lk_buf, vec![0, 1, 1, 0]);
    assert!(alex_at_2.is_finite() && alex_at_2.abs() > 0.01);
}
