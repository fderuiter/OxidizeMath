//! Verification tests for Catmull-Rom 3D lifts, general-position $SO(3)$ projections,
//! Euler rotation-system planarity, Reidemeister I/II/III edge cases, and the $5_2$ Lissajous knot.

use pure_math::pure_math::algebra::traits::Ring;
use pure_math::pure_math::algebra::{BraidWord, LaurentPolynomial};
use pure_math::pure_math::knot_theory::{
    ClosedLink3D, Crossing, CrossingSign, DiagramSegment, Endpoint, KnotError, LissajousKnot,
    MAX_DIAGRAM_CROSSINGS, PlanarDiagram, alexander_polynomial_burau, alexander_polynomial_fox,
    gauss_writhe_quadrature, jones_polynomial, pairwise_linking_matrix,
};

fn enumerate_3_strand_words_up_to_len_4() -> Vec<Vec<i32>> {
    let alphabet = [-2_i32, -1, 1, 2];
    let mut words = Vec::with_capacity(341);
    words.push(Vec::new());
    for len in 1..=4 {
        let total = 4_usize.pow(len);
        for mask in 0..total {
            let mut w = Vec::with_capacity(len as usize);
            let mut cur = mask;
            for _ in 0..len {
                w.push(alphabet[cur % 4]);
                cur /= 4;
            }
            words.push(w);
        }
    }
    words
}

#[test]
fn all_341_three_strand_braids_roundtrip_lift_and_project_without_corruption() {
    let words = enumerate_3_strand_words_up_to_len_4();
    assert_eq!(words.len(), 341);

    for gens in &words {
        let braid = BraidWord::new(3, gens.clone()).expect("valid 3-strand word");
        let base_diag = PlanarDiagram::from_braid(&braid);
        base_diag.validate().expect("valid braid diagram");

        let base_alex = alexander_polynomial_fox(&base_diag).expect("Fox Alexander");
        let burau_alex = alexander_polynomial_burau(&braid).expect("Burau Alexander");
        assert_eq!(base_alex, burau_alex, "Fox/Burau mismatch on {gens:?}");
        let base_jones = jones_polynomial(&base_diag).ok();

        // 1. lift_to_3d_link(10) -> from_3d_link preserves crossings, linking, and invariants.
        let lifted = base_diag.lift_to_3d_link(10).expect("Catmull-Rom 3D lift");
        let roundtrip = PlanarDiagram::from_3d_link(&lifted)
            .unwrap_or_else(|e| panic!("lift->project failed on {gens:?}: {e}"));
        assert_eq!(
            roundtrip.crossing_count(),
            base_diag.crossing_count(),
            "lift roundtrip crossing count mismatch on {gens:?}"
        );
        assert_eq!(roundtrip.writhe(), base_diag.writhe());
        assert_eq!(roundtrip.component_count(), base_diag.component_count());
        assert_eq!(
            alexander_polynomial_fox(&roundtrip).expect("roundtrip Fox"),
            base_alex,
            "lift roundtrip Alexander mismatch on {gens:?}"
        );
        assert_eq!(
            jones_polynomial(&roundtrip).ok(),
            base_jones,
            "lift roundtrip Jones mismatch on {gens:?}"
        );

        // 2. ClosedLink3D::from_braid_closure at curve_samples = 72 achieves 0 vertex duplications.
        let link_72 =
            ClosedLink3D::from_braid_closure(&braid, 2.0, 0.34, 72).expect("3D braid closure");
        let proj_72 = PlanarDiagram::from_3d_link(&link_72)
            .unwrap_or_else(|e| panic!("from_3d_link(72) failed on {gens:?}: {e}"));
        assert_eq!(
            proj_72.crossing_count(),
            gens.len(),
            "vertex-crossing duplication at curve_samples=72 on {gens:?}"
        );
        assert_eq!(proj_72.writhe(), base_diag.writhe());
        assert_eq!(
            alexander_polynomial_fox(&proj_72).expect("projected Fox"),
            base_alex
        );
        assert_eq!(jones_polynomial(&proj_72).ok(), base_jones);
    }
}

#[test]
fn canonical_lifts_figure_eight_isotopy_and_borromean_linking_roundtrip() {
    for positive in [true, false] {
        let tref = PlanarDiagram::trefoil(positive);
        let lifted = tref.lift_to_3d_link(10).unwrap();
        let proj = PlanarDiagram::from_3d_link(&lifted).unwrap();
        assert_eq!(proj.crossing_count(), 3);
        assert_eq!(
            jones_polynomial(&proj).unwrap(),
            jones_polynomial(&tref).unwrap()
        );
    }

    // Figure-eight with Reidemeister-I curl (matching FIGURE_EIGHT_ISOTOPY_PRESET_JSON).
    let mut fig8_iso = PlanarDiagram::figure_eight();
    let expected_alex = alexander_polynomial_fox(&fig8_iso).unwrap();
    let expected_jones = jones_polynomial(&fig8_iso).unwrap();
    let w_before = gauss_writhe_quadrature(&fig8_iso.lift_to_3d_link(10).unwrap().components[0])
        .expect("base 3D writhe");

    fig8_iso
        .apply_reidemeister_i_add(0, CrossingSign::Positive)
        .expect("R-I add on segment 0");
    let lifted_iso = fig8_iso.lift_to_3d_link(10).expect("lift figure-8 isotopy");
    let w_after = gauss_writhe_quadrature(&lifted_iso.components[0]).expect("isotopy 3D writhe");
    assert!(w_after > w_before + 0.1);

    let proj_iso = PlanarDiagram::from_3d_link(&lifted_iso).expect("project lifted isotopy");
    assert_eq!(alexander_polynomial_fox(&proj_iso).unwrap(), expected_alex);
    assert_eq!(jones_polynomial(&proj_iso).unwrap(), expected_jones);

    // Borromean rings lift preserves zero pairwise Gauss linking numbers [[0, 0, 0]; 3].
    let borr = PlanarDiagram::borromean_rings();
    let lifted_borr = borr.lift_to_3d_link(10).expect("lift Borromean rings");
    assert_eq!(
        pairwise_linking_matrix(&lifted_borr).unwrap(),
        vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]
    );
    let proj_borr = PlanarDiagram::from_3d_link(&lifted_borr).expect("project Borromean lift");
    assert_eq!(proj_borr.crossing_count(), 6);
    assert_eq!(
        proj_borr.combinatorial_linking_matrix(),
        vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]]
    );
    assert_eq!(
        alexander_polynomial_fox(&proj_borr).unwrap(),
        alexander_polynomial_fox(&borr).unwrap()
    );

    // Detached free loops stay outside lifted crossing components with zero false linking.
    let hopf_plus_free =
        PlanarDiagram::from_braid(&BraidWord::new(3, vec![1, 1]).expect("3-strand Hopf + free"));
    assert_eq!(hopf_plus_free.free_loops, 1);
    let lifted_hpf = hopf_plus_free.lift_to_3d_link(10).unwrap();
    assert_eq!(
        pairwise_linking_matrix(&lifted_hpf).unwrap(),
        vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 0]]
    );
}

#[test]
fn euler_planarity_and_deserialize_reject_virtual_and_malformed_diagrams() {
    // 1-crossing virtual knot on the torus: V = 1, E = 2, F = 1 => V - E + F = 0 != 2.
    let virtual_one = PlanarDiagram {
        crossings: vec![Crossing {
            sign: CrossingSign::Positive,
            slots: [1, 0, 1, 0],
            position: [0.0, 0.0],
        }],
        segments: vec![
            DiagramSegment {
                tail: Endpoint {
                    crossing: 0,
                    slot: 1,
                },
                head: Endpoint {
                    crossing: 0,
                    slot: 3,
                },
                waypoints: Vec::new(),
            },
            DiagramSegment {
                tail: Endpoint {
                    crossing: 0,
                    slot: 2,
                },
                head: Endpoint {
                    crossing: 0,
                    slot: 0,
                },
                waypoints: Vec::new(),
            },
        ],
        free_loops: 0,
    };
    assert!(matches!(
        virtual_one.validate(),
        Err(KnotError::InvalidDiagram { .. })
    ));
    let raw_json = serde_json::to_string(&virtual_one).unwrap();
    assert!(serde_json::from_str::<PlanarDiagram>(&raw_json).is_err());

    // Excessive free_loops and non-finite coordinates are rejected.
    let too_many_loops = PlanarDiagram {
        crossings: Vec::new(),
        segments: Vec::new(),
        free_loops: MAX_DIAGRAM_CROSSINGS + 1,
    };
    assert!(too_many_loops.validate().is_err());
    let loops_json = serde_json::to_string(&too_many_loops).unwrap();
    assert!(serde_json::from_str::<PlanarDiagram>(&loops_json).is_err());

    let mut nan_diag = PlanarDiagram::trefoil(true);
    nan_diag.crossings[0].position[0] = f64::NAN;
    assert!(nan_diag.validate().is_err());

    // Classical jones_polynomial rejects non-planar virtual diagrams (which produce odd A/q powers).
    assert!(matches!(
        jones_polynomial(&virtual_one),
        Err(KnotError::InvalidDiagram { .. })
    ));
}

#[test]
fn reidemeister_i_and_ii_edge_cases_on_crossing_free_unlinks() {
    let mut u2 = PlanarDiagram::unlink(2).unwrap();
    assert!(
        u2.apply_reidemeister_i_add(2, CrossingSign::Positive)
            .is_err()
    );
    let c0 = u2
        .apply_reidemeister_i_add(1, CrossingSign::Positive)
        .expect("unlink(2) accepts segment_id = 1");
    assert_eq!(c0, 0);
    u2.validate().unwrap();
    assert_eq!(u2.crossing_count(), 1);
    assert_eq!(u2.free_loops, 1);
    assert_eq!(u2.component_count(), 2);
    assert_eq!(u2.writhe(), 1);

    let mut unknot = PlanarDiagram::unknot();
    let (ca, cb) = unknot.apply_reidemeister_ii_add(0, 0).expect("unknot R-II");
    unknot.validate().unwrap();
    assert_eq!((ca, cb), (0, 1));
    assert_eq!(unknot.crossing_count(), 2);
    assert_eq!(unknot.writhe(), 0);
    assert_eq!(unknot.component_count(), 1);
    assert_eq!(jones_polynomial(&unknot).unwrap(), LaurentPolynomial::one());
    unknot
        .apply_reidemeister_ii_remove(ca, cb)
        .expect("cancel unknot R-II");
    assert_eq!(unknot, PlanarDiagram::unknot());
}

#[test]
fn reidemeister_iii_multi_slot_triangles_and_three_twist_lissajous_5_2() {
    for gens in [vec![1, 2, 1], vec![-2, -1, -2]] {
        let braid = BraidWord::new(3, gens.clone()).unwrap();
        let mut diag = PlanarDiagram::from_braid(&braid);
        let base_w = diag.writhe();
        let base_a = alexander_polynomial_fox(&diag).unwrap();
        let base_j = jones_polynomial(&diag).unwrap();
        let tris = diag.find_reidemeister_iii_triangles();
        assert!(!tris.is_empty(), "expected R-III triangle on {gens:?}");
        let t = tris[0];
        diag.apply_reidemeister_iii(t.c_top_bot, t.c_top_mid, t.c_mid_bot)
            .unwrap();
        diag.validate().unwrap();
        assert_eq!(diag.writhe(), base_w);
        assert_eq!(alexander_polynomial_fox(&diag).unwrap(), base_a);
        assert_eq!(jones_polynomial(&diag).unwrap(), base_j);
    }

    let liss_link = LissajousKnot::three_twist().to_link(160).unwrap();
    let liss_diag = PlanarDiagram::from_3d_link(&liss_link).unwrap();
    assert_eq!(liss_diag.crossing_count(), 7);
    let alex_5_2 = alexander_polynomial_fox(&liss_diag).unwrap();
    assert_eq!(alex_5_2, LaurentPolynomial::new(-1, vec![2_i64, -3, 2]));
    assert_eq!(alex_5_2.eval_with_inverse(-1_i64, -1_i64).abs(), 7);
}
