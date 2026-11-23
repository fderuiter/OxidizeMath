#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_real_singular_classification_cusp_crunode_acnode() {
    assert!(RealSingularCubic::classify(-1.0, 1.0).is_none());
    assert!(RealSingularCubic::classify(f64::NAN, 0.0).is_none());

    let cusp = RealSingularCubic::classify(0.0, 0.0).expect("origin cusp");
    assert_eq!(cusp.kind, SingularityKind::Cusp);
    assert_eq!(cusp.singular_x, 0.0);
    assert_eq!(cusp.branch_min_x, 0.0);
    assert!(cusp.is_singular_vertex(CurvePoint::Affine { x: 0.0, y: 0.0 }));

    let crunode = RealSingularCubic::classify(-3.0, 2.0).expect("(-3, 2) crunode");
    assert_eq!(crunode.kind, SingularityKind::Crunode);
    assert!((crunode.singular_x - 1.0).abs() < 1e-12);
    assert!((crunode.branch_min_x - (-2.0)).abs() < 1e-12);
    assert!((crunode.tangent_slope_abs - 3.0_f64.sqrt()).abs() < 1e-12);
    assert!(
        crunode
            .secant_or_tangent_slope(
                CurvePoint::Affine { x: 1.0, y: 0.0 },
                CurvePoint::Affine { x: 2.0, y: 2.0 },
            )
            .is_none()
    );
    let smooth_p = CurvePoint::Affine { x: 2.0, y: 2.0 };
    let doubled = crunode
        .add_smooth(smooth_p, smooth_p)
        .expect("smooth tangent");
    let CurvePoint::Affine { x: dx, y: dy } = doubled else {
        panic!("expected affine double on smooth locus");
    };
    assert!((dy * dy - crunode.rhs(dx)).abs() < 1e-9);

    let acnode = RealSingularCubic::classify(-3.0, -2.0).expect("(-3, -2) acnode");
    assert_eq!(acnode.kind, SingularityKind::Acnode);
    assert!((acnode.singular_x - (-1.0)).abs() < 1e-12);
    assert!((acnode.branch_min_x - 2.0).abs() < 1e-12);
    assert!(acnode.rhs(-1.0).abs() < 1e-12);
    assert!(acnode.rhs(0.0) < 0.0);
    assert!(acnode.rhs(2.0).abs() < 1e-12);
    assert!(acnode.rhs(3.0) > 0.0);
}

#[test]
fn test_fp_singular_slider_pairs_p5_and_exact_affine_counts() {
    let mut singular_slider_pairs = 0_usize;
    let mut total_slider_pairs = 0_usize;
    let mut scratch = [CurvePoint::Infinity; 16];

    for a in -15..=15 {
        for b in -15..=15 {
            total_slider_pairs += 1;
            if let Some(sing) = FpSingularCubic::classify(a, b, 5).unwrap() {
                singular_slider_pairs += 1;
                let written = sing.write_affine_points(&mut scratch).unwrap();
                assert_eq!(written, sing.affine_point_count());
                assert!(
                    scratch[..written]
                        .iter()
                        .any(|&pt| sing.is_singular_vertex(pt))
                );
            }
        }
    }
    assert_eq!(total_slider_pairs, 961);
    assert_eq!(singular_slider_pairs, 193);
    let pct = (singular_slider_pairs as f64 / total_slider_pairs as f64) * 100.0;
    assert!((pct - 20.0832466).abs() < 1e-3);
}

#[test]
fn test_fp_singular_zero_heap_allocations_under_verified_engine() {
    let mut affine_buf = [CurvePoint::Infinity; 64];
    let mut proj_buf = [AffinePointFp::Infinity; 64];

    let ((), metrics) = VerifiedEngine::run_verified(|| {
        let cr = RealSingularCubic::classify(-3.0, 2.0).unwrap();
        let _ = cr.add_smooth(
            CurvePoint::Affine { x: 2.0, y: 2.0 },
            CurvePoint::Affine { x: 2.0, y: 2.0 },
        );

        let fp_sing = FpSingularCubic::classify(-3, 2, 23).unwrap().unwrap();
        assert_eq!(fp_sing.kind, SingularityKind::Crunode);
        assert_eq!(fp_sing.singular_x, 1);
        let n_aff = fp_sing.write_affine_points(&mut affine_buf).unwrap();
        assert_eq!(n_aff, 22);
        let n_proj = fp_sing.enumerate_projective_points(&mut proj_buf).unwrap();
        assert_eq!(n_proj, 23);

        // Singular vertex suppresses slope division
        let s_pt = AffinePointFp::Finite { x: 1, y: 0 };
        let other = proj_buf[1];
        assert!(fp_sing.chord_step_fp(s_pt, other).is_none());
    });
    assert!(metrics.function_calls > 0);
}
