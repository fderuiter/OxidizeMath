//! Integration tests for Weierstrass elliptic curves and modular invariants.
// [cite:hurwitz_kronecker]
// [cite:modular_polynomials_review]

use pure_math::pure_math::algebra::fields::Fp;
use pure_math::pure_math::elliptic_curves::*;
use pure_math::pure_math::number_theory::primes::primes_up_to;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_singular_and_invalid_modulus_rejection() {
    assert_eq!(
        WeierstrassCurve::new(0.0, 0.0),
        Err(EllipticCurveError::SingularRealCurve)
    );
    assert_eq!(
        WeierstrassCurve::new(-3.0, 2.0),
        Err(EllipticCurveError::SingularRealCurve)
    );
    assert_eq!(
        WeierstrassCurve::new(f64::NAN, 1.0),
        Err(EllipticCurveError::SingularRealCurve)
    );

    assert_eq!(
        FpWeierstrassCurve::<3>::new(1, 1),
        Err(EllipticCurveError::InvalidModulus { p: 3 })
    );
    assert_eq!(
        FpWeierstrassCurve::<15>::new(1, 1),
        Err(EllipticCurveError::InvalidModulus { p: 15 })
    );
    assert_eq!(
        FpWeierstrassCurve::<17>::new(-3, 2),
        Err(EllipticCurveError::SingularCurve { p: 17, a: 14, b: 2 })
    );

    for bad_p in [-5, 0, 2, 3, 9, 21] {
        assert_eq!(
            PrimeWeierstrassCurve::new(1, 1, bad_p),
            Err(EllipticCurveError::InvalidModulus { p: bad_p })
        );
    }
    assert_eq!(
        PrimeWeierstrassCurve::new(0, 0, 11),
        Err(EllipticCurveError::SingularCurve { p: 11, a: 0, b: 0 })
    );

    let large_curve = PrimeWeierstrassCurve::new(1, 1, 1009).unwrap();
    let mut buf = [CurvePoint::Infinity; 8];
    assert_eq!(
        large_curve.write_affine_points(&mut buf),
        Err(EllipticCurveError::ModulusExceedsLatticeCap {
            modulus: 1009,
            max_modulus: MAX_LATTICE_PRIME,
        })
    );
    assert_eq!(
        large_curve.point_count(),
        Err(EllipticCurveError::ModulusExceedsLatticeCap {
            modulus: 1009,
            max_modulus: MAX_LATTICE_PRIME,
        })
    );
}

#[test]
#[verified_engine::verified]
fn test_real_weierstrass_group_identities_and_two_torsion() {
    let two_torsion_curve = WeierstrassCurve::new(-1.0, 0.0).unwrap();
    let t0 = CurvePoint::Affine { x: 0.0, y: 0.0 };
    assert!(two_torsion_curve.is_on_curve(t0, 1e-12));
    assert_eq!(two_torsion_curve.double(t0), CurvePoint::Infinity);

    let curve = WeierstrassCurve::new(-2.0, 2.0).unwrap();
    let p = CurvePoint::Affine {
        x: 1.0,
        y: 1.0_f64.sqrt(),
    };
    let q = CurvePoint::Affine {
        x: 2.0,
        y: 6.0_f64.sqrt(),
    };
    let r = CurvePoint::Affine {
        x: 3.0,
        y: 23.0_f64.sqrt(),
    };

    assert_eq!(curve.add(p, CurvePoint::Infinity), p);
    assert_eq!(curve.add(CurvePoint::Infinity, p), p);
    assert_eq!(curve.add(p, curve.negate(p)), CurvePoint::Infinity);

    let pq = curve.add(p, q);
    let qp = curve.add(q, p);
    assert!(curve.is_on_curve(pq, 1e-9));
    let ((x_pq, y_pq), (x_qp, y_qp)) = (pq.coords().unwrap(), qp.coords().unwrap());
    assert!((x_pq - x_qp).abs() < 1e-9 && (y_pq - y_qp).abs() < 1e-9);

    let lhs = curve.add(curve.add(p, q), r);
    let rhs = curve.add(p, curve.add(q, r));
    let ((lx, ly), (rx, ry)) = (lhs.coords().unwrap(), rhs.coords().unwrap());
    assert!((lx - rx).abs() < 1e-8 && (ly - ry).abs() < 1e-8);

    let pos_3p = curve.scalar_mul(3, p);
    let neg_3p = curve.scalar_mul(-3, p);
    assert_eq!(curve.add(pos_3p, neg_3p), CurvePoint::Infinity);
}

#[test]
#[verified_engine::verified]
fn test_fp_and_runtime_prime_group_laws_agree() {
    let fp_curve = FpWeierstrassCurve::<23>::new(1, 1).unwrap();
    let rt_curve = PrimeWeierstrassCurve::new(1, 1, 23).unwrap();

    let p_fp = CurvePoint::Affine {
        x: Fp::<23>::new(0),
        y: Fp::<23>::new(1),
    };
    let q_fp = CurvePoint::Affine {
        x: Fp::<23>::new(1),
        y: Fp::<23>::new(7),
    };
    let p_rt = CurvePoint::Affine { x: 0, y: 1 };
    let q_rt = CurvePoint::Affine { x: 1, y: 7 };

    assert!(fp_curve.is_on_curve(p_fp) && fp_curve.is_on_curve(q_fp));
    assert!(rt_curve.is_on_curve(p_rt) && rt_curve.is_on_curve(q_rt));
    assert_eq!(fp_curve.discriminant().value, rt_curve.discriminant_mod_p());

    // 2-torsion tangent at y = 0: on y^2 = x^3 + x + 1 mod 23, x = 4 -> 64+4+1 = 69 = 0 mod 23
    let torsion_fp = CurvePoint::Affine {
        x: Fp::<23>::new(4),
        y: Fp::<23>::new(0),
    };
    let torsion_rt = CurvePoint::Affine { x: 4, y: 0 };
    assert_eq!(fp_curve.double(torsion_fp), CurvePoint::Infinity);
    assert_eq!(rt_curve.double(torsion_rt), CurvePoint::Infinity);
    assert_eq!(
        rt_curve.add(p_rt, rt_curve.negate(p_rt)),
        CurvePoint::Infinity
    );

    let order = rt_curve.point_count().unwrap() as i64;
    assert_eq!(order, 28);
    for k in -10..=30 {
        let m_fp = fp_curve.scalar_mul(k, p_fp);
        let m_rt = rt_curve.scalar_mul(k, p_rt);
        assert_eq!(
            m_fp.coords().map(|(x, y)| (x.value, y.value)),
            m_rt.coords()
        );
    }
    assert_eq!(rt_curve.scalar_mul(order, p_rt), CurvePoint::Infinity);
    assert_eq!(fp_curve.scalar_mul(order, p_fp), CurvePoint::Infinity);
}

#[test]
#[verified_engine::verified]
fn test_hasse_weil_bounds_across_primes_up_to_997() {
    for p in primes_up_to(MAX_LATTICE_PRIME as u64) {
        if p <= 3 {
            continue;
        }
        if let Ok(curve) = PrimeWeierstrassCurve::new(-1, 1, p as i64) {
            let trace = curve.frobenius_trace().unwrap();
            assert!(
                curve.satisfies_hasse_weil().unwrap(),
                "Hasse-Weil violated at p={p}: a_p={trace}"
            );
            assert!((trace as f64).abs() <= hasse_weil_bound(p) + 1e-12);
        }
    }

    let c13 = PrimeWeierstrassCurve::new(1, 1, 13).unwrap();
    let mut lattice = [CurvePoint::Infinity; 32];
    let affine_len = c13.write_affine_points(&mut lattice).unwrap();
    assert_eq!(affine_len as u64 + 1, c13.point_count().unwrap());
}

#[test]
fn test_verified_engine_zero_allocation_arithmetic_and_orbits() {
    let real_curve = WeierstrassCurve::new(-1.0, 1.0).unwrap();
    let fp_curve = FpWeierstrassCurve::<97>::new(2, 3).unwrap();
    let rt_curve = PrimeWeierstrassCurve::new(2, 3, 97).unwrap();
    let mut real_orbit = [CurvePoint::Infinity; 16];
    let mut fp_orbit = [CurvePoint::Infinity; 128];
    let mut rt_orbit = [CurvePoint::Infinity; 128];
    let mut affine_pts = [CurvePoint::Infinity; 128];

    let ((real_n, fp_n, rt_n, trace), metrics) = VerifiedEngine::run_verified(|| {
        let rp = CurvePoint::Affine { x: 0.0, y: 1.0 };
        let _r_sum = real_curve.scalar_mul(7, rp);
        let rn = real_curve.write_orbit(rp, &mut real_orbit);

        let count_aff = rt_curve.write_affine_points(&mut affine_pts).unwrap();
        assert!(count_aff > 0);
        let gen_rt = affine_pts[0];
        let (gx, gy) = gen_rt.coords().unwrap();
        let gen_fp = CurvePoint::Affine {
            x: Fp::<97>::new(gx),
            y: Fp::<97>::new(gy),
        };

        let _fp_mul = fp_curve.scalar_mul(19, gen_fp);
        let _rt_mul = rt_curve.scalar_mul(19, gen_rt);
        let fn_len = fp_curve.write_orbit(gen_fp, &mut fp_orbit);
        let rn_len = rt_curve.write_orbit(gen_rt, &mut rt_orbit);
        let a_p = rt_curve.frobenius_trace().unwrap();
        (rn, fn_len, rn_len, a_p)
    });

    assert!(
        metrics.function_calls > 0,
        "Expected verified function calls to be recorded"
    );
    assert_eq!(real_n, 16);
    assert_eq!(fp_n, rt_n);
    assert!(trace.unsigned_abs() <= 20);
}

#[test]
#[verified_engine::verified]
fn test_modular_bridge_invariants_class_numbers_and_q_series() {
    let cm_1728 = WeierstrassCurve::new(-1.0, 0.0).unwrap();
    assert!((j_invariant_real(&cm_1728) - 1728.0).abs() < 1e-9);
    let cm_zero = WeierstrassCurve::new(0.0, 1.0).unwrap();
    assert!(j_invariant_real(&cm_zero).abs() < 1e-9);

    let fp_cm = FpWeierstrassCurve::<29>::new(-1, 0).unwrap();
    let rt_cm = PrimeWeierstrassCurve::new(-1, 0, 29).unwrap();
    assert_eq!(j_invariant_fp(&fp_cm).value, 1728_i64.rem_euclid(29));
    assert_eq!(j_invariant_mod_p(&rt_cm), 1728_i64.rem_euclid(29));

    let mod_summary = modular_valuation_summary(&rt_cm, 5, 1, 1).unwrap();
    assert_eq!(mod_summary.psi_n, 6);
    assert_eq!(mod_summary.theorem_1_1.v2_bound, Some(60));
    assert_eq!(mod_summary.theorem_1_2.v7_bound, Some(8));

    let frob = analyze_frobenius_prime(&rt_cm).unwrap();
    assert_eq!(frob.p, 29);
    assert!(frob.frobenius_discriminant < 0);
    assert!(frob.principal_form.as_ref().is_some_and(|f| f.is_reduced()));
    assert!(frob.imaginary_class_number >= 1);
    assert!(frob.hurwitz_kronecker > 0.0);

    for p in [5_u64, 7, 11, 13, 17, 23, 29] {
        let (sum, expected, ok) = hurwitz_trace_sum(p);
        assert!(ok, "Hurwitz-Kronecker identity failed at p={p}");
        assert!((sum - expected).abs() < 1e-9);
    }

    let delta = ramanujan_delta_q_series(7);
    assert_eq!(delta.coeffs, vec![0, 1, -24, 252, -1472, 4830, -6048]);

    let e4 = eisenstein_e4_q_series(5);
    assert_eq!(e4.coeffs, vec![1, 240, 2160, 6720, 17520]);

    let scaled_j = j_invariant_q_scaled(5);
    assert_eq!(
        scaled_j.coeffs,
        vec![1, 744, 196_884, 21_493_760, 864_299_970]
    );
}
