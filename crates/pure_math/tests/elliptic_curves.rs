#![allow(missing_docs)]
use pure_math::math_types::Integer;
use pure_math::pure_math::elliptic_curves::*;
use pure_math::pure_math::number_theory::ambs::{AmbsDsp, modular_inverse};
use pure_math::pure_math::number_theory::primes::primes_up_to;
use std::collections::HashMap;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_prime_factors() {
    let mut factors = HashMap::new();
    factors.insert(2, 2);
    factors.insert(3, 1);
    factors.insert(5, 1);
    assert_eq!(prime_factors(60), factors);

    let mut factors_2 = HashMap::new();
    factors_2.insert(97, 1);
    assert_eq!(prime_factors(97), factors_2);

    let factors_3 = HashMap::new();
    assert_eq!(prime_factors(1), factors_3);

    let factors_4 = HashMap::new();
    assert_eq!(prime_factors(0), factors_4);
}

#[test]
#[verified_engine::verified]
fn test_psi_function() {
    assert_eq!(psi(1), 1);
    assert_eq!(psi(2), 3);
    assert_eq!(psi(3), 4);
    assert_eq!(psi(4), 6);
    assert_eq!(psi(5), 6);
    assert_eq!(psi(6), 12);
    assert_eq!(psi(10), 18);
    assert_eq!(psi(11), 12);
}

#[test]
#[verified_engine::verified]
fn test_theorem_1_1_bounds_simple() {
    // For N=5, psi(5) = 6. Let's check a_{1,1} where i+j=2. diff = 4
    let bounds = theorem_1_1_bounds(5, 1, 1).unwrap();
    assert_eq!(bounds.v2_bound, Some(15 * 4)); // 60
    assert_eq!(bounds.v3_bound, Some(3 * 4)); // 12
    assert_eq!(bounds.v5_bound, None);
}

#[test]
#[verified_engine::verified]
fn test_theorem_1_1_bounds_n_is_1_mod_3() {
    // For N=7, psi(7) = 8. N=7 is 1 mod 3. Check a_{2,1} where i+j=3. diff = 5
    let bounds = theorem_1_1_bounds(7, 2, 1).unwrap();
    assert_eq!(bounds.v2_bound, Some(15 * 5)); // 75
    // ceil(4.5 * 5) = ceil(22.5) = 23
    assert_eq!(bounds.v3_bound, Some(23));
    assert_eq!(bounds.v5_bound, Some(3 * 5)); // 15
}

#[test]
#[verified_engine::verified]
fn test_theorem_1_1_bounds_panic() {
    // N=5, psi(5)=6. i+j=6, which should panic.
    assert!(theorem_1_1_bounds(5, 3, 3).is_err());
}

#[test]
#[verified_engine::verified]
fn test_theorem_1_2_bounds_simple() {
    // For N=11, psi(11) = 12. N=11 is 3 mod 4. Check a_{1,0}. diff = 11
    let bounds = theorem_1_2_bounds(11, 1, 0).unwrap();
    assert_eq!(bounds.v2_bound, Some(9 * 11)); // 99
    assert_eq!(bounds.v3_bound, Some(6 * 11)); // 66
    assert_eq!(bounds.v7_bound, Some(2 * 11)); // 22
}

#[test]
#[verified_engine::verified]
fn test_theorem_1_2_bounds_n_is_1_mod_4() {
    // For N=5, psi(5) = 6. N=5 is 1 mod 4. Check a_{0,0}. diff = 6
    let bounds = theorem_1_2_bounds(5, 0, 0).unwrap();
    assert_eq!(bounds.v2_bound, Some(10 * 6)); // 60
    assert_eq!(bounds.v3_bound, Some(6 * 6)); // 36
    assert_eq!(bounds.v7_bound, Some(2 * 6)); // 12
}

#[test]
#[verified_engine::verified]
fn test_theorem_1_2_bounds_panic() {
    // N=5, psi(5)=6. i+j=7, which should panic.
    assert!(theorem_1_2_bounds(5, 4, 3).is_err());
}

#[test]
#[verified_engine::verified]
fn test_weierstrass_singular_and_invalid_rejection() {
    assert_eq!(
        WeierstrassCurveFp::new(3, 1, 1),
        Err(EllipticCurveError::InvalidModulus { p: 3 })
    );
    assert_eq!(
        WeierstrassCurveFp::new(15, 1, 1),
        Err(EllipticCurveError::InvalidModulus { p: 15 })
    );
    assert_eq!(
        WeierstrassCurveFp::new(MAX_WEIERSTRASS_PRIME + 7, 1, 1),
        Err(EllipticCurveError::InvalidModulus {
            p: MAX_WEIERSTRASS_PRIME + 7
        })
    );
    assert_eq!(
        WeierstrassCurveFp::new(17, 0, 0),
        Err(EllipticCurveError::SingularCurve { p: 17, a: 0, b: 0 })
    );
    assert_eq!(
        WeierstrassCurveFp::new(19, -3, 2),
        Err(EllipticCurveError::SingularCurve { p: 19, a: 16, b: 2 })
    );
    assert_eq!(
        WeierstrassCurveFp::new(11, 8, 2),
        Err(EllipticCurveError::SingularCurve { p: 11, a: 8, b: 2 })
    );

    let curve = WeierstrassCurveFp::new(17, 2, 2).unwrap();
    assert_ne!(curve.discriminant(), 0);
    let bad_pt = AffinePointFp::from_coords(5, 2, 17);
    assert!(!curve.contains_point(bad_pt));
    assert_eq!(
        curve.add_points(bad_pt, AffinePointFp::Infinity),
        Err(EllipticCurveError::PointNotOnCurve { x: 5, y: 2, p: 17 })
    );
    let mut tiny = [AffinePointFp::Infinity; 2];
    assert!(matches!(
        curve.enumerate_points(&mut tiny),
        Err(EllipticCurveError::BufferTooSmall { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_weierstrass_identity_inverse_and_associativity() {
    let curve = WeierstrassCurveFp::new(7, -1, 0).unwrap();
    let mut pts = [AffinePointFp::Infinity; 16];
    let n = curve.enumerate_points(&mut pts).unwrap();
    assert_eq!(n, 8);

    // 2-torsion point (0, 0): -P == P and P + P == Infinity
    let p_zero = AffinePointFp::from_coords(0, 0, 7);
    assert_eq!(curve.negate_point(p_zero).unwrap(), p_zero);
    assert_eq!(
        curve.add_points(p_zero, p_zero).unwrap(),
        AffinePointFp::Infinity
    );

    for i in 0..n {
        let p = pts[i];
        let neg_p = curve.negate_point(p).unwrap();
        assert_eq!(curve.add_points(p, AffinePointFp::Infinity).unwrap(), p);
        assert_eq!(curve.add_points(AffinePointFp::Infinity, p).unwrap(), p);
        assert_eq!(curve.add_points(p, neg_p).unwrap(), AffinePointFp::Infinity);
        for j in 0..n {
            let q = pts[j];
            let pq = curve.add_points(p, q).unwrap();
            assert_eq!(pq, curve.add_points(q, p).unwrap());
            for r in &pts[..n] {
                let lhs = curve.add_points(pq, *r).unwrap();
                let qr = curve.add_points(q, *r).unwrap();
                let rhs = curve.add_points(p, qr).unwrap();
                assert_eq!(lhs, rhs);
            }
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_weierstrass_scalar_mul_and_lagrange_divisibility() {
    for (p, a, b) in [(17_i64, 2_i64, 2_i64), (23, 1, 1), (43, -1, 1)] {
        let curve = WeierstrassCurveFp::new(p, a, b).unwrap();
        let group_order = curve.cardinality();
        let mut pts = [AffinePointFp::Infinity; 64];
        let count = curve.enumerate_points(&mut pts).unwrap();
        assert_eq!(count, group_order);

        let mut orbit = [AffinePointFp::Infinity; 64];
        for &pt in &pts[..count] {
            let ord = curve.point_order(pt).unwrap();
            assert_eq!(group_order % ord, 0);
            let orbit_len = curve.torsion_orbit(pt, &mut orbit).unwrap();
            assert_eq!(orbit_len, ord);
            assert_eq!(orbit[ord - 1], AffinePointFp::Infinity);
            assert_eq!(
                curve.scalar_mul(group_order as i64, pt).unwrap(),
                AffinePointFp::Infinity
            );
            let k3 = curve.scalar_mul(3, pt).unwrap();
            let neg_k3 = curve.scalar_mul(-3, pt).unwrap();
            assert_eq!(curve.negate_point(k3).unwrap(), neg_k3);
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_weierstrass_hasse_bounds_all_primes_5_to_997() {
    let primes: Vec<i64> = primes_up_to(997)
        .into_iter()
        .filter(|&p| p >= 5)
        .map(|p| p as i64)
        .collect();
    assert_eq!(primes.len(), 166);

    let mut entries = vec![
        HasseSweepEntry {
            p: 5,
            is_singular: false,
            group_order: 0,
            frobenius_trace: 0,
            hasse_bound: 0.0,
        };
        primes.len()
    ];

    for (a, b) in [(2_i64, 3_i64), (-1, 0), (0, 7), (1, -1)] {
        let written = sweep_hasse_bounds(a, b, &primes, &mut entries).unwrap();
        assert_eq!(written, primes.len());
        for entry in &entries[..written] {
            if !entry.is_singular {
                let curve = WeierstrassCurveFp::new(entry.p, a, b).unwrap();
                let (lo, hi) = curve.hasse_bounds();
                let ap = entry.frobenius_trace as f64;
                assert!(ap >= lo - 1e-12 && ap <= hi + 1e-12);
                assert_eq!(entry.frobenius_trace, curve.frobenius_trace());
                if a == -1 && b == 0 && entry.p.rem_euclid(4) == 3 {
                    assert_eq!(entry.frobenius_trace, 0);
                }
            }
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_weierstrass_ambs_tonelli_and_inverse_parity() {
    for p in [5_i64, 7, 17, 41, 73, 97] {
        let p_int = Integer::from(p);
        for n in 0..p {
            let n_int = Integer::from(n);
            let expected_inv =
                modular_inverse(&n_int, &p_int).and_then(|v| v.to_u64().map(|x| x as i64));
            assert_eq!(mod_inv_fp(n, p), expected_inv);

            let mut ambs_roots: Vec<i64> = AmbsDsp::tonelli_shanks(&n_int, &p_int)
                .unwrap()
                .into_iter()
                .map(|r| r.to_u64().unwrap() as i64)
                .collect();
            ambs_roots.sort_unstable();
            let (fp_roots, count) = tonelli_shanks_fp(n, p);
            assert_eq!(&fp_roots[..count], ambs_roots.as_slice());
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_weierstrass_zero_allocations_under_verified_allocator() {
    let curve = WeierstrassCurveFp::new(97, 2, 3).unwrap();
    let mut points_buf = [AffinePointFp::Infinity; 128];
    let mut orbit_buf = [AffinePointFp::Infinity; 128];
    let primes = [5_i64, 7, 11, 13, 17, 19, 23, 29, 31, 97];
    let mut sweep_buf = [HasseSweepEntry {
        p: 5,
        is_singular: false,
        group_order: 0,
        frobenius_trace: 0,
        hasse_bound: 0.0,
    }; 10];

    let ((sum, k_pt, orbit_len, swept), metrics) = VerifiedEngine::run_verified(|| {
        let count = curve.enumerate_points(&mut points_buf).unwrap();
        let p1 = points_buf[1];
        let p2 = points_buf[count / 2];
        let chord = curve.chord_step(p1, p2).unwrap();
        let k_mul = curve.scalar_mul(17, p1).unwrap();
        let o_len = curve.torsion_orbit(p1, &mut orbit_buf).unwrap();
        let s_len = sweep_hasse_bounds(2, 3, &primes, &mut sweep_buf).unwrap();
        (chord.sum, k_mul, o_len, s_len)
    });

    assert!(curve.contains_point(sum));
    assert!(curve.contains_point(k_pt));
    assert!(orbit_len >= 2);
    assert_eq!(swept, 10);
    assert!(metrics.function_calls > 0);
}

// [cite:modular_polynomials_review]
