//! Zero-allocation integration tests for stack-allocated `math_scalars` types.
#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:math_scalars]

use core::mem::size_of;
use math_scalars::{
    Complex, EuclideanDomain, FieldScalar, Fp, FpExt, I256, IrreducibleModulus, Octonion, Qp,
    Quaternion, ScalarError, StackPoly, U256, UnitQuaternion,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModF7X2Plus1;
impl IrreducibleModulus<7, 2> for ModF7X2Plus1 {
    const MODULUS_LOW: [u64; 2] = [1, 0];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModF2X3PlusXPlus1;
impl IrreducibleModulus<2, 3> for ModF2X3PlusXPlus1 {
    const MODULUS_LOW: [u64; 3] = [1, 1, 0];
}

#[test]
fn test_u256_and_i256_zero_alloc_and_euclidean_laws() {
    VerifiedEngine::run_verified(|| {
        let a = U256::from_u128(u128::MAX);
        let prod = a.checked_mul(a).unwrap();
        assert_eq!(prod.checked_div(a).unwrap(), a);
        assert_eq!(prod.checked_rem(a).unwrap(), U256::ZERO);
        assert_eq!(
            U256::MAX.checked_add(U256::ONE),
            Err(ScalarError::ArithmeticOverflow)
        );
        assert_eq!(
            U256::ZERO.checked_sub(U256::ONE),
            Err(ScalarError::ArithmeticOverflow)
        );
        assert_eq!(
            U256::ONE.checked_div(U256::ZERO),
            Err(ScalarError::DivisionByZero)
        );

        for &(x, y) in &[(-17i64, 5i64), (-17, -5), (17, -5), (17, 5), (0, -7)] {
            let ix = I256::from_i64(x);
            let iy = I256::from_i64(y);
            let q = ix.checked_div_euclid(iy).unwrap();
            let r = ix.checked_rem_euclid(iy).unwrap();
            assert_eq!(q.to_i64().unwrap(), x.div_euclid(y));
            assert_eq!(r.to_i64().unwrap(), x.rem_euclid(y));
            assert!(!r.is_negative());
            assert_eq!(q * iy + r, ix);
        }

        assert_eq!(
            I256::MIN.checked_div_euclid(I256::MINUS_ONE),
            Err(ScalarError::ArithmeticOverflow)
        );
        assert_eq!(
            I256::from_i64(-1).checked_rem_euclid(I256::MIN).unwrap(),
            I256::MAX
        );
    });
}

#[test]
fn test_hypercomplex_repr_c_division_and_moufang_zero_alloc() {
    assert_eq!(size_of::<Complex<f64>>(), 2 * size_of::<f64>());
    assert_eq!(size_of::<Quaternion<f64>>(), 4 * size_of::<f64>());
    assert_eq!(size_of::<UnitQuaternion<f64>>(), 4 * size_of::<f64>());
    assert_eq!(size_of::<Octonion<f64>>(), 8 * size_of::<f64>());

    VerifiedEngine::run_verified(|| {
        let z = Complex::new(3.0_f64, -4.0_f64);
        let z_inv = z.try_inv().unwrap();
        assert!((z * z_inv - Complex::ONE).abs() < 1e-14);
        assert_eq!(
            Complex::<f64>::ZERO.try_inv(),
            Err(ScalarError::NonInvertible)
        );

        let q1 = Quaternion::new(1.5_f64, -2.0, 0.5, 3.0);
        let q2 = Quaternion::new(-0.5_f64, 1.25, -2.5, 0.75);
        assert_ne!(q1 * q2, q2 * q1);
        assert!(((q1 * q2) * q2.checked_inv().unwrap() - q1).norm() < 1e-13);
        assert!((((q1 * q2) * Quaternion::I) - (q1 * (q2 * Quaternion::I))).norm() < 1e-13);

        let x = Octonion::new(1.0_f64, -0.5, 2.0, 0.25, -1.5, 0.75, -0.25, 1.25);
        let y = Octonion::new(-0.75_f64, 1.5, -0.25, 1.0, 0.5, -1.25, 2.0, -0.5);
        let w = Octonion::new(0.5_f64, 0.25, -1.0, 1.5, -0.5, 0.75, 0.5, -1.25);

        // Norm multiplicativity ||x y||^2 == ||x||^2 ||y||^2
        assert!(((x * y).norm_sqr() - x.norm_sqr() * y.norm_sqr()).abs() < 1e-11);
        // Non-associativity on basis units: (e1 * e2) * e4 != e1 * (e2 * e4)
        let e1 = Octonion::<f64>::basis(1);
        let e2 = Octonion::<f64>::basis(2);
        let e4 = Octonion::<f64>::basis(4);
        assert_ne!((e1 * e2) * e4, e1 * (e2 * e4));

        // All three Moufang identities
        assert!(((w * (x * (w * y))) - (((w * x) * w) * y)).norm() < 1e-11);
        assert!(((((x * w) * y) * w) - (x * (w * (y * w)))).norm() < 1e-11);
        assert!((((w * x) * (y * w)) - (w * ((x * y) * w))).norm() < 1e-11);
        assert!(((x * x.checked_inv().unwrap()) - Octonion::ONE).norm() < 1e-13);
    });
}

#[test]
fn test_montgomery_fp_large_primes_and_galois_zero_alloc() {
    const P_OVER_I64_SQRT: u64 = 3_037_000_507;
    const P_MAX_U64: u64 = 18_446_744_073_709_551_557;

    VerifiedEngine::run_verified(|| {
        // Explicit P = 2 path
        assert_eq!((Fp::<2>::ONE + Fp::<2>::ONE).value(), 0);
        assert_eq!(Fp::<2>::ONE.checked_inverse(), Ok(Fp::<2>::ONE));
        assert_eq!(
            Fp::<2>::ZERO.checked_inverse(),
            Err(ScalarError::NonInvertible)
        );

        // Prime immediately above sqrt(i64::MAX) that previously overflowed i64
        let a = Fp::<P_OVER_I64_SQRT>::new(P_OVER_I64_SQRT - 2);
        let b = Fp::<P_OVER_I64_SQRT>::new(P_OVER_I64_SQRT - 5);
        assert_eq!((a * b).value(), 10);
        assert_eq!((a * a.checked_inverse().unwrap()).value(), 1);

        // Largest 64-bit prime (2^64 - 59, top bit set: P > 2^63)
        let hi_a = Fp::<P_MAX_U64>::new(P_MAX_U64 - 7);
        let hi_b = Fp::<P_MAX_U64>::new(P_MAX_U64 - 11);
        assert_eq!((hi_a * hi_b).value(), 77);
        assert_eq!((hi_a + hi_b).value(), P_MAX_U64 - 18);
        assert_eq!((hi_a - hi_b).value(), 4);
        assert_eq!((hi_a * hi_a.checked_inverse().unwrap()).value(), 1);
        assert_eq!(hi_a.pow(P_MAX_U64 - 1), Fp::<P_MAX_U64>::ONE);

        // StackPoly Euclidean division & extended GCD over Fp<17>
        let num =
            StackPoly::<Fp<17>, 8>::from_slice(&[Fp::new(1), Fp::new(0), Fp::new(5), Fp::new(3)])
                .unwrap();
        let den = StackPoly::<Fp<17>, 8>::from_slice(&[Fp::new(2), Fp::new(1)]).unwrap();
        let (q, r) = num.checked_div_rem(&den).unwrap();
        assert_eq!(q.checked_mul(&den).unwrap() + r, num);
        let (g, s, t) = num.xgcd(&den).unwrap();
        assert_eq!(
            s.checked_mul(&num).unwrap() + t.checked_mul(&den).unwrap(),
            g
        );

        // Galois extension fields GF(7^2) and GF(2^3)
        let z = FpExt::<7, 2, ModF7X2Plus1>::from_u64_coeffs([3, 4]);
        let z_inv = z.checked_inverse().unwrap();
        assert_eq!(z * z_inv, FpExt::ONE);
        assert_eq!(z.pow(48), FpExt::ONE);

        let g = FpExt::<2, 3, ModF2X3PlusXPlus1>::generator();
        assert_eq!(g.pow(7), FpExt::ONE);
        assert_eq!(g * g.checked_inverse().unwrap(), FpExt::ONE);
    });
}

#[test]
fn test_qp_ultrametric_and_hensel_lifting_zero_alloc() {
    VerifiedEngine::run_verified(|| {
        let a = Qp::<7, 8>::from_i64(49 * 3);
        let b = Qp::<7, 8>::from_i64(7 * 5);
        assert_eq!(a.valuation(), Some(2));
        assert_eq!(b.valuation(), Some(1));
        assert!(a.satisfies_ultrametric(b));
        assert!((a - a).is_zero());

        let ratio = Qp::<7, 8>::from_ratio(1, 3).unwrap();
        assert_eq!(ratio * Qp::<7, 8>::from_i64(3), Qp::<7, 8>::ONE);

        // Lift simple root r0 = 3 of x^2 - 2 == 0 in Q_7 to 7^8 precision
        let root = Qp::<7, 8>::hensel_lift_root(&[-2, 0, 1], 3).unwrap();
        let residual = root * root - Qp::<7, 8>::from_i64(2);
        assert!(residual.is_zero());
        let r_mod = root.to_mod_pk_u128(8).unwrap();
        let p8 = 7u128.pow(8);
        assert_eq!((r_mod * r_mod) % p8, 2);

        // Reject non-root and ramified/singular derivative (x^2 - 1 at P = 2 has f'(1) == 2 == 0 mod 2)
        assert_eq!(
            Qp::<7, 6>::hensel_lift_root(&[-2, 0, 1], 2),
            Err(ScalarError::NotARootModP)
        );
        assert_eq!(
            Qp::<2, 6>::hensel_lift_root(&[-1, 0, 1], 1),
            Err(ScalarError::SingularDerivative)
        );
    });
}
