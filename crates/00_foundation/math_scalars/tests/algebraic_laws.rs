#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:advanced_linear_algebra]
// [cite:modular_polynomials_review]
//! Algebraic, precision, memory-layout, finite-field, Galois extension, and $p$-adic
//! verification tests for `math_scalars`.

use core::mem::{align_of, size_of};
use math_scalars::{
    Complex, FieldScalar, Fp, FpExt, I256, IrreducibleModulus, MAX_PRIME_U64, Octonion, Qp,
    Quaternion, ScalarError, StackPoly, U256, UnitQuaternion, eval_int_poly_qp, hensel_lift_root,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ModP5Deg3;

// x^3 + x + 1 is irreducible over F_5 (roots mod 5: 0->1, 1->3, 2->1, 3->1, 4->4).
impl IrreducibleModulus<5, 3> for ModP5Deg3 {
    const COEFFS: [u64; 3] = [1, 1, 0];
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ReducibleModP5Deg2;

// x^2 - 1 = (x - 1)(x + 1) is reducible over F_5.
impl IrreducibleModulus<5, 2> for ReducibleModP5Deg2 {
    const COEFFS: [u64; 2] = [4, 0];
}

#[test]
fn repr_c_hypercomplex_layouts_and_cayley_dickson_identities_hold() {
    assert_eq!(size_of::<Complex<f64>>(), 2 * size_of::<f64>());
    assert_eq!(align_of::<Complex<f64>>(), align_of::<f64>());
    assert_eq!(size_of::<Quaternion<f64>>(), 4 * size_of::<f64>());
    assert_eq!(size_of::<UnitQuaternion<f64>>(), 4 * size_of::<f64>());
    assert_eq!(size_of::<Octonion<f64>>(), 8 * size_of::<f64>());

    // Quaternion non-commutative division-ring laws
    let q1 = Quaternion::new(1.5_f64, -2.0, 0.75, 3.25);
    let q2 = Quaternion::new(-0.5_f64, 1.25, -2.5, 0.5);
    let q3 = Quaternion::new(2.0_f64, 0.5, -1.0, -1.5);
    assert_ne!(q1 * q2, q2 * q1);
    let assoc_lhs = (q1 * q2) * q3;
    let assoc_rhs = q1 * (q2 * q3);
    for (a, b) in assoc_lhs.as_wxyz().iter().zip(assoc_rhs.as_wxyz().iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(((q1 * q2).norm_sqr() - q1.norm_sqr() * q2.norm_sqr()).abs() < 1e-10);
    assert_eq!(
        Quaternion::<f64>::ZERO.try_inverse(),
        Err(ScalarError::DivisionByZero)
    );

    // Octonion non-associative, norm-multiplicative, alternative, and Moufang laws
    let x = Octonion::new(1.0_f64, 2.0, -1.0, 0.5, -1.5, 0.25, 2.5, -0.75);
    let y = Octonion::new(-0.5_f64, 1.0, 1.5, -2.0, 0.75, -1.25, 0.5, 1.0);
    let z = Octonion::new(0.25_f64, -0.5, 2.0, 1.0, -0.25, 1.5, -1.0, 0.5);

    assert!(((x * y).norm_sqr() - x.norm_sqr() * y.norm_sqr()).abs() < 1e-9);

    // Left and right alternative laws: x(xy) = (xx)y and (yx)x = y(xx)
    let left_alt = (x * (x * y)) - ((x * x) * y);
    let right_alt = ((y * x) * x) - (y * (x * x));
    assert!(left_alt.norm() < 1e-10);
    assert!(right_alt.norm() < 1e-10);

    // Three Moufang identities
    let m1 = (z * (x * (z * y))) - (((z * x) * z) * y);
    let m2 = (x * (z * (y * z))) - (((x * z) * y) * z);
    let m3 = ((z * x) * (y * z)) - ((z * (x * y)) * z);
    assert!(m1.norm() < 1e-9);
    assert!(m2.norm() < 1e-9);
    assert!(m3.norm() < 1e-9);

    let x_inv = x.try_inverse().unwrap();
    assert!(((x * x_inv) - Octonion::ONE).norm() < 1e-11);
    assert_eq!(
        Octonion::<f64>::ZERO.try_inverse(),
        Err(ScalarError::DivisionByZero)
    );
}

#[test]
fn u256_and_i256_checked_and_euclidean_edge_cases_hold() {
    assert_eq!(U256::MAX.checked_add(U256::ONE), None);
    assert_eq!(U256::ZERO.checked_sub(U256::ONE), None);
    assert_eq!(U256::MAX.checked_div(U256::ZERO), None);
    assert_eq!(I256::MIN.checked_neg(), None);
    assert_eq!(I256::MIN.checked_abs(), None);
    assert_eq!(I256::MIN.checked_div(I256::NEG_ONE), None);
    assert_eq!(I256::MIN.checked_div_euclid(I256::NEG_ONE), None);

    // Euclidean remainder is strictly non-negative for all sign combinations
    for (a, b) in [(-7_i64, 4_i64), (-7, -4), (7, -4), (7, 4)] {
        let ia = I256::from_i64(a);
        let ib = I256::from_i64(b);
        let q = ia.checked_div_euclid(ib).unwrap();
        let r = ia.checked_rem_euclid(ib).unwrap();
        assert_eq!(q, I256::from_i64(a.div_euclid(b)));
        assert_eq!(r, I256::from_i64(a.rem_euclid(b)));
    }
}

#[test]
fn fp_montgomery_handles_p2_above_i64_sqrt_and_max_64_bit_prime() {
    // Explicit P = 2 path
    let z2 = Fp::<2>::ZERO;
    let o2 = Fp::<2>::ONE;
    assert_eq!(o2 + o2, z2);
    assert_eq!(o2 * o2, o2);
    assert_eq!(o2.try_inverse(), Ok(o2));
    assert_eq!(z2.try_inverse(), Err(ScalarError::DivisionByZero));

    // Prime just above floor(sqrt(i64::MAX)) = 3_037_000_499 where naive i64*i64 overflows!
    const P_OVER_SQRT_I64: u64 = 3_037_000_507;
    let a = Fp::<P_OVER_SQRT_I64>::from_u64(P_OVER_SQRT_I64 - 2);
    let b = Fp::<P_OVER_SQRT_I64>::from_u64(P_OVER_SQRT_I64 - 3);
    assert_eq!((a * b).value(), 6);
    assert_eq!(a * a.try_inverse().unwrap(), Fp::<P_OVER_SQRT_I64>::ONE);

    // Largest 64-bit prime 18_446_744_073_709_551_557 = 2^64 - 59 (> 2^63!)
    let hi_a = Fp::<MAX_PRIME_U64>::from_u64(MAX_PRIME_U64 - 1);
    let hi_b = Fp::<MAX_PRIME_U64>::from_u64(MAX_PRIME_U64 - 7);
    assert_eq!((hi_a + hi_b).value(), MAX_PRIME_U64 - 8);
    assert_eq!((hi_a - hi_b).value(), 6);
    assert_eq!((hi_a * hi_b).value(), 7);
    assert_eq!(hi_b.pow(MAX_PRIME_U64 - 1), Fp::<MAX_PRIME_U64>::ONE);
    let hi_inv = hi_b.try_inverse().unwrap();
    assert_eq!(hi_b * hi_inv, Fp::<MAX_PRIME_U64>::ONE);
}

#[test]
fn stack_poly_and_fp_ext_galois_inversion_hold() {
    // GF(5^3) = F_5[x] / (x^3 + x + 1) has multiplicative group order 5^3 - 1 = 124
    let alpha = FpExt::<5, 3, ModP5Deg3>::generator();
    let elem = FpExt::<5, 3, ModP5Deg3>::from_u64_coeffs([2, 3, 4]);
    assert_eq!(alpha.pow(124), FpExt::<5, 3, ModP5Deg3>::ONE);
    assert_eq!(elem.pow(124), FpExt::<5, 3, ModP5Deg3>::ONE);
    let inv = elem.try_inverse().unwrap();
    assert_eq!(elem * inv, FpExt::<5, 3, ModP5Deg3>::ONE);
    // Frobenius repeated 3 times is identity on GF(5^3)
    assert_eq!(elem.frobenius().frobenius().frobenius(), elem);

    // Zero divisor x - 1 in reducible quotient F_5[x]/(x^2 - 1) returns NonInvertible without panic
    let zero_div = FpExt::<5, 2, ReducibleModP5Deg2>::from_u64_coeffs([4, 1]);
    assert_eq!(zero_div.try_inverse(), Err(ScalarError::NonInvertible));

    // StackPoly extended GCD Bézout identity over F_11
    let p1 = StackPoly::<Fp<11>, 8>::from_slice(&[Fp::new(1), Fp::new(2), Fp::new(1)]).unwrap();
    let p2 = StackPoly::<Fp<11>, 8>::from_slice(&[Fp::new(10), Fp::new(0), Fp::new(1)]).unwrap();
    let (g, s, t) = p1.extended_gcd(&p2).unwrap();
    assert_eq!(g, StackPoly::from_slice(&[Fp::new(1), Fp::new(1)]).unwrap());
    assert_eq!(s * p1 + t * p2, g);
}

#[test]
fn qp_ultrametric_valuation_and_hensel_lifting_hold() {
    let a = Qp::<7, 8>::from_i64(49 * 3);
    let b = Qp::<7, 8>::from_i64(49 * 4);
    assert_eq!(a.valuation(), Some(2));
    assert_eq!(b.valuation(), Some(2));
    // 49*3 + 49*4 = 343 = 7^3 -> strict ultrametric jump v_7(a + b) = 3 > min(2, 2)!
    assert_eq!((a + b).valuation(), Some(3));
    assert!(Qp::satisfies_ultrametric(a, b));

    let rat = Qp::<7, 8>::from_rational(3, 7).unwrap();
    assert_eq!(rat.valuation(), Some(-1));
    assert_eq!(rat * Qp::<7, 8>::from_i64(7), Qp::<7, 8>::from_i64(3));

    // Lift x^3 - 2 = 0 in Q_5 from r_0 = 3 (3^3 = 27 ≡ 2 mod 5, f'(3) = 27 ≡ 2 != 0 mod 5)
    let cubic = [-2_i64, 0, 0, 1];
    let root5: Qp<5, 10> = hensel_lift_root::<5, 10, 8>(&cubic, 3).unwrap();
    assert!(eval_int_poly_qp(&cubic, root5).is_zero());
    let r_mod = root5.to_u128_mod_pn().unwrap();
    let five10 = 5_u128.pow(10);
    assert_eq!((((r_mod * r_mod) % five10) * r_mod) % five10, 2);

    // Reject non-root and singular derivative
    assert_eq!(
        hensel_lift_root::<7, 6, 4>(&[-2, 0, 1], 2),
        Err(ScalarError::RootNotFound)
    );
    // f(x) = x^2 - 7 at r_0 = 0 has f(0) ≡ 0 (mod 7) and f'(0) = 0 ≡ 0 (mod 7)
    assert_eq!(
        hensel_lift_root::<7, 6, 4>(&[-7, 0, 1], 0),
        Err(ScalarError::SingularDerivative)
    );

    // Complex FieldScalar sanity check
    let c = Complex::new(1.0_f64, 2.0_f64);
    assert!((FieldScalar::norm_sqr(c) - 5.0).abs() < 1e-12);
}

#[test]
fn complex_transcendentals_smith_division_and_unit_quaternion_rotations_hold() {
    use math_scalars::RealScalar;

    assert_eq!(f64::from_usize(7), 7.0);
    assert_eq!(f32::from_f64(2.5), 2.5_f32);
    assert!(f64::NAN.is_nan());
    assert!(!f64::NAN.is_finite());
    assert_eq!(0.0_f64.checked_inv(), None);
    assert_eq!(0.0_f64.try_inverse(), Err(ScalarError::DivisionByZero));
    assert!((2.0_f64.checked_inv().unwrap() - 0.5).abs() < 1e-15);
    assert!((RealScalar::hypot(3.0_f64, 4.0) - 5.0).abs() < 1e-15);

    // Smith division avoids overflow near DBL_MAX and underflow near MIN_POSITIVE
    let big = Complex::new(1e300_f64, -2e300_f64);
    let big_q = big.checked_div(big).unwrap();
    assert!((big_q.re - 1.0).abs() < 1e-12);
    assert!(big_q.im.abs() < 1e-12);
    let tiny = Complex::new(1e-300_f64, 2e-300_f64);
    let tiny_q = tiny.checked_div(tiny).unwrap();
    assert!((tiny_q.re - 1.0).abs() < 1e-12);
    assert!(tiny_q.im.abs() < 1e-12);
    assert_eq!(big.checked_div(Complex::ZERO), None);

    // Polar, exp/ln, and principal sqrt roundtrips
    let polar = Complex::from_polar(2.0_f64, core::f64::consts::FRAC_PI_3);
    assert!((polar.norm() - 2.0).abs() < 1e-12);
    assert!((polar.arg() - core::f64::consts::FRAC_PI_3).abs() < 1e-12);
    assert!(((polar.ln().exp()) - polar).norm() < 1e-12);
    let root = Complex::new(-3.0_f64, 4.0).sqrt();
    assert!(((root * root) - Complex::new(-3.0, 4.0)).norm() < 1e-12);

    // UnitQuaternion rotation_between (orthogonal + antiparallel) and SO(3) matrix parity
    let r90 = UnitQuaternion::rotation_between([1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]).unwrap();
    let v90 = r90.rotate_vec3([1.0, 0.0, 0.0]);
    assert!(v90[0].abs() < 1e-12 && (v90[1] - 1.0).abs() < 1e-12 && v90[2].abs() < 1e-12);
    let anti = UnitQuaternion::rotation_between([1.0_f64, 0.0, 0.0], [-1.0, 0.0, 0.0]).unwrap();
    let v_anti = anti.rotate_vec3([1.0, 0.0, 0.0]);
    assert!((v_anti[0] + 1.0).abs() < 1e-12 && v_anti[1].abs() < 1e-12 && v_anti[2].abs() < 1e-12);
    assert_eq!(
        UnitQuaternion::rotation_between([0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]),
        None
    );
    let mat = r90.to_rotation_matrix3();
    assert!((mat[1][0] - 1.0).abs() < 1e-12 && (mat[0][1] + 1.0).abs() < 1e-12);
    assert_eq!(
        UnitQuaternion::from_wxyz_or_identity([0.0_f64, 0.0, 0.0, 0.0]),
        UnitQuaternion::identity()
    );
}

#[test]
fn fixed256_bit_shifts_and_limb_roundtrips_hold() {
    let one = U256::ONE;
    let hi_bit = one.checked_shl(255).unwrap();
    assert_eq!(hi_bit.bits(), 256);
    assert_eq!(hi_bit.leading_zeros(), 0);
    assert!(hi_bit.bit(255));
    assert!(!hi_bit.bit(254));
    assert_eq!(hi_bit.checked_shr(255), Some(one));
    assert_eq!(one.checked_shl(256), None);
    assert_eq!(one.checked_shr(256), None);
    assert_eq!(U256::from_limbs(hi_bit.limbs()), hi_bit);
    assert_eq!(U256::MAX.wrapping_add(one), U256::ZERO);
    assert_eq!(U256::ZERO.wrapping_sub(one), U256::MAX);
    assert_eq!(U256::from_u64(1024).to_f64(), 1024.0);

    let neg5 = I256::from_i64(-5);
    assert!(neg5.is_negative());
    assert!(!neg5.is_positive());
    assert_eq!(neg5.unsigned_abs(), U256::from_u64(5));
    assert_eq!(neg5.checked_abs(), Some(I256::from_i64(5)));
    assert_eq!(
        neg5.checked_rem(I256::from_i64(3)),
        Some(I256::from_i64(-2))
    );
    assert_eq!(I256::from_raw_u256(neg5.to_raw_u256()), neg5);
}

#[test]
fn modular_primality_and_tonelli_shanks_hold() {
    use math_scalars::{
        is_prime_u64, mod_add_u64, mod_inv_u64, mod_mul_u64, mod_neg_u64, mod_pow_u64,
        mod_reduce_i64, mod_reduce_i128, mod_sub_u64, tonelli_shanks_u64,
    };

    assert!(!is_prime_u64(0) && !is_prime_u64(1));
    assert!(is_prime_u64(2) && is_prime_u64(3) && is_prime_u64(MAX_PRIME_U64));
    assert!(!is_prime_u64(561) && !is_prime_u64(1105) && !is_prime_u64(1729));
    assert!(!is_prime_u64(u64::MAX));

    assert_eq!(mod_reduce_i64(-3, 7), 4);
    assert_eq!(mod_reduce_i128(-10, 7), 4);
    assert_eq!(mod_add_u64(MAX_PRIME_U64 - 1, 5, MAX_PRIME_U64), 4);
    assert_eq!(mod_sub_u64(3, 5, MAX_PRIME_U64), MAX_PRIME_U64 - 2);
    assert_eq!(mod_neg_u64(0, 17), 0);
    assert_eq!(
        mod_mul_u64(MAX_PRIME_U64 - 2, MAX_PRIME_U64 - 3, MAX_PRIME_U64),
        6
    );
    assert_eq!(mod_pow_u64(3, 16, 17), 1);
    assert_eq!(mod_inv_u64(6, 9), None);
    let (roots, count) = tonelli_shanks_u64(2, 17);
    assert_eq!(count, 2);
    assert_eq!((roots[0] * roots[0]) % 17, 2);
    assert_eq!((roots[1] * roots[1]) % 17, 2);
}

#[test]
fn fp_legendre_poly_derivatives_qp_divisions_and_error_formatting_hold() {
    // Fp Legendre symbols, square roots, and const helpers (p = 41 ≡ 1 mod 8 -> 2 is QR)
    let two41 = Fp::<41>::new(2);
    let non_qr41 = Fp::<41>::new(3);
    assert_eq!(two41.legendre_symbol(), 1);
    assert_eq!(non_qr41.legendre_symbol(), -1);
    assert_eq!(Fp::<41>::ZERO.legendre_symbol(), 0);
    let (roots, n_roots) = two41.sqrt_roots();
    assert_eq!(n_roots, 2);
    assert_eq!(roots[0] * roots[0], two41);
    assert_eq!(roots[1] * roots[1], two41);
    assert_eq!(non_qr41.sqrt_roots().1, 0);
    assert_eq!(Fp::<41>::from_i128(-42).value(), 40);
    assert_eq!(
        two41.checked_div(Fp::<41>::ZERO),
        Err(ScalarError::DivisionByZero)
    );

    // StackPoly derivative, monic normalization, evaluation, and capacity bounds
    let poly = StackPoly::<Fp<13>, 4>::from_slice(&[Fp::new(4), Fp::new(6), Fp::new(2)]).unwrap();
    assert_eq!(poly.degree(), Some(2));
    assert_eq!(poly.leading_coeff(), Some(Fp::new(2)));
    assert_eq!(poly.eval(Fp::new(3)), Fp::new(1));
    let deriv = poly.formal_derivative();
    assert_eq!(deriv.coeffs(), &[Fp::new(6), Fp::new(4)]);
    let monic = poly.to_monic().unwrap();
    assert_eq!(monic.coeffs(), &[Fp::new(2), Fp::new(3), Fp::new(1)]);
    assert_eq!(
        StackPoly::<Fp<13>, 2>::monomial(Fp::ONE, 2),
        Err(ScalarError::DegreeOverflow {
            capacity: 2,
            required: 3,
        })
    );

    // Qp uniformizer, p-adic norm, division, and zero-division rejection
    let p7 = Qp::<7, 6>::uniformizer();
    assert_eq!(p7.valuation(), Some(1));
    assert!((p7.padic_abs_f64() - (1.0 / 7.0)).abs() < 1e-12);
    assert_eq!(Qp::<7, 6>::zero().padic_abs_f64(), 0.0);
    let inv3 = Qp::<7, 6>::from_u64(3).try_inverse().unwrap();
    assert_eq!(inv3 * Qp::<7, 6>::from_u64(3), Qp::<7, 6>::one());
    assert_eq!(
        Qp::<7, 6>::one().checked_div(Qp::<7, 6>::zero()),
        Err(ScalarError::DivisionByZero)
    );
    assert_eq!(
        Qp::<7, 6>::from_rational(1, 0),
        Err(ScalarError::ZeroDenominator)
    );
}

#[cfg(feature = "alloc")]
#[test]
fn bigfloat_preserves_256_bit_precision_and_bigrational_preserves_exact_256_bit_integers() {
    use math_scalars::{BigFloat, BigInt, BigRational};

    assert_eq!(
        BigFloat::try_with_prec(0, 1.0),
        Err(ScalarError::ZeroPrecision)
    );
    assert_eq!(
        BigFloat::try_with_prec(256, f64::NAN),
        Err(ScalarError::NonFiniteFloat)
    );
    assert_eq!(
        BigFloat::try_with_prec(256, f64::INFINITY),
        Err(ScalarError::NonFiniteFloat)
    );
    assert_eq!(
        BigFloat::try_with_prec(256, f64::NEG_INFINITY),
        Err(ScalarError::NonFiniteFloat)
    );

    let one = BigFloat::try_with_prec(256, 1.0).unwrap();
    let three = BigFloat::try_with_prec(256, 3.0).unwrap();
    assert_eq!(one.precision(), 256);
    let third = one.clone() / three.clone();
    assert_eq!(third.precision(), 256);
    // Verify precision far exceeds f64's 53 bits: (1/3)*3 - 1 has magnitude < 2^-254
    let diff = (third * three - one).abs();
    assert!(
        diff.to_f64() < 1e-75,
        "256-bit roundtrip residual was {}",
        diff.to_f64()
    );

    // Exact 256-bit numerator and denominator that would collide in f64!
    let u_num = U256::MAX;
    let u_den = U256::MAX.checked_sub(U256::from_u64(2)).unwrap();
    let num_bi = BigInt::from_u256(u_num);
    let den_bi = BigInt::from_u256(u_den);
    let rat = BigRational::try_new(num_bi.clone(), den_bi.clone()).unwrap();
    assert_eq!(rat.numerator(), num_bi);
    assert_eq!(rat.denominator(), den_bi);
    assert_eq!(
        BigRational::try_new(BigInt::from(1_i64), BigInt::zero()),
        Err(ScalarError::ZeroDenominator)
    );
}
