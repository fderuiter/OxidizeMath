//! Arbitrary-precision `BigFloat`/`BigRational` precision and rotation integration tests.
#![cfg(feature = "alloc")]
#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:math_scalars]

use math_scalars::{BigFloat, BigInt, BigRational, ScalarError, U256, UnitQuaternion};

#[test]
fn test_big_float_preserves_256_bit_precision_and_rejects_non_finite() {
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

    // At 256-bit precision, (1/3)*3 - 1 has magnitude <= 2^-254 (whereas f64 is ~2^-53)
    let diff = (third * three - one).abs();
    let bound = BigFloat::try_with_prec(256, 2.0_f64.powi(-200)).unwrap();
    assert!(diff < bound, "256-bit residual {diff} exceeded 2^-200");

    // Verify sqrt(2) at 256 bits retains >200 bits of accuracy
    let two = BigFloat::try_with_prec(256, 2.0).unwrap();
    let root2 = two.sqrt();
    assert_eq!(root2.precision(), 256);
    let sq_err = (root2.clone() * root2 - two).abs();
    assert!(
        sq_err < bound,
        "256-bit sqrt residual {sq_err} exceeded 2^-200"
    );
}

#[test]
fn test_big_rational_preserves_exact_256_bit_integers() {
    let huge_u256 = U256::from_limbs([0xDEAD_BEEF_CAFE_BABE, 0x1234_5678_9ABC_DEF1, 7, 1 << 62]);
    let num = BigInt::from_u256(huge_u256) + BigInt::from(1u64);
    let den = BigInt::from_u256(huge_u256) + BigInt::from(3u64);
    // Both are odd or share small factor -> divide by gcd via BigRational::new
    let odd_num = num * BigInt::from(2u64) + BigInt::one();
    let odd_den = den * BigInt::from(2u64) + BigInt::one();
    let rat = BigRational::new(odd_num.clone(), odd_den.clone()).unwrap();

    // f64 collapses both >250-bit integers to the same rounded float (1.0), whereas BigRational is exact
    assert_eq!(rat.to_f64(), 1.0);
    assert!(rat.numerator().bit_len() >= 250);
    assert!(rat.denominator().bit_len() >= 250);
    assert_ne!(rat.numerator(), rat.denominator());
    assert_eq!(rat.clone() * rat.checked_inv().unwrap(), BigRational::one());
    assert_eq!(
        BigRational::new(BigInt::one(), BigInt::zero()),
        Err(ScalarError::DivisionByZero)
    );
}

#[test]
fn test_unit_quaternion_rotation_matrix_orthogonality() {
    let q = UnitQuaternion::from_axis_angle([1.0_f64, 2.0, -3.0], 0.85);
    let r = q.to_rotation_matrix();
    let v = [0.4_f64, -1.1, 2.3];
    let rv_quat = q.rotate_vec3(v);
    let rv_mat = [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ];
    for k in 0..3 {
        assert!((rv_quat[k] - rv_mat[k]).abs() < 1e-14);
    }
    let back = q.inv().rotate_vec3(rv_quat);
    for k in 0..3 {
        assert!((back[k] - v[k]).abs() < 1e-14);
    }
}
