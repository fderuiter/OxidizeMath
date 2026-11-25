//! Unit tests for pure_math math_types module.

use pure_math::math_types::ops::{Pow, RemRounding};
use pure_math::math_types::{BigRational, Float, Integer, Rational, ScalarError};

#[test]
fn test_integer_basics_and_conversions() {
    let mut a = Integer::new();
    assert_eq!(a.to_i32(), Some(0));
    assert_eq!(a.to_usize(), Some(0));
    assert_eq!(a.to_u64(), Some(0));
    assert!((a.to_f64() - 0.0).abs() < 1e-9);

    a.assign(42);
    assert_eq!(a.to_i32(), Some(42));
    assert_eq!(a.to_usize(), Some(42));
    assert_eq!(a.to_u64(), Some(42));
    assert!((a.to_f64() - 42.0).abs() < 1e-9);

    let b = Integer::from(100i64);
    assert_eq!(b.to_i32(), Some(100));

    let parsed = Integer::parse("12345").expect("valid parse");
    assert_eq!(parsed.to_i32(), Some(12345));
    assert!(Integer::parse("invalid_num").is_err());

    let hex_parsed = Integer::from_str_radix("1f", 16).expect("valid hex");
    assert_eq!(hex_parsed.to_i32(), Some(31));
    assert!(Integer::from_str_radix("zz", 16).is_err());

    let def: Integer = Default::default();
    assert_eq!(def.to_i32(), Some(0));
}

#[test]
fn test_integer_arithmetic() {
    let a = Integer::from(15);
    let b = Integer::from(4);

    assert_eq!((&a + &b).to_i32(), Some(19));
    assert_eq!((&a - &b).to_i32(), Some(11));
    assert_eq!((&a * &b).to_i32(), Some(60));
    assert_eq!((&a / &b).to_i32(), Some(3));

    assert_eq!((a.clone() + b.clone()).to_i32(), Some(19));
    assert_eq!((a.clone() - b.clone()).to_i32(), Some(11));
    assert_eq!((a.clone() * b.clone()).to_i32(), Some(60));
    assert_eq!((a.clone() / b.clone()).to_i32(), Some(3));

    let mut c = a.clone();
    c += &b;
    assert_eq!(c.to_i32(), Some(19));
    c -= &b;
    assert_eq!(c.to_i32(), Some(15));
    c *= &b;
    assert_eq!(c.to_i32(), Some(60));
    c /= b.clone();
    assert_eq!(c.to_i32(), Some(15));
    c %= b.clone();
    assert_eq!(c.to_i32(), Some(3));

    let mut d = a.clone();
    d += b.clone();
    assert_eq!(d.to_i32(), Some(19));
    d -= b.clone();
    assert_eq!(d.to_i32(), Some(15));
    d *= b.clone();
    assert_eq!(d.to_i32(), Some(60));
    d /= b.clone();
    assert_eq!(d.to_i32(), Some(15));

    // Pow & RemRounding
    let base = Integer::from(3);
    assert_eq!(base.pow(3u32).to_i32(), Some(27));

    let neg = Integer::from(-7);
    let mod_val = Integer::from(4);
    assert_eq!(neg.clone().rem_euc(mod_val.clone()).to_i32(), Some(1));
    assert_eq!(neg.rem_euc(&mod_val).to_i32(), Some(1));
}

#[test]
fn test_rational_basics_and_arithmetic() {
    let r = Rational::new();
    assert!((r.numer() - 0.0).abs() < 1e-9);
    assert!((r.denom() - 1.0).abs() < 1e-9);

    let r1 = Rational::from((3, 4));
    assert!((r1.numer() - 3.0).abs() < 1e-9);
    assert!((r1.denom() - 4.0).abs() < 1e-9);

    let r_neg = Rational::from((3, -4));
    assert!((r_neg.numer() - -3.0).abs() < 1e-9);
    assert!((r_neg.denom() - 4.0).abs() < 1e-9);

    let def: Rational = Default::default();
    assert_eq!(def, r);

    let r2 = Rational::from((1, 2));
    let sum = r1.clone() + r2.clone();
    assert!((sum.numer() - 5.0).abs() < 1e-9);
    assert!((sum.denom() - 4.0).abs() < 1e-9);

    let sum_ref = &r1 + &r2;
    assert_eq!(sum, sum_ref);

    let sum_mix = &r1 + r2.clone();
    assert_eq!(sum, sum_mix);

    let prod = r1.clone() * r2.clone();
    assert!((prod.numer() - 3.0).abs() < 1e-9);
    assert!((prod.denom() - 8.0).abs() < 1e-9);

    let prod_ref = &r1 * &r2;
    assert_eq!(prod, prod_ref);

    let prod_mix = &r1 * r2.clone();
    assert_eq!(prod, prod_mix);
}

#[test]
fn test_float_basics_and_arithmetic() {
    let f1 = Float::with_val(53, 2.5);
    assert_eq!(f1.clone().round(), 3.0);

    let f2 = Float::with_val(53, -4.75);
    let f_abs = f2.clone().abs();
    assert_eq!(f_abs.round(), 5.0);

    let sum = f1.clone() + f2.clone();
    assert_eq!(sum.round(), -2.0);

    let diff = f1.clone() - f2.clone();
    assert_eq!(diff.round(), 7.0);

    let prod = f1.clone() * Float::with_val(53, 2.0);
    assert_eq!(prod.round(), 5.0);

    let div = f1.clone() / Float::with_val(53, 2.0);
    assert_eq!(div.round(), 1.0);

    assert!(f1 > f2);
    assert_eq!(f1, Float::with_val(53, 2.5));

    // Preserve 256-bit caller precision across 1/3 * 3 and reject NaN / Infinity / prec == 0:
    assert_eq!(
        Float::try_with_prec(0, 1.0),
        Err(ScalarError::ZeroPrecision)
    );
    assert_eq!(
        Float::try_with_prec(256, f64::NAN),
        Err(ScalarError::NonFiniteFloat)
    );
    assert_eq!(
        Float::try_with_prec(256, f64::INFINITY),
        Err(ScalarError::NonFiniteFloat)
    );
    assert_eq!(
        Float::try_with_prec(256, f64::NEG_INFINITY),
        Err(ScalarError::NonFiniteFloat)
    );
    let one_256 = Float::try_with_prec(256, 1.0).unwrap();
    let three_256 = Float::try_with_prec(256, 3.0).unwrap();
    assert_eq!(one_256.precision(), 256);
    let third_256 = &one_256 / &three_256;
    assert_eq!(third_256.precision(), 256);
    let err_256 = (&third_256 * &three_256 - one_256).abs();
    let bound = Float::try_with_prec(256, 2.0_f64.powi(-250)).unwrap();
    assert!(err_256 < bound);

    // Exact large integer numerator and denominator on Rational and BigRational without f64 loss:
    let big_num = Integer::from(1_i64).shl_bits(100) + Integer::from(3_i64);
    let big_den = Integer::from(1_i64).shl_bits(100) + Integer::from(7_i64);
    let exact_rat = Rational::try_from_integers(big_num.clone(), big_den.clone()).unwrap();
    assert_eq!(exact_rat.numerator(), big_num);
    assert_eq!(exact_rat.denominator(), big_den);
    let big_rat: BigRational = exact_rat.to_big_rational();
    assert_eq!(big_rat.numerator().to_string(), big_num.0.to_string());
    assert_eq!(big_rat.denominator().to_string(), big_den.0.to_string());
}
