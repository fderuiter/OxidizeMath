#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:advanced_linear_algebra]
// [cite:modular_polynomials_review]
//! Optional `#![no_std]` `serde` round-trip and algebraic invariant validation tests for `math_scalars`.

#[cfg(feature = "serde")]
use math_scalars::{
    Complex, Fp, FpExt, I256, IrreducibleModulus, MAX_PRIME_U64, Octonion, Qp, ScalarError,
    StackPoly, U256, UnitQuaternion, hensel_lift_root,
};

#[cfg(feature = "serde")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ModP5Deg3;

#[cfg(feature = "serde")]
impl IrreducibleModulus<5, 3> for ModP5Deg3 {
    const COEFFS: [u64; 3] = [1, 1, 0];
}

#[test]
fn math_scalars_serde_json_dev_dep_linked() {
    let parsed: u32 = serde_json::from_str("42").expect("valid json integer");
    assert_eq!(parsed, 42);
}

#[cfg(feature = "serde")]
#[test]
fn no_std_serde_fixed_and_hypercomplex_roundtrip() {
    let u = U256::from_u128(u128::MAX);
    assert_eq!(
        serde_json::from_str::<U256>(&serde_json::to_string(&u).unwrap()).unwrap(),
        u
    );
    let i = I256::MIN;
    assert_eq!(
        serde_json::from_str::<I256>(&serde_json::to_string(&i).unwrap()).unwrap(),
        i
    );

    let z = Complex::new(1.5_f64, -2.25_f64);
    assert_eq!(
        serde_json::from_str::<Complex<f64>>(&serde_json::to_string(&z).unwrap()).unwrap(),
        z
    );
    let rot = UnitQuaternion::from_axis_angle([0.0_f64, 0.0, 1.0], core::f64::consts::FRAC_PI_4);
    let rot_back: UnitQuaternion<f64> =
        serde_json::from_str(&serde_json::to_string(&rot).unwrap()).unwrap();
    assert!((rot_back.quaternion().norm_sqr() - 1.0).abs() < 1e-12);
    assert!(
        serde_json::from_str::<UnitQuaternion<f64>>(r#"{"w":0.0,"x":0.0,"y":0.0,"z":0.0}"#)
            .is_err()
    );
    assert!(
        serde_json::from_str::<UnitQuaternion<f64>>(r#"{"w":3.0,"x":0.0,"y":0.0,"z":0.0}"#)
            .is_err()
    );

    let oct = Octonion::new(1.0_f64, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5);
    assert_eq!(
        serde_json::from_str::<Octonion<f64>>(&serde_json::to_string(&oct).unwrap()).unwrap(),
        oct
    );
}

#[cfg(feature = "serde")]
#[test]
fn no_std_serde_finite_field_poly_and_padic_roundtrip() {
    let fp = Fp::<MAX_PRIME_U64>::from_u64(MAX_PRIME_U64 - 5);
    let fp_json = serde_json::to_string(&fp).unwrap();
    assert_eq!(
        serde_json::from_str::<Fp<MAX_PRIME_U64>>(&fp_json).unwrap(),
        fp
    );
    assert!(serde_json::from_str::<Fp<7>>("7").is_err());

    let poly = StackPoly::<Fp<11>, 4>::from_slice(&[Fp::new(3), Fp::new(0), Fp::new(5)]).unwrap();
    assert_eq!(
        serde_json::from_str::<StackPoly<Fp<11>, 4>>(&serde_json::to_string(&poly).unwrap())
            .unwrap(),
        poly
    );
    assert!(serde_json::from_str::<StackPoly<Fp<11>, 2>>("[1, 2, 3]").is_err());

    let ext = FpExt::<5, 3, ModP5Deg3>::from_u64_coeffs([2, 3, 4]);
    assert_eq!(
        serde_json::from_str::<FpExt<5, 3, ModP5Deg3>>(&serde_json::to_string(&ext).unwrap())
            .unwrap(),
        ext
    );

    let qp = hensel_lift_root::<7, 6, 4>(&[-2, 0, 1], 3).unwrap();
    assert_eq!(
        serde_json::from_str::<Qp<7, 6>>(&serde_json::to_string(&qp).unwrap()).unwrap(),
        qp
    );
    assert!(
        serde_json::from_str::<Qp<7, 2>>(r#"{"digits":[0,1],"valuation":0,"is_zero":false}"#)
            .is_err()
    );
    assert!(
        serde_json::from_str::<Qp<7, 2>>(r#"{"digits":[1,0],"valuation":2,"is_zero":true}"#)
            .is_err()
    );

    let err = ScalarError::DegreeOverflow {
        capacity: 4,
        required: 9,
    };
    assert_eq!(
        serde_json::from_str::<ScalarError>(&serde_json::to_string(&err).unwrap()).unwrap(),
        err
    );
}

#[cfg(all(feature = "alloc", feature = "serde"))]
#[test]
fn bigfloat_serde_preserves_mantissa_bit_precision_and_rejects_zero_prec() {
    use math_scalars::{BigFloat, BigInt, BigRational};

    let bf =
        BigFloat::try_with_prec(256, 1.0).unwrap() / BigFloat::try_with_prec(256, 3.0).unwrap();
    let json = serde_json::to_string(&bf).unwrap();
    let decoded: BigFloat = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.precision(), 256);
    assert_eq!(decoded, bf);
    assert!(serde_json::from_str::<BigFloat>(&json.replace("\"prec\":256", "\"prec\":0")).is_err());

    let rat = BigRational::try_new(BigInt::from(-22_i64), BigInt::from(7_i64)).unwrap();
    assert_eq!(
        serde_json::from_str::<BigRational>(&serde_json::to_string(&rat).unwrap()).unwrap(),
        rat
    );
}
