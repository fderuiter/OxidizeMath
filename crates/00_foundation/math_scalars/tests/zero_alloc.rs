#![cfg_attr(any(), verified(opt_out = "tests"))]
// [cite:advanced_linear_algebra]
//! Verifies zero dynamic heap allocations across all stack-allocated `math_scalars` types
//! (`U256`, `I256`, `Complex`, `Quaternion`, `UnitQuaternion`, `Octonion`, `Fp`, `StackPoly`,
//! `FpExt`, and `Qp` including Newton-Hensel root lifting) under [`VerifiedAllocator`].

use math_scalars::{
    Complex, Fp, FpExt, I256, IrreducibleModulus, MAX_PRIME_U64, Octonion, Qp, Quaternion,
    StackPoly, U256, UnitQuaternion, eval_int_poly_qp, hensel_lift_root,
};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ModP7Deg2;

// x^2 + 1 is irreducible over F_7 since -1 = 6 is a quadratic non-residue mod 7.
impl IrreducibleModulus<7, 2> for ModP7Deg2 {
    const COEFFS: [u64; 2] = [1, 0];
}

#[test]
fn fixed256_and_hypercomplex_run_with_zero_heap_allocations() {
    VerifiedEngine::run_verified(|| {
        let a = U256::from_u128(u128::MAX);
        let b = U256::from_u64(17);
        let prod = a.checked_mul(b).unwrap();
        let (q, r) = prod.checked_div_rem(b).unwrap();
        assert_eq!(q, a);
        assert!(r.is_zero());

        let s1 = I256::from_i128(-1_000_000_000_000_000_007);
        let s2 = I256::from_i64(42);
        let eq = s1.checked_div_euclid(s2).unwrap();
        let er = s1.checked_rem_euclid(s2).unwrap();
        assert!(!er.is_negative());
        assert_eq!(eq.checked_mul(s2).unwrap().checked_add(er).unwrap(), s1);

        let z = Complex::new(3.0_f64, -4.0_f64);
        let z_inv = z.try_inverse().unwrap();
        assert!(((z * z_inv).re - 1.0).abs() < 1e-12);

        let quat = Quaternion::new(1.0_f64, 2.0, -3.0, 4.0);
        let q_inv = quat.try_inverse().unwrap();
        assert!(((quat * q_inv).w - 1.0).abs() < 1e-12);

        let rot =
            UnitQuaternion::from_axis_angle([0.0_f64, 0.0, 1.0], core::f64::consts::FRAC_PI_2);
        let v = rot.rotate_vec3([1.0, 0.0, 0.0]);
        assert!((v[0]).abs() < 1e-12 && (v[1] - 1.0).abs() < 1e-12);

        let oct = Octonion::new(1.0_f64, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5);
        let oct_inv = oct.try_inverse().unwrap();
        assert!(((oct * oct_inv).e0 - 1.0).abs() < 1e-12);
    });
}

#[test]
fn montgomery_fp_galois_fp_ext_and_padic_hensel_run_with_zero_heap_allocations() {
    VerifiedEngine::run_verified(|| {
        let x = Fp::<MAX_PRIME_U64>::from_u64(MAX_PRIME_U64 - 3);
        let y = Fp::<MAX_PRIME_U64>::from_u64(MAX_PRIME_U64 - 5);
        let xy = x * y;
        assert_eq!(xy.value(), 15);
        let x_inv = x.try_inverse().unwrap();
        assert_eq!(x * x_inv, Fp::<MAX_PRIME_U64>::ONE);

        let p =
            StackPoly::<Fp<7>, 8>::from_slice(&[Fp::new(6), Fp::new(0), Fp::new(0), Fp::new(1)])
                .unwrap();
        let d = StackPoly::<Fp<7>, 8>::from_slice(&[Fp::new(6), Fp::new(1)]).unwrap();
        let (quot, rem) = p.checked_div_rem(&d).unwrap();
        assert!(rem.is_zero());
        assert_eq!(quot * d, p);

        let elem = FpExt::<7, 2, ModP7Deg2>::from_u64_coeffs([3, 5]);
        let elem_inv = elem.try_inverse().unwrap();
        assert_eq!(elem * elem_inv, FpExt::<7, 2, ModP7Deg2>::ONE);

        // Lift x^2 - 2 ≡ 0 (mod 7) from r_0 = 3 to Q_7 with 6 digits
        let coeffs = [-2_i64, 0, 1];
        let root: Qp<7, 6> = hensel_lift_root::<7, 6, 4>(&coeffs, 3).unwrap();
        let residual = eval_int_poly_qp(&coeffs, root);
        assert!(residual.is_zero());
    });
}

#[test]
fn rotations_octonions_and_padic_field_ops_run_with_zero_heap_allocations() {
    use math_scalars::{ScalarError, is_prime_u64, tonelli_shanks_u64};

    VerifiedEngine::run_verified(|| {
        let q_lo = Quaternion::new(1.0_f64, 0.5, -0.5, 0.25);
        let q_hi = Quaternion::new(-0.25_f64, 1.0, 0.75, -0.5);
        let oct = Octonion::from_quaternions(q_lo, q_hi);
        let (lo_back, hi_back) = oct.to_quaternions();
        assert_eq!(lo_back, q_lo);
        assert_eq!(hi_back, q_hi);
        assert!(oct.is_finite());
        assert!((oct.dot(oct) - oct.norm_sqr()).abs() < 1e-12);
        assert_eq!(Octonion::from_array(oct.to_array()), oct);

        let rot = UnitQuaternion::from_scaled_axis([0.0_f64, 0.0, core::f64::consts::FRAC_PI_2]);
        let inv_rot = rot.inverse();
        let prod = rot * inv_rot;
        assert!((prod.as_wxyz()[0] - 1.0).abs() < 1e-12);
        let m = rot.to_rotation_matrix3();
        assert!((m[0][0]).abs() < 1e-12);
        assert!((m[1][0] - 1.0).abs() < 1e-12);

        assert!(is_prime_u64(MAX_PRIME_U64));
        assert!(!is_prime_u64(MAX_PRIME_U64 - 1));
        let (sq_roots, n) = tonelli_shanks_u64(4, 29);
        assert_eq!(n, 2);
        assert!(sq_roots[0] == 2 || sq_roots[1] == 2);

        let alpha = FpExt::<7, 2, ModP7Deg2>::generator();
        let base = FpExt::<7, 2, ModP7Deg2>::from_base(Fp::new(3));
        assert_eq!(base.mul_by_alpha(), alpha.scale(Fp::new(3)));
        assert!(!alpha.is_zero());
        assert_eq!(
            FpExt::<7, 2, ModP7Deg2>::ZERO.try_inverse(),
            Err(ScalarError::DivisionByZero)
        );

        let a = Qp::<11, 6>::from_rational(5, 11).unwrap();
        let b = Qp::<11, 6>::from_rational(6, 11).unwrap();
        assert_eq!((a + b).valuation(), Some(0));
        assert_eq!(a + b, Qp::<11, 6>::one());
        assert_eq!((a - a), Qp::<11, 6>::zero());
        assert_eq!((-a) + a, Qp::<11, 6>::zero());
    });
}
