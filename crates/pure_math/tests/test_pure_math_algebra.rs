//! Integration tests for test pure math algebra.
use pure_math::pure_math::algebra::group::{
    Permutation, Zn, generate_right_coset, is_normal_subgroup, is_subgroup,
};
use pure_math::pure_math::algebra::traits::{Field, Group, Monoid, Ring, Semigroup};
use pure_math::pure_math::algebra::*;

#[test]
#[verified_engine::verified]
fn test_cyclic_group_z5() {
    // Z5 is a group under addition mod 5
    let a = Zn::<5>::new(2);
    let b = Zn::<5>::new(3);

    // Closure: 2 + 3 = 5 = 0 mod 5
    let c = Zn::<5>::operate(&a, &b);
    assert_eq!(c.value, 0);
    assert_eq!(c, Zn::<5>::identity());

    // Inverse: 2 + 3 = 0, so 3 is inverse of 2
    assert_eq!(a.inverse(), b);
    assert_eq!(b.inverse(), a);
}

#[test]
#[verified_engine::verified]
fn test_subgroups_z6() {
    // Z6 = {0, 1, 2, 3, 4, 5}
    // H = {0, 2, 4} should be a subgroup
    let h_vals = [0, 2, 4];
    let h: Vec<Zn<6>> = h_vals.iter().map(|&x| Zn::new(x)).collect();

    assert!(is_subgroup(&h));

    // K = {0, 3} should be a subgroup
    let k_vals = [0, 3];
    let k: Vec<Zn<6>> = k_vals.iter().map(|&x| Zn::new(x)).collect();
    assert!(is_subgroup(&k));

    // L = {0, 1} is NOT a subgroup (1+1=2 not in L)
    let l_vals = [0, 1];
    let l: Vec<Zn<6>> = l_vals.iter().map(|&x| Zn::new(x)).collect();
    assert!(!is_subgroup(&l));
}

#[test]
#[verified_engine::verified]
fn test_permutation_s3() {
    // S3 perms of {0, 1, 2}
    let p1 = Permutation::new(vec![1, 0, 2]); // (0 1)
    let p2 = Permutation::new(vec![0, 2, 1]); // (1 2)

    // p1 * p2 = (0 1)(1 2) = (0 1 2) -> map: 0->1->2, 1->0->0, 2->2->1 => [2, 0, 1]
    // Wait, composition order: (f*g)(x) = f(g(x))
    // p2 maps 0->0, p1 maps 0->1 => 0->1
    // p2 maps 1->2, p1 maps 2->2 => 1->2
    // p2 maps 2->1, p1 maps 1->0 => 2->0
    // Result map: [1, 2, 0]

    let prod = Permutation::operate(&p1, &p2);
    assert_eq!(prod.map, vec![1, 2, 0]);

    // Verify non-abelian: p2 * p1
    // p1 maps 0->1, p2 maps 1->2 => 0->2
    // p1 maps 1->0, p2 maps 0->0 => 1->0
    // p1 maps 2->2, p2 maps 2->1 => 2->1
    // Result map: [2, 0, 1]
    let prod_rev = Permutation::operate(&p2, &p1);
    assert_eq!(prod_rev.map, vec![2, 0, 1]);

    assert_ne!(prod, prod_rev);
}

#[test]
#[verified_engine::verified]
fn test_finite_field_f7() {
    let a = Fp::<7>::new(3);
    let b = Fp::<7>::new(4);

    // Addition: 3 + 4 = 7 = 0
    assert_eq!(a + b, Fp::<7>::zero());

    // Multiplication: 3 * 4 = 12 = 5
    assert_eq!(a * b, Fp::<7>::new(5));

    // Inverse: 3 * 5 = 15 = 1. So inv(3) = 5.
    let inv_a = a.multiplicative_inverse();
    assert_eq!(inv_a, Fp::<7>::new(5));
    assert_eq!(a * inv_a, Fp::<7>::one());
}

#[test]
#[verified_engine::verified]
fn test_polynomial_arithmetic() {
    // P(x) = 1 + x (in Z_5)
    let p1 = Polynomial::new(vec![Fp::<5>::new(1), Fp::<5>::new(1)]);

    // Q(x) = 1 - x = 1 + 4x (in Z_5)
    let p2 = Polynomial::new(vec![Fp::<5>::new(1), Fp::<5>::new(4)]); // 4 == -1 mod 5

    // P * Q = (1+x)(1-x) = 1 - x^2 = 1 + 4x^2
    let prod = p1.clone() * p2.clone();

    // Expected coeffs: [1, 0, 4]
    assert_eq!(prod.coeffs.len(), 3);
    assert_eq!(prod.coeffs[0], Fp::<5>::new(1));
    assert_eq!(prod.coeffs[1], Fp::<5>::new(0));
    assert_eq!(prod.coeffs[2], Fp::<5>::new(4));
}

#[test]
#[verified_engine::verified]
fn test_lagrange_theorem_check() {
    // Lagrange: |H| divides |G|
    // Z6 has order 6.
    // Subgroup {0, 2, 4} has order 3. 3 divides 6.
    // Subgroup {0, 3} has order 2. 2 divides 6.

    let h_vals = [0, 2, 4];
    let h: Vec<Zn<6>> = h_vals.iter().map(|&x| Zn::new(x)).collect();

    assert!(is_subgroup(&h));

    // Cosets of H in Z6:
    // H + 0 = {0, 2, 4}
    // H + 1 = {1, 3, 5}
    let coset1 = generate_right_coset(&h, &Zn::new(1));
    assert_eq!(coset1.len(), 3);
    // Check elements
    let coset1_vals: Vec<usize> = coset1.iter().map(|x| x.value).collect();
    // Since map order is preserved (0+1, 2+1, 4+1) -> 1, 3, 5
    // Order might depend on input order.
    assert!(coset1_vals.contains(&1));
    assert!(coset1_vals.contains(&3));
    assert!(coset1_vals.contains(&5));
}

#[test]
#[verified_engine::verified]
fn test_normal_subgroup_check() {
    // In Z6 (abelian), all subgroups are normal.
    let h_vals = [0, 2, 4];
    let h: Vec<Zn<6>> = h_vals.iter().map(|&x| Zn::new(x)).collect();
    let g_vals = [0, 1, 2, 3, 4, 5];
    let g: Vec<Zn<6>> = g_vals.iter().map(|&x| Zn::new(x)).collect();

    assert!(is_normal_subgroup(&g, &h));
}

fn assert_abelian_commutativity<G: AbelianGroup + PartialEq + core::fmt::Debug>(a: &G, b: &G) {
    assert_eq!(G::operate(a, b), G::operate(b, a));
}

#[test]
#[verified_engine::verified]
fn test_abelian_group_reexport_and_left_cosets() {
    let a = Zn::<6>::new(2);
    let b = Zn::<6>::new(5);
    assert_abelian_commutativity(&a, &b);

    let subgroup = [Zn::<6>::new(0), Zn::<6>::new(3)];
    assert!(is_subgroup(&subgroup));
    let left = generate_left_coset(&subgroup, &a);
    assert_eq!(left, vec![Zn::<6>::new(2), Zn::<6>::new(5)]);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ModF32Irred;
impl IrreducibleModulus<3, 2> for ModF32Irred {
    const COEFFS: [u64; 2] = [1, 0]; // x^2 + 1 irreducible over F_3
}

#[test]
#[verified_engine::verified]
fn test_montgomery_fp64_fp2_fp_ext_stack_poly_and_ambs_qp() {
    // Explicit P = 2 path:
    let one2 = Fp::<2>::new(1);
    assert_eq!(one2 + one2, Fp::<2>::zero());
    assert_eq!(one2.checked_inverse(), Ok(one2));
    assert!(Fp::<2>::zero().checked_inverse().is_err());

    // Carry-safe 64-bit primes above sqrt(i64::MAX) and up to 2^64 - 59:
    const P_MID: u64 = 3_037_000_507;
    const P_MAX: u64 = 18_446_744_073_709_551_557;
    let x_mid = Fp::<P_MID>::from_u64(P_MID - 2);
    assert_eq!(x_mid * x_mid, Fp::<P_MID>::from_u64(4));
    assert_eq!(x_mid * x_mid.checked_inverse().unwrap(), Fp::<P_MID>::one());

    let x_max = Fp::<P_MAX>::from_u64(P_MAX - 3);
    assert_eq!(x_max * x_max, Fp::<P_MAX>::from_u64(9));
    assert_eq!(x_max * x_max.checked_inverse().unwrap(), Fp::<P_MAX>::one());

    // Polynomial Euclidean division + StackPoly:
    let dividend = Polynomial::new(vec![Fp::<7>::new(6), Fp::<7>::new(0), Fp::<7>::new(1)]); // x^2 - 1
    let divisor = Polynomial::new(vec![Fp::<7>::new(6), Fp::<7>::new(1)]); // x - 1
    let (quot, rem) = dividend.checked_div_rem(&divisor).unwrap();
    assert!(rem.is_zero());
    assert_eq!(
        quot,
        Polynomial::new(vec![Fp::<7>::new(1), Fp::<7>::new(1)])
    );

    let sp_a = StackPoly::<Fp<7>, 8>::from_slice(&[Fp::new(6), Fp::new(0), Fp::new(1)]).unwrap();
    let sp_b = StackPoly::<Fp<7>, 8>::from_slice(&[Fp::new(6), Fp::new(1)]).unwrap();
    assert_eq!(
        sp_a / sp_b,
        StackPoly::<Fp<7>, 8>::from_slice(&[Fp::new(1), Fp::new(1)]).unwrap()
    );
    assert!((sp_a % sp_b).is_zero());

    // Galois extension F_{3^2} = F_3[x] / (x^2 + 1):
    let i_ext = FpExt::<3, 2, ModF32Irred>::generator();
    assert_eq!(i_ext * i_ext, -FpExt::<3, 2, ModF32Irred>::one());
    let u_ext = FpExt::<3, 2, ModF32Irred>::from_base(Fp::new(1)) + i_ext;
    assert_eq!(u_ext * u_ext.checked_inverse().unwrap(), FpExt::one());

    // Stack Qp quadratic residue Hensel lifting in ambs:
    let (roots, count) =
        pure_math::pure_math::number_theory::ambs::lift_quadratic_residue_qp::<7, 6>(2);
    assert_eq!(count, 2);
    for r in roots {
        let diff = r * r - pure_math::pure_math::number_theory::ambs::Qp::<7, 6>::from_i64(2);
        assert!(diff.is_zero() || diff.valuation().unwrap() >= 6);
    }
}
