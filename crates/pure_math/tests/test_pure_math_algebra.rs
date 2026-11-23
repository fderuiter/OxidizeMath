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
