#![allow(missing_docs, clippy::needless_range_loop)]

use pure_math::pure_math::algebra::polynomial::Polynomial;
use pure_math::pure_math::algebra::traits::Ring;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    if a == 0 { 1 } else { a.abs() }
}

/// A simplified rational number ring for testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TestRational {
    num: i64,
    den: i64,
}

impl TestRational {
    fn new(num: i64, den: i64) -> Self {
        assert_ne!(den, 0, "denominator cannot be zero");
        let g = gcd(num.abs(), den.abs());
        let sign = if den < 0 { -1 } else { 1 };
        Self {
            num: sign * (num / g),
            den: (den.abs()) / g,
        }
    }

    fn from_int(n: i64) -> Self {
        Self::new(n, 1)
    }
}

impl Add for TestRational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)
    }
}

impl Sub for TestRational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
    }
}

impl Mul for TestRational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.num * rhs.num, self.den * rhs.den)
    }
}

impl AddAssign for TestRational {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for TestRational {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for TestRational {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for TestRational {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.num, self.den)
    }
}

impl Ring for TestRational {
    fn zero() -> Self {
        Self::new(0, 1)
    }

    fn one() -> Self {
        Self::new(1, 1)
    }
}

/// A 2x2 integer matrix ring for testing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TestMat2 {
    data: [[i64; 2]; 2],
}

impl TestMat2 {
    fn new(data: [[i64; 2]; 2]) -> Self {
        Self { data }
    }

    fn scalar(s: i64) -> Self {
        Self {
            data: [[s, 0], [0, s]],
        }
    }
}

impl Add for TestMat2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut res = [[0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                res[r][c] = self.data[r][c] + rhs.data[r][c];
            }
        }
        Self::new(res)
    }
}

impl Sub for TestMat2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut res = [[0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                res[r][c] = self.data[r][c] - rhs.data[r][c];
            }
        }
        Self::new(res)
    }
}

impl Mul for TestMat2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut res = [[0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                for k in 0..2 {
                    res[r][c] += self.data[r][k] * rhs.data[k][c];
                }
            }
        }
        Self::new(res)
    }
}

impl AddAssign for TestMat2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for TestMat2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for TestMat2 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for TestMat2 {
    type Output = Self;
    fn neg(self) -> Self {
        let mut res = [[0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                res[r][c] = -self.data[r][c];
            }
        }
        Self::new(res)
    }
}

impl Ring for TestMat2 {
    fn zero() -> Self {
        Self::scalar(0)
    }

    fn one() -> Self {
        Self::scalar(1)
    }
}

#[test]
#[verified_engine::verified]
fn test_polynomial_ring_equivalence() {
    // P(x) = 2 + 3x + x^2
    // Q(x) = 1 - 2x + 4x^3
    let p_z = Polynomial::new(vec![2i64, 3, 1]);
    let q_z = Polynomial::new(vec![1i64, -2, 0, 4]);

    let p_q = Polynomial::new(vec![
        TestRational::from_int(2),
        TestRational::from_int(3),
        TestRational::from_int(1),
    ]);
    let q_q = Polynomial::new(vec![
        TestRational::from_int(1),
        TestRational::from_int(-2),
        TestRational::from_int(0),
        TestRational::from_int(4),
    ]);

    let p_m = Polynomial::new(vec![
        TestMat2::scalar(2),
        TestMat2::scalar(3),
        TestMat2::scalar(1),
    ]);
    let q_m = Polynomial::new(vec![
        TestMat2::scalar(1),
        TestMat2::scalar(-2),
        TestMat2::scalar(0),
        TestMat2::scalar(4),
    ]);

    // 1. Addition equivalence
    let sum_z = &p_z + &q_z;
    let sum_q = &p_q + &q_q;
    let sum_m = &p_m + &q_m;

    assert_eq!(sum_z.coeffs.len(), sum_q.coeffs.len());
    assert_eq!(sum_z.coeffs.len(), sum_m.coeffs.len());
    for i in 0..sum_z.coeffs.len() {
        assert_eq!(sum_q.coeffs[i], TestRational::from_int(sum_z.coeffs[i]));
        assert_eq!(sum_m.coeffs[i], TestMat2::scalar(sum_z.coeffs[i]));
    }

    // 2. Subtraction equivalence
    let diff_z = &p_z - &q_z;
    let diff_q = &p_q - &q_q;
    let diff_m = &p_m - &q_m;

    assert_eq!(diff_z.coeffs.len(), diff_q.coeffs.len());
    assert_eq!(diff_z.coeffs.len(), diff_m.coeffs.len());
    for i in 0..diff_z.coeffs.len() {
        assert_eq!(diff_q.coeffs[i], TestRational::from_int(diff_z.coeffs[i]));
        assert_eq!(diff_m.coeffs[i], TestMat2::scalar(diff_z.coeffs[i]));
    }

    // 3. Multiplication equivalence
    let prod_z = &p_z * &q_z;
    let prod_q = &p_q * &q_q;
    let prod_m = &p_m * &q_m;

    assert_eq!(prod_z.coeffs.len(), prod_q.coeffs.len());
    assert_eq!(prod_z.coeffs.len(), prod_m.coeffs.len());
    for i in 0..prod_z.coeffs.len() {
        assert_eq!(prod_q.coeffs[i], TestRational::from_int(prod_z.coeffs[i]));
        assert_eq!(prod_m.coeffs[i], TestMat2::scalar(prod_z.coeffs[i]));
    }

    // 4. Scaling equivalence
    let scaled_z = p_z.scale(&5);
    let scaled_q = p_q.scale(&TestRational::from_int(5));
    let scaled_m = p_m.scale(&TestMat2::scalar(5));

    assert_eq!(scaled_z.coeffs.len(), scaled_q.coeffs.len());
    assert_eq!(scaled_z.coeffs.len(), scaled_m.coeffs.len());
    for i in 0..scaled_z.coeffs.len() {
        assert_eq!(
            scaled_q.coeffs[i],
            TestRational::from_int(scaled_z.coeffs[i])
        );
        assert_eq!(scaled_m.coeffs[i], TestMat2::scalar(scaled_z.coeffs[i]));
    }
}

#[test]
#[verified_engine::verified]
fn test_in_place_vector_arithmetic_equivalence() {
    let p_z = Polynomial::new(vec![3i64, -1, 4]);
    let q_z = Polynomial::new(vec![2i64, 5, -4, 1]);

    let p_q = Polynomial::new(vec![
        TestRational::from_int(3),
        TestRational::from_int(-1),
        TestRational::from_int(4),
    ]);
    let q_q = Polynomial::new(vec![
        TestRational::from_int(2),
        TestRational::from_int(5),
        TestRational::from_int(-4),
        TestRational::from_int(1),
    ]);

    let p_m = Polynomial::new(vec![
        TestMat2::scalar(3),
        TestMat2::scalar(-1),
        TestMat2::scalar(4),
    ]);
    let q_m = Polynomial::new(vec![
        TestMat2::scalar(2),
        TestMat2::scalar(5),
        TestMat2::scalar(-4),
        TestMat2::scalar(1),
    ]);

    // Test in-place AddAssign
    let mut add_acc_z = p_z.clone();
    add_acc_z += &q_z;
    let mut add_acc_q = p_q.clone();
    add_acc_q += &q_q;
    let mut add_acc_m = p_m.clone();
    add_acc_m += &q_m;

    assert_eq!(add_acc_z, &p_z + &q_z);
    assert_eq!(add_acc_q, &p_q + &q_q);
    assert_eq!(add_acc_m, &p_m + &q_m);

    // Test in-place SubAssign
    let mut sub_acc_z = p_z.clone();
    sub_acc_z -= &q_z;
    let mut sub_acc_q = p_q.clone();
    sub_acc_q -= &q_q;
    let mut sub_acc_m = p_m.clone();
    sub_acc_m -= &q_m;

    assert_eq!(sub_acc_z, &p_z - &q_z);
    assert_eq!(sub_acc_q, &p_q - &q_q);
    assert_eq!(sub_acc_m, &p_m - &q_m);

    // Test in-place MulAssign
    let mut mul_acc_z = p_z.clone();
    mul_acc_z *= &q_z;
    let mut mul_acc_q = p_q.clone();
    mul_acc_q *= &q_q;
    let mut mul_acc_m = p_m.clone();
    mul_acc_m *= &q_m;

    assert_eq!(mul_acc_z, &p_z * &q_z);
    assert_eq!(mul_acc_q, &p_q * &q_q);
    assert_eq!(mul_acc_m, &p_m * &q_m);
}

#[test]
#[verified_engine::verified]
fn test_edge_cases_and_vector_resizing() {
    let zero_poly = Polynomial::<i64>::zero();
    let poly_a = Polynomial::new(vec![1i64, 2]);
    let poly_b = Polynomial::new(vec![3i64, 4, 5, 6]);

    // Empty polynomial added to non-empty
    let mut acc = zero_poly.clone();
    acc += poly_b.clone();
    assert_eq!(acc, poly_b);

    // Non-empty added to empty
    let mut acc2 = poly_b.clone();
    acc2 += zero_poly.clone();
    assert_eq!(acc2, poly_b);

    // Smaller degree += larger degree
    let mut acc3 = poly_a.clone();
    acc3 += poly_b.clone();
    assert_eq!(acc3, Polynomial::new(vec![4, 6, 5, 6]));

    // Larger degree -= smaller degree
    let mut acc4 = poly_b.clone();
    acc4 -= poly_a.clone();
    assert_eq!(acc4, Polynomial::new(vec![2, 2, 5, 6]));

    // Leading term cancellation trims vector length
    let p1 = Polynomial::new(vec![1i64, 2, 3]);
    let p2 = Polynomial::new(vec![4i64, 5, -3]);
    let p_sum = p1 + p2;
    assert_eq!(p_sum.degree(), Some(1));
    assert_eq!(p_sum.coeffs, vec![5, 7]);

    // Total cancellation reduces to zero polynomial
    let mut p_cancel = Polynomial::new(vec![1i64, 2, 3]);
    p_cancel -= Polynomial::new(vec![1i64, 2, 3]);
    assert_eq!(p_cancel.degree(), None);
    assert_eq!(p_cancel, Polynomial::zero());

    // Scaling edge cases
    let p_scale = Polynomial::new(vec![1i64, 2, 3]);
    assert_eq!(p_scale.scale(&0), Polynomial::zero());
    assert_eq!(Polynomial::<i64>::zero().scale(&5), Polynomial::zero());
}

#[test]
#[verified_engine::verified]
fn test_matrix_ring_non_commutative_multiplication() {
    // Non-commuting matrices:
    // A = [[0, 1], [0, 0]]
    // B = [[0, 0], [1, 0]]
    // A * B = [[1, 0], [0, 0]]
    // B * A = [[0, 0], [0, 1]]
    let a = TestMat2::new([[0, 1], [0, 0]]);
    let b = TestMat2::new([[0, 0], [1, 0]]);

    assert_ne!(a * b, b * a);

    let p_a = Polynomial::new(vec![TestMat2::zero(), a]); // A * x
    let p_b = Polynomial::new(vec![TestMat2::zero(), b]); // B * x

    let prod_ab = &p_a * &p_b;
    let prod_ba = &p_b * &p_a;

    assert_eq!(prod_ab.coeffs.len(), 3);
    assert_eq!(prod_ba.coeffs.len(), 3);
    assert_eq!(prod_ab.coeffs[2], a * b);
    assert_eq!(prod_ba.coeffs[2], b * a);
    assert_ne!(prod_ab, prod_ba);
}
