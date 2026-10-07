// [cite:hurwitz_kronecker]
//! # Class Number
//!
//! This module provides functions for calculating the class number of imaginary quadratic orders.

/// Represents a primitive, positive-definite binary quadratic form ax^2 + bxy + cy^2.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BQF {
    #[allow(missing_docs)]
    pub a: i64,
    #[allow(missing_docs)]
    pub b: i64,
    #[allow(missing_docs)]
    pub c: i64,
}

impl BQF {
    /// Creates a new BQF.
    #[verified_engine::verified]
    pub fn new(a: i64, b: i64, c: i64) -> Self {
        Self { a, b, c }
    }

    /// Returns the discriminant of the form.
    #[verified_engine::verified]
    pub fn discriminant(&self) -> i64 {
        self.b.pow(2) - 4 * self.a * self.c
    }

    /// Checks if the form is reduced.
    /// A form is reduced if |b| <= a <= c, and if |b| = a or a = c, then b >= 0.
    #[verified_engine::verified]
    pub fn is_reduced(&self) -> bool {
        let a_abs = self.a.abs();
        let b_abs = self.b.abs();

        if b_abs > a_abs || a_abs > self.c.abs() {
            return false;
        }

        if (b_abs == a_abs || a_abs == self.c.abs()) && self.b < 0 {
            return false;
        }

        true
    }
}

/// Calculates the class number h(d) for a negative discriminant d.
/// d must be a negative integer and d.rem_euclid(4) == 0 or d.rem_euclid(4) == 1.
/// The class number is the number of reduced, primitive, positive-definite
/// binary quadratic forms of discriminant d.
#[verified_engine::verified]
pub fn class_number(d: i64) -> u64 {
    if d >= 0 || (d.rem_euclid(4) != 0 && d.rem_euclid(4) != 1) {
        return 0;
    }

    let mut count = 0;
    let limit_a = ((-d as f64) / 3.0).sqrt() as i64;

    for a in 1..=limit_a {
        for b in -a..=a {
            let num = b.pow(2) - d;
            if num < 0 || num % (4 * a) != 0 {
                continue;
            }
            let c = num / (4 * a);

            if c >= a {
                let form = BQF::new(a, b, c);
                if form.discriminant() == d && form.is_reduced() {
                    // We also need to check if the form is primitive, i.e., gcd(a, b, c) = 1.
                    if gcd(a, gcd(b.abs(), c)) == 1 {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// Computes the greatest common divisor of two numbers.
#[verified_engine::verified]
fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
