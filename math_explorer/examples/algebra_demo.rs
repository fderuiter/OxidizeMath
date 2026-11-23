//! Example target for algebra demo.
use math_explorer::pure_math::algebra::{Fp, Polynomial};

fn main() {
    // 1. Arithmetic in Finite Field F_7
    let a = Fp::<7>::new(3);
    let b = Fp::<7>::new(5);
    // (3 * 5) % 7 = 15 % 7 = 1
    assert_eq!(a * b, Fp::<7>::new(1));

    // 2. Polynomials over F_7: P(x) = 3x^2 + 5
    let _p = Polynomial::new(vec![b, Fp::<7>::new(0), a]);
}
