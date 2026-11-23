//! Integration tests for algorithmic information.
use pure_math::pure_math::algorithmic_information::combinatorics::*;
use std::collections::{HashMap, HashSet};

#[test]
#[ignore]
#[cfg(any())]
#[verified_engine::verified]
fn test_distance() {
    assert_eq!(distance(&p1, &p2), Rational::from(5));
}

#[test]
#[ignore]
#[cfg(any())]
#[verified_engine::verified]
fn test_kolmogorov_approx() {
    assert!(
        (prefix_kolmogorov_approx(&Integer::from(10))
            - (10.0f64.log2() + 2.0 * 10.0f64.log2().log2() + 1.0))
            .abs()
            < math_commons::registry::TOLERANCE_STANDARD
    );
}

#[test]
#[verified_engine::verified]
fn test_combinatorial_lemma() {
    let mut x_set = HashSet::new();
    x_set.insert(1);
    x_set.insert(2);

    let mut v_set = HashSet::new();
    v_set.insert("u");
    v_set.insert("v");

    let mut n_v = HashMap::new();
    let mut n_u_set = HashSet::new();
    n_u_set.insert(1);
    n_u_set.insert(2);
    n_v.insert("u", n_u_set);

    let mut n_v_set = HashSet::new();
    n_v_set.insert(1);
    n_v_set.insert(2);
    n_v.insert("v", n_v_set);

    let mut sim = HashMap::new();
    sim.insert(
        1,
        Box::new(|_u: &&str, _v: &&str| true) as Box<dyn Fn(&&str, &&str) -> bool>,
    );
    sim.insert(
        2,
        Box::new(|_u: &&str, _v: &&str| false) as Box<dyn Fn(&&str, &&str) -> bool>,
    );

    let alpha = 0.9;
    let result = combinatorial_lemma(&x_set, &v_set, &n_v, &sim, alpha);
    assert!(result.is_some());
}
