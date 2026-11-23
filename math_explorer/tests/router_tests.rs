//! Integration tests for router tests.
#[cfg(feature = "pure_math")]
#[test]
fn test_primality_routing() {
    use math_explorer::pure_math::number_theory::primes::is_prime;
    assert!(is_prime(2));
    assert!(is_prime(3));
    assert!(is_prime(5));
    assert!(is_prime(7));
    assert!(!is_prime(4));
    assert!(!is_prime(9));
}

#[cfg(feature = "physics")]
#[test]
fn test_physics_coefficient_routing() {
    use math_explorer::physics::quantum::clebsch_gordan;
    let cg = clebsch_gordan(1.5, -0.5, 1.0, 1.0, 2.5, 0.5);
    let expected = (0.3f64).sqrt();
    assert!((cg - expected).abs() < 1e-9);
}
