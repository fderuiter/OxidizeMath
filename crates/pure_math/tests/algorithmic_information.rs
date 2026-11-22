#![allow(missing_docs)]
#![cfg_attr(any(), verified(opt_out = "tests"))]

use pure_math::math_types::{Integer, Rational};
use pure_math::pure_math::algorithmic_information::combinatorics::*;
use pure_math::pure_math::algorithmic_information::fractran::{
    FractranError, FractranProgram, FractranState, FractranTrajectoryOutcome, MAX_REGISTER_EXPONENT,
};
use pure_math::pure_math::algorithmic_information::geometry::{
    DyadicPoint2D, DyadicRational, Line, Point2D, distance, project_point_onto_line,
};
use pure_math::pure_math::algorithmic_information::kolmogorov::{
    bdm_complexity, integer_to_bits, k_floor_r, kolmogorov_complexity, lz76_complexity,
    prefix_kolmogorov_approx, shannon_entropy_bits,
};
use std::collections::{HashMap, HashSet};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_distance() {
    let p1 = Point2D::from_ints(0, 0);
    let p2 = Point2D::from_ints(3, 4);
    assert_eq!(distance(&p1, &p2), Rational::from((5, 1)));
}

#[test]
#[verified_engine::verified]
fn test_project_point_and_dyadic_quantization() {
    let line = Line::new(Point2D::from_ints(1, 1), Point2D::from_ints(2, 0));
    let projected = project_point_onto_line(&Point2D::from_ints(5, 9), &line);
    assert_eq!(projected, Point2D::from_ints(5, 1));

    let neg = Point2D::new(Rational::from((-1, 4)), Rational::from((3, 8)));
    let dyadic = DyadicPoint2D::from_point_floor(&neg, 2);
    assert_eq!(dyadic.x, DyadicRational::new(-1, 2));
    assert_eq!(dyadic.y, DyadicRational::new(1, 2));
}

#[test]
#[verified_engine::verified]
fn test_kolmogorov_approx() {
    assert!(
        (prefix_kolmogorov_approx(&Integer::from(10))
            - (10.0f64.log2() + 2.0 * 10.0f64.log2().log2() + 1.0))
            .abs()
            < math_commons::registry::TOLERANCE_STANDARD
    );
    assert_eq!(kolmogorov_complexity(&Integer::from(1_000_000)), 30);
}

#[test]
#[verified_engine::verified]
fn test_bdm_lz76_and_k_floor_r_bounds() {
    let periodic = b"010101010101010101010101010101010101010101010101";
    let irregular = b"011010011100101100011010110100101110010101101100";

    let bdm_per = bdm_complexity(periodic);
    let bdm_irr = bdm_complexity(irregular);
    let lz_per = lz76_complexity(periodic);
    let lz_irr = lz76_complexity(irregular);

    assert!(bdm_per < bdm_irr);
    assert!(lz_per < lz_irr);
    assert!(bdm_per < shannon_entropy_bits(periodic));

    let dyadic_pt = Point2D::new(Rational::from((1, 2)), Rational::from((3, 4)));
    let generic_pt = Point2D::new(Rational::from((1, 3)), Rational::from((1, 7)));
    assert!(k_floor_r(&dyadic_pt, 18) < k_floor_r(&generic_pt, 18));
    assert_eq!(integer_to_bits(&Integer::from(13)), vec![1, 1, 0, 1]);
}

#[test]
#[verified_engine::verified]
fn test_primegame_generates_initial_primes() {
    let program = FractranProgram::primegame();
    let mut regs = FractranProgram::encode_u64(2).unwrap();
    let mut primes = Vec::new();

    for _ in 0..3_000 {
        assert!(program.step_exponents(&mut regs).unwrap().is_some());
        if let Some(p) = FractranProgram::power_of_two_exponent(&regs) {
            primes.push(p);
            if primes.len() == 5 {
                break;
            }
        }
    }

    assert_eq!(primes, vec![2, 3, 5, 7, 11]);
}

#[test]
#[verified_engine::verified]
fn test_adder_multiplier_and_fibonacci_programs() {
    let adder = FractranProgram::adder();
    let mut add_regs = FractranProgram::encode_u64(16 * 27).unwrap(); // 2^4 * 3^3
    while adder.step_exponents(&mut add_regs).unwrap().is_some() {}
    assert_eq!(add_regs[1], 7);

    let mult = FractranProgram::multiplier();
    let mut mul_regs = FractranProgram::encode_u64(4 * 27).unwrap(); // 2^2 * 3^3
    while mult.step_exponents(&mut mul_regs).unwrap().is_some() {}
    assert_eq!(mul_regs[2], 6);

    let fib = FractranProgram::fibonacci();
    let mut fib_regs = FractranProgram::encode_u64(2 * 3_125).unwrap(); // 2^1 * 5^5
    while fib.step_exponents(&mut fib_regs).unwrap().is_some() {}
    assert_eq!(fib_regs[0], 8); // F_6 = 8
    assert_eq!(fib_regs[1], 5); // F_5 = 5
}

#[test]
#[verified_engine::verified]
fn test_floyd_cycle_detection_and_zero_allocation_verified_lock() {
    let oscillator = FractranProgram::try_from_pairs(&[(3, 2), (2, 3)]).unwrap();
    let init = FractranProgram::encode_u64(2).unwrap();

    let ((outcome, reg_sum), metrics) = VerifiedEngine::run_verified(|| {
        let mut regs = init;
        for _ in 0..64 {
            let _ = oscillator.step_exponents(&mut regs).unwrap();
        }
        let detected = oscillator.detect_cycle(&init, 128).unwrap();
        (detected, regs[0] + regs[1])
    });

    assert_eq!(reg_sum, 1);
    assert!(metrics.function_calls > 0);
    match outcome {
        FractranTrajectoryOutcome::CycleDetected(info) => {
            assert_eq!(info.start_step, 0);
            assert_eq!(info.period, 2);
        }
        other => panic!("expected CycleDetected, got {other:?}"),
    }

    let mut live = FractranState::from_u64(2).unwrap();
    for _ in 0..4 {
        live.step(&oscillator).unwrap();
    }
    assert_eq!(live.cycle.map(|c| c.period), Some(2));
}

#[test]
#[verified_engine::verified]
fn test_fractran_validation_and_exponent_overflow_errors() {
    assert_eq!(
        FractranProgram::try_from_pairs(&[(5, 0)]),
        Err(FractranError::ZeroDenominator { rule_index: 0 })
    );
    assert_eq!(
        FractranProgram::try_from_pairs(&[(-3, 2)]),
        Err(FractranError::NegativeFraction { rule_index: 0 })
    );
    assert_eq!(
        FractranProgram::try_from_pairs(&[(313, 2)]),
        Err(FractranError::UnsupportedPrimeFactor {
            rule_index: 0,
            factor: 313
        })
    );

    let doubler = FractranProgram::parse("2/1").unwrap();
    let mut regs = [0u32; 64];
    regs[0] = MAX_REGISTER_EXPONENT;
    assert_eq!(
        doubler.step_exponents(&mut regs),
        Err(FractranError::ExponentOverflow {
            prime: 2,
            attempted: u64::from(MAX_REGISTER_EXPONENT) + 1
        })
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
