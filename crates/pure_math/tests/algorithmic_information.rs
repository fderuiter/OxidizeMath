//! Integration tests for algorithmic information.
#![cfg_attr(any(), verified(opt_out = "tests"))]

use dashu::integer::{IBig, UBig};
use dashu::rational::RBig;
use pure_math::math_types::{Integer, Rational};
use pure_math::pure_math::algorithmic_information::combinatorics::*;
use pure_math::pure_math::algorithmic_information::fractran::{
    FractranError, FractranPreset, FractranProgram, FractranState, FractranTrajectoryOutcome,
    MAX_REGISTER_EXPONENT, MAX_SMOOTH_PRIME,
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

static ALLOC_LOCK_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
#[verified_engine::verified]
fn test_distance() {
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
    let p1 = Point2D::from_ints(0, 0);
    let p2 = Point2D::from_ints(3, 4);
    assert_eq!(distance(&p1, &p2), Rational::from((5, 1)));
}

#[test]
#[verified_engine::verified]
fn test_project_point_and_dyadic_quantization() {
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
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

    let result = combinatorial_lemma(&x_set, &v_set, &n_v, &sim, 0.9);
    assert!(result.is_some());
}

#[test]
fn test_primegame_generates_primes_with_zero_allocations_under_lock() {
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
    let (program, state, initial) = FractranPreset::PrimeGame.build();
    let vm = program.compile().expect("PRIMEGAME must compile");
    let mut regs = vm.encode_value(initial).expect("initial 2 must encode");

    let mut found_primes = [0_u32; 5];
    let mut found_steps = [0_usize; 5];
    let mut found_count = 0_usize;

    verified_engine::metrics::enable_verification();
    verified_engine::metrics::reset_metrics();
    verified_engine::allocator::lock_allocations();

    for step in 1..=3000 {
        let fired = vm
            .step_registers(&mut regs)
            .expect("PRIMEGAME step must not overflow");
        assert!(fired.is_some(), "PRIMEGAME never halts");
        if let Some(exp) = vm.power_of_two_exponent(&regs) {
            found_primes[found_count] = exp;
            found_steps[found_count] = step;
            found_count += 1;
            if found_count == found_primes.len() {
                break;
            }
        }
    }

    verified_engine::allocator::unlock_allocations();
    let metrics = verified_engine::metrics::get_metrics();
    verified_engine::metrics::disable_verification();

    assert_eq!(found_primes, [2, 3, 5, 7, 11]);
    assert_eq!(found_steps, [19, 69, 281, 710, 2375]);
    assert!(metrics.function_calls >= 2375);
    assert!(metrics.arithmetic_ops > 0);

    let decoded = state.decode(&vm, &regs);
    assert!(decoded.active_state_primes.is_empty());
    assert_eq!(decoded.counters, vec![(2, 11), (3, 0), (5, 0), (7, 0)]);
    assert_eq!(state.transitions(&vm).len(), 14);
}

#[test]
fn test_add_mult_and_fibonacci_presets_and_minsky_decoding() {
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
    for (preset, expected_checks) in [
        (FractranPreset::Add, vec![(2_u64, 5_u32), (3, 0)]),
        (FractranPreset::Mult, vec![(2, 0), (3, 0), (5, 6), (7, 0)]),
        (
            FractranPreset::Fibonacci,
            vec![(2, 5), (3, 8), (5, 0), (7, 0), (11, 0)],
        ),
    ] {
        let (program, state, initial) = preset.build();
        let vm = program.compile().expect("preset must compile");
        let mut regs = vm.encode_value(initial).expect("initial value must encode");
        let mut saw_active_state = false;
        let mut steps = 0_usize;
        while steps < 5000 {
            if vm.step_registers(&mut regs).unwrap().is_none() {
                break;
            }
            if !state.decode(&vm, &regs).active_state_primes.is_empty() {
                saw_active_state = true;
            }
            steps += 1;
        }
        assert!(
            steps < 5000,
            "{} did not halt within budget",
            preset.label()
        );
        if !state.state_primes.is_empty() {
            assert!(
                saw_active_state,
                "{} should enter Minsky states",
                preset.label()
            );
        }
        let final_decoded = state.decode(&vm, &regs);
        assert!(final_decoded.active_state_primes.is_empty());
        assert_eq!(final_decoded.counters, expected_checks);
    }
}

#[test]
fn test_fractran_rbig_large_exponents_validation_and_overflow_safety() {
    let _guard = ALLOC_LOCK_MUTEX.lock().unwrap();
    let huge_num = IBig::from(2_u8).pow(80);
    let huge_den = UBig::from(3_u8).pow(70);
    let huge_prog = FractranProgram::new(vec![Rational(RBig::from_parts(huge_num, huge_den))]);
    let huge_vm = huge_prog
        .compile()
        .expect("smooth RBig powers must compile");
    assert_eq!(huge_vm.primes(), &[2, 3]);
    assert_eq!(
        huge_vm.rule_exponents(0),
        Some((&[80_u32, 0][..], &[0_u32, 70][..]))
    );
    assert_eq!(
        FractranState::from_vm_heuristic(&huge_vm),
        FractranState::new(Vec::new(), vec![2, 3])
    );
    let mut huge_regs = [0_u32, 70_u32];
    assert_eq!(huge_vm.step_registers(&mut huge_regs), Ok(Some(0)));
    assert_eq!(huge_regs, [80, 0]);

    let zero_prog = FractranProgram::new(vec![Rational::from((0, 5))]);
    assert_eq!(
        zero_prog.compile(),
        Err(FractranError::NonPositiveFraction { rule_index: 0 })
    );
    let neg_prog = FractranProgram::new(vec![Rational::from((3, 2)), Rational::from((-5, 7))]);
    assert_eq!(
        neg_prog.compile(),
        Err(FractranError::NonPositiveFraction { rule_index: 1 })
    );
    let unsmooth = FractranProgram::parse_fractions("10007/2").unwrap();
    assert_eq!(
        unsmooth.compile(),
        Err(FractranError::UnsmoothFactor {
            max_prime: MAX_SMOOTH_PRIME
        })
    );

    let inc_vm = FractranProgram::parse_fractions("4/3")
        .unwrap()
        .compile()
        .unwrap();
    let mut near_max = [u32::MAX - 1, 1];
    verified_engine::metrics::enable_verification();
    verified_engine::allocator::lock_allocations();
    let overflow_res = inc_vm.step_registers(&mut near_max);
    verified_engine::allocator::unlock_allocations();
    verified_engine::metrics::disable_verification();
    assert_eq!(
        overflow_res,
        Err(FractranError::RegisterOverflow {
            rule_index: 0,
            prime: 2
        })
    );
    assert_eq!(near_max, [u32::MAX - 1, 1]);
}
