#![allow(missing_docs)]
use dashu::integer::{IBig, UBig};
use dashu::rational::RBig;
use pure_math::math_types::Rational;
use pure_math::pure_math::algorithmic_information::combinatorics::*;
use pure_math::pure_math::algorithmic_information::fractran::{
    FractranError, FractranPreset, FractranProgram, FractranState, MAX_SMOOTH_PRIME,
};
use std::collections::{HashMap, HashSet};

#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

static ALLOC_LOCK_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

    let alpha = 0.9;
    let result = combinatorial_lemma(&x_set, &v_set, &n_v, &sim, alpha);
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
    // Exact RBig fraction 2^80 / 3^70 exceeds u64/i64 numerators without lossy float conversion.
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

    // Reject zero, negative, and unsmooth prime fractions.
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

    // Checked u32 overflow leaves registers uncorrupted and allocates zero heap memory.
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
