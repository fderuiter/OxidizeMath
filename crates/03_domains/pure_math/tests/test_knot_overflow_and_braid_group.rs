//! Verification tests for `BraidWord` Artin relations, Dehornoy group equality, Burau hashing,
//! zero-allocation `BurauWorkspace` at $t = -1.0$, `IBig` Bareiss solvers, and checked arithmetic.

use pure_math::pure_math::algebra::traits::{Ring, Semigroup};
use pure_math::pure_math::algebra::{
    BraidError, BraidWord, LaurentError, LaurentPolynomial, MAX_BRAID_WORD_LEN, MAX_LAURENT_SPAN,
    Polynomial,
};
use pure_math::pure_math::knot_theory::{
    BurauWorkspace, KnotError, PlanarDiagram, alexander_polynomial_burau, alexander_polynomial_fox,
    jones_polynomial, kauffman_bracket, normalized_kauffman_polynomial, reduced_burau_matrix,
    try_reduced_burau_matrix,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn hash_of<T: Hash>(val: &T) -> u64 {
    let mut h = DefaultHasher::new();
    val.hash(&mut h);
    h.finish()
}

#[test]
fn braid_word_artin_triples_and_group_equality_hashing() {
    // All 8 mixed-sign conjugate Yang-Baxter triples + 4 homogeneous triples in B_3.
    let mut mixed_count = 0_usize;
    for (i, j) in [(1_i32, 2_i32), (2_i32, 1_i32)] {
        for s0 in [-1_i32, 1_i32] {
            let mut hom = BraidWord::new(3, vec![s0 * i, s0 * j, s0 * i]).unwrap();
            let orig_hom = hom.clone();
            assert!(hom.can_apply_artin_relation(0));
            hom.apply_artin_relation(0).unwrap();
            assert_eq!(hom.generators(), &[s0 * j, s0 * i, s0 * j]);
            assert_eq!(orig_hom, hom);
            assert_eq!(hash_of(&orig_hom), hash_of(&hom));

            for s1 in [-1_i32, 1_i32] {
                let mut mix = BraidWord::new(3, vec![s0 * i, s1 * j, -s0 * i]).unwrap();
                let orig_mix = mix.clone();
                assert!(mix.can_apply_artin_relation(0));
                mix.apply_artin_relation(0).unwrap();
                assert_eq!(mix.generators(), &[-s0 * j, s1 * i, s0 * j]);
                assert_eq!(orig_mix, mix);
                assert_eq!(hash_of(&orig_mix), hash_of(&mix));
                mixed_count += 1;
            }
        }
    }
    assert_eq!(mixed_count, 8);

    let b_comm_1 = BraidWord::new(4, vec![1, 3, -2, 2]).unwrap();
    let b_comm_2 = BraidWord::new(4, vec![3, 1]).unwrap();
    assert_eq!(b_comm_1, b_comm_2);
    assert_eq!(hash_of(&b_comm_1), hash_of(&b_comm_2));
    assert_ne!(
        BraidWord::new(3, vec![1, 1, 1]).unwrap(),
        BraidWord::new(3, vec![-1, -1, -1]).unwrap()
    );
}

#[test]
fn braid_word_try_operate_and_deserialize_bounds() {
    let half = BraidWord::new(3, vec![1; MAX_BRAID_WORD_LEN / 2]).unwrap();
    let full = half.try_operate(&half).expect("128 generators fits");
    assert_eq!(full.len(), MAX_BRAID_WORD_LEN);
    let one_more = BraidWord::new(3, vec![2]).unwrap();
    assert!(matches!(
        full.try_operate(&one_more),
        Err(BraidError::WordTooLong { len: 129 })
    ));
    let cancel_one = BraidWord::new(3, vec![-1]).unwrap();
    assert_eq!(Semigroup::operate(&full, &cancel_one).len(), 127);

    assert!(serde_json::from_str::<BraidWord>(r#"{"strands":0,"generators":[]}"#).is_err());
    assert!(serde_json::from_str::<BraidWord>(r#"{"strands":3,"generators":[0]}"#).is_err());
    assert!(serde_json::from_str::<BraidWord>(r#"{"strands":3,"generators":[3]}"#).is_err());
    let over_json = format!(
        r#"{{"strands":3,"generators":[{}]}}"#,
        vec!["1"; MAX_BRAID_WORD_LEN + 1].join(",")
    );
    assert!(serde_json::from_str::<BraidWord>(&over_json).is_err());
}

#[test]
fn burau_workspace_at_minus_one_is_exact_and_zero_allocation_across_all_even_strands() {
    for n in (2..=16).step_by(2) {
        let mut gens = Vec::with_capacity(n + 1);
        for i in 1..(n as i32) {
            gens.push(if i % 2 == 1 { i } else { -i });
        }
        gens.push(1);
        gens.push(1);
        let braid = BraidWord::new(n, gens).unwrap();
        let exact_poly = alexander_polynomial_burau(&braid).unwrap();
        let expected_det = (exact_poly.eval_with_inverse(-1_i64, -1_i64) as f64).abs();

        let mut ws = BurauWorkspace::new(n).unwrap();
        let (val, _telemetry) =
            VerifiedEngine::run_verified(|| ws.eval_alexander_in_place(&braid, -1.0).unwrap());
        assert!(
            val.is_finite(),
            "non-finite Burau value at t = -1.0 for n = {n}"
        );
        assert!(
            (val.abs() - expected_det).abs() < 1e-7,
            "t = -1.0 determinant mismatch for n = {n}: got {}, expected {expected_det}",
            val.abs()
        );

        assert_eq!(
            try_reduced_burau_matrix(&braid).unwrap(),
            reduced_burau_matrix(&braid)
        );
    }
}

#[test]
fn kauffman_loop_overflow_and_ibig_bareiss_boundaries() {
    let unlink_67 = PlanarDiagram::unlink(67).unwrap();
    assert!(matches!(
        kauffman_bracket(&unlink_67),
        Err(KnotError::CoefficientOverflow)
    ));
    let unlink_128 = PlanarDiagram::unlink(128).unwrap();
    assert!(matches!(
        kauffman_bracket(&unlink_128),
        Err(KnotError::CoefficientOverflow)
    ));
    assert!(matches!(
        normalized_kauffman_polynomial(&unlink_67),
        Err(KnotError::CoefficientOverflow)
    ));
    let unlink_64 = PlanarDiagram::unlink(64).unwrap();
    assert!(matches!(
        jones_polynomial(&unlink_64),
        Err(KnotError::CoefficientOverflow)
    ));

    let gens_62: Vec<i32> = (0..31).flat_map(|_| [1_i32, -2_i32]).collect();
    let braid_62 = BraidWord::new(3, gens_62).unwrap();
    let diag_62 = PlanarDiagram::from_braid(&braid_62);
    let fox_62 = alexander_polynomial_fox(&diag_62).expect("IBig Fox Bareiss on 62 crossings");
    let burau_62 = alexander_polynomial_burau(&braid_62).expect("IBig Burau Bareiss");
    assert_eq!(fox_62, burau_62);
    assert!(fox_62.eval_with_inverse(-1_i64, -1_i64).abs() > 1_000_000_000_000_i64);

    let gens_110: Vec<i32> = (0..55).flat_map(|_| [1_i32, -2_i32]).collect();
    let braid_110 = BraidWord::new(3, gens_110).unwrap();
    let diag_110 = PlanarDiagram::from_braid(&braid_110);
    assert!(matches!(
        alexander_polynomial_fox(&diag_110),
        Err(KnotError::CoefficientOverflow)
    ));
    assert!(matches!(
        alexander_polynomial_burau(&braid_110),
        Err(KnotError::CoefficientOverflow)
    ));
}

#[test]
fn laurent_exponent_and_span_checked_boundaries() {
    assert_eq!(
        LaurentPolynomial::try_new(i32::MAX, vec![1_i64, 2]),
        Err(LaurentError::ExponentOverflow)
    );
    let top_edge = LaurentPolynomial::try_new(i32::MAX - 1, vec![1_i64, 2]).unwrap();
    assert_eq!(top_edge.max_degree(), Some(i32::MAX));
    assert_eq!(top_edge.coeff(i32::MIN), 0);
    assert!(!top_edge.is_reciprocal());

    assert_eq!(top_edge.try_shift(1), Err(LaurentError::ExponentOverflow));
    let bot_edge = LaurentPolynomial::try_new(i32::MIN, vec![3_i64]).unwrap();
    assert_eq!(bot_edge.try_shift(-1), Err(LaurentError::ExponentOverflow));
    assert_eq!(
        bot_edge.try_invert_variable(),
        Err(LaurentError::ExponentOverflow)
    );
    assert_eq!(
        LaurentPolynomial::try_from_terms(&[(i32::MIN, 1_i64), (i32::MAX, 1_i64)]),
        Err(LaurentError::SpanTooLarge {
            span: usize::try_from(4_294_967_296_u64).unwrap_or(usize::MAX),
            max: MAX_LAURENT_SPAN
        })
    );
    assert_eq!(
        LaurentPolynomial::try_from_terms(&[(0, 1_i64), ((MAX_LAURENT_SPAN as i32) + 1, 1_i64)]),
        Err(LaurentError::SpanTooLarge {
            span: MAX_LAURENT_SPAN + 2,
            max: MAX_LAURENT_SPAN
        })
    );
    assert!(
        LaurentPolynomial::try_from_polynomial(Polynomial::new(vec![1_i64; MAX_LAURENT_SPAN + 2]))
            .is_err()
    );
}

#[test]
fn laurent_coefficient_and_deserialize_checked_boundaries() {
    assert_eq!(
        LaurentPolynomial::checked_from_terms(&[(0, i64::MAX), (0, 1_i64)]),
        Err(LaurentError::CoefficientOverflow)
    );
    let max_c = LaurentPolynomial::new(0, vec![i64::MAX]);
    let min_c = LaurentPolynomial::new(0, vec![i64::MIN]);
    let two_c = LaurentPolynomial::new(0, vec![2_i64]);
    assert_eq!(
        max_c.checked_add(&LaurentPolynomial::one()),
        Err(LaurentError::CoefficientOverflow)
    );
    assert_eq!(
        min_c.checked_sub(&LaurentPolynomial::one()),
        Err(LaurentError::CoefficientOverflow)
    );
    assert_eq!(min_c.checked_neg(), Err(LaurentError::CoefficientOverflow));
    assert_eq!(
        max_c.checked_mul(&two_c),
        Err(LaurentError::CoefficientOverflow)
    );
    assert_eq!(
        min_c.try_normalize_alexander(),
        Err(LaurentError::CoefficientOverflow)
    );
    assert!(!LaurentPolynomial::new(0, vec![i64::MIN, 1]).is_reciprocal_up_to_unit());
    assert_eq!(
        LaurentPolynomial::new(0, vec![i64::MAX, 1]).try_eval_at_one(),
        Err(LaurentError::CoefficientOverflow)
    );

    let norm_de: LaurentPolynomial<i64> =
        serde_json::from_str(r#"{"min_exp":-2,"coeffs":[0,0,7,0]}"#).unwrap();
    assert_eq!(norm_de.min_exp(), 0);
    assert_eq!(norm_de.coeffs(), &[7]);
    let bad_de = format!(r#"{{"min_exp":{},"coeffs":[1,2]}}"#, i32::MAX);
    assert!(serde_json::from_str::<LaurentPolynomial<i64>>(&bad_de).is_err());
}

#[cfg(panic = "unwind")]
#[test]
#[should_panic(expected = "MAX_BRAID_WORD_LEN")]
fn semigroup_operate_panics_when_braid_word_exceeds_max_len() {
    let full = BraidWord::new(3, vec![1; MAX_BRAID_WORD_LEN]).unwrap();
    let extra = BraidWord::new(3, vec![2]).unwrap();
    let _ = Semigroup::operate(&full, &extra);
}
