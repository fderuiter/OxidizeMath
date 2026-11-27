use super::*;

#[test]
#[verified_engine::verified]
fn test_rating_valid() {
    let r = Rating::new(1500.0).unwrap();
    assert_eq!(r.value(), 1500.0);
}

#[test]
#[verified_engine::verified]
fn test_rating_invalid() {
    assert!(Rating::new(f64::NAN).is_err());
    assert!(Rating::new(f64::INFINITY).is_err());
}

#[test]
#[verified_engine::verified]
fn test_rating_scale_conversion() {
    let r = Rating::new(1500.0).unwrap();
    let mu = r.to_glicko2_scale();
    assert!((mu - 0.0).abs() < math_commons::registry::TOLERANCE_FAST);

    let r2 = Rating::from_glicko2_scale(mu).unwrap();
    assert!((r2.value() - 1500.0).abs() < 1e-3);
}

#[test]
#[verified_engine::verified]
fn test_rd_valid() {
    let rd = RatingDeviation::new(350.0).unwrap();
    assert_eq!(rd.value(), 350.0);
}

#[test]
#[verified_engine::verified]
fn test_rd_invalid() {
    assert!(RatingDeviation::new(0.0).is_err());
    assert!(RatingDeviation::new(-1.0).is_err());
    assert!(RatingDeviation::new(f64::NAN).is_err());
}

#[test]
#[verified_engine::verified]
fn test_rd_scale_conversion() {
    let rd = RatingDeviation::new(173.7178).unwrap();
    let phi = rd.to_glicko2_scale();
    assert!((phi - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);

    let rd2 = RatingDeviation::from_glicko2_scale(phi).unwrap();
    assert!((rd2.value() - 173.7178).abs() < 1e-3);
}

#[test]
#[verified_engine::verified]
fn test_volatility_valid() {
    let vol = Volatility::new(0.06).unwrap();
    assert_eq!(vol.value(), 0.06);
}

#[test]
#[verified_engine::verified]
fn test_volatility_invalid() {
    assert!(Volatility::new(0.0).is_err());
    assert!(Volatility::new(-0.1).is_err());
    assert!(Volatility::new(f64::NAN).is_err());
}

#[test]
#[verified_engine::verified]
fn test_system_constant_valid() {
    let tau = SystemConstant::new(0.5).unwrap();
    assert_eq!(tau.value(), 0.5);
}

#[test]
#[verified_engine::verified]
fn test_system_constant_invalid() {
    assert!(SystemConstant::new(0.2).is_err()); // Too small
    assert!(SystemConstant::new(1.5).is_err()); // Too large
    assert!(SystemConstant::new(f64::NAN).is_err());
}

#[test]
#[verified_engine::verified]
fn test_glicko_player_default() {
    let player = GlickoPlayer::default();
    assert_eq!(player.rating.value(), 1500.0);
    assert_eq!(player.rating_deviation.value(), 350.0);
    assert_eq!(player.volatility.value(), 0.06);
}

#[test]
#[verified_engine::verified]
fn test_match_result_valid() {
    let opponent = GlickoPlayer::default();
    assert!(MatchResult::new(opponent, 0.0).is_ok()); // Loss
    assert!(MatchResult::new(opponent, 0.5).is_ok()); // Draw
    assert!(MatchResult::new(opponent, 1.0).is_ok()); // Win
}

#[test]
#[verified_engine::verified]
fn test_match_result_invalid() {
    let opponent = GlickoPlayer::default();
    assert!(MatchResult::new(opponent, -0.1).is_err());
    assert!(MatchResult::new(opponent, 1.1).is_err());
    assert!(MatchResult::new(opponent, f64::NAN).is_err());
}

#[test]
#[verified_engine::verified]
fn test_g_function() {
    // When phi = 0, g(phi) = 1
    assert!((g_function(0.0) - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);

    // As phi increases, g(phi) decreases
    let g1 = g_function(1.0);
    let g2 = g_function(2.0);
    assert!(g1 > g2);
}

#[test]
#[verified_engine::verified]
fn test_expected_outcome() {
    // Equal ratings and RD should give 0.5 probability
    let e = expected_outcome(0.0, 0.0, 1.0);
    assert!((e - 0.5).abs() < math_commons::registry::TOLERANCE_FAST);

    // Higher rating should give > 0.5 probability
    let e = expected_outcome(1.0, 0.0, 1.0);
    assert!(e > 0.5);

    // Lower rating should give < 0.5 probability
    let e = expected_outcome(0.0, 1.0, 1.0);
    assert!(e < 0.5);
}

#[test]
#[verified_engine::verified]
fn test_serde_roundtrip_and_validation_all_six_core_types() {
    let r = Rating::new(1625.5).unwrap();
    let rd = RatingDeviation::new(85.0).unwrap();
    let vol = Volatility::new(0.065).unwrap();
    let tau = SystemConstant::new(0.6).unwrap();
    let player = GlickoPlayer::new(r, rd, vol);
    let match_res = MatchResult::new(player, 0.5).unwrap();

    assert_eq!(
        serde_json::from_str::<Rating>(&serde_json::to_string(&r).unwrap()).unwrap(),
        r
    );
    assert_eq!(
        serde_json::from_str::<RatingDeviation>(&serde_json::to_string(&rd).unwrap()).unwrap(),
        rd
    );
    assert_eq!(
        serde_json::from_str::<Volatility>(&serde_json::to_string(&vol).unwrap()).unwrap(),
        vol
    );
    assert_eq!(
        serde_json::from_str::<SystemConstant>(&serde_json::to_string(&tau).unwrap()).unwrap(),
        tau
    );
    assert_eq!(
        serde_json::from_str::<GlickoPlayer>(&serde_json::to_string(&player).unwrap()).unwrap(),
        player
    );
    assert_eq!(
        serde_json::from_str::<MatchResult>(&serde_json::to_string(&match_res).unwrap()).unwrap(),
        match_res
    );

    // Reject invalid RatingDeviation, Volatility, SystemConstant, MatchResult, and unknown fields
    assert!(serde_json::from_str::<RatingDeviation>("0.0").is_err());
    assert!(serde_json::from_str::<RatingDeviation>("-25.0").is_err());
    assert!(serde_json::from_str::<Volatility>("0.0").is_err());
    assert!(serde_json::from_str::<Volatility>("-0.01").is_err());
    assert!(serde_json::from_str::<SystemConstant>("0.25").is_err());
    assert!(serde_json::from_str::<SystemConstant>("1.35").is_err());
    assert!(
        serde_json::from_str::<GlickoPlayer>(
            r#"{"rating":1500.0,"rating_deviation":-10.0,"volatility":0.06}"#
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<GlickoPlayer>(
            r#"{"rating":1500.0,"rating_deviation":200.0,"volatility":0.06,"extra":1}"#
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<MatchResult>(
            r#"{"opponent":{"rating":1500.0,"rating_deviation":200.0,"volatility":0.06},"score":1.5}"#
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<PeriodMatch>(r#"{"player_a":1,"player_b":1,"score_a":0.5}"#)
            .is_err()
    );
}

#[test]
#[verified_engine::verified]
fn test_frozen_period_order_invariance_inactivity_and_sequential_drift() {
    let roster = vec![
        GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(200.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1400.0).unwrap(),
            RatingDeviation::new(150.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1550.0).unwrap(),
            RatingDeviation::new(180.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1650.0).unwrap(),
            RatingDeviation::new(90.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
    ];
    let tau = SystemConstant::default();

    // Player 3 is inactive; players 0, 1, 2 play a round-robin period
    let forward_matches = vec![
        PeriodMatch::new(0, 1, 1.0).unwrap(),
        PeriodMatch::new(1, 2, 0.0).unwrap(),
        PeriodMatch::new(0, 2, 0.5).unwrap(),
    ];
    let permuted_matches = vec![
        PeriodMatch::new(2, 0, 0.5).unwrap(),
        PeriodMatch::new(2, 1, 1.0).unwrap(),
        PeriodMatch::new(1, 0, 0.0).unwrap(),
    ];

    let batch_forward = evaluate_rating_period(&roster, &forward_matches, &tau).unwrap();
    let batch_permuted = evaluate_rating_period(&roster, &permuted_matches, &tau).unwrap();
    assert_eq!(batch_forward, batch_permuted);

    // Inactive player 3 keeps rating and increases RD
    assert_eq!(batch_forward[3].rating, roster[3].rating);
    assert!(batch_forward[3].rating_deviation.value() > roster[3].rating_deviation.value());

    // Sequential update exhibits order-dependent drift relative to frozen batch update
    let seq_forward = evaluate_sequential_period(&roster, &forward_matches, &tau).unwrap();
    let seq_permuted = evaluate_sequential_period(&roster, &permuted_matches, &tau).unwrap();
    let max_drift = batch_forward
        .iter()
        .zip(seq_forward.iter())
        .map(|(b, s)| (b.rating.value() - s.rating.value()).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_drift > 1.0);
    assert_ne!(seq_forward, seq_permuted);
}

#[test]
fn test_verified_allocator_zero_allocation_period_evaluation() {
    let roster = [
        GlickoPlayer::default(),
        GlickoPlayer::new(
            Rating::new(1600.0).unwrap(),
            RatingDeviation::new(120.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1420.0).unwrap(),
            RatingDeviation::new(180.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
    ];
    let matches = [
        PeriodMatch::new(0, 1, 1.0).unwrap(),
        PeriodMatch::new(1, 2, 0.5).unwrap(),
    ];
    let tau = SystemConstant::default();
    let mut scratch = [MatchResult::new(GlickoPlayer::default(), 0.5).unwrap(); 2];
    let mut out = [GlickoPlayer::default(); 3];

    let (res, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        evaluate_rating_period_into(&roster, &matches, &tau, &mut scratch, &mut out)
    });
    assert!(res.is_ok());
    assert!(out[0].rating.value() > 1500.0);
}

#[test]
#[verified_engine::verified]
fn test_pairwise_win_matrix_and_deterministic_pairings() {
    let roster = vec![
        GlickoPlayer::new(
            Rating::new(1700.0).unwrap(),
            RatingDeviation::new(80.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1600.0).unwrap(),
            RatingDeviation::new(90.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1500.0).unwrap(),
            RatingDeviation::new(100.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1450.0).unwrap(),
            RatingDeviation::new(110.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
        GlickoPlayer::new(
            Rating::new(1400.0).unwrap(),
            RatingDeviation::new(120.0).unwrap(),
            Volatility::new(0.06).unwrap(),
        ),
    ];

    let win_mat = pairwise_win_matrix(&roster);
    assert_eq!(win_mat.nrows(), 5);
    for i in 0..5 {
        assert!((win_mat[(i, i)] - 0.5).abs() < 1e-12);
        for j in (i + 1)..5 {
            assert!((win_mat[(i, j)] + win_mat[(j, i)] - 1.0).abs() < 1e-12);
            assert!(win_mat[(i, j)] > 0.5);
        }
    }

    // Round-robin schedule for odd (5) and even (4) players covers each pair once
    let rr_odd = round_robin_schedule(5).unwrap();
    assert_eq!(rr_odd.len(), 5);
    assert_eq!(rr_odd, round_robin_schedule(5).unwrap());
    let total_games: usize = rr_odd.iter().map(Vec::len).sum();
    assert_eq!(total_games, 10);

    let rr_even = round_robin_schedule(4).unwrap();
    assert_eq!(rr_even.len(), 3);
    assert_eq!(rr_even.iter().map(Vec::len).sum::<usize>(), 6);

    // Odd-sized Swiss assigns deterministic non-repeating bye and avoids rematches
    let r1 = swiss_pairings(&roster, &[0.0; 5], &[]).unwrap();
    assert_eq!(r1.bye_player, Some(4));
    assert_eq!(r1.pairings.len(), 2);

    let mut rng_a = oxidize_core::rng::OxidizeRng::new(77);
    let mut rng_b = oxidize_core::rng::OxidizeRng::new(77);
    let m1_a = simulate_period_matches(&roster, &r1.pairings, 0.2, &mut rng_a).unwrap();
    let m1_b = simulate_period_matches(&roster, &r1.pairings, 0.2, &mut rng_b).unwrap();
    assert_eq!(m1_a, m1_b);

    let r2 = swiss_pairings(&roster, &[1.0, 0.0, 1.0, 0.0, 1.0], &m1_a).unwrap();
    assert!(r2.bye_player.is_some());
    assert_ne!(r2.bye_player, r1.bye_player);
}
