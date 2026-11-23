use super::*;
use crate::error::EloError;

#[test]
#[verified_engine::verified]
fn test_elo_expected_score_and_win_margin_benchmarks() {
    let r1500 = EloRating::new(1500.0).unwrap();
    let exp = expected_score(r1500, r1500, 65.0, 400.0).unwrap();
    assert!((exp - 0.592466).abs() < 1e-6);

    let log_cfg = EloConfig::new(400.0, 20.0, 65.0, MarginScaling::Logarithmic).unwrap();
    let (h_1_0, a_1_0) = update_elo_from_goals(r1500, r1500, 1, 0, Venue::Home, &log_cfg).unwrap();
    assert!(((h_1_0.value() - 1500.0) - 5.6496).abs() < 1e-4);
    assert!(((1500.0 - a_1_0.value()) - 5.6496).abs() < 1e-4);

    let (h_4_0, a_4_0) = update_elo_from_goals(r1500, r1500, 4, 0, Venue::Home, &log_cfg).unwrap();
    assert!(((h_4_0.value() - 1500.0) - 13.1180).abs() < 1e-4);
    assert!(((1500.0 - a_4_0.value()) - 13.1180).abs() < 1e-4);

    // Autocorrelation-adjusted multiplier coincides with log multiplier at delta_r_adj = 0
    // and damps rating inflation as |delta_r_adj| grows.
    assert!((autocorrelated_margin_multiplier(1, 0.0) - 2.0_f64.ln()).abs() < 1e-12);
    assert!((autocorrelated_margin_multiplier(4, 0.0) - 5.0_f64.ln()).abs() < 1e-12);
    let auto_cfg = EloConfig::new(400.0, 20.0, 65.0, MarginScaling::Autocorrelated).unwrap();
    let (ha_1_0, _) = update_elo_from_goals(r1500, r1500, 1, 0, Venue::Home, &auto_cfg).unwrap();
    let expected_damped = (h_1_0.value() - 1500.0) * (2.2 / (65.0 * 0.001 + 2.2));
    assert!(((ha_1_0.value() - 1500.0) - expected_damped).abs() < 1e-9);
    assert!(ha_1_0.value() < h_1_0.value());
}

#[test]
#[verified_engine::verified]
fn test_elo_draws_neutral_venue_and_seasonal_regression() {
    let auto_cfg = EloConfig::new(400.0, 20.0, 65.0, MarginScaling::Autocorrelated).unwrap();
    let fav = EloRating::new(1700.0).unwrap();
    let dog = EloRating::new(1400.0).unwrap();

    // Draw between unequal clubs still transfers rating toward the underdog
    let (fav_next, dog_next) =
        update_elo_from_goals(fav, dog, 1, 1, Venue::Home, &auto_cfg).unwrap();
    assert!(fav_next.value() < fav.value());
    assert!(dog_next.value() > dog.value());
    assert!(((fav_next.value() + dog_next.value()) - 3100.0).abs() < 1e-12);

    // Neutral venue produces 0.5 expected score for equal ratings
    let neutral_e = expected_score_with_config(dog, dog, Venue::Neutral, &auto_cfg).unwrap();
    assert!((neutral_e - 0.5).abs() < 1e-12);

    // Seasonal mean regression r_next = (1 - theta) * r + theta * mean
    let regressed =
        regress_to_mean(EloRating::new(1800.0).unwrap(), EloRating::default(), 0.25).unwrap();
    assert!((regressed.value() - 1725.0).abs() < 1e-12);
    assert!(regress_to_mean(regressed, EloRating::default(), -0.1).is_err());
    assert!(regress_to_mean(regressed, EloRating::default(), 1.1).is_err());
}

#[test]
#[verified_engine::verified]
fn test_elo_frozen_batch_order_invariance_and_sequential_drift() {
    let roster = [
        EloRating::new(1620.0).unwrap(),
        EloRating::new(1540.0).unwrap(),
        EloRating::new(1480.0).unwrap(),
        EloRating::new(1360.0).unwrap(),
    ];
    let cfg = EloConfig::new(400.0, 24.0, 60.0, MarginScaling::Autocorrelated).unwrap();
    let forward = [
        EloBatchMatch::from_goals(0, 1, 2, 0, Venue::Home).unwrap(),
        EloBatchMatch::from_goals(1, 2, 1, 1, Venue::Home).unwrap(),
        EloBatchMatch::from_goals(2, 0, 3, 1, Venue::Home).unwrap(),
    ];
    let permuted = [forward[2], forward[0], forward[1]];

    let batch_fwd = evaluate_elo_batch(&roster, &forward, &cfg).unwrap();
    let batch_perm = evaluate_elo_batch(&roster, &permuted, &cfg).unwrap();
    assert_eq!(batch_fwd, batch_perm);
    assert_eq!(batch_fwd[3], roster[3]);
    let sum_before: f64 = roster.iter().map(EloRating::value).sum();
    let sum_after: f64 = batch_fwd.iter().map(EloRating::value).sum();
    assert!((sum_before - sum_after).abs() < 1e-10);

    let seq_fwd = evaluate_elo_sequential(&roster, &forward, &cfg).unwrap();
    let seq_perm = evaluate_elo_sequential(&roster, &permuted, &cfg).unwrap();
    assert_ne!(seq_fwd, seq_perm);
    assert_ne!(batch_fwd, seq_fwd);
}

#[test]
fn test_verified_allocator_zero_allocation_elo_batch_and_regression() {
    let roster = [
        EloRating::new(1580.0).unwrap(),
        EloRating::new(1510.0).unwrap(),
        EloRating::new(1410.0).unwrap(),
    ];
    let matches = [
        EloBatchMatch::from_goals(0, 1, 2, 1, Venue::Home).unwrap(),
        EloBatchMatch::from_goals(1, 2, 0, 0, Venue::Neutral).unwrap(),
    ];
    let cfg = EloConfig::new(400.0, 20.0, 65.0, MarginScaling::Autocorrelated).unwrap();
    let mut scratch = [0.0_f64; 2];
    let mut out_batch = [EloRating::default(); 3];
    let mut out_seq = [EloRating::default(); 3];
    let mut out_reg = [EloRating::default(); 3];

    let (res, _) =
        verified_engine::engine::VerifiedEngine::run_verified(|| -> Result<(), EloError> {
            evaluate_elo_batch_into(&roster, &matches, &cfg, &mut scratch, &mut out_batch)?;
            evaluate_elo_sequential_into(&out_batch, &matches, &cfg, &mut out_seq)?;
            regress_roster_into(&out_seq, EloRating::default(), 0.2, &mut out_reg)?;
            let _ = elo_moneyline_odds(out_reg[0], out_reg[2], Venue::Home, &cfg)?;
            let _ = elo_point_spread(out_reg[0], out_reg[2], Venue::Home, &cfg, 25.0)?;
            Ok(())
        });
    assert!(res.is_ok());
    assert!(out_batch[0].value() > roster[0].value());
}

#[test]
#[verified_engine::verified]
fn test_elo_markets_and_serde_validation() {
    let cfg = EloConfig::default();
    let home = EloRating::new(1600.0).unwrap();
    let away = EloRating::new(1465.0).unwrap();

    let p = elo_win_probability(home, away, Venue::Home, &cfg).unwrap();
    let (h_odds, a_odds) = elo_moneyline_odds(home, away, Venue::Home, &cfg).unwrap();
    assert!((h_odds.implied_probability() - p.value()).abs() < 1e-12);
    assert!((h_odds.implied_probability() + a_odds.implied_probability() - 1.0).abs() < 1e-12);

    // Adjusted diff = (1600 + 65) - 1465 = 200 Elo points -> -8.0 point spread at 25 Elo/pt
    let spread = elo_point_spread(home, away, Venue::Home, &cfg, 25.0).unwrap();
    assert!((spread - (-8.0)).abs() < 1e-12);
    assert!((spread_to_elo_diff(spread, 25.0).unwrap() - 200.0).abs() < 1e-12);

    // Serde round-trip and boundary rejections
    let cfg_json = serde_json::to_string(&cfg).unwrap();
    assert_eq!(serde_json::from_str::<EloConfig>(&cfg_json).unwrap(), cfg);
    assert!(
        serde_json::from_str::<EloConfig>(
            r#"{"scale":0.0,"k_factor":20.0,"home_advantage":65.0,"margin_scaling":"None"}"#
        )
        .is_err()
    );
    assert!(serde_json::from_str::<EloConfig>(
        r#"{"scale":400.0,"k_factor":20.0,"home_advantage":65.0,"margin_scaling":"None","extra":1}"#
    )
    .is_err());
    assert!(
        serde_json::from_str::<EloBatchMatch>(
            r#"{"home_team":1,"away_team":1,"score_home":0.5,"goal_diff":0,"venue":"Home"}"#
        )
        .is_err()
    );
}
