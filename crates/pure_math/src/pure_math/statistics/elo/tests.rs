use super::*;

#[test]
#[verified_engine::verified]
fn test_elo_expected_symmetry_and_home_advantage() {
    let r_a = EloRating::new(1650.0).unwrap();
    let r_b = EloRating::new(1450.0).unwrap();
    let neutral = HomeAdvantage::neutral();

    let p_ab = elo_expected_score(r_a, r_b, neutral);
    let p_ba = elo_expected_score(r_b, r_a, neutral);
    assert!((p_ab.value() + p_ba.value() - 1.0).abs() < 1e-12);
    assert!(p_ab.value() > 0.75);

    let hfa = HomeAdvantage::new(65.0).unwrap();
    let p_home = elo_expected_score(r_a, r_b, hfa);
    assert!(p_home.value() > p_ab.value());
}

#[test]
#[verified_engine::verified]
fn test_elo_update_conserves_total_rating() {
    let r_a = EloRating::default();
    let r_b = EloRating::new(1580.0).unwrap();
    let k = EloKFactor::new(32.0).unwrap();
    let hfa = HomeAdvantage::new(40.0).unwrap();

    let (new_a, new_b) = elo_update_rating(r_a, r_b, UnitInterval::new(1.0).unwrap(), k, hfa);
    assert!(((new_a.value() + new_b.value()) - (r_a.value() + r_b.value())).abs() < 1e-12);
    assert!(new_a.value() > r_a.value());
    assert!(new_b.value() < r_b.value());
}

#[test]
#[verified_engine::verified]
fn test_elo_mov_upset_rewards_underdog_more_than_favorite_blowout() {
    let favorite = EloRating::new(1700.0).unwrap();
    let underdog = EloRating::new(1400.0).unwrap();
    let k = EloKFactor::new(20.0).unwrap();
    let neutral = HomeAdvantage::neutral();

    // Underdog wins by 3 goals (winner_elo_lead = -300 -> higher autocorr factor)
    let m_upset = elo_mov_multiplier(3, -300.0);
    let m_fav = elo_mov_multiplier(3, 300.0);
    assert!(m_upset > m_fav);

    let (fav_after_loss, und_after_win) = elo_update_with_mov(
        favorite,
        underdog,
        UnitInterval::new(0.0).unwrap(),
        3,
        k,
        neutral,
    );
    assert!(und_after_win.value() - underdog.value() > 25.0);
    assert!(((fav_after_loss.value() + und_after_win.value()) - 3100.0).abs() < 1e-12);

    assert!(EloRating::new(f64::NAN).is_err());
    assert!(EloKFactor::new(0.0).is_err());
    assert!(HomeAdvantage::new(f64::INFINITY).is_err());
}
