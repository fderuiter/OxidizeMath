use super::*;

#[test]
#[verified_engine::verified]
fn test_two_way_symmetric_devig_all_methods_equal_half() {
    let line = Odds::from_american(-110.0).unwrap();
    for method in [
        DevigMethod::Multiplicative,
        DevigMethod::Additive,
        DevigMethod::Power,
        DevigMethod::Shin,
    ] {
        let [p1, p2] = devig_two_way(line, line, method).unwrap();
        assert!((p1.value() - 0.5).abs() < 1e-10);
        assert!((p2.value() - 0.5).abs() < 1e-10);
    }
}

#[test]
#[verified_engine::verified]
fn test_shin_and_power_penalize_longshot_more_than_multiplicative() {
    // 1X2 soccer market with a heavy favorite and longshot away side
    let home = Odds::new(1.40).unwrap();
    let draw = Odds::new(4.50).unwrap();
    let away = Odds::new(8.00).unwrap();

    let mult = devig_three_way(home, draw, away, DevigMethod::Multiplicative).unwrap();
    let pow = devig_three_way(home, draw, away, DevigMethod::Power).unwrap();
    let shin = devig_three_way(home, draw, away, DevigMethod::Shin).unwrap();

    for probs in [mult, pow, shin] {
        let sum = probs[0].value() + probs[1].value() + probs[2].value();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // Favorite gets higher true probability under Power and Shin than proportional devig
    assert!(pow[0].value() > mult[0].value());
    assert!(shin[0].value() > mult[0].value());
    // Longshot gets lower true probability under Power and Shin (favorite-longshot bias)
    assert!(pow[2].value() < mult[2].value());
    assert!(shin[2].value() < mult[2].value());
}

#[test]
#[verified_engine::verified]
fn test_shin_two_way_matches_clarke_exact_identity() {
    let home = Odds::new(1.55).unwrap();
    let away = Odds::new(2.55).unwrap();
    let odds = [home, away];
    let mut shin_probs = [UnitInterval::new_clamped(0.5); 2];
    let z = devig_shin_into(&odds, &mut shin_probs).unwrap();
    assert!(z.value() > 0.0 && z.value() < 1.0);

    let add_probs = devig_two_way(home, away, DevigMethod::Additive).unwrap();
    assert!((shin_probs[0].value() - add_probs[0].value()).abs() < 1e-8);
    assert!((shin_probs[1].value() - add_probs[1].value()).abs() < 1e-8);
}

#[test]
#[verified_engine::verified]
fn test_overround_vigorish_fair_odds_and_clv() {
    let home = Odds::new(1.91).unwrap();
    let away = Odds::new(1.91).unwrap();
    let s = book_overround(&[home, away]).unwrap();
    assert!(s > 1.04);
    let vig = book_vigorish(&[home, away]).unwrap();
    assert!(vig.value() > 0.04 && vig.value() < 0.06);

    let fair = fair_odds(UnitInterval::new(0.4).unwrap()).unwrap();
    assert!((fair.value() - 2.5).abs() < 1e-12);
    assert!(fair_odds(UnitInterval::new(0.0).unwrap()).is_err());
    assert!(fair_odds(UnitInterval::new(1.0).unwrap()).is_err());

    let bet = Odds::new(2.10).unwrap();
    let (clv, fair_close) = closing_line_edge_two_way(
        bet,
        Odds::new(1.85).unwrap(),
        Odds::new(2.05).unwrap(),
        DevigMethod::Shin,
    )
    .unwrap();
    assert!(fair_close.value() > 0.5);
    assert!(clv > 0.05);
}
