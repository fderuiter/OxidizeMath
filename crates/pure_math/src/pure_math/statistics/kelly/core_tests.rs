#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_odds_valid_and_invalid() {
    let odds = Odds::new(2.5).unwrap();
    assert_eq!(odds.value(), 2.5);
    assert_eq!(odds.net_profit_multiplier(), 1.5);

    assert_eq!(Odds::new(1.0), Err(KellyError::InvalidOdds { value: 1.0 }));
    assert_eq!(Odds::new(0.5), Err(KellyError::InvalidOdds { value: 0.5 }));
    assert!(Odds::new(f64::NAN).is_err());
    assert!(Odds::new(f64::INFINITY).is_err());
}

#[test]
#[verified_engine::verified]
fn test_odds_american_boundaries_and_roundtrip() {
    let neg_even = Odds::from_american(-100.0).unwrap();
    let pos_even = Odds::from_american(100.0).unwrap();
    assert!((neg_even.value() - 2.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert!((pos_even.value() - 2.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert!((pos_even.to_american() - 100.0).abs() < math_commons::registry::TOLERANCE_HIGH);

    let underdog = Odds::from_american(200.0).unwrap();
    assert!((underdog.value() - 3.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert!((underdog.to_american() - 200.0).abs() < math_commons::registry::TOLERANCE_HIGH);

    let favorite = Odds::from_american(-150.0).unwrap();
    assert!((favorite.value() - (5.0 / 3.0)).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert!((favorite.to_american() - (-150.0)).abs() < math_commons::registry::TOLERANCE_FAST);

    assert_eq!(
        Odds::from_american(-99.9),
        Err(KellyError::InvalidAmericanOdds { value: -99.9 })
    );
    assert_eq!(
        Odds::from_american(0.0),
        Err(KellyError::InvalidAmericanOdds { value: 0.0 })
    );
    assert_eq!(
        Odds::from_american(99.9),
        Err(KellyError::InvalidAmericanOdds { value: 99.9 })
    );
    assert!(matches!(
        Odds::from_american(f64::NAN),
        Err(KellyError::InvalidAmericanOdds { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_odds_fractional_guards_and_gcd_reduction() {
    let odds = Odds::from_fractional(5.0, 2.0).unwrap();
    assert!((odds.value() - 3.5).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert_eq!(odds.to_fractional(1000), (5, 2));

    let unreduced = Odds::from_fractional(6.0, 4.0).unwrap();
    assert_eq!(unreduced.to_fractional(1000), (3, 2));

    let fav = Odds::from_american(-110.0).unwrap();
    assert_eq!(fav.to_fractional(1000), (10, 11));

    let repeating = Odds::new(2.333_333_333_333).unwrap();
    assert_eq!(repeating.to_fractional(100), (4, 3));

    assert_eq!(odds.to_fractional(0), (2, 1));

    assert_eq!(
        Odds::from_fractional(0.0, 2.0),
        Err(KellyError::InvalidFractionalOdds {
            numerator: 0.0,
            denominator: 2.0,
        })
    );
    assert_eq!(
        Odds::from_fractional(5.0, 0.0),
        Err(KellyError::InvalidFractionalOdds {
            numerator: 5.0,
            denominator: 0.0,
        })
    );
    assert_eq!(
        Odds::from_fractional(0.0, 0.0),
        Err(KellyError::InvalidFractionalOdds {
            numerator: 0.0,
            denominator: 0.0,
        })
    );
    assert_eq!(
        Odds::from_fractional(-1.0, 2.0),
        Err(KellyError::InvalidFractionalOdds {
            numerator: -1.0,
            denominator: 2.0,
        })
    );
}

#[test]
#[verified_engine::verified]
fn test_odds_implied_probability_and_net_odds() {
    let odds = Odds::from_implied_probability(0.25).unwrap();
    assert!((odds.value() - 4.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert!((odds.implied_probability() - 0.25).abs() < math_commons::registry::TOLERANCE_HIGH);

    assert_eq!(
        Odds::from_implied_probability(0.0),
        Err(KellyError::InvalidImpliedProbability { value: 0.0 })
    );
    assert_eq!(
        Odds::from_implied_probability(1.0),
        Err(KellyError::InvalidImpliedProbability { value: 1.0 })
    );
    assert_eq!(
        Odds::from_implied_probability(-0.2),
        Err(KellyError::InvalidImpliedProbability { value: -0.2 })
    );

    let even = Odds::from_net_odds(1.0).unwrap();
    assert!((even.value() - 2.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    assert_eq!(
        Odds::from_net_odds(0.0),
        Err(KellyError::InvalidNetOdds { value: 0.0 })
    );
    assert_eq!(
        Odds::from_net_odds(-0.5),
        Err(KellyError::InvalidNetOdds { value: -0.5 })
    );
}

#[test]
fn test_odds_and_bankroll_fraction_serde_validation() {
    let odds = Odds::new(2.75).unwrap();
    let odds_json = serde_json::to_string(&odds).unwrap();
    assert_eq!(odds_json, "2.75");
    assert_eq!(serde_json::from_str::<Odds>(&odds_json).unwrap(), odds);
    assert!(serde_json::from_str::<Odds>("1.0").is_err());
    assert!(serde_json::from_str::<Odds>("0.5").is_err());
    assert!(serde_json::from_str::<Odds>("-2.0").is_err());

    let frac = BankrollFraction::new(0.25).unwrap();
    let frac_json = serde_json::to_string(&frac).unwrap();
    assert_eq!(frac_json, "0.25");
    assert_eq!(
        serde_json::from_str::<BankrollFraction>(&frac_json).unwrap(),
        frac
    );
    assert!(serde_json::from_str::<BankrollFraction>("0.0").is_ok());
    assert!(serde_json::from_str::<BankrollFraction>("1.0").is_ok());
    assert!(serde_json::from_str::<BankrollFraction>("-0.01").is_err());
    assert!(serde_json::from_str::<BankrollFraction>("1.01").is_err());
}

#[test]
fn test_error_messages_are_format_specific() {
    assert!(
        KellyError::InvalidOdds { value: 0.8 }
            .to_string()
            .contains("> 1.0")
    );
    assert!(
        KellyError::InvalidAmericanOdds { value: 50.0 }
            .to_string()
            .contains("|odds| >= 100.0")
    );
    assert!(
        KellyError::InvalidFractionalOdds {
            numerator: 0.0,
            denominator: 2.0,
        }
        .to_string()
        .contains("0/2")
    );
    assert!(
        KellyError::InvalidImpliedProbability { value: 1.0 }
            .to_string()
            .contains("0.0 and 1.0")
    );
    assert!(
        KellyError::InvalidNetOdds { value: 0.0 }
            .to_string()
            .contains("> 0.0")
    );
}

#[test]
fn test_zero_allocation_verified_conversions_and_guards() {
    let (ok, _) = VerifiedEngine::run_verified(|| {
        let a = Odds::from_american(-100.0).unwrap();
        let b = Odds::from_american(-150.0).unwrap();
        let c = Odds::from_fractional(7.0, 4.0).unwrap();
        let d = Odds::from_implied_probability(0.4).unwrap();
        let (num, den) = b.to_fractional(1000);
        let (c_num, c_den) = c.to_fractional(1000);
        let am = d.to_american();
        let imp = a.implied_probability();
        let err_am = Odds::from_american(0.0).is_err();
        let err_fr = Odds::from_fractional(0.0, 2.0).is_err();
        let err_imp = Odds::from_implied_probability(1.0).is_err();
        let frac = BankrollFraction::new(0.1).unwrap();
        let bet = frac.bet_amount(1000.0).unwrap();

        num == 2
            && den == 3
            && c_num == 7
            && c_den == 4
            && (am - 150.0).abs() < 1e-9
            && (imp - 0.5).abs() < 1e-12
            && err_am
            && err_fr
            && err_imp
            && (bet - 100.0).abs() < 1e-12
    });
    assert!(ok);
}

// [cite:clinical_trials_statistics]
