use super::*;
use crate::pure_math::statistics::zip_regression::PoissonRate;

#[test]
#[verified_engine::verified]
fn test_dixon_coles_negative_rho_increases_draws_over_independent_poisson() {
    let lh = PoissonRate::new(1.45).unwrap();
    let la = PoissonRate::new(1.25).unwrap();
    let mut indep = [0.0_f64; 100]; // max_goals = 9
    let mut dc = [0.0_f64; 100];

    independent_poisson_matrix_into(lh, la, 9, &mut indep).unwrap();
    dixon_coles_matrix_into(lh, la, DixonColesRho::new(-0.15).unwrap(), 9, &mut dc).unwrap();

    let out_indep = match_1x2_probabilities(&indep, 9).unwrap();
    let out_dc = match_1x2_probabilities(&dc, 9).unwrap();

    // Negative rho boosts (0,0) and (1,1) draw cells per Dixon & Coles (1997)
    assert!(out_dc.draw.value() > out_indep.draw.value());
    let sum_dc = out_dc.home_win.value() + out_dc.draw.value() + out_dc.away_win.value();
    assert!((sum_dc - 1.0).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_bivariate_poisson_reduces_to_independent_when_lambda3_zero() {
    let l1 = PoissonRate::new(1.6).unwrap();
    let l2 = PoissonRate::new(1.1).unwrap();
    let mut indep = [0.0_f64; 81];
    let mut biv_zero = [0.0_f64; 81];
    let mut biv_pos = [0.0_f64; 81];

    independent_poisson_matrix_into(l1, l2, 8, &mut indep).unwrap();
    bivariate_poisson_matrix_into(l1, l2, SharedGoalRate::new(0.0).unwrap(), 8, &mut biv_zero)
        .unwrap();
    bivariate_poisson_matrix_into(l1, l2, SharedGoalRate::new(0.25).unwrap(), 8, &mut biv_pos)
        .unwrap();

    for (a, b) in indep.iter().zip(biv_zero.iter()) {
        assert!((a - b).abs() < 1e-12);
    }

    let (over_zero, under_zero) = over_under_probabilities(&biv_zero, 8, 2.5).unwrap();
    let (over_pos, _) = over_under_probabilities(&biv_pos, 8, 2.5).unwrap();
    assert!((over_zero.value() + under_zero.value() - 1.0).abs() < 1e-12);
    // Adding shared goal rate lambda_3 > 0 increases total expected goals E[X+Y] = l1 + l2 + 2*l3
    assert!(over_pos.value() > over_zero.value());
}

#[test]
#[verified_engine::verified]
fn test_btts_and_asian_handicap_consistency() {
    let lh = PoissonRate::new(1.8).unwrap();
    let la = PoissonRate::new(1.0).unwrap();
    let mut grid = [0.0_f64; 81];
    independent_poisson_matrix_into(lh, la, 8, &mut grid).unwrap();

    let outcomes = match_1x2_probabilities(&grid, 8).unwrap();
    let ah_minus_half = asian_handicap_home_cover(&grid, 8, -0.5).unwrap();
    // Home -0.5 cover probability equals exact Home Win probability!
    assert!((ah_minus_half.value() - outcomes.home_win.value()).abs() < 1e-10);

    let ah_dnb = asian_handicap_home_cover(&grid, 8, 0.0).unwrap();
    // Draw-No-Bet (0.0) cover probability is strictly higher than -0.5 when P(draw) > 0
    assert!(ah_dnb.value() > ah_minus_half.value());

    let btts = both_teams_to_score_probability(&grid, 8).unwrap();
    assert!(btts.value() > 0.4 && btts.value() < 0.7);

    assert!(DixonColesRho::new(1.5).is_err());
    assert!(SharedGoalRate::new(-0.1).is_err());
    assert!(match_1x2_probabilities(&grid[..10], 8).is_err());
}
