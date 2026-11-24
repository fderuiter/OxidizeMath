use super::*;
use crate::error::MatchModelError;
use crate::pure_math::statistics::zip_regression::PoissonRate;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

#[test]
#[verified_engine::verified]
fn test_independent_poisson_benchmarks_and_tail_mass() {
    let params = IndependentPoissonParams::new(1.35, 1.15).unwrap();
    let grid = params.default_grid().unwrap();
    assert_eq!(grid.max_goals(), 15);
    assert!(grid.tail_mass() < 1e-9);
    assert!((grid.expected_home_goals() - 1.35).abs() < 1e-6);
    assert!((grid.expected_away_goals() - 1.15).abs() < 1e-6);

    let m1x2 = one_x_two(&grid);
    assert!((m1x2.home_win - 0.413382).abs() < 1e-6);
    assert!((m1x2.draw - 0.268397).abs() < 1e-6);
    assert!((m1x2.away_win - 0.318221).abs() < 1e-6);

    let ou25 = over_under(&grid, 2.5).unwrap();
    assert!((ou25.under - 0.543813).abs() < 1e-6);
    assert!((ou25.over - (1.0 - 0.543813)).abs() < 1e-6);

    let ah_minus_1 = asian_handicap(&grid, -1.0).unwrap();
    assert!((ah_minus_1.win - 0.191174).abs() < 1e-6);
    assert!((ah_minus_1.push - 0.222207).abs() < 1e-6);
    assert!((ah_minus_1.loss - 0.586618).abs() < 1e-6);

    // High-scoring truncated grid reports positive tail_mass and normalizes cleanly
    let high = IndependentPoissonParams::new(3.2, 2.8).unwrap();
    let trunc = independent_poisson_grid(&high, 4, false).unwrap();
    assert!(trunc.tail_mass() > 0.05);
    let norm = trunc.normalized().unwrap();
    assert!(norm.is_normalized());
    assert!((norm.active_mass() - 1.0).abs() < 1e-12);
    assert_eq!(norm.tail_mass(), trunc.tail_mass());
}

#[test]
#[verified_engine::verified]
fn test_dixon_coles_benchmarks_and_rho_bounds() {
    let dc = DixonColesParams::new(1.35, 1.15, -0.13).unwrap();
    let grid = dc.default_grid().unwrap();
    assert!((grid.prob(0, 0) - 0.098652).abs() < 1e-6);
    assert!((grid.prob(1, 1) - 0.144004).abs() < 1e-6);
    assert!((one_x_two(&grid).draw - 0.301531).abs() < 1e-6);
    assert!((grid.active_mass() - 1.0).abs() < 1e-9);

    let (lo, hi) = dixon_coles_rho_bounds(1.35, 1.15).unwrap();
    assert!((lo - (-1.0 / (1.35 * 1.15))).abs() < 1e-12);
    assert!((hi - (1.0 / 1.35)).abs() < 1e-12);

    // Reject any rho outside [max(-1/(lambda*mu), -1), min(1/lambda, 1/mu)]
    assert!(DixonColesParams::new(1.35, 1.15, lo - 1e-4).is_err());
    assert!(DixonColesParams::new(1.35, 1.15, hi + 1e-4).is_err());
    assert!(DixonColesParams::new(0.6, 0.5, -1.05).is_err());
    assert!(DixonColesParams::new(0.0, 1.15, 0.0).is_err());
}

#[test]
#[verified_engine::verified]
fn test_bivariate_poisson_benchmarks_and_max_goals_stability() {
    let bp = BivariatePoissonParams::from_marginals(1.35, 1.15, 0.15).unwrap();
    assert!((bp.lambda_1() - 1.20).abs() < 1e-12);
    assert!((bp.lambda_2() - 1.00).abs() < 1e-12);
    assert!((bp.lambda_3() - 0.15).abs() < 1e-12);

    let grid = bp.default_grid().unwrap();
    assert!((one_x_two(&grid).draw - 0.289393).abs() < 1e-6);
    assert!((over_under(&grid, 2.5).unwrap().under - 0.550280).abs() < 1e-6);

    let additive = BivariatePoissonParams::from_components(1.20, 1.00, 0.15).unwrap();
    assert!(
        (one_x_two(&additive.default_grid().unwrap()).draw - one_x_two(&grid).draw).abs() < 1e-15
    );
    assert!(BivariatePoissonParams::from_marginals(1.35, 1.15, 1.20).is_err());

    // Stress-test MAX_GOALS_LIMIT = 31 for zero NaN / overflow across extreme and degenerate rates
    for case in [
        BivariatePoissonParams::from_components(1.20, 1.00, 0.15).unwrap(),
        BivariatePoissonParams::from_components(8.5, 7.2, 2.5).unwrap(),
        BivariatePoissonParams::from_components(0.0, 1.4, 0.3).unwrap(),
        BivariatePoissonParams::from_components(1.4, 0.0, 0.3).unwrap(),
        BivariatePoissonParams::from_components(1.4, 1.1, 0.0).unwrap(),
    ] {
        let g31 = bivariate_poisson_grid(&case, MAX_GOALS_LIMIT, false).unwrap();
        assert_eq!(g31.max_goals(), 31);
        for h in 0..=MAX_GOALS_LIMIT {
            for a in 0..=MAX_GOALS_LIMIT {
                let p = g31.prob(h, a);
                assert!(p.is_finite() && (0.0..=1.0).contains(&p));
            }
        }
    }
    assert!(bivariate_poisson_grid(&bp, MAX_GOALS_LIMIT + 1, false).is_err());
}

#[test]
#[verified_engine::verified]
fn test_quarter_lines_dnb_btts_and_serde() {
    let grid = DixonColesParams::new(1.45, 1.10, -0.10)
        .unwrap()
        .default_grid()
        .unwrap();

    for (q_line, lo_line, hi_line) in [(-0.25, -0.5, 0.0), (-0.75, -1.0, -0.5), (0.25, 0.0, 0.5)] {
        let q = asian_handicap(&grid, q_line).unwrap();
        let lo = asian_handicap(&grid, lo_line).unwrap();
        let hi = asian_handicap(&grid, hi_line).unwrap();
        assert!((q.win - 0.5 * (lo.win + hi.win)).abs() < 1e-12);
        assert!((q.push - 0.5 * (lo.push + hi.push)).abs() < 1e-12);
        assert!((q.loss - 0.5 * (lo.loss + hi.loss)).abs() < 1e-12);
    }

    for (q_line, lo_line, hi_line) in [(2.25, 2.0, 2.5), (2.75, 2.5, 3.0)] {
        let q = over_under(&grid, q_line).unwrap();
        let lo = over_under(&grid, lo_line).unwrap();
        let hi = over_under(&grid, hi_line).unwrap();
        assert!((q.over - 0.5 * (lo.over + hi.over)).abs() < 1e-12);
        assert!((q.push - 0.5 * (lo.push + hi.push)).abs() < 1e-12);
        assert!((q.under - 0.5 * (lo.under + hi.under)).abs() < 1e-12);
    }

    let dnb = draw_no_bet(&grid).unwrap();
    let ah0 = asian_handicap(&grid, 0.0).unwrap();
    assert!((dnb.home_win - ah0.win).abs() < 1e-12);
    assert!((dnb.push - ah0.push).abs() < 1e-12);
    assert!((dnb.conditional_home.value() + dnb.conditional_away.value() - 1.0).abs() < 1e-12);

    let btts_q = btts(&grid);
    assert!((btts_q.yes + btts_q.no - grid.active_mass()).abs() < 1e-12);
    assert!(btts_q.fair_odds().is_ok());
    assert!(asian_handicap(&grid, -0.15).is_err());
    assert!(over_under(&grid, -0.5).is_err());

    // Serde round-trip and validation
    let encoded = serde_json::to_string(&grid).unwrap();
    let decoded: ScoreGrid = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.max_goals(), grid.max_goals());
    assert_eq!(decoded.is_normalized(), grid.is_normalized());
    assert!((decoded.tail_mass() - grid.tail_mass()).abs() < 1e-15);
    for h in 0..=grid.max_goals() {
        for a in 0..=grid.max_goals() {
            assert!((decoded.prob(h, a) - grid.prob(h, a)).abs() < 1e-15);
        }
    }
    assert!(
        serde_json::from_str::<DixonColesParams>(r#"{"lambda":1.35,"mu":1.15,"rho":-0.95}"#)
            .is_err()
    );
    assert!(
        serde_json::from_str::<DixonColesParams>(
            r#"{"lambda":1.35,"mu":1.15,"rho":-0.13,"extra":true}"#
        )
        .is_err()
    );
}

#[test]
fn test_verified_allocator_zero_allocation_score_grid_and_markets() {
    let indep = IndependentPoissonParams::new(1.35, 1.15).unwrap();
    let biv = BivariatePoissonParams::from_marginals(1.35, 1.15, 0.15).unwrap();
    let dc = DixonColesParams::new(1.35, 1.15, -0.13).unwrap();

    let (res, _) = verified_engine::engine::VerifiedEngine::run_verified(
        || -> Result<f64, MatchModelError> {
            let g1 = independent_poisson_grid(&indep, DEFAULT_MAX_GOALS, false)?;
            let g2 = bivariate_poisson_grid(&biv, DEFAULT_MAX_GOALS, true)?;
            let g3 = dixon_coles_grid(&dc, DEFAULT_MAX_GOALS, false)?;
            let m1 = one_x_two(&g1);
            let dnb = draw_no_bet(&g2)?;
            let ah = asian_handicap(&g3, -0.75)?;
            let ou = over_under(&g3, 2.25)?;
            let bt = btts(&g3);
            let _ = ah.fair_odds()?;
            let _ = ou.fair_odds()?;
            let _ = dnb.fair_odds()?;
            Ok(m1.home_win + bt.yes)
        },
    );
    assert!(res.unwrap() > 0.8);
}

#[verified_engine::verified]
fn sample_from_grid(grid: &ScoreGrid, rng: &mut OxidizeRng) -> (u32, u32) {
    let u = rng.r#gen::<f64>() * grid.active_mass();
    let mut acc = 0.0;
    for h in 0..=grid.max_goals() {
        for a in 0..=grid.max_goals() {
            acc += grid.prob_unchecked(h, a);
            if u <= acc {
                return (h as u32, a as u32);
            }
        }
    }
    (0, 0)
}

#[test]
#[verified_engine::verified]
fn test_time_decayed_dixon_coles_mle_recovers_synthetic_parameters() {
    let true_attack = [0.35_f64, 0.15, -0.15, -0.35];
    let true_defense = [0.05_f64, 0.15, 0.25, 0.35];
    let true_ha = 0.24_f64;
    let true_rho = -0.15_f64;

    let mut rng = OxidizeRng::new(20261122);
    let mut matches = Vec::with_capacity(480);
    for season in 0..40 {
        let age = (39 - season) as f64 * 7.0;
        for home in 0..4 {
            for away in 0..4 {
                if home == away {
                    continue;
                }
                let lam = (true_attack[home] + true_defense[away] + true_ha).exp();
                let mu = (true_attack[away] + true_defense[home]).exp();
                let grid = DixonColesParams::new(lam, mu, true_rho)
                    .unwrap()
                    .default_grid()
                    .unwrap();
                let (hg, ag) = sample_from_grid(&grid, &mut rng);
                matches.push(HistoricalMatch::new(home, away, hg, ag, age).unwrap());
            }
        }
    }

    let cfg = DixonColesMleConfig::new(0.001, 0.05, 700, 1e-6, 5.0).unwrap();
    let fitted = fit_dixon_coles_mle(&matches, 4, &cfg).unwrap();

    let attack_sum: f64 = fitted.attack.iter().sum();
    assert!(attack_sum.abs() < 1e-10);
    assert!(fitted.log_likelihood.is_finite());

    assert!(fitted.attack[0] > fitted.attack[3]);
    assert!(fitted.defense[0] < fitted.defense[3]);
    for i in 0..4 {
        assert!((fitted.attack[i] - true_attack[i]).abs() < 0.15);
        assert!((fitted.defense[i] - true_defense[i]).abs() < 0.15);
    }
    assert!((fitted.home_advantage - true_ha).abs() < 0.12);
    assert!(fitted.rho < 0.0 && (fitted.rho - true_rho).abs() < 0.15);

    let pred = fitted.predict_grid(0, 3, DEFAULT_MAX_GOALS, true).unwrap();
    assert!(one_x_two(&pred).home_win > 0.60);
}

#[test]
#[verified_engine::verified]
fn test_flat_slice_score_matrices_and_unit_interval_aggregators() {
    let lh = PoissonRate::new(1.45).unwrap();
    let la = PoissonRate::new(1.25).unwrap();
    let mut indep = [0.0_f64; 100];
    let mut dc = [0.0_f64; 100];
    independent_poisson_matrix_into(lh, la, 9, &mut indep).unwrap();
    dixon_coles_matrix_into(lh, la, DixonColesRho::new(-0.15).unwrap(), 9, &mut dc).unwrap();

    let out_indep = match_1x2_probabilities(&indep, 9).unwrap();
    let out_dc = match_1x2_probabilities(&dc, 9).unwrap();
    assert!(out_dc.draw.value() > out_indep.draw.value());
    let sum_dc = out_dc.home_win.value() + out_dc.draw.value() + out_dc.away_win.value();
    assert!((sum_dc - 1.0).abs() < 1e-12);

    let l1 = PoissonRate::new(1.6).unwrap();
    let l2 = PoissonRate::new(1.1).unwrap();
    let mut indep8 = [0.0_f64; 81];
    let mut biv_zero = [0.0_f64; 81];
    let mut biv_pos = [0.0_f64; 81];
    independent_poisson_matrix_into(l1, l2, 8, &mut indep8).unwrap();
    bivariate_poisson_matrix_into(l1, l2, SharedGoalRate::new(0.0).unwrap(), 8, &mut biv_zero)
        .unwrap();
    bivariate_poisson_matrix_into(l1, l2, SharedGoalRate::new(0.25).unwrap(), 8, &mut biv_pos)
        .unwrap();
    for (a, b) in indep8.iter().zip(biv_zero.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    let (over_zero, under_zero) = over_under_probabilities(&biv_zero, 8, 2.5).unwrap();
    let (over_pos, _) = over_under_probabilities(&biv_pos, 8, 2.5).unwrap();
    assert!((over_zero.value() + under_zero.value() - 1.0).abs() < 1e-12);
    assert!(over_pos.value() > over_zero.value());

    let outcomes8 = match_1x2_probabilities(&indep8, 8).unwrap();
    let ah_minus_half = asian_handicap_home_cover(&indep8, 8, -0.5).unwrap();
    assert!((ah_minus_half.value() - outcomes8.home_win.value()).abs() < 1e-10);
    let ah_dnb = asian_handicap_home_cover(&indep8, 8, 0.0).unwrap();
    assert!(ah_dnb.value() > ah_minus_half.value());
    let btts_p = both_teams_to_score_probability(&indep8, 8).unwrap();
    assert!(btts_p.value() > 0.4 && btts_p.value() < 0.7);
    assert!(DixonColesRho::new(1.5).is_err());
    assert!(SharedGoalRate::new(-0.1).is_err());
    assert!(match_1x2_probabilities(&indep8[..10], 8).is_err());
}

#[test]
#[verified_engine::verified]
fn test_flat_slice_score_matrix_rejects_overflowing_max_goals() {
    let lh = PoissonRate::new(1.45).unwrap();
    let la = PoissonRate::new(1.15).unwrap();
    let mut buf = [0.0_f64; 4];
    for bad_max in [usize::MAX, usize::MAX / 2] {
        assert!(independent_poisson_matrix_into(lh, la, bad_max, &mut buf).is_err());
        assert!(
            dixon_coles_matrix_into(lh, la, DixonColesRho::default(), bad_max, &mut buf).is_err()
        );
        assert!(
            bivariate_poisson_matrix_into(lh, la, SharedGoalRate::default(), bad_max, &mut buf)
                .is_err()
        );
        assert!(match_1x2_probabilities(&buf, bad_max).is_err());
        assert!(over_under_probabilities(&buf, bad_max, 2.5).is_err());
        assert!(asian_handicap_home_cover(&buf, bad_max, -0.5).is_err());
        assert!(both_teams_to_score_probability(&buf, bad_max).is_err());
    }
}
