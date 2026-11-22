#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::io::{format_leaderboard_csv, format_matchup_csv, parse_match_csv};
use super::model::{RosterPlayer, ScheduledMatch};
use super::Glicko2TournamentTool;
use egui_plot::PlotPoint;

#[test]
fn test_glickman_canonical_period_and_inactivity_rd_growth() {
    let tool = Glicko2TournamentTool::default();
    assert!(tool.eval_error.is_none());

    let hero = tool
        .standings
        .iter()
        .find(|s| s.name == "Player A")
        .expect("Player A must be in standings");
    assert!(
        (hero.post_rating - 1464.06).abs() < 0.15,
        "Expected canonical Glickman rating ~1464.06, got {}",
        hero.post_rating
    );
    assert!(
        (hero.post_rd - 151.52).abs() < 0.15,
        "Expected canonical Glickman RD ~151.52, got {}",
        hero.post_rd
    );
    assert!((hero.ci_low - (hero.post_rating - 2.0 * hero.post_rd)).abs() < 1e-12);
    assert!((hero.ci_high - (hero.post_rating + 2.0 * hero.post_rd)).abs() < 1e-12);

    let inactive = tool
        .standings
        .iter()
        .find(|s| s.name == "Inactive Reserve")
        .expect("Inactive Reserve must be in standings");
    assert_eq!(inactive.matches_played, 0);
    assert_eq!(inactive.post_rating, inactive.pre_rating);
    assert!(
        inactive.post_rd > inactive.pre_rd,
        "Inactive player's RD must grow: pre={}, post={}",
        inactive.pre_rd,
        inactive.post_rd
    );
}

#[test]
fn test_simultaneous_updates_invariant_to_match_order() {
    let mut forward = Glicko2TournamentTool::default();
    let mut reversed = Glicko2TournamentTool {
        matches: vec![
            ScheduledMatch {
                player_a: 3,
                player_b: 0,
                score_a: 1.0,
            },
            ScheduledMatch {
                player_a: 2,
                player_b: 0,
                score_a: 1.0,
            },
            ScheduledMatch {
                player_a: 1,
                player_b: 0,
                score_a: 0.0,
            },
        ],
        ..Default::default()
    };
    forward.recalculate();
    reversed.recalculate();

    assert_eq!(forward.standings.len(), reversed.standings.len());
    for (a, b) in forward.standings.iter().zip(reversed.standings.iter()) {
        assert_eq!(a.player_idx, b.player_idx);
        assert_eq!(a.post_rating, b.post_rating);
        assert_eq!(a.post_rd, b.post_rd);
        assert_eq!(a.post_volatility, b.post_volatility);
    }

    let slice: &[PlotPoint] = &forward.post_rating_points[..];
    let borrowed: egui_plot::PlotPoints<'_> = slice.into();
    assert!(matches!(borrowed, egui_plot::PlotPoints::Borrowed(_)));
}

#[test]
fn test_csv_import_validation_and_formula_injection_protection() {
    let csv = "\u{feff}player_a,player_b,score\nAlice,Bob,1.0\nBob,Carol,0.5\n";
    let (roster, matches) = parse_match_csv(csv, &[]).expect("valid UTF-8 BOM CSV should parse");
    assert_eq!(roster.len(), 3);
    assert_eq!(matches.len(), 2);

    // Reject self-match, invalid score, and >64 KiB payload
    assert!(parse_match_csv("Alice,Alice,1.0\n", &[]).is_err());
    assert!(parse_match_csv("Alice,Bob,1.5\n", &[]).is_err());
    assert!(parse_match_csv("Alice,Bob,NaN\n", &[]).is_err());
    let oversized = "a,b,1.0\n".repeat(9_000);
    assert!(parse_match_csv(&oversized, &[]).is_err());

    // Formula-injection protection on CSV exports
    let mut tool = Glicko2TournamentTool::default();
    tool.roster[0] = RosterPlayer {
        name: "=HYPERLINK(\"http://evil\",\"win\")".into(),
        rating: 1500.0,
        rd: 200.0,
        volatility: 0.06,
    };
    tool.recalculate();

    let standings_csv = format_leaderboard_csv(&tool.standings);
    let matrix_csv = format_matchup_csv(&tool.roster, &tool.win_prob_matrix);
    assert!(
        standings_csv.contains("\"'=HYPERLINK("),
        "Leaderboard CSV must escape formula trigger: {standings_csv}"
    );
    assert!(
        matrix_csv.contains("\"'=HYPERLINK("),
        "Matchup CSV must escape formula trigger: {matrix_csv}"
    );
}
