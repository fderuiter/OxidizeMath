use super::*;
use approx::assert_relative_eq;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_STANDARD};

#[test]
#[verified_engine::verified]
fn test_iterated_strict_and_weak_dominance() {
    // Prisoner's Dilemma: Defect (index 1) strictly dominates Cooperate (index 0) for both players.
    let a = [3.0, 0.0, 5.0, 1.0];
    let b = [3.0, 5.0, 0.0, 1.0];
    let game = BimatrixGameView::new(&a, &b, 2, 2).unwrap();
    let mut ws = DominanceWorkspace::new(3, 3);
    let mut r_alive = [false; 2];
    let mut c_alive = [false; 2];

    let res = iterated_dominance(
        game,
        DominanceMode::Strict,
        &mut ws,
        &mut r_alive,
        &mut c_alive,
    )
    .unwrap();
    assert_eq!(res.surviving_rows, 1);
    assert_eq!(res.surviving_cols, 1);
    assert_eq!(r_alive, [false, true]);
    assert_eq!(c_alive, [false, true]);

    // Weak dominance game where Row 0 weakly dominates Row 1, then Col 0 strictly dominates Col 1.
    let aw = [2.0, 1.0, 2.0, 0.0];
    let bw = [3.0, 1.0, 0.0, 2.0];
    let weak_game = BimatrixGameView::new(&aw, &bw, 2, 2).unwrap();
    let strict_res = iterated_dominance(
        weak_game,
        DominanceMode::Strict,
        &mut ws,
        &mut r_alive,
        &mut c_alive,
    )
    .unwrap();
    assert_eq!(strict_res.surviving_rows, 2);

    let weak_res = iterated_dominance(
        weak_game,
        DominanceMode::Weak,
        &mut ws,
        &mut r_alive,
        &mut c_alive,
    )
    .unwrap();
    assert_eq!(weak_res.surviving_rows, 1);
    assert_eq!(weak_res.surviving_cols, 1);
    assert_eq!(r_alive, [true, false]);
    assert_eq!(c_alive, [true, false]);
}

#[test]
#[verified_engine::verified]
fn test_zero_sum_simplex_minimax_rps_and_rectangular() {
    // Rock-Paper-Scissors: unique minimax value 0 and uniform (1/3, 1/3, 1/3) strategies.
    let rps = [0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0];
    let mut ws = MinimaxWorkspace::new(3, 3);
    let mut p = [0.0; 3];
    let mut q = [0.0; 3];
    let res = solve_zero_sum_minimax(&rps, 3, 3, &mut ws, &mut p, &mut q).unwrap();
    assert_relative_eq!(res.game_value, 0.0, epsilon = TOLERANCE_STANDARD);
    for i in 0..3 {
        assert_relative_eq!(p[i], 1.0 / 3.0, epsilon = TOLERANCE_FAST);
        assert_relative_eq!(q[i], 1.0 / 3.0, epsilon = TOLERANCE_FAST);
    }

    // 2x3 zero-sum game: A = [[3, 0, 2], [0, 4, 2]]; col 2 is dominated or active at value 12/7?
    // Wait: against p=(4/7, 3/7), col payoffs are 12/7, 12/7, 14/7 -> col plays (4/7, 3/7, 0), value = 12/7!
    let rect = [3.0, 0.0, 2.0, 0.0, 4.0, 2.0];
    let mut ws_rect = MinimaxWorkspace::new(2, 3);
    let mut p2 = [0.0; 2];
    let mut q3 = [0.0; 3];
    let rect_res = solve_zero_sum_minimax(&rect, 2, 3, &mut ws_rect, &mut p2, &mut q3).unwrap();
    assert_relative_eq!(rect_res.game_value, 12.0 / 7.0, epsilon = TOLERANCE_FAST);
    assert_relative_eq!(p2[0], 4.0 / 7.0, epsilon = TOLERANCE_FAST);
    assert_relative_eq!(q3[2], 0.0, epsilon = TOLERANCE_FAST);
}

#[test]
#[verified_engine::verified]
fn test_support_enumeration_battle_of_the_sexes_and_cap() {
    // Battle of the Sexes has 3 Nash equilibria: (1,0)-(1,0), (0,1)-(0,1), and (2/3, 1/3)-(1/3, 2/3).
    let a = [2.0, 0.0, 0.0, 1.0];
    let b = [1.0, 0.0, 0.0, 2.0];
    let game = BimatrixGameView::new(&a, &b, 2, 2).unwrap();
    let mut eqs = [SupportEquilibrium {
        rows: 0,
        cols: 0,
        row_strategy: [0.0; MAX_SUPPORT_ENUM_DIM],
        col_strategy: [0.0; MAX_SUPPORT_ENUM_DIM],
        row_payoff: 0.0,
        col_payoff: 0.0,
    }; 8];

    let count = support_enumeration(game, &mut eqs).unwrap();
    assert_eq!(count, 3);
    for eq in &eqs[..count] {
        let (_, _, regret) =
            evaluate_bimatrix_profile(game, eq.row_probs(), eq.col_probs()).unwrap();
        assert!(regret <= TOLERANCE_FAST);
    }

    // Verify 9x2 exceeds the 8x8 cap.
    let big = [0.0; 18];
    let big_game = BimatrixGameView::new(&big, &big, 9, 2).unwrap();
    assert!(matches!(
        support_enumeration(big_game, &mut eqs),
        Err(GameTheoryError::SupportEnumerationCapExceeded { .. })
    ));
}

#[test]
#[verified_engine::verified]
fn test_lemke_howson_nondegenerate_and_degenerate_bimatrix() {
    // Matching Pennies: unique mixed Nash equilibrium (0.5, 0.5), payoffs (0, 0).
    let a_mp = [1.0, -1.0, -1.0, 1.0];
    let b_mp = [-1.0, 1.0, 1.0, -1.0];
    let mp = BimatrixGameView::new(&a_mp, &b_mp, 2, 2).unwrap();
    let mut ws = LemkeHowsonWorkspace::new(3, 3);
    let mut p = [0.0; 2];
    let mut q = [0.0; 2];

    for init_label in 0..4 {
        let eq = lemke_howson_equilibrium(mp, init_label, &mut ws, &mut p, &mut q).unwrap();
        assert_relative_eq!(p[0], 0.5, epsilon = TOLERANCE_FAST);
        assert_relative_eq!(q[0], 0.5, epsilon = TOLERANCE_FAST);
        assert_relative_eq!(eq.row_payoff, 0.0, epsilon = TOLERANCE_FAST);
        assert_relative_eq!(eq.col_payoff, 0.0, epsilon = TOLERANCE_FAST);
    }

    // Highly degenerate 3x3 game (all zero / tied payoffs) must not cycle thanks to lexicographic perturbation.
    let deg_a = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let deg_b = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let deg_game = BimatrixGameView::new(&deg_a, &deg_b, 3, 3).unwrap();
    let mut p3 = [0.0; 3];
    let mut q3 = [0.0; 3];
    for label in 0..6 {
        let _ = lemke_howson_equilibrium(deg_game, label, &mut ws, &mut p3, &mut q3).unwrap();
        let (_, _, regret) = evaluate_bimatrix_profile(deg_game, &p3, &q3).unwrap();
        assert!(regret <= TOLERANCE_FAST);
    }
}

// [cite:graph_parameters_rust]
