use super::*;
use crate::pure_math::analysis::stochastic::GillespieSolver;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_jump_ratchet_stack_and_ssa_walkers_zero_alloc() {
    let (win_a, win_b, _) = parrondo_variant_win_probs(ParrondoVariant::HistoryDependent, 0.005);
    let sys = ParrondoGillespieSystem {
        win_a,
        win_b,
        variant: ParrondoVariant::HistoryDependent,
        switch_ab: 1.92,
        switch_ba: 0.96,
    };
    let mut solver = GillespieSolver::with_capacity(OxidizeRng::new(7), 8);
    let mut ssa_state = ParrondoSsaState::default();
    let mut walkers = [ParrondoWalker::default(); 32];
    let mut rng = OxidizeRng::new(42);

    let ((da_c, db_c, dmix_c, dabb_c, dssa_c, da_h, db_h, dmix_h, dabb_h, dssa_h), _) =
        VerifiedEngine::run_verified(|| {
            let _ = advance_parrondo_walkers(
                &mut walkers,
                &mut rng,
                ParrondoVariant::HistoryDependent,
                &win_a,
            );
            let _ = advance_parrondo_mix_walkers(
                &mut walkers,
                &mut rng,
                ParrondoVariant::HistoryDependent,
                &win_a,
                &win_b,
                0.5,
            );
            let _ = solver.step(&sys, &mut ssa_state);
            let cap =
                evaluate_jump_ratchet_stack(ParrondoVariant::CapitalDependent, 0.005, 0.5, 1.2);
            let hist =
                evaluate_jump_ratchet_stack(ParrondoVariant::HistoryDependent, 0.001, 0.5, 1.2);
            (
                cap.0, cap.1, cap.2, cap.3, cap.4, hist.0, hist.1, hist.2, hist.3, hist.4,
            )
        });
    assert!(da_c < 0.0 && db_c < 0.0 && dmix_c > 0.0 && dabb_c > 0.0 && dssa_c > 0.0);
    assert!(da_h < 0.0 && db_h < 0.0 && dmix_h > 0.0 && dabb_h > 0.0 && dssa_h > 0.0);
}
