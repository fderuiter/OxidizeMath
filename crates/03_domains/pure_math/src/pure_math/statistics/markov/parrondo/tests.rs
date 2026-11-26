use super::*;
use approx::assert_relative_eq;
use oxidize_core::rng::OxidizeRng;

#[test]
#[verified_engine::verified]
fn test_canonical_m3_exact_gains_and_broken_detailed_balance() {
    let params = ParrondoParams::canonical();
    let schedule = ParrondoSchedule::parse("ABB").unwrap();
    let report = solve_parrondo_ratchet(&params, &schedule).unwrap();

    assert_relative_eq!(report.gain_a, -0.0100, epsilon = 1e-4);
    assert_relative_eq!(report.gain_b, -0.008695, epsilon = 1e-4);
    assert_relative_eq!(report.gain_mix, 0.015704, epsilon = 1e-4);
    assert_relative_eq!(report.gain_schedule, 0.057431, epsilon = 1e-4);
    assert!(report.is_paradox());
    assert!(report.detailed_balance_schedule > 0.004);
    assert!(report.detailed_balance_mix > 0.005);

    let fair_params = ParrondoParams::new(3, 0.5, 0.1, 0.75, 0.0).unwrap();
    let fair_report = solve_parrondo_ratchet(&fair_params, &schedule).unwrap();
    assert!(fair_report.detailed_balance_b < 1e-11);
    assert!(fair_report.detailed_balance_mix > 0.008);
    assert!(fair_report.detailed_balance_schedule > 0.002);
}

#[test]
#[verified_engine::verified]
fn test_schedule_parsing_and_bounds() {
    let abb = ParrondoSchedule::parse("ABB").unwrap();
    assert_eq!(abb.len(), 3);
    assert!(!abb.is_empty());
    assert_eq!(abb.as_string(), "ABB");

    let max_sched = ParrondoSchedule::parse("ABABABABABABABAB").unwrap();
    assert_eq!(max_sched.len(), MAX_SCHEDULE_LEN);

    assert!(ParrondoSchedule::parse("").is_err());
    assert!(ParrondoSchedule::parse("ABBC").is_err());
    assert!(ParrondoSchedule::parse("abb").is_err());
    assert!(ParrondoSchedule::parse("ABABABABABABABABA").is_err());
}

#[test]
#[verified_engine::verified]
fn test_params_validation_and_odd_moduli_convergence() {
    for m in ALLOWED_MODULI {
        let params = ParrondoParams::new(m, 0.5, 0.1, 0.75, 0.005).unwrap();
        for spec in ["ABB", "AABB", "ABAB"] {
            let sched = ParrondoSchedule::parse(spec).unwrap();
            let report = solve_parrondo_ratchet(&params, &sched).unwrap();
            assert_relative_eq!(report.pi_b.sum(), 1.0, epsilon = 1e-9);
            assert_relative_eq!(report.pi_mix.sum(), 1.0, epsilon = 1e-9);
            assert_relative_eq!(report.pi_schedule.sum(), 1.0, epsilon = 1e-9);
            assert_relative_eq!(report.pi_schedule_mean.sum(), 1.0, epsilon = 1e-9);
        }
    }

    assert!(ParrondoParams::new(4, 0.5, 0.1, 0.75, 0.005).is_err());
    assert!(ParrondoParams::new(2, 0.5, 0.1, 0.75, 0.005).is_err());
    assert!(ParrondoParams::new(11, 0.5, 0.1, 0.75, 0.005).is_err());
    assert!(ParrondoParams::new(3, 0.5, 0.1, 0.75, 0.15).is_err());
    assert!(ParrondoParams::new(3, 0.5, 0.1, 0.75, -0.30).is_err());
    assert!(ParrondoParams::new(3, f64::NAN, 0.1, 0.75, 0.005).is_err());
}

#[test]
#[verified_engine::verified]
fn test_negative_wealth_rem_euclid_and_deterministic_rng() {
    let losing = ParrondoParams::new(5, 0.15, 0.05, 0.2, 0.01).unwrap();
    let sched = ParrondoSchedule::parse("AB").unwrap();
    let mut ens_a = ParrondoEnsemble::with_capacity(64, 120).unwrap();
    let mut ens_b = ParrondoEnsemble::with_capacity(64, 120).unwrap();

    let mut rng_a = OxidizeRng::new(0xDEAD_BEEF_CAFE_2026);
    let mut rng_b = OxidizeRng::new(0xDEAD_BEEF_CAFE_2026);

    let final_a = ens_a.simulate_schedule(&losing, &sched, 120, &mut rng_a);
    let final_b = ens_b.simulate_schedule(&losing, &sched, 120, &mut rng_b);

    assert!(final_a < -40.0);
    assert_eq!(final_a.to_bits(), final_b.to_bits());
    assert_eq!(ens_a.wealth(), ens_b.wealth());
    assert_eq!(ens_a.occupancy_counts(5), ens_b.occupancy_counts(5));
    let total_visits: u64 = ens_a.occupancy_counts(5).iter().sum();
    assert_eq!(total_visits, 64 * 120);
}

#[test]
#[verified_engine::verified]
fn test_ensemble_empirical_gain_matches_markov_ratchet() {
    let params = ParrondoParams::canonical();
    let sched = ParrondoSchedule::abb();
    let mut ens = ParrondoEnsemble::with_capacity(1_500, 300).unwrap();
    let mut rng = OxidizeRng::new(42);

    let final_abb = ens.simulate_schedule(&params, &sched, 300, &mut rng);
    let empirical_rate = final_abb / 300.0;
    assert!(
        (empirical_rate - 0.057431).abs() < 0.02,
        "Empirical ABB drift {empirical_rate} deviated from exact +0.0574"
    );

    let final_mix = ens.simulate_mixture(&params, 300, &mut rng);
    assert!(
        final_mix > 0.0,
        "Expected positive mean wealth for 50/50 mixture"
    );
}

#[test]
#[verified_engine::verified]
fn test_scan_parrondo_phase_region_identifies_wedge() {
    let cfg = ParrondoPhaseGridConfig::default();
    let sched = ParrondoSchedule::abb();
    let pts = scan_parrondo_phase_region(&cfg, &sched).unwrap();

    assert_eq!(pts.len(), cfg.p_a_steps * cfg.epsilon_steps);
    let paradox_count = pts.iter().filter(|p| p.is_paradox).count();
    assert!(
        paradox_count >= 20 && paradox_count < pts.len(),
        "Expected non-trivial paradox wedge, got {paradox_count}/{}",
        pts.len()
    );
}

#[test]
#[verified_engine::verified]
fn test_ensemble_cross_walker_standard_errors_zero_allocation() {
    use verified_engine::engine::VerifiedEngine;

    let params = ParrondoParams::canonical();
    let sched = ParrondoSchedule::abb();
    let mut ens = ParrondoEnsemble::with_capacity(256, 60).unwrap();
    let mut rng = OxidizeRng::new(909);

    let (est, _) = VerifiedEngine::run_verified(|| {
        let _ = ens.simulate_schedule(&params, &sched, 60, &mut rng);
        ens.terminal_estimate(0.95).unwrap()
    });

    assert_eq!(ens.stderr_trajectory().len(), 61);
    assert_eq!(ens.stderr_trajectory()[0], 0.0);
    assert!(ens.terminal_standard_error() > 0.0);
    assert_relative_eq!(
        est.standard_error,
        ens.terminal_standard_error(),
        epsilon = 1e-12
    );
    assert!(est.ci_lower < est.mean && est.mean < est.ci_upper);
}

// [cite:clinical_trials_statistics]
