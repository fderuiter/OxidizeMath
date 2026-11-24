use super::*;
use oxidize_core::rng::OxidizeRng;
use pure_math::pure_math::statistics::bootstrap::BootstrapMethod;

#[test]
#[verified_engine::verified]
fn test_kaplan_meier_bootstrap_percentile_and_bca_bands() {
    let raw = [
        (1.0, true),
        (2.0, true),
        (2.5, false),
        (3.0, true),
        (4.0, true),
        (5.0, false),
        (6.0, true),
        (7.5, true),
        (8.0, false),
        (10.0, true),
    ];
    let obs: Vec<Observation> = raw
        .into_iter()
        .map(|(t, ev)| Observation {
            time: SurvivalTime::new(t).unwrap(),
            event_occurred: ev,
        })
        .collect();

    let mut rng = OxidizeRng::new(2026);
    let pct = kaplan_meier_bootstrap(&obs, 600, 0.95, BootstrapMethod::Percentile, &mut rng)
        .expect("percentile survival bands");
    let bca = kaplan_meier_bootstrap(&obs, 600, 0.95, BootstrapMethod::BCa, &mut rng)
        .expect("bca survival bands");

    assert_eq!(pct.len(), bca.len());
    for (p, b) in pct.iter().zip(&bca) {
        assert!(p.ci_lower >= 0.0 && p.ci_upper <= 1.0);
        assert!(b.ci_lower >= 0.0 && b.ci_upper <= 1.0);
        assert!(p.ci_lower <= p.survival_probability && p.survival_probability <= p.ci_upper);
        assert!(b.ci_lower <= b.survival_probability && b.survival_probability <= b.ci_upper);
    }
    assert!(pct[0].standard_error > 0.0);

    // Reject invalid sample sizes or confidence levels.
    assert!(
        kaplan_meier_bootstrap(&obs[..1], 100, 0.95, BootstrapMethod::Percentile, &mut rng)
            .is_err()
    );
    assert!(kaplan_meier_bootstrap(&obs, 0, 0.95, BootstrapMethod::Percentile, &mut rng).is_err());
    assert!(kaplan_meier_bootstrap(&obs, 100, 1.2, BootstrapMethod::Percentile, &mut rng).is_err());
}

#[cfg(feature = "serde")]
#[test]
#[verified_engine::verified]
fn test_survival_analysis_serde_roundtrip() {
    let arm_a = vec![
        Observation {
            time: SurvivalTime::new(2.0).unwrap(),
            event_occurred: true,
        },
        Observation {
            time: SurvivalTime::new(5.0).unwrap(),
            event_occurred: true,
        },
    ];
    let arm_b = vec![
        Observation {
            time: SurvivalTime::new(4.0).unwrap(),
            event_occurred: true,
        },
        Observation {
            time: SurvivalTime::new(8.0).unwrap(),
            event_occurred: false,
        },
    ];
    let lr = log_rank_test(&arm_a, &arm_b).expect("log-rank result");
    let json = serde_json::to_string(&lr).expect("serialize LogRankResult");
    let restored: LogRankResult = serde_json::from_str(&json).expect("deserialize LogRankResult");
    assert_eq!(lr, restored);
}

// [cite:clinical_trials_statistics]
