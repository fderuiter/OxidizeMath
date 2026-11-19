#![allow(missing_docs)]
use domain_applied::applied::clinical_trials::survival_analysis::{
    Observation, kaplan_meier, try_estimate_hazard_ratio,
};
use domain_applied::applied::clinical_trials::types::{ClinicalTrialError, SurvivalTime};

#[test]
#[verified_engine::verified]
fn test_hazard_ratio_valid() {
    let group1 = vec![
        Observation {
            time: SurvivalTime::new(10.0).unwrap(),
            event_occurred: true,
        },
        Observation {
            time: SurvivalTime::new(20.0).unwrap(),
            event_occurred: false,
        },
    ];
    let group2 = vec![
        Observation {
            time: SurvivalTime::new(5.0).unwrap(),
            event_occurred: true,
        },
        Observation {
            time: SurvivalTime::new(5.0).unwrap(),
            event_occurred: true,
        },
    ];

    // HR = (1 event / 30 time) / (2 events / 10 time) = 0.0333... / 0.2 = 0.1666...
    let hr = try_estimate_hazard_ratio(&group1, &group2).unwrap();
    assert!((hr - 0.166666).abs() < 1e-4);
}

#[test]
#[verified_engine::verified]
fn test_hazard_ratio_error() {
    let group1 = vec![Observation {
        time: SurvivalTime::new(10.0).unwrap(),
        event_occurred: true,
    }];
    let group_empty = vec![];

    let res = try_estimate_hazard_ratio(&group1, &group_empty);
    assert!(res.is_err());
    match res.unwrap_err() {
        ClinicalTrialError::InvalidData(msg) => {
            assert_eq!(msg, "Group 2 total time is zero or negative");
        }
        e => panic!("Expected InvalidData, got {:?}", e),
    }
}

#[test]
#[verified_engine::verified]
fn test_hazard_ratio_negative_time() {
    // This test now verifies that we cannot construct an invalid SurvivalTime,
    // which prevents invalid Observations from being created in the first place.
    let res = SurvivalTime::new(-10.0);
    assert!(res.is_err());

    match res {
        Err(e) => assert!(e.to_string().contains("Time must be non-negative")),
        _ => panic!("Expected error for negative time"),
    }
}

#[test]
#[verified_engine::verified]
fn test_survival_time_rejects_non_finite_and_overflow() {
    assert!(SurvivalTime::new(f64::INFINITY).is_err());
    assert!(SurvivalTime::new(f64::NEG_INFINITY).is_err());
    assert!(SurvivalTime::new(f64::NAN).is_err());

    // Arithmetic overflow and domain errors delegating to SurvivalTime::new
    assert!(SurvivalTime::new(1000.0).unwrap().exp().is_err());
    assert!(SurvivalTime::new(0.0).unwrap().ln().is_err());
    assert!((SurvivalTime::new(1.0).unwrap() / 0.0).is_err());
    assert!((SurvivalTime::new(f64::MAX).unwrap() * 2.0).is_err());
}

#[test]
#[verified_engine::verified]
fn test_survival_time_normalizes_negative_zero() {
    let neg_zero = SurvivalTime::new(-0.0).unwrap();
    let pos_zero = SurvivalTime::new(0.0).unwrap();
    assert_eq!(neg_zero.as_f64().to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        neg_zero.as_f64().total_cmp(&pos_zero.as_f64()),
        std::cmp::Ordering::Equal
    );

    let obs = vec![
        Observation {
            time: neg_zero,
            event_occurred: false,
        },
        Observation {
            time: pos_zero,
            event_occurred: true,
        },
    ];
    let curve = kaplan_meier(&obs);
    assert_eq!(curve.len(), 1);
    assert_eq!(curve[0].time.to_bits(), 0.0_f64.to_bits());
    assert_eq!(curve[0].n_events, 1);
    assert_eq!(curve[0].n_censored, 1);
    assert!((curve[0].survival_probability - 0.5).abs() < 1e-12);
}

#[test]
#[verified_engine::verified]
fn test_kaplan_meier_near_ties_total_order() {
    let mut x: u64 = 1;
    let obs: Vec<Observation> = (0..40)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            let t = ((x >> 48) % 6) as f64 * 6e-10;
            Observation {
                time: SurvivalTime::new(t).unwrap(),
                event_occurred: true,
            }
        })
        .collect();

    let curve = kaplan_meier(&obs);
    assert!(!curve.is_empty());
    assert_eq!(curve[0].n_at_risk, 40);

    let total_events: usize = curve.iter().map(|pt| pt.n_events).sum();
    assert_eq!(total_events, 40);
    assert_eq!(curve.last().unwrap().survival_probability, 0.0);

    for pair in curve.windows(2) {
        assert!(pair[0].time < pair[1].time);
    }
}
