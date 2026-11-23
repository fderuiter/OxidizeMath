#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::super::model::HiddenMarkovModel;
    use approx::assert_relative_eq;
    use nalgebra::{DMatrix, DVector};

    #[test]
    #[verified_engine::verified]
    fn test_hmm_creation() {
        let initial = DVector::from_vec(vec![0.5, 0.5]);
        let transitions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emissions = DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.3, 0.7]);

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();

        assert_eq!(hmm.num_states(), 2);
        assert_eq!(hmm.num_observations(), 2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_hmm_creation_f32() {
        let initial = DVector::from_vec(vec![0.5f32, 0.5]);
        let transitions = DMatrix::from_row_slice(2, 2, &[0.7f32, 0.3, 0.4, 0.6]);
        let emissions = DMatrix::from_row_slice(2, 2, &[0.8f32, 0.2, 0.3, 0.7]);

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();

        assert_eq!(hmm.num_states(), 2);
        assert_eq!(hmm.num_observations(), 2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_forward_algorithm() {
        // Simple HMM: two states, two observations
        let initial = DVector::from_vec(vec![0.6, 0.4]);
        let transitions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emissions = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.2, 0.8]);

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();

        let observations = vec![0, 1, 0];
        let prob = hmm.forward(&observations).unwrap();
        let log_prob = hmm.forward_log_likelihood(&observations).unwrap();

        // Exact sum over all 2^3 = 8 hidden state paths
        let init = [0.6, 0.4];
        let trans = [[0.7, 0.3], [0.4, 0.6]];
        let emit = [[0.9, 0.1], [0.2, 0.8]];
        let mut expected: f64 = 0.0;
        for s0 in 0..2 {
            for s1 in 0..2 {
                for s2 in 0..2 {
                    expected += init[s0]
                        * emit[s0][0]
                        * trans[s0][s1]
                        * emit[s1][1]
                        * trans[s1][s2]
                        * emit[s2][0];
                }
            }
        }
        assert!(prob > 0.0 && prob < 1.0);
        assert_relative_eq!(prob, expected, epsilon = 1e-12);
        assert_relative_eq!(log_prob, expected.ln(), epsilon = 1e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_3state_forbidden_marginal_paradox_and_zero_alloc() {
        use super::super::workspace::HmmScratch;

        let initial = DVector::from_vec(vec![0.25, 0.05, 0.70]);
        let transitions = DMatrix::from_row_slice(
            3,
            3,
            &[
                0.20, 0.40, 0.40, // State 0
                0.60, 0.00, 0.40, // State 1 (A[1, 1] = 0.00 forbidden self-loop!)
                0.05, 0.95, 0.00, // State 2
            ],
        );
        let emissions = DMatrix::from_row_slice(
            3,
            3,
            &[
                0.50, 0.15, 0.35, // State 0
                0.10, 0.85, 0.05, // State 1
                0.10, 0.70, 0.20, // State 2
            ],
        );
        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
        let observations = [0usize, 1, 1];

        let fwd = hmm.forward(&observations).unwrap();
        let fwd_log = hmm.forward_log_likelihood(&observations).unwrap();
        assert_relative_eq!(fwd, 0.0693765, epsilon = 1e-12);
        assert_relative_eq!(fwd_log, (0.0693765_f64).ln(), epsilon = 1e-12);

        let viterbi_path = hmm.viterbi(&observations).unwrap();
        assert_eq!(viterbi_path, vec![0, 2, 1]);

        let gamma = hmm.posterior_probabilities(&observations).unwrap();
        let map_path: Vec<usize> = (0..3)
            .map(|t| {
                (0..3)
                    .max_by(|&a, &b| gamma[(a, t)].partial_cmp(&gamma[(b, t)]).unwrap())
                    .unwrap()
            })
            .collect();
        assert_eq!(map_path, vec![0, 1, 1]);

        let mut scratch = HmmScratch::new(4, 8);
        let (eval, _cert) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            hmm.evaluate_with_scratch(&observations, &mut scratch)
                .unwrap()
        });

        assert_relative_eq!(eval.likelihood, 0.0693765, epsilon = 1e-12);
        assert_relative_eq!(eval.viterbi_joint_prob, 0.0282625, epsilon = 1e-12);
        assert_relative_eq!(eval.marginal_joint_prob, 0.0, epsilon = 1e-15);
        assert!(eval.has_forbidden_marginal_step);
        assert_eq!(eval.first_forbidden_step, Some(1));
        assert_eq!(scratch.viterbi_slice(3), &[0, 2, 1]);
        assert_eq!(scratch.marginal_slice(3), &[0, 1, 1]);
    }

    #[test]
    #[verified_engine::verified]
    fn test_viterbi() {
        // Classic example: two states (Fair/Loaded dice), two observations (0-5)
        // Simplified to 2 observations for testing
        let initial = DVector::from_vec(vec![0.5, 0.5]);
        let transitions = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.1, 0.9]);
        let emissions = DMatrix::from_row_slice(
            2,
            2,
            &[
                0.5, 0.5, // Fair: uniform
                0.1, 0.9, // Loaded: biased toward 1
            ],
        );

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();

        // Observations: 1, 1, 1 (likely loaded)
        let observations = vec![1, 1, 1];
        let path = hmm.viterbi(&observations).unwrap();

        assert_eq!(path.len(), 3);
        // Most likely state sequence should end in loaded (state 1)
        // Due to persistence (0.9 stay probability), likely all state 1
        assert_eq!(path[2], 1);
    }

    #[test]
    #[verified_engine::verified]
    fn test_hot_hand_detection() {
        // Basketball shooting: Cold (0) vs Hot (1)
        // Observations: Miss (0), Make (1)
        let initial = DVector::from_vec(vec![0.5, 0.5]);

        let transitions = DMatrix::from_row_slice(
            2,
            2,
            &[
                0.8, 0.2, // Cold → Cold/Hot
                0.3, 0.7, // Hot → Cold/Hot
            ],
        );

        let emissions = DMatrix::from_row_slice(
            2,
            2,
            &[
                0.7, 0.3, // Cold: 30% makes
                0.2, 0.8, // Hot: 80% makes
            ],
        );

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();

        // Shooting sequence: Make, Make, Make, Miss
        let observations = vec![1, 1, 1, 0];

        // Viterbi: most likely states
        let states = hmm.viterbi(&observations).unwrap();

        // After three makes, should likely be in hot state
        assert_eq!(states[2], 1); // Hot after 3 makes

        // Filtering: current belief
        let posterior = hmm.filter(&observations).unwrap();

        // After 3 makes then a miss, should still believe somewhat in hot
        assert!(posterior[1] > 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_posterior_probabilities() {
        let initial = DVector::from_vec(vec![0.6, 0.4]);
        let transitions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emissions = DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.3, 0.7]);

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();

        let observations = vec![0, 1];
        let gamma = hmm.posterior_probabilities(&observations).unwrap();

        // Check that posteriors sum to 1 at each time
        for t in 0..2 {
            let sum: f64 = gamma.column(t).iter().sum();
            assert_relative_eq!(sum, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_generate_sequence() {
        let initial = DVector::from_vec(vec![0.5, 0.5]);
        let transitions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emissions = DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.3, 0.7]);

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();
        let mut rng = oxidize_core::rng::OxidizeRng::default();

        let (states, observations) = hmm.generate(10, &mut rng).unwrap();

        assert_eq!(states.len(), 10);
        assert_eq!(observations.len(), 10);

        // All states should be valid
        for &s in &states {
            assert!(s < 2);
        }

        // All observations should be valid
        for &o in &observations {
            assert!(o < 2);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_deterministic_generation() {
        let initial = DVector::from_vec(vec![0.5, 0.5]);
        let transitions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emissions = DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.3, 0.7]);

        let hmm = HiddenMarkovModel::new(initial, transitions, emissions).unwrap();

        let mut rng1 = oxidize_core::rng::OxidizeRng::new(123);
        let (states1, obs1) = hmm.generate(20, &mut rng1).unwrap();

        let mut rng2 = oxidize_core::rng::OxidizeRng::new(123);
        let (states2, obs2) = hmm.generate(20, &mut rng2).unwrap();

        // Same seed should produce same sequence
        assert_eq!(states1, states2);
        assert_eq!(obs1, obs2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_validation_errors() {
        let initial = DVector::from_vec(vec![0.5, 0.4]); // Doesn't sum to 1
        let transitions = DMatrix::from_row_slice(2, 2, &[0.7, 0.3, 0.4, 0.6]);
        let emissions = DMatrix::from_row_slice(2, 2, &[0.8, 0.2, 0.3, 0.7]);

        assert!(HiddenMarkovModel::new(initial, transitions, emissions).is_err());
    }

    fn market_regime_hmm() -> HiddenMarkovModel<f64> {
        let initial = DVector::from_vec(vec![0.33, 0.33, 0.34]);
        let transitions = DMatrix::from_row_slice(
            3,
            3,
            &[0.70, 0.20, 0.10, 0.20, 0.70, 0.10, 0.25, 0.25, 0.50],
        );
        let emissions = DMatrix::from_row_slice(
            3,
            5,
            &[
                0.30, 0.30, 0.20, 0.10, 0.10, // Bull
                0.10, 0.10, 0.20, 0.30, 0.30, // Bear
                0.10, 0.20, 0.40, 0.20, 0.10, // Sideways
            ],
        );
        HiddenMarkovModel::new(initial, transitions, emissions).unwrap()
    }

    #[test]
    #[verified_engine::verified]
    fn test_market_regime_forward_and_log_likelihood() {
        let hmm = market_regime_hmm();
        let observations = [1, 0, 2, 3, 4];
        let prob = hmm.forward(&observations).unwrap();
        let ll = hmm.log_likelihood(&observations).unwrap();

        assert_relative_eq!(prob, 0.000_326, epsilon = 1e-6);
        assert_relative_eq!(ll, -8.028_610, epsilon = 1e-6);
        assert_relative_eq!(ll.exp(), prob, epsilon = 1e-15);
    }

    #[test]
    fn test_viterbi_long_sequences_no_underflow_collapse() {
        let hmm = market_regime_hmm();
        let mut rng = oxidize_core::rng::OxidizeRng::new(2026);
        let (_, obs_600) = hmm.generate(600, &mut rng).unwrap();
        let path_600 = hmm.viterbi(&obs_600).unwrap();
        assert_eq!(path_600.len(), 600);
        assert!(path_600[400..].iter().any(|&s| s != 0));

        // 10,000-step sequence alternating bear/sideways emissions must stay out of state 0
        let mut obs_10k = Vec::with_capacity(10_000);
        for t in 0..10_000 {
            obs_10k.push(if t % 50 < 35 { 4 } else { 2 });
        }
        let path_10k = hmm.viterbi(&obs_10k).unwrap();
        assert_eq!(path_10k.len(), 10_000);
        assert_eq!(path_10k[9_960], 1);
        assert!(path_10k[5_000..].contains(&1));
    }

    #[test]
    fn test_zero_allocation_viterbi_and_baum_welch_into() {
        use super::super::{BaumWelchConfig, BaumWelchWorkspace, ViterbiWorkspace};

        let true_hmm = market_regime_hmm();
        let mut rng = oxidize_core::rng::OxidizeRng::new(77);
        let (_, observations) = true_hmm.generate(120, &mut rng).unwrap();

        let mut model = HiddenMarkovModel::new(
            DVector::from_vec(vec![0.40, 0.35, 0.25]),
            DMatrix::from_row_slice(
                3,
                3,
                &[0.55, 0.25, 0.20, 0.25, 0.55, 0.20, 0.30, 0.30, 0.40],
            ),
            DMatrix::from_row_slice(
                3,
                5,
                &[
                    0.26, 0.24, 0.20, 0.15, 0.15, 0.15, 0.15, 0.20, 0.24, 0.26, 0.18, 0.21, 0.24,
                    0.19, 0.18,
                ],
            ),
        )
        .unwrap();

        let config = BaumWelchConfig::new(15, 1e-8, 1e-6).unwrap();
        let mut bw_ws = BaumWelchWorkspace::new(3, 5, observations.len());
        let mut vit_ws = ViterbiWorkspace::new(3, 5, observations.len());
        let mut history = [0.0_f64; 15];
        let mut decoded = vec![0_usize; observations.len()];

        let (res, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let report =
                model.fit_baum_welch_into(&observations, &config, &mut bw_ws, &mut history)?;
            model.viterbi_into(&observations, &mut vit_ws, &mut decoded)?;
            Ok::<_, crate::error::MarkovError>(report)
        });
        let report = res.expect("zero-alloc Baum-Welch + Viterbi succeeded");
        assert!(report.iterations >= 1);
        assert!(report.final_log_likelihood >= report.initial_log_likelihood - 1e-12);
        let mut prev = report.initial_log_likelihood;
        for &ll in &history[..report.iterations] {
            assert!(ll >= prev - 1e-12, "non-monotonic step: {ll} < {prev}");
            prev = ll;
        }
    }

    #[test]
    fn test_hmm_serde_roundtrip_and_validation_rejections() {
        use serde::Deserialize;
        use serde::de::IntoDeserializer;
        use serde::de::value::{Error as ValErr, MapDeserializer, SeqDeserializer};

        let hmm = market_regime_hmm();
        let json = serde_json::to_string(&hmm).unwrap();
        let decoded: HiddenMarkovModel<f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.num_states(), 3);
        assert_eq!(decoded.num_observations(), 5);
        assert_relative_eq!(decoded.transitions()[(0, 0)], 0.70, epsilon = 1e-12);

        // Reject non-stochastic transition row via MarkovError
        let bad_row = r#"{"initial":[0.5,0.5],"transitions":[[0.8,0.5],[0.4,0.6]],"emissions":[[0.5,0.5],[0.5,0.5]]}"#;
        let err = serde_json::from_str::<HiddenMarkovModel<f64>>(bad_row).unwrap_err();
        assert!(err.to_string().contains("not stochastic"));

        // Reject unknown fields
        let unknown = r#"{"initial":[0.5,0.5],"transitions":[[0.5,0.5],[0.5,0.5]],"emissions":[[0.5,0.5],[0.5,0.5]],"extra":1}"#;
        assert!(serde_json::from_str::<HiddenMarkovModel<f64>>(unknown).is_err());

        // Reject NaN probability reaching Deserialize via MarkovError::InvalidProbability
        let map_de: MapDeserializer<'_, _, ValErr> = MapDeserializer::new(
            [
                ("initial", vec![f64::NAN, 0.5]),
                ("transitions", vec![1.0]),
                ("emissions", vec![1.0]),
            ]
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    SeqDeserializer::new(v.into_iter().map(IntoDeserializer::into_deserializer)),
                )
            }),
        );
        assert!(HiddenMarkovModel::<f64>::deserialize(map_de).is_err());
        let nan_raw = r#"{"initial":[NaN,0.5],"transitions":[[0.5,0.5],[0.5,0.5]],"emissions":[[0.5,0.5],[0.5,0.5]]}"#;
        assert!(serde_json::from_str::<HiddenMarkovModel<f64>>(nan_raw).is_err());
    }
}
