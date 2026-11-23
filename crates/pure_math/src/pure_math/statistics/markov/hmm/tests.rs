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
}
