//! Expected Value of Perfect Information (`EVPI`) and Sample Information (`EVSI`).

use super::DecisionTheoryError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};

/// Result returned by [`evpi`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvpiResult {
    /// Maximum prior expected payoff $\max_a \sum_\theta P(\theta) u(a, \theta)$.
    pub baseline_value: f64,
    /// Expected payoff under clairvoyance $\sum_\theta P(\theta) \max_a u(a, \theta)$.
    pub perfect_info_value: f64,
    /// Expected Value of Perfect Information (`perfect_info_value - baseline_value >= 0`).
    pub evpi: f64,
    /// Optimal prior action index in `0..n_actions`.
    pub baseline_action: usize,
}

/// Result returned by [`evsi`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvsiResult {
    /// Maximum prior expected payoff without observing the signal.
    pub baseline_value: f64,
    /// Expected payoff when acting optimally after observing signal $Z$.
    pub sample_info_value: f64,
    /// Expected Value of Sample Information (`sample_info_value - baseline_value >= 0`).
    pub evsi: f64,
    /// Optimal prior action index in `0..n_actions`.
    pub baseline_action: usize,
}

/// Preallocated workspace storing marginal signal probabilities and Bayes-optimal posterior actions.
#[derive(Clone, Debug)]
pub struct VoiWorkspace {
    signal_probs: Vec<f64>,
    optimal_signal_actions: Vec<usize>,
}

impl VoiWorkspace {
    /// Allocates workspace buffers for up to `max_signals` signal realizations.
    #[must_use]
    pub fn new(max_signals: usize) -> Self {
        Self {
            signal_probs: vec![0.0; max_signals],
            optimal_signal_actions: vec![0; max_signals],
        }
    }

    /// Marginal signal probability $P(Z = z)$ populated by the latest [`evsi`] call.
    #[must_use]
    pub fn signal_probability(&self, signal: usize) -> Option<f64> {
        self.signal_probs.get(signal).copied()
    }

    /// Optimal posterior action $\pi^*(z)$ populated by the latest [`evsi`] call.
    #[must_use]
    pub fn optimal_action_for_signal(&self, signal: usize) -> Option<usize> {
        self.optimal_signal_actions.get(signal).copied()
    }
}

#[verified_engine::verified]
fn validate_payoff_and_prior(
    payoffs: &[f64],
    n_actions: usize,
    n_states: usize,
    prior: &[f64],
) -> Result<(), DecisionTheoryError> {
    if n_actions == 0 || n_states == 0 {
        return Err(DecisionTheoryError::EmptyOutcomes);
    }
    let expected = n_actions * n_states;
    if payoffs.len() != expected {
        return Err(DecisionTheoryError::DimensionMismatch {
            expected,
            actual: payoffs.len(),
        });
    }
    if prior.len() != n_states {
        return Err(DecisionTheoryError::DimensionMismatch {
            expected: n_states,
            actual: prior.len(),
        });
    }
    for &u in payoffs {
        if !u.is_finite() {
            return Err(DecisionTheoryError::InvalidRiskParameter { value: u });
        }
    }
    let mut sum = 0.0;
    for &p in prior {
        if !p.is_finite() || p < -TOLERANCE_HIGH || p > 1.0 + TOLERANCE_FAST {
            return Err(DecisionTheoryError::InvalidPrior { sum: f64::NAN });
        }
        sum += p.max(0.0);
    }
    if (sum - 1.0).abs() > TOLERANCE_FAST {
        return Err(DecisionTheoryError::InvalidPrior { sum });
    }
    Ok(())
}

/// Computes Expected Value of Perfect Information (`EVPI`) for a row-major `n_actions x n_states` payoff table.
#[verified_engine::verified]
pub fn evpi(
    payoffs: &[f64],
    n_actions: usize,
    n_states: usize,
    prior: &[f64],
) -> Result<EvpiResult, DecisionTheoryError> {
    validate_payoff_and_prior(payoffs, n_actions, n_states, prior)?;

    let mut baseline_action = 0usize;
    let mut baseline_value = f64::NEG_INFINITY;
    for a in 0..n_actions {
        let mut ev_a = 0.0;
        for (s, &p_s) in prior.iter().enumerate() {
            ev_a += p_s.max(0.0) * payoffs[flatten_2d_index(s, a, n_states)];
        }
        if ev_a > baseline_value {
            baseline_value = ev_a;
            baseline_action = a;
        }
    }

    let mut perfect_info_value = 0.0;
    for (s, &p_s) in prior.iter().enumerate() {
        let mut best_for_state = f64::NEG_INFINITY;
        for a in 0..n_actions {
            best_for_state = best_for_state.max(payoffs[flatten_2d_index(s, a, n_states)]);
        }
        perfect_info_value += p_s.max(0.0) * best_for_state;
    }

    Ok(EvpiResult {
        baseline_value,
        perfect_info_value,
        evpi: (perfect_info_value - baseline_value).max(0.0),
        baseline_action,
    })
}

/// Computes Expected Value of Sample Information (`EVSI`) given row-stochastic signal likelihoods
/// `likelihood[state * n_signals + signal] = P(Z = signal | Theta = state)`.
#[verified_engine::verified]
pub fn evsi(
    payoffs: &[f64],
    n_actions: usize,
    n_states: usize,
    prior: &[f64],
    likelihood: &[f64],
    n_signals: usize,
    workspace: &mut VoiWorkspace,
) -> Result<EvsiResult, DecisionTheoryError> {
    if n_signals == 0 {
        return Err(DecisionTheoryError::EmptyOutcomes);
    }
    let expected_lik = n_states * n_signals;
    if likelihood.len() != expected_lik {
        return Err(DecisionTheoryError::DimensionMismatch {
            expected: expected_lik,
            actual: likelihood.len(),
        });
    }
    if workspace.signal_probs.len() < n_signals
        || workspace.optimal_signal_actions.len() < n_signals
    {
        return Err(DecisionTheoryError::WorkspaceTooSmall {
            required: n_signals,
            actual: workspace.signal_probs.len(),
        });
    }

    let base = evpi(payoffs, n_actions, n_states, prior)?;

    for s in 0..n_states {
        let mut row_sum = 0.0;
        for z in 0..n_signals {
            let pz = likelihood[flatten_2d_index(z, s, n_signals)];
            if !pz.is_finite() || pz < -TOLERANCE_HIGH || pz > 1.0 + TOLERANCE_FAST {
                return Err(DecisionTheoryError::InvalidLikelihoodMatrix {
                    state: s,
                    sum: f64::NAN,
                });
            }
            row_sum += pz.max(0.0);
        }
        if (row_sum - 1.0).abs() > TOLERANCE_FAST {
            return Err(DecisionTheoryError::InvalidLikelihoodMatrix {
                state: s,
                sum: row_sum,
            });
        }
    }

    let mut sample_info_value = 0.0;
    for z in 0..n_signals {
        let mut p_z = 0.0;
        for (s, &p_s) in prior.iter().enumerate() {
            p_z += p_s.max(0.0) * likelihood[flatten_2d_index(z, s, n_signals)].max(0.0);
        }
        workspace.signal_probs[z] = p_z;

        let mut best_joint_u = f64::NEG_INFINITY;
        let mut best_action = base.baseline_action;
        for a in 0..n_actions {
            let mut joint_u = 0.0;
            for (s, &p_s) in prior.iter().enumerate() {
                let p_sz = p_s.max(0.0) * likelihood[flatten_2d_index(z, s, n_signals)].max(0.0);
                joint_u += p_sz * payoffs[flatten_2d_index(s, a, n_states)];
            }
            if joint_u > best_joint_u {
                best_joint_u = joint_u;
                best_action = a;
            }
        }
        workspace.optimal_signal_actions[z] = best_action;
        if p_z > 0.0 {
            sample_info_value += best_joint_u;
        }
    }

    Ok(EvsiResult {
        baseline_value: base.baseline_value,
        sample_info_value,
        evsi: (sample_info_value - base.baseline_value).max(0.0),
        baseline_action: base.baseline_action,
    })
}

// [cite:graph_parameters_rust]
