//! Deterministic `OxidizeRng`-seeded `UCB1` and Beta-Bernoulli Thompson Sampling bandits.

use super::DecisionTheoryError;
use math_commons::registry::TOLERANCE_HIGH;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

#[inline]
#[verified_engine::verified]
fn uniform_open01(rng: &mut OxidizeRng) -> f64 {
    // Map 53 random bits into ((0, 1)) strictly away from 0 and 1.
    let bits = (rng.next_u64() >> 11) as f64;
    (bits + 0.5) * (1.0 / ((1u64 << 53) as f64))
}

#[inline]
#[verified_engine::verified]
fn standard_normal(rng: &mut OxidizeRng) -> f64 {
    let u1 = uniform_open01(rng);
    let u2 = uniform_open01(rng);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[verified_engine::verified]
fn sample_gamma_ge_one(shape: f64, rng: &mut OxidizeRng) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    for _ in 0..256 {
        let z = standard_normal(rng);
        let one_plus_cz = 1.0 + c * z;
        if one_plus_cz <= 0.0 {
            continue;
        }
        let v = one_plus_cz * one_plus_cz * one_plus_cz;
        let u = uniform_open01(rng);
        let z2 = z * z;
        if u < 1.0 - 0.0331 * z2 * z2 {
            return d * v;
        }
        if u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
    d
}

#[verified_engine::verified]
fn sample_gamma(shape: f64, rng: &mut OxidizeRng) -> f64 {
    if shape >= 1.0 {
        sample_gamma_ge_one(shape, rng)
    } else {
        let u = uniform_open01(rng);
        sample_gamma_ge_one(shape + 1.0, rng) * u.powf(1.0 / shape)
    }
}

#[verified_engine::verified]
fn sample_beta(alpha: f64, beta: f64, rng: &mut OxidizeRng) -> f64 {
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    let denom = x + y;
    if denom <= 0.0 { 0.5 } else { x / denom }
}

/// Upper Confidence Bound (`UCB1`) multi-armed bandit with seeded tie-breaking.
#[derive(Clone, Debug)]
pub struct Ucb1Bandit {
    counts: Vec<u64>,
    values: Vec<f64>,
    total_pulls: u64,
    exploration_c: f64,
}

impl Ucb1Bandit {
    /// Creates a `UCB1` bandit over `n_arms` with exploration bonus multiplier `exploration_c >= 0`.
    pub fn new(n_arms: usize, exploration_c: f64) -> Result<Self, DecisionTheoryError> {
        if n_arms == 0 {
            return Err(DecisionTheoryError::EmptyArms);
        }
        if !exploration_c.is_finite() || exploration_c < 0.0 {
            return Err(DecisionTheoryError::InvalidExplorationParameter {
                value: exploration_c,
            });
        }
        Ok(Self {
            counts: vec![0; n_arms],
            values: vec![0.0; n_arms],
            total_pulls: 0,
            exploration_c,
        })
    }

    /// Resets all pull counts and empirical reward estimates to zero without reallocating.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.counts.fill(0);
        self.values.fill(0.0);
        self.total_pulls = 0;
    }

    /// Empirical mean rewards per arm.
    #[must_use]
    pub fn empirical_means(&self) -> &[f64] {
        &self.values
    }

    /// Pull counts per arm.
    #[must_use]
    pub fn pull_counts(&self) -> &[u64] {
        &self.counts
    }

    /// Selects the highest-UCB1 arm, breaking initial/exact ties uniformly via `rng`.
    #[verified_engine::verified]
    pub fn select_arm(&self, rng: &mut OxidizeRng) -> usize {
        let mut unvisited_ties = 0u64;
        let mut chosen_unvisited = 0usize;
        for (i, &c) in self.counts.iter().enumerate() {
            if c == 0 {
                unvisited_ties += 1;
                if rng.next_u64().is_multiple_of(unvisited_ties) {
                    chosen_unvisited = i;
                }
            }
        }
        if unvisited_ties > 0 {
            return chosen_unvisited;
        }

        let ln_n = (self.total_pulls as f64).ln();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_arm = 0usize;
        let mut ties = 0u64;
        for (i, (&c, &mean)) in self.counts.iter().zip(self.values.iter()).enumerate() {
            let score = mean + self.exploration_c * (ln_n / (c as f64)).sqrt();
            if score > best_score + TOLERANCE_HIGH {
                best_score = score;
                best_arm = i;
                ties = 1;
            } else if (score - best_score).abs() <= TOLERANCE_HIGH {
                ties += 1;
                if rng.next_u64().is_multiple_of(ties) {
                    best_arm = i;
                }
            }
        }
        best_arm
    }

    /// Updates arm `arm` with finite observed `reward`.
    #[verified_engine::verified]
    pub fn update(&mut self, arm: usize, reward: f64) -> Result<(), DecisionTheoryError> {
        if arm >= self.counts.len() {
            return Err(DecisionTheoryError::InvalidArmIndex {
                index: arm,
                arm_count: self.counts.len(),
            });
        }
        if !reward.is_finite() {
            return Err(DecisionTheoryError::InvalidReward { value: reward });
        }
        self.counts[arm] += 1;
        self.total_pulls += 1;
        let n_k = self.counts[arm] as f64;
        self.values[arm] += (reward - self.values[arm]) / n_k;
        Ok(())
    }
}

/// Conjugate Beta-Bernoulli Thompson Sampling bandit driven by [`OxidizeRng`].
#[derive(Clone, Debug)]
pub struct ThompsonBandit {
    alphas: Vec<f64>,
    betas: Vec<f64>,
    counts: Vec<u64>,
    prior_alpha: f64,
    prior_beta: f64,
}

impl ThompsonBandit {
    /// Creates a Thompson Sampling bandit with uniform `Beta(1, 1)` priors over `n_arms`.
    pub fn new(n_arms: usize) -> Result<Self, DecisionTheoryError> {
        Self::with_prior(n_arms, 1.0, 1.0)
    }

    /// Creates a Thompson Sampling bandit with symmetric `Beta(prior_alpha, prior_beta)` priors.
    pub fn with_prior(
        n_arms: usize,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> Result<Self, DecisionTheoryError> {
        if n_arms == 0 {
            return Err(DecisionTheoryError::EmptyArms);
        }
        if !prior_alpha.is_finite() || prior_alpha <= 0.0 {
            return Err(DecisionTheoryError::InvalidExplorationParameter { value: prior_alpha });
        }
        if !prior_beta.is_finite() || prior_beta <= 0.0 {
            return Err(DecisionTheoryError::InvalidExplorationParameter { value: prior_beta });
        }
        Ok(Self {
            alphas: vec![prior_alpha; n_arms],
            betas: vec![prior_beta; n_arms],
            counts: vec![0; n_arms],
            prior_alpha,
            prior_beta,
        })
    }

    /// Resets posterior parameters back to the initial Beta prior without reallocating.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.alphas.fill(self.prior_alpha);
        self.betas.fill(self.prior_beta);
        self.counts.fill(0);
    }

    /// Posterior `alpha` parameters per arm.
    #[must_use]
    pub fn alphas(&self) -> &[f64] {
        &self.alphas
    }

    /// Posterior `beta` parameters per arm.
    #[must_use]
    pub fn betas(&self) -> &[f64] {
        &self.betas
    }

    /// Pull counts per arm.
    #[must_use]
    pub fn pull_counts(&self) -> &[u64] {
        &self.counts
    }

    /// Samples $\theta_i \sim \mathrm{Beta}(\alpha_i, \beta_i)$ via `rng` and returns $\arg\max_i \theta_i$.
    #[verified_engine::verified]
    pub fn select_arm(&self, rng: &mut OxidizeRng) -> usize {
        let mut best_arm = 0usize;
        let mut best_sample = f64::NEG_INFINITY;
        for (i, (&a, &b)) in self.alphas.iter().zip(self.betas.iter()).enumerate() {
            let theta = sample_beta(a, b, rng);
            if theta > best_sample {
                best_sample = theta;
                best_arm = i;
            }
        }
        best_arm
    }

    /// Updates arm `arm` with bounded reward `reward` in `[0.0, 1.0]`.
    #[verified_engine::verified]
    pub fn update(&mut self, arm: usize, reward: f64) -> Result<(), DecisionTheoryError> {
        if arm >= self.alphas.len() {
            return Err(DecisionTheoryError::InvalidArmIndex {
                index: arm,
                arm_count: self.alphas.len(),
            });
        }
        if !reward.is_finite() || !(0.0..=1.0).contains(&reward) {
            return Err(DecisionTheoryError::InvalidReward { value: reward });
        }
        self.alphas[arm] += reward;
        self.betas[arm] += 1.0 - reward;
        self.counts[arm] += 1;
        Ok(())
    }
}

#[verified_engine::verified]
fn validate_arm_probs(arm_probs: &[f64]) -> Result<f64, DecisionTheoryError> {
    if arm_probs.is_empty() {
        return Err(DecisionTheoryError::EmptyArms);
    }
    let mut best = 0.0_f64;
    for &p in arm_probs {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(DecisionTheoryError::InvalidReward { value: p });
        }
        best = best.max(p);
    }
    Ok(best)
}

/// Simulates `UCB1` on stationary Bernoulli `arm_probs`, writing cumulative pseudo-regret into `regret_out`.
#[verified_engine::verified]
pub fn simulate_bernoulli_ucb1(
    bandit: &mut Ucb1Bandit,
    arm_probs: &[f64],
    rng: &mut OxidizeRng,
    regret_out: &mut [f64],
) -> Result<f64, DecisionTheoryError> {
    let mu_star = validate_arm_probs(arm_probs)?;
    if bandit.counts.len() != arm_probs.len() {
        return Err(DecisionTheoryError::DimensionMismatch {
            expected: bandit.counts.len(),
            actual: arm_probs.len(),
        });
    }
    bandit.reset();
    let mut cum_regret = 0.0;
    for slot in regret_out.iter_mut() {
        let arm = bandit.select_arm(rng);
        let reward = if uniform_open01(rng) < arm_probs[arm] {
            1.0
        } else {
            0.0
        };
        bandit.update(arm, reward)?;
        cum_regret += mu_star - arm_probs[arm];
        *slot = cum_regret;
    }
    Ok(cum_regret)
}

/// Simulates Thompson Sampling on stationary Bernoulli `arm_probs`, writing cumulative pseudo-regret into `regret_out`.
#[verified_engine::verified]
pub fn simulate_bernoulli_thompson(
    bandit: &mut ThompsonBandit,
    arm_probs: &[f64],
    rng: &mut OxidizeRng,
    regret_out: &mut [f64],
) -> Result<f64, DecisionTheoryError> {
    let mu_star = validate_arm_probs(arm_probs)?;
    if bandit.counts.len() != arm_probs.len() {
        return Err(DecisionTheoryError::DimensionMismatch {
            expected: bandit.counts.len(),
            actual: arm_probs.len(),
        });
    }
    bandit.reset();
    let mut cum_regret = 0.0;
    for slot in regret_out.iter_mut() {
        let arm = bandit.select_arm(rng);
        let reward = if uniform_open01(rng) < arm_probs[arm] {
            1.0
        } else {
            0.0
        };
        bandit.update(arm, reward)?;
        cum_regret += mu_star - arm_probs[arm];
        *slot = cum_regret;
    }
    Ok(cum_regret)
}

// [cite:graph_parameters_rust]
