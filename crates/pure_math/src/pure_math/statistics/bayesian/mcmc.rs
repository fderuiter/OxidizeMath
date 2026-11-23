use super::intervals_and_bf::{CredibleInterval, validate_credible_level};
use crate::error::BayesianError;
use crate::pure_math::statistics::distributions::special::{
    sample_standard_normal, sample_uniform_open,
};
use oxidize_core::rng::OxidizeRng;

/// Posterior MCMC summary diagnostics computed over a pre-allocated chain buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McmcSummary {
    /// Empirical posterior mean.
    pub mean: f64,
    /// Empirical posterior sample variance.
    pub variance: f64,
    /// Proposal acceptance rate in `[0, 1]`.
    pub acceptance_rate: f64,
    /// Effective Sample Size (ESS) estimated via truncated positive autocorrelation sums.
    pub effective_sample_size: f64,
}

/// Zero-allocation univariate Random-Walk Metropolis-Hastings sampler.
///
/// Steps in-place and writes retained samples into caller-supplied slices without touching the
/// heap allocator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetropolisHastings1D {
    current: f64,
    current_log_prob: Option<f64>,
    proposal_std: f64,
    accepted: u64,
    total_steps: u64,
}

impl MetropolisHastings1D {
    /// Constructs a 1D Metropolis-Hastings sampler starting at `initial` with Gaussian random-walk
    /// standard deviation `proposal_std > 0`.
    #[verified_engine::verified]
    pub fn new(initial: f64, proposal_std: f64) -> Result<Self, BayesianError> {
        if !initial.is_finite() {
            return Err(BayesianError::InvalidHyperparameter {
                name: "initial",
                value: initial,
            });
        }
        if !proposal_std.is_finite() || proposal_std <= 0.0 {
            return Err(BayesianError::InvalidProposalScale {
                scale: proposal_std,
            });
        }
        Ok(Self {
            current: initial,
            current_log_prob: None,
            proposal_std,
            accepted: 0,
            total_steps: 0,
        })
    }

    /// Resets the Markov chain to `initial` and clears acceptance counters.
    #[verified_engine::verified]
    pub fn reset(&mut self, initial: f64) {
        self.current = initial;
        self.current_log_prob = None;
        self.accepted = 0;
        self.total_steps = 0;
    }

    /// Returns the current chain state.
    #[verified_engine::verified]
    pub fn state(&self) -> f64 {
        self.current
    }

    /// Returns proposal standard deviation.
    #[verified_engine::verified]
    pub fn proposal_std(&self) -> f64 {
        self.proposal_std
    }

    /// Updates proposal standard deviation if positive and finite.
    #[verified_engine::verified]
    pub fn set_proposal_std(&mut self, proposal_std: f64) -> Result<(), BayesianError> {
        if !proposal_std.is_finite() || proposal_std <= 0.0 {
            return Err(BayesianError::InvalidProposalScale {
                scale: proposal_std,
            });
        }
        self.proposal_std = proposal_std;
        Ok(())
    }

    /// Returns cumulative proposal acceptance ratio in `[0, 1]`.
    #[verified_engine::verified]
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.accepted as f64) / (self.total_steps as f64)
        }
    }

    /// Returns total MCMC transitions executed since initialization or reset.
    #[verified_engine::verified]
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Advances the chain by one Metropolis-Hastings step against unnormalized `log_target`.
    #[verified_engine::verified]
    pub fn step<F>(&mut self, log_target: F, rng: &mut OxidizeRng) -> Result<f64, BayesianError>
    where
        F: Fn(f64) -> f64,
    {
        let lp_curr = match self.current_log_prob {
            Some(lp) => lp,
            None => {
                let init_lp = log_target(self.current);
                if init_lp.is_nan() || init_lp == f64::INFINITY {
                    return Err(BayesianError::NumericalInstability {
                        reason: "initial state has invalid log-target density",
                    });
                }
                self.current_log_prob = Some(init_lp);
                init_lp
            }
        };

        let candidate = self.current + self.proposal_std * sample_standard_normal(rng);
        let lp_cand = log_target(candidate);
        self.total_steps += 1;

        if lp_cand.is_finite() {
            let log_alpha = lp_cand - lp_curr;
            if log_alpha >= 0.0 || sample_uniform_open(rng).ln() < log_alpha {
                self.current = candidate;
                self.current_log_prob = Some(lp_cand);
                self.accepted += 1;
            }
        }
        Ok(self.current)
    }

    /// Executes `burn_in` warmup transitions and populates pre-allocated `out_chain` with zero heap allocations.
    #[verified_engine::verified]
    pub fn sample_into<F>(
        &mut self,
        log_target: F,
        burn_in: usize,
        out_chain: &mut [f64],
        rng: &mut OxidizeRng,
    ) -> Result<McmcSummary, BayesianError>
    where
        F: Fn(f64) -> f64,
    {
        if out_chain.len() < 2 {
            return Err(BayesianError::InvalidBufferLength {
                expected: 2,
                actual: out_chain.len(),
            });
        }
        for _ in 0..burn_in {
            self.step(&log_target, rng)?;
        }
        self.accepted = 0;
        self.total_steps = 0;
        for slot in out_chain.iter_mut() {
            *slot = self.step(&log_target, rng)?;
        }
        Ok(summarize_chain(out_chain, self.acceptance_rate()))
    }
}

/// Computes an empirical equal-tailed credible interval from `chain` using caller-supplied `scratch`
/// buffer (same length as `chain`) with zero heap allocations.
#[verified_engine::verified]
pub fn empirical_credible_interval(
    chain: &[f64],
    scratch: &mut [f64],
    level: f64,
) -> Result<CredibleInterval, BayesianError> {
    validate_credible_level(level)?;
    if chain.len() < 2 || scratch.len() < chain.len() {
        return Err(BayesianError::InvalidBufferLength {
            expected: chain.len().max(2),
            actual: scratch.len().min(chain.len()),
        });
    }
    let n = chain.len();
    let work = &mut scratch[..n];
    work.copy_from_slice(chain);
    work.sort_unstable_by(f64::total_cmp);
    let tail = 0.5 * (1.0 - level);
    let lo_idx = ((tail * ((n - 1) as f64)).floor() as usize).min(n - 1);
    let hi_idx = (((1.0 - tail) * ((n - 1) as f64)).ceil() as usize).min(n - 1);
    Ok(CredibleInterval {
        lower: work[lo_idx],
        upper: work[hi_idx],
        level,
    })
}

fn summarize_chain(chain: &[f64], acceptance_rate: f64) -> McmcSummary {
    let n = chain.len() as f64;
    let mean = chain.iter().sum::<f64>() / n;
    let var = chain
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    let ess = if var <= 1e-30 {
        1.0
    } else {
        let max_lag = chain.len().min(64);
        let mut rho_sum = 0.0;
        for lag in 1..max_lag {
            let cov: f64 = chain[..chain.len() - lag]
                .iter()
                .zip(&chain[lag..])
                .map(|(&a, &b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / n;
            let rho = cov / var;
            if rho <= 0.0 {
                break;
            }
            rho_sum += rho;
        }
        (n / (1.0 + 2.0 * rho_sum)).clamp(1.0, n)
    };
    McmcSummary {
        mean,
        variance: var,
        acceptance_rate,
        effective_sample_size: ess,
    }
}

/// Zero-allocation $D$-dimensional Random-Walk Metropolis-Hastings sampler on `[f64; D]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetropolisHastingsND<const D: usize> {
    current: [f64; D],
    current_log_prob: Option<f64>,
    proposal_scales: [f64; D],
    accepted: u64,
    total_steps: u64,
}

impl<const D: usize> MetropolisHastingsND<D> {
    /// Constructs a $D$-dimensional sampler ($D \ge 1$) with coordinate proposal standard deviations.
    #[verified_engine::verified]
    pub fn new(initial: [f64; D], proposal_scales: [f64; D]) -> Result<Self, BayesianError> {
        if D == 0 {
            return Err(BayesianError::InvalidHyperparameter {
                name: "dimension_d",
                value: 0.0,
            });
        }
        for &v in &initial {
            if !v.is_finite() {
                return Err(BayesianError::InvalidHyperparameter {
                    name: "initial",
                    value: v,
                });
            }
        }
        for &s in &proposal_scales {
            if !s.is_finite() || s <= 0.0 {
                return Err(BayesianError::InvalidProposalScale { scale: s });
            }
        }
        Ok(Self {
            current: initial,
            current_log_prob: None,
            proposal_scales,
            accepted: 0,
            total_steps: 0,
        })
    }

    /// Returns current $D$-dimensional state.
    #[verified_engine::verified]
    pub fn state(&self) -> &[f64; D] {
        &self.current
    }

    /// Returns cumulative acceptance rate in `[0, 1]`.
    #[verified_engine::verified]
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_steps == 0 {
            0.0
        } else {
            (self.accepted as f64) / (self.total_steps as f64)
        }
    }

    /// Advances the $D$-dimensional chain by one transition with zero heap allocations.
    #[verified_engine::verified]
    pub fn step<F>(
        &mut self,
        log_target: F,
        rng: &mut OxidizeRng,
    ) -> Result<[f64; D], BayesianError>
    where
        F: Fn(&[f64; D]) -> f64,
    {
        let lp_curr = match self.current_log_prob {
            Some(lp) => lp,
            None => {
                let init_lp = log_target(&self.current);
                if init_lp.is_nan() || init_lp == f64::INFINITY {
                    return Err(BayesianError::NumericalInstability {
                        reason: "initial ND state has invalid log-target density",
                    });
                }
                self.current_log_prob = Some(init_lp);
                init_lp
            }
        };
        let mut candidate = self.current;
        for j in 0..D {
            candidate[j] += self.proposal_scales[j] * sample_standard_normal(rng);
        }
        let lp_cand = log_target(&candidate);
        self.total_steps += 1;
        if lp_cand.is_finite() {
            let log_alpha = lp_cand - lp_curr;
            if log_alpha >= 0.0 || sample_uniform_open(rng).ln() < log_alpha {
                self.current = candidate;
                self.current_log_prob = Some(lp_cand);
                self.accepted += 1;
            }
        }
        Ok(self.current)
    }

    /// Runs `burn_in` transitions and fills `out_chain` with zero heap allocations, returning coordinate means.
    #[verified_engine::verified]
    pub fn sample_into<F>(
        &mut self,
        log_target: F,
        burn_in: usize,
        out_chain: &mut [[f64; D]],
        rng: &mut OxidizeRng,
    ) -> Result<[f64; D], BayesianError>
    where
        F: Fn(&[f64; D]) -> f64,
    {
        if out_chain.is_empty() {
            return Err(BayesianError::InvalidBufferLength {
                expected: 1,
                actual: 0,
            });
        }
        for _ in 0..burn_in {
            self.step(&log_target, rng)?;
        }
        self.accepted = 0;
        self.total_steps = 0;
        let mut sums = [0.0_f64; D];
        for slot in out_chain.iter_mut() {
            let draw = self.step(&log_target, rng)?;
            *slot = draw;
            for j in 0..D {
                sums[j] += draw[j];
            }
        }
        let inv_n = 1.0 / (out_chain.len() as f64);
        for s in &mut sums {
            *s *= inv_n;
        }
        Ok(sums)
    }
}

// [cite:clinical_trials_statistics]
