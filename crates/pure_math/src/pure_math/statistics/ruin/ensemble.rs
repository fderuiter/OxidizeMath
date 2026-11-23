//! Preallocated zero-allocation [`OxidizeRng`] Monte Carlo bankroll ensemble workspace.

use super::analytical::PortfolioBet;
use super::barrier::ProportionalPortfolioBet;
use crate::error::KellyError;
use crate::pure_math::statistics::validation::Z_95_TWO_SIDED;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use statrs::distribution::{ContinuousCDF, Normal};

const INV_TWO_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Wilson score binomial proportion confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WilsonInterval {
    /// Empirical proportion $\hat{p} = k / n$.
    pub estimate: f64,
    /// Lower confidence bound in $[0, 1]$.
    pub lower: f64,
    /// Upper confidence bound in $[0, 1]$.
    pub upper: f64,
}

impl WilsonInterval {
    /// Full interval width (`upper - lower`).
    #[must_use]
    #[verified_engine::verified]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

/// Realized peak-to-trough maximum drawdown quantiles across simulated ensemble paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawdownQuantiles {
    /// Median (50th percentile) pathwise maximum drawdown in $[0, 1]$.
    pub p50: f64,
    /// 90th percentile pathwise maximum drawdown in $[0, 1]$.
    pub p90: f64,
    /// 95th percentile pathwise maximum drawdown in $[0, 1]$.
    pub p95: f64,
    /// 99th percentile pathwise maximum drawdown in $[0, 1]$.
    pub p99: f64,
}

/// Finite-horizon solvency and drawdown summary produced by [`BankrollEnsembleWorkspace`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnsembleSolvencyReport {
    /// Number of Monte Carlo paths simulated.
    pub num_paths: usize,
    /// Finite wager horizon simulated.
    pub num_steps: usize,
    /// Number of paths that hit or crossed below `barrier_floor`.
    pub ruin_count: usize,
    /// Empirical finite-horizon barrier/ruin probability (`ruin_count / num_paths`).
    pub ruin_probability: f64,
    /// 95% Wilson score confidence interval around `ruin_probability`.
    pub wilson_95: WilsonInterval,
    /// Median terminal wealth across all paths at step `num_steps`.
    pub terminal_median_wealth: f64,
    /// Arithmetic mean terminal wealth across all paths at step `num_steps`.
    pub terminal_mean_wealth: f64,
    /// Empirical peak-to-trough maximum drawdown quantiles.
    pub drawdown_quantiles: DrawdownQuantiles,
}

fn wilson_from_z(successes: usize, trials: usize, z: f64) -> Result<WilsonInterval, KellyError> {
    if trials == 0 {
        return Err(KellyError::InsufficientBets {
            required: 1,
            actual: 0,
        });
    }
    if successes > trials {
        return Err(KellyError::DimensionMismatch {
            expected: trials,
            actual: successes,
        });
    }
    let nf = trials as f64;
    let p_hat = (successes as f64) / nf;
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let center = (p_hat + z2 / (2.0 * nf)) / denom;
    let rad = (p_hat * (1.0 - p_hat) / nf + z2 / (4.0 * nf * nf))
        .max(0.0)
        .sqrt();
    let half = (z / denom) * rad;
    Ok(WilsonInterval {
        estimate: p_hat,
        lower: (center - half).clamp(0.0, 1.0),
        upper: (center + half).clamp(0.0, 1.0),
    })
}

/// Computes the 95% Wilson score interval for `successes` out of `trials` with zero allocations.
#[verified_engine::verified]
pub fn wilson_interval_95(successes: usize, trials: usize) -> Result<WilsonInterval, KellyError> {
    wilson_from_z(successes, trials, Z_95_TWO_SIDED)
}

/// Computes a Wilson score interval at arbitrary `confidence` level in $(0, 1)$.
#[verified_engine::verified]
pub fn wilson_score_interval(
    successes: usize,
    trials: usize,
    confidence: f64,
) -> Result<WilsonInterval, KellyError> {
    if !confidence.is_finite() || confidence <= 0.0 || confidence >= 1.0 {
        return Err(KellyError::InvalidProbability { value: confidence });
    }
    let normal = Normal::new(0.0, 1.0).map_err(|_| KellyError::ConvergenceFailed {
        reason: "standard normal initialization failed",
    })?;
    let z = normal.inverse_cdf(0.5 * (1.0 + confidence));
    wilson_from_z(successes, trials, z)
}

fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn validate_wealth_and_barrier(initial_wealth: f64, barrier_floor: f64) -> Result<(), KellyError> {
    if !initial_wealth.is_finite() || initial_wealth <= 0.0 {
        return Err(KellyError::InvalidBankroll {
            value: initial_wealth,
        });
    }
    if !barrier_floor.is_finite() || barrier_floor < 0.0 || barrier_floor >= initial_wealth {
        return Err(KellyError::InvalidBarrier {
            floor: barrier_floor,
            initial_wealth,
        });
    }
    Ok(())
}

/// Preallocated zero-allocation Monte Carlo bankroll ensemble workspace driven by [`OxidizeRng`].
///
/// Modeled on [`crate::pure_math::statistics::markov::parrondo::ParrondoEnsemble`]: all walker
/// wealth, peak, drawdown, barrier-hit, trajectory, and selection buffers are allocated once in
/// [`Self::with_capacity`] and reused across simulations without heap allocations.
#[derive(Debug, Clone)]
pub struct BankrollEnsembleWorkspace {
    wealth: Vec<f64>,
    peak_wealth: Vec<f64>,
    max_drawdown: Vec<f64>,
    hit_barrier: Vec<u8>,
    median_trajectory: Vec<f64>,
    scratch: Vec<f64>,
    steps_recorded: usize,
}

impl BankrollEnsembleWorkspace {
    /// Allocates buffers for `num_paths` walkers and optional per-step median history up to `max_steps`.
    pub fn with_capacity(num_paths: usize, max_steps: usize) -> Result<Self, KellyError> {
        if num_paths == 0 {
            return Err(KellyError::InsufficientBets {
                required: 1,
                actual: 0,
            });
        }
        let traj_len = if max_steps > 0 {
            max_steps.saturating_add(1)
        } else {
            0
        };
        Ok(Self {
            wealth: vec![0.0; num_paths],
            peak_wealth: vec![0.0; num_paths],
            max_drawdown: vec![0.0; num_paths],
            hit_barrier: vec![0; num_paths],
            median_trajectory: vec![0.0; traj_len],
            scratch: vec![0.0; num_paths],
            steps_recorded: 0,
        })
    }

    /// Resets walker wealth, peaks, drawdowns, and barrier flags in place with zero allocations.
    #[verified_engine::verified]
    pub fn reset(&mut self, initial_wealth: f64) {
        self.wealth.fill(initial_wealth);
        self.peak_wealth.fill(initial_wealth);
        self.max_drawdown.fill(0.0);
        self.hit_barrier.fill(0);
        if !self.median_trajectory.is_empty() {
            self.median_trajectory[0] = initial_wealth;
        }
        self.steps_recorded = 0;
    }

    fn record_walker_wealth(&mut self, idx: usize, new_wealth: f64, barrier_floor: f64) {
        let w = new_wealth.max(0.0);
        self.wealth[idx] = w;
        if w > self.peak_wealth[idx] {
            self.peak_wealth[idx] = w;
        }
        let peak = self.peak_wealth[idx];
        if peak > 0.0 {
            let dd = (peak - w) / peak;
            if dd > self.max_drawdown[idx] {
                self.max_drawdown[idx] = dd;
            }
        }
        if w <= barrier_floor {
            self.hit_barrier[idx] = 1;
        }
    }

    fn fast_median_from_wealth(&mut self) -> f64 {
        let n = self.wealth.len();
        self.scratch.copy_from_slice(&self.wealth);
        let mid = n / 2;
        let (left, mid_val, _) = self.scratch.select_nth_unstable_by(mid, f64::total_cmp);
        if n % 2 == 1 {
            *mid_val
        } else {
            let max_left = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            0.5 * (max_left + *mid_val)
        }
    }

    fn finalize_report(&mut self, num_steps: usize) -> Result<EnsembleSolvencyReport, KellyError> {
        let n = self.wealth.len();
        let ruin_count = self.hit_barrier.iter().filter(|&&h| h != 0).count();
        let terminal_mean_wealth = self.wealth.iter().sum::<f64>() / (n as f64);
        let terminal_median_wealth = self.fast_median_from_wealth();

        self.scratch.copy_from_slice(&self.max_drawdown);
        self.scratch.sort_unstable_by(f64::total_cmp);
        let drawdown_quantiles = DrawdownQuantiles {
            p50: sorted_quantile(&self.scratch, 0.50),
            p90: sorted_quantile(&self.scratch, 0.90),
            p95: sorted_quantile(&self.scratch, 0.95),
            p99: sorted_quantile(&self.scratch, 0.99),
        };
        let wilson_95 = wilson_interval_95(ruin_count, n)?;

        Ok(EnsembleSolvencyReport {
            num_paths: n,
            num_steps,
            ruin_count,
            ruin_probability: wilson_95.estimate,
            wilson_95,
            terminal_median_wealth,
            terminal_mean_wealth,
            drawdown_quantiles,
        })
    }

    /// Simulates `num_steps` of proportional-Kelly wagering with zero heap allocations.
    #[verified_engine::verified]
    pub fn simulate_proportional(
        &mut self,
        initial_wealth: f64,
        barrier_floor: f64,
        bet: &ProportionalPortfolioBet,
        num_steps: usize,
        rng: &mut OxidizeRng,
    ) -> Result<EnsembleSolvencyReport, KellyError> {
        validate_wealth_and_barrier(initial_wealth, barrier_floor)?;
        self.reset(initial_wealth);
        let p = bet.win_prob.value();
        let f = bet.fraction.value();
        let win_mult = 1.0 + f * bet.odds.net_profit_multiplier();
        let loss_mult = (1.0 - f).max(0.0);
        let record_traj =
            self.median_trajectory.len() > num_steps && !self.median_trajectory.is_empty();

        for step in 1..=num_steps {
            for idx in 0..self.wealth.len() {
                let u = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
                let w = self.wealth[idx] * if u < p { win_mult } else { loss_mult };
                self.record_walker_wealth(idx, w, barrier_floor);
            }
            if record_traj {
                self.median_trajectory[step] = self.fast_median_from_wealth();
                self.steps_recorded = step;
            }
        }
        self.finalize_report(num_steps)
    }

    /// Simulates `num_steps` across a heterogeneous proportional-Kelly portfolio with zero allocations.
    #[verified_engine::verified]
    pub fn simulate_proportional_portfolio(
        &mut self,
        initial_wealth: f64,
        barrier_floor: f64,
        portfolio: &[ProportionalPortfolioBet],
        num_steps: usize,
        rng: &mut OxidizeRng,
    ) -> Result<EnsembleSolvencyReport, KellyError> {
        if portfolio.is_empty() {
            return Err(KellyError::InsufficientBets {
                required: 1,
                actual: 0,
            });
        }
        validate_wealth_and_barrier(initial_wealth, barrier_floor)?;
        self.reset(initial_wealth);
        let total_w: f64 = portfolio.iter().map(|b| b.weight).sum();

        for _ in 0..num_steps {
            for idx in 0..self.wealth.len() {
                let u_pick = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53 * total_w;
                let bet = pick_weighted_prop_bet(portfolio, u_pick);
                let u_win = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
                let mult = if u_win < bet.win_prob.value() {
                    1.0 + bet.fraction.value() * bet.odds.net_profit_multiplier()
                } else {
                    (1.0 - bet.fraction.value()).max(0.0)
                };
                self.record_walker_wealth(idx, self.wealth[idx] * mult, barrier_floor);
            }
        }
        self.finalize_report(num_steps)
    }

    /// Simulates `num_steps` of fixed-stake portfolio wagering with absorbing lower/upper barriers.
    #[verified_engine::verified]
    pub fn simulate_fixed_stake_portfolio(
        &mut self,
        initial_wealth: f64,
        barrier_floor: f64,
        upper_target: Option<f64>,
        portfolio: &[PortfolioBet],
        num_steps: usize,
        rng: &mut OxidizeRng,
    ) -> Result<EnsembleSolvencyReport, KellyError> {
        if portfolio.is_empty() {
            return Err(KellyError::InsufficientBets {
                required: 1,
                actual: 0,
            });
        }
        validate_wealth_and_barrier(initial_wealth, barrier_floor)?;
        if let Some(target) = upper_target
            && (!target.is_finite() || target <= initial_wealth)
        {
            return Err(KellyError::InvalidBarrier {
                floor: target,
                initial_wealth,
            });
        }

        self.reset(initial_wealth);
        let total_w: f64 = portfolio.iter().map(|b| b.weight).sum();
        for _ in 0..num_steps {
            for idx in 0..self.wealth.len() {
                if self.hit_barrier[idx] != 0
                    || upper_target.is_some_and(|tgt| self.wealth[idx] >= tgt)
                {
                    continue;
                }
                let u_pick = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53 * total_w;
                let bet = pick_weighted_fixed_bet(portfolio, u_pick);
                let u_win = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
                let delta = if u_win < bet.win_prob.value() {
                    bet.stake * bet.odds.net_profit_multiplier()
                } else {
                    -bet.stake
                };
                self.record_walker_wealth(idx, self.wealth[idx] + delta, barrier_floor);
            }
        }
        self.finalize_report(num_steps)
    }

    /// Computes an arbitrary pathwise max-drawdown quantile $q \in [0, 1]$ with zero allocations.
    #[verified_engine::verified]
    pub fn drawdown_quantile(&mut self, q: f64) -> Result<f64, KellyError> {
        if !q.is_finite() || !(0.0..=1.0).contains(&q) {
            return Err(KellyError::InvalidProbability { value: q });
        }
        self.scratch.copy_from_slice(&self.max_drawdown);
        self.scratch.sort_unstable_by(f64::total_cmp);
        Ok(sorted_quantile(&self.scratch, q))
    }

    /// Returns the recorded step-by-step median wealth slice `0..=steps_recorded`.
    #[must_use]
    #[verified_engine::verified]
    pub fn median_trajectory(&self) -> &[f64] {
        if self.median_trajectory.is_empty() {
            &[]
        } else {
            &self.median_trajectory[..=self.steps_recorded]
        }
    }

    /// Returns terminal wealths across all walkers.
    #[must_use]
    #[verified_engine::verified]
    pub fn terminal_wealths(&self) -> &[f64] {
        &self.wealth
    }

    /// Returns realized pathwise maximum drawdowns across all walkers.
    #[must_use]
    #[verified_engine::verified]
    pub fn max_drawdowns(&self) -> &[f64] {
        &self.max_drawdown
    }
}

fn pick_weighted_prop_bet(
    portfolio: &[ProportionalPortfolioBet],
    target_w: f64,
) -> &ProportionalPortfolioBet {
    let mut acc = 0.0;
    for item in portfolio {
        acc += item.weight;
        if target_w <= acc {
            return item;
        }
    }
    &portfolio[portfolio.len() - 1]
}

fn pick_weighted_fixed_bet(portfolio: &[PortfolioBet], target_w: f64) -> &PortfolioBet {
    let mut acc = 0.0;
    for item in portfolio {
        acc += item.weight;
        if target_w <= acc {
            return item;
        }
    }
    &portfolio[portfolio.len() - 1]
}

// [cite:clinical_trials_statistics]
