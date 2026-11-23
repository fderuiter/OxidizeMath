//! Zero-allocation [`OxidizeRng`] ensembles and `(p_A, epsilon)` phase-wedge scanners.

use super::{
    MAX_MODULUS, ParrondoGame, ParrondoParams, ParrondoSchedule, Result, solve_parrondo_ratchet,
};
use crate::error::MarkovError;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;

const INV_TWO_POW_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Preallocated zero-allocation Monte Carlo ensemble workspace driven by [`OxidizeRng`].
#[derive(Debug, Clone)]
pub struct ParrondoEnsemble {
    wealth: Vec<i64>,
    mean_trajectory: Vec<f64>,
    occupancy_counts: [u64; MAX_MODULUS],
    steps_recorded: usize,
}

impl ParrondoEnsemble {
    /// Allocates an ensemble workspace for `num_walkers` and up to `max_steps` steps.
    ///
    /// # Errors
    ///
    /// Returns [`MarkovError::InvalidState`] when `num_walkers == 0`.
    pub fn with_capacity(num_walkers: usize, max_steps: usize) -> Result<Self> {
        if num_walkers == 0 {
            return Err(MarkovError::InvalidState {
                reason: "Ensemble num_walkers must be at least 1".to_string(),
            });
        }
        Ok(Self {
            wealth: vec![0_i64; num_walkers],
            mean_trajectory: vec![0.0_f64; max_steps.saturating_add(1)],
            occupancy_counts: [0_u64; MAX_MODULUS],
            steps_recorded: 0,
        })
    }

    /// Resets walker wealth, trajectory history, and residue occupancy in place with zero allocations.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.wealth.fill(0);
        self.occupancy_counts = [0_u64; MAX_MODULUS];
        if !self.mean_trajectory.is_empty() {
            self.mean_trajectory[0] = 0.0;
        }
        self.steps_recorded = 0;
    }

    /// Advances all walkers by one step of `game` without heap allocations and returns mean wealth.
    #[verified_engine::verified]
    pub fn step_game(
        &mut self,
        params: &ParrondoParams,
        game: ParrondoGame,
        rng: &mut OxidizeRng,
    ) -> f64 {
        let m_i64 = params.modulus() as i64;
        let mut sum_wealth: i64 = 0;
        for w in &mut self.wealth {
            let residue = w.rem_euclid(m_i64) as usize;
            self.occupancy_counts[residue] = self.occupancy_counts[residue].saturating_add(1);
            let p_win = params.win_probability(game, residue);
            let u = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
            *w += if u < p_win { 1 } else { -1 };
            sum_wealth += *w;
        }
        self.record_mean_step(sum_wealth)
    }

    /// Advances all walkers by one 50/50 random-mixture step without heap allocations.
    #[verified_engine::verified]
    pub fn step_mixture(&mut self, params: &ParrondoParams, rng: &mut OxidizeRng) -> f64 {
        let m_i64 = params.modulus() as i64;
        let mut sum_wealth: i64 = 0;
        for w in &mut self.wealth {
            let residue = w.rem_euclid(m_i64) as usize;
            self.occupancy_counts[residue] = self.occupancy_counts[residue].saturating_add(1);
            let game = if (rng.next_u64() & 1) == 0 {
                ParrondoGame::GameA
            } else {
                ParrondoGame::GameB
            };
            let p_win = params.win_probability(game, residue);
            let u = ((rng.next_u64() >> 11) as f64) * INV_TWO_POW_53;
            *w += if u < p_win { 1 } else { -1 };
            sum_wealth += *w;
        }
        self.record_mean_step(sum_wealth)
    }

    #[verified_engine::verified]
    fn record_mean_step(&mut self, sum_wealth: i64) -> f64 {
        let mean_w = (sum_wealth as f64) / (self.wealth.len() as f64);
        let next_idx = self.steps_recorded.saturating_add(1);
        if next_idx < self.mean_trajectory.len() {
            self.mean_trajectory[next_idx] = mean_w;
            self.steps_recorded = next_idx;
        }
        mean_w
    }

    /// Resets and simulates `num_steps` of `schedule` in place with zero heap allocations.
    #[verified_engine::verified]
    pub fn simulate_schedule(
        &mut self,
        params: &ParrondoParams,
        schedule: &ParrondoSchedule,
        num_steps: usize,
        rng: &mut OxidizeRng,
    ) -> f64 {
        self.reset();
        let steps = schedule.steps();
        let capped = num_steps.min(self.mean_trajectory.len().saturating_sub(1));
        let mut last_mean = 0.0;
        for t in 0..capped {
            last_mean = self.step_game(params, steps[t % steps.len()], rng);
        }
        last_mean
    }

    /// Resets and simulates `num_steps` of pure `game` in place with zero heap allocations.
    #[verified_engine::verified]
    pub fn simulate_pure_game(
        &mut self,
        params: &ParrondoParams,
        game: ParrondoGame,
        num_steps: usize,
        rng: &mut OxidizeRng,
    ) -> f64 {
        self.reset();
        let capped = num_steps.min(self.mean_trajectory.len().saturating_sub(1));
        let mut last_mean = 0.0;
        for _ in 0..capped {
            last_mean = self.step_game(params, game, rng);
        }
        last_mean
    }

    /// Resets and simulates `num_steps` of the 50/50 random mixture in place with zero allocations.
    #[verified_engine::verified]
    pub fn simulate_mixture(
        &mut self,
        params: &ParrondoParams,
        num_steps: usize,
        rng: &mut OxidizeRng,
    ) -> f64 {
        self.reset();
        let capped = num_steps.min(self.mean_trajectory.len().saturating_sub(1));
        let mut last_mean = 0.0;
        for _ in 0..capped {
            last_mean = self.step_mixture(params, rng);
        }
        last_mean
    }

    /// Returns the recorded mean wealth trajectory slice `0..=steps_recorded`.
    #[must_use]
    #[verified_engine::verified]
    pub fn mean_trajectory(&self) -> &[f64] {
        &self.mean_trajectory[..=self.steps_recorded]
    }

    /// Returns walker wealth slice for inspection.
    #[must_use]
    #[verified_engine::verified]
    pub fn wealth(&self) -> &[i64] {
        &self.wealth
    }

    /// Returns empirical residue visit counts for `0..modulus`.
    #[must_use]
    #[verified_engine::verified]
    pub fn occupancy_counts(&self, modulus: usize) -> &[u64] {
        &self.occupancy_counts[..modulus.min(MAX_MODULUS)]
    }
}

/// Grid configuration for scanning the `(p_A, epsilon)` Parrondo paradox region.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParrondoPhaseGridConfig {
    /// Modulus $M \in \{3, 5, 7, 9\}$.
    pub modulus: usize,
    /// Unperturbed Game B trap win probability $p_{B0,0}$.
    pub p_b0_base: f64,
    /// Unperturbed Game B non-trap win probability $p_{B1,0}$.
    pub p_b1_base: f64,
    /// Minimum effective Game A win probability $p_A$.
    pub p_a_min: f64,
    /// Maximum effective Game A win probability $p_A$.
    pub p_a_max: f64,
    /// Minimum bias parameter $\epsilon$.
    pub epsilon_min: f64,
    /// Maximum bias parameter $\epsilon$.
    pub epsilon_max: f64,
    /// Horizontal sample count along $p_A$.
    pub p_a_steps: usize,
    /// Vertical sample count along $\epsilon$.
    pub epsilon_steps: usize,
}

impl Default for ParrondoPhaseGridConfig {
    fn default() -> Self {
        Self {
            modulus: 3,
            p_b0_base: 0.1,
            p_b1_base: 0.75,
            p_a_min: 0.42,
            p_a_max: 0.51,
            epsilon_min: -0.005,
            epsilon_max: 0.045,
            p_a_steps: 25,
            epsilon_steps: 21,
        }
    }
}

/// Evaluated drift sample at a single `(p_A, epsilon)` phase-plane coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParrondoPhasePoint {
    /// Effective Game A win probability $p_A$.
    pub p_a: f64,
    /// Perturbation $\epsilon$ applied to Game B branches.
    pub epsilon: f64,
    /// Expected step drift $E[\Delta W_A] = 2 p_A - 1$.
    pub gain_a: f64,
    /// Expected step drift $E[\Delta W_B]$ under pure Game B.
    pub gain_b: f64,
    /// Expected step drift $E[\Delta W_{\text{sched}}]$ under the periodic schedule.
    pub gain_schedule: f64,
    /// `true` when $E[\Delta W_A] < 0$, $E[\Delta W_B] < 0$, and $E[\Delta W_{\text{sched}}] > 0$.
    pub is_paradox: bool,
}

/// Scans the `(p_A, epsilon)` plane to map the Parrondo paradox wedge.
///
/// # Errors
///
/// Returns [`MarkovError`] if `config` bounds or grid dimensions are invalid.
#[verified_engine::verified]
pub fn scan_parrondo_phase_region(
    config: &ParrondoPhaseGridConfig,
    schedule: &ParrondoSchedule,
) -> Result<Vec<ParrondoPhasePoint>> {
    if config.p_a_steps < 2 || config.epsilon_steps < 2 {
        return Err(MarkovError::InvalidState {
            reason: "Phase grid requires at least 2x2 samples".to_string(),
        });
    }
    if !config.p_a_min.is_finite()
        || !config.p_a_max.is_finite()
        || config.p_a_min >= config.p_a_max
        || !config.epsilon_min.is_finite()
        || !config.epsilon_max.is_finite()
        || config.epsilon_min >= config.epsilon_max
    {
        return Err(MarkovError::InvalidState {
            reason: "Phase grid ranges must be finite and strictly ordered".to_string(),
        });
    }
    let mut points = Vec::with_capacity(config.p_a_steps * config.epsilon_steps);
    for iy in 0..config.epsilon_steps {
        let fy = (iy as f64) / ((config.epsilon_steps - 1) as f64);
        let eps = config.epsilon_min + fy * (config.epsilon_max - config.epsilon_min);
        for ix in 0..config.p_a_steps {
            let fx = (ix as f64) / ((config.p_a_steps - 1) as f64);
            let p_a = config.p_a_min + fx * (config.p_a_max - config.p_a_min);
            if let Ok(params) = ParrondoParams::new(
                config.modulus,
                p_a + eps,
                config.p_b0_base,
                config.p_b1_base,
                eps,
            ) && let Ok(report) = solve_parrondo_ratchet(&params, schedule)
            {
                points.push(ParrondoPhasePoint {
                    p_a,
                    epsilon: eps,
                    gain_a: report.gain_a,
                    gain_b: report.gain_b,
                    gain_schedule: report.gain_schedule,
                    is_paradox: report.is_paradox(),
                });
            }
        }
    }
    Ok(points)
}

// [cite:clinical_trials_statistics]
