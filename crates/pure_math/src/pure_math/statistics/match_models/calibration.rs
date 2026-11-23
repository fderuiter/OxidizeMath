//! Time-decayed Maximum Likelihood Estimation (`fit_dixon_coles_mle`) for the Dixon-Coles (1997)
//! league model with sum-to-zero attack identifiability and `optimization.rs`-compatible updates.

use super::dixon_coles::{DixonColesParams, dixon_coles_grid, dixon_coles_tau};
use super::grid::ScoreGrid;
use super::poisson::poisson_log_pmf;
use crate::error::MatchModelError;
use crate::pure_math::analysis::optimization::{Adam, Optimizer};
use nalgebra::DVector;
use serde::{Deserialize, Deserializer, Serialize};

/// Historical match record used to calibrate team attack, defense, home-advantage, and `rho`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct HistoricalMatch {
    /// Home club index in `0..num_teams`.
    pub home_team: usize,
    /// Away club index in `0..num_teams`.
    pub away_team: usize,
    /// Observed home goals.
    pub home_goals: u32,
    /// Observed away goals.
    pub away_goals: u32,
    /// Non-negative elapsed age (e.g., in days or matchweeks) weighted by `exp(-xi * age)`.
    pub age: f64,
}

impl HistoricalMatch {
    /// Creates a validated historical match fixture (`home_team != away_team`, `age >= 0.0` finite).
    #[verified_engine::verified]
    pub fn new(
        home_team: usize,
        away_team: usize,
        home_goals: u32,
        away_goals: u32,
        age: f64,
    ) -> Result<Self, MatchModelError> {
        if home_team == away_team {
            return Err(MatchModelError::SelfMatch {
                team_index: home_team,
            });
        }
        if age < 0.0 || !age.is_finite() {
            return Err(MatchModelError::InvalidMatchAge { value: age });
        }
        Ok(Self {
            home_team,
            away_team,
            home_goals,
            away_goals,
            age,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHistoricalMatch {
    home_team: usize,
    away_team: usize,
    home_goals: u32,
    away_goals: u32,
    age: f64,
}

impl<'de> Deserialize<'de> for HistoricalMatch {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawHistoricalMatch::deserialize(deserializer)?;
        Self::new(
            raw.home_team,
            raw.away_team,
            raw.home_goals,
            raw.away_goals,
            raw.age,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Hyperparameters for time-decayed Dixon-Coles maximum likelihood calibration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DixonColesMleConfig {
    decay_xi: f64,
    learning_rate: f64,
    max_iterations: usize,
    tolerance: f64,
    grad_clip: f64,
}

impl DixonColesMleConfig {
    /// Creates a validated [`DixonColesMleConfig`].
    #[verified_engine::verified]
    pub fn new(
        decay_xi: f64,
        learning_rate: f64,
        max_iterations: usize,
        tolerance: f64,
        grad_clip: f64,
    ) -> Result<Self, MatchModelError> {
        if decay_xi < 0.0 || !decay_xi.is_finite() {
            return Err(MatchModelError::InvalidDecayRate { value: decay_xi });
        }
        if learning_rate <= 0.0
            || !learning_rate.is_finite()
            || max_iterations == 0
            || tolerance <= 0.0
            || !tolerance.is_finite()
            || grad_clip <= 0.0
            || !grad_clip.is_finite()
        {
            return Err(MatchModelError::CalibrationFailed {
                iterations: max_iterations,
            });
        }
        Ok(Self {
            decay_xi,
            learning_rate,
            max_iterations,
            tolerance,
            grad_clip,
        })
    }

    /// Exponential time-decay rate $\xi \ge 0$ (`w = exp(-xi * age)`).
    #[verified_engine::verified]
    pub fn decay_xi(&self) -> f64 {
        self.decay_xi
    }

    /// Optimizer step size.
    #[verified_engine::verified]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Maximum gradient optimization iterations.
    #[verified_engine::verified]
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Infinity-norm gradient convergence tolerance.
    #[verified_engine::verified]
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Per-coordinate gradient clipping bound.
    #[verified_engine::verified]
    pub fn grad_clip(&self) -> f64 {
        self.grad_clip
    }
}

impl Default for DixonColesMleConfig {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            decay_xi: 0.002,
            learning_rate: 0.05,
            max_iterations: 600,
            tolerance: 1e-6,
            grad_clip: 5.0,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDixonColesMleConfig {
    decay_xi: f64,
    learning_rate: f64,
    max_iterations: usize,
    tolerance: f64,
    grad_clip: f64,
}

impl<'de> Deserialize<'de> for DixonColesMleConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawDixonColesMleConfig::deserialize(deserializer)?;
        Self::new(
            raw.decay_xi,
            raw.learning_rate,
            raw.max_iterations,
            raw.tolerance,
            raw.grad_clip,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Fitted league-wide Dixon-Coles parameters with sum-to-zero log-attack identifiability
/// ($\sum_i \alpha_i = 0$), log-defense ($\beta_j$), home-advantage ($\gamma$), and $\rho$.
///
/// Expected goal rates for `home_team = i` vs `away_team = j` follow:
/// $\lambda_{ij} = \exp(\alpha_i + \beta_j + \gamma)$ and $\mu_{ij} = \exp(\alpha_j + \beta_i)$.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedDixonColes {
    /// Sum-to-zero log-attack strengths $\alpha_i$ ($\sum_i \alpha_i = 0$).
    pub attack: Vec<f64>,
    /// Log-defense (concession) parameters $\beta_j$ (lower values indicate stronger defense).
    pub defense: Vec<f64>,
    /// Log-scale home-venue advantage $\gamma$.
    pub home_advantage: f64,
    /// Calibrated low-score correlation $\rho$.
    pub rho: f64,
    /// Final time-decayed log-likelihood achieved at convergence.
    pub log_likelihood: f64,
    /// Number of optimization iterations executed.
    pub iterations: usize,
    /// Whether the projected gradient norm fell below `config.tolerance()`.
    pub converged: bool,
}

impl FittedDixonColes {
    /// Constructs validated [`DixonColesParams`] for `home_team` hosting `away_team`.
    #[verified_engine::verified]
    pub fn match_params(
        &self,
        home_team: usize,
        away_team: usize,
    ) -> Result<DixonColesParams, MatchModelError> {
        let n = self.attack.len();
        if home_team >= n || away_team >= n {
            let index = home_team.max(away_team);
            return Err(MatchModelError::TeamIndexOutOfBounds {
                index,
                num_teams: n,
            });
        }
        if home_team == away_team {
            return Err(MatchModelError::SelfMatch {
                team_index: home_team,
            });
        }
        let lambda = (self.attack[home_team] + self.defense[away_team] + self.home_advantage).exp();
        let mu = (self.attack[away_team] + self.defense[home_team]).exp();
        DixonColesParams::new(lambda, mu, self.rho)
    }

    /// Evaluates the predicted [`ScoreGrid`] for `home_team` vs `away_team`.
    #[verified_engine::verified]
    pub fn predict_grid(
        &self,
        home_team: usize,
        away_team: usize,
        max_goals: usize,
        normalize: bool,
    ) -> Result<ScoreGrid, MatchModelError> {
        let params = self.match_params(home_team, away_team)?;
        dixon_coles_grid(&params, max_goals, normalize)
    }
}

#[verified_engine::verified]
fn validate_calibration_inputs(
    matches: &[HistoricalMatch],
    num_teams: usize,
) -> Result<(), MatchModelError> {
    if num_teams < 2 {
        return Err(MatchModelError::InsufficientTeams { count: num_teams });
    }
    if matches.len() < num_teams {
        return Err(MatchModelError::InsufficientMatches {
            required: num_teams,
            actual: matches.len(),
        });
    }
    for m in matches {
        if m.home_team >= num_teams || m.away_team >= num_teams {
            let index = m.home_team.max(m.away_team);
            return Err(MatchModelError::TeamIndexOutOfBounds { index, num_teams });
        }
        if m.home_team == away_check(m) {
            return Err(MatchModelError::SelfMatch {
                team_index: m.home_team,
            });
        }
        if m.age < 0.0 || !m.age.is_finite() {
            return Err(MatchModelError::InvalidMatchAge { value: m.age });
        }
    }
    Ok(())
}

#[inline]
#[verified_engine::verified]
fn away_check(m: &HistoricalMatch) -> usize {
    m.away_team
}

#[inline]
#[verified_engine::verified]
fn tau_log_partials(home: usize, away: usize, lambda: f64, mu: f64, rho: f64) -> (f64, f64, f64) {
    let t = dixon_coles_tau(home, away, lambda, mu, rho).max(1e-9);
    match (home, away) {
        (0, 0) => ((-mu * rho) / t, (-lambda * rho) / t, (-lambda * mu) / t),
        (0, 1) => (rho / t, 0.0, lambda / t),
        (1, 0) => (0.0, rho / t, mu / t),
        (1, 1) => (0.0, 0.0, -1.0 / t),
        _ => (0.0, 0.0, 0.0),
    }
}

#[verified_engine::verified]
fn evaluate_nll_and_grad(
    params: &[f64],
    matches: &[HistoricalMatch],
    num_teams: usize,
    xi: f64,
    grad_clip: f64,
    grad_out: &mut [f64],
) -> f64 {
    grad_out.fill(0.0);
    let ha_idx = num_teams + num_teams;
    let rho_idx = ha_idx + 1;
    let (ha, rho) = (params[ha_idx], params[rho_idx]);
    let mut weight_sum = 0.0_f64;
    let mut log_lik = 0.0_f64;

    for m in matches {
        let w = (-xi * m.age).exp();
        weight_sum += w;
        let (i, j) = (m.home_team, m.away_team);
        let (x, y) = (m.home_goals as usize, m.away_goals as usize);
        let lambda = (params[i] + params[num_teams + j] + ha).exp();
        let mu = (params[j] + params[num_teams + i]).exp();
        let tau = dixon_coles_tau(x, y, lambda, mu, rho).max(1e-12);
        log_lik += w * (tau.ln() + poisson_log_pmf(lambda, x) + poisson_log_pmf(mu, y));

        let (dt_dlam, dt_dmu, dt_drho) = tau_log_partials(x, y, lambda, mu, rho);
        let g_ln_lam = -w * ((x as f64) - lambda + lambda * dt_dlam);
        let g_ln_mu = -w * ((y as f64) - mu + mu * dt_dmu);
        grad_out[i] += g_ln_lam;
        grad_out[num_teams + j] += g_ln_lam;
        grad_out[ha_idx] += g_ln_lam;
        grad_out[j] += g_ln_mu;
        grad_out[num_teams + i] += g_ln_mu;
        grad_out[rho_idx] += -w * dt_drho;
    }

    let inv_w = if weight_sum > 0.0 {
        1.0 / weight_sum
    } else {
        1.0
    };
    for g in grad_out.iter_mut() {
        *g = (*g * inv_w).clamp(-grad_clip, grad_clip);
    }
    let mean_att_grad = grad_out[..num_teams].iter().sum::<f64>() / (num_teams as f64);
    for g in &mut grad_out[..num_teams] {
        *g -= mean_att_grad;
    }
    log_lik
}

#[verified_engine::verified]
fn project_identifiable_and_feasible(params: &mut [f64], num_teams: usize) {
    let ha_idx = num_teams + num_teams;
    let rho_idx = ha_idx + 1;
    let mean_att = params[..num_teams].iter().sum::<f64>() / (num_teams as f64);
    for i in 0..num_teams {
        params[i] = (params[i] - mean_att).clamp(-3.5, 3.5);
        params[num_teams + i] = (params[num_teams + i] + mean_att).clamp(-3.5, 3.5);
    }
    let re_mean = params[..num_teams].iter().sum::<f64>() / (num_teams as f64);
    for p in &mut params[..num_teams] {
        *p -= re_mean;
    }
    params[ha_idx] = params[ha_idx].clamp(-2.5, 2.5);

    let (mut lo, mut hi) = (-0.98_f64, 0.98_f64);
    for i in 0..num_teams {
        for j in 0..num_teams {
            if i == j {
                continue;
            }
            let lam = (params[i] + params[num_teams + j] + params[ha_idx]).exp();
            let mu = (params[j] + params[num_teams + i]).exp();
            lo = lo.max(-1.0 / (lam * mu)).max(-1.0 / lam).max(-1.0 / mu);
            hi = hi.min(1.0 / lam).min(1.0 / mu).min(1.0 / (lam * mu));
        }
    }
    let margin = ((hi - lo) * 1e-3).min(1e-4);
    params[rho_idx] = params[rho_idx].clamp(lo + margin, hi - margin);
}

/// Fits a time-decayed Dixon-Coles league model using any [`Optimizer`] from `pure_math::analysis::optimization`.
#[verified_engine::verified]
pub fn fit_dixon_coles_with_optimizer(
    matches: &[HistoricalMatch],
    num_teams: usize,
    config: &DixonColesMleConfig,
    optimizer: &mut dyn Optimizer<f64, usize>,
) -> Result<FittedDixonColes, MatchModelError> {
    validate_calibration_inputs(matches, num_teams)?;
    let dim = num_teams + num_teams + 2;
    let mut state = DVector::zeros(dim);
    state[num_teams + num_teams] = 0.15;
    project_identifiable_and_feasible(state.as_mut_slice(), num_teams);

    let mut grad = DVector::zeros(dim);
    let mut best_state = state.clone();
    let mut best_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iters_run = 0usize;

    for iter in 0..config.max_iterations() {
        iters_run = iter + 1;
        let ll = evaluate_nll_and_grad(
            state.as_slice(),
            matches,
            num_teams,
            config.decay_xi(),
            config.grad_clip(),
            grad.as_mut_slice(),
        );
        if ll > best_ll && ll.is_finite() {
            best_ll = ll;
            best_state.copy_from(&state);
        }
        let max_grad = grad.iter().fold(0.0_f64, |acc, &g| acc.max(g.abs()));
        if max_grad < config.tolerance() {
            converged = true;
            break;
        }
        optimizer.update_vector(0, &mut state, &grad).map_err(|_| {
            MatchModelError::CalibrationFailed {
                iterations: iters_run,
            }
        })?;
        project_identifiable_and_feasible(state.as_mut_slice(), num_teams);
    }

    let final_ll = evaluate_nll_and_grad(
        state.as_slice(),
        matches,
        num_teams,
        config.decay_xi(),
        config.grad_clip(),
        grad.as_mut_slice(),
    );
    if final_ll >= best_ll && final_ll.is_finite() {
        best_ll = final_ll;
        best_state = state;
    }
    if !best_ll.is_finite() {
        return Err(MatchModelError::CalibrationFailed {
            iterations: iters_run,
        });
    }
    let slice = best_state.as_slice();
    let ha_idx = num_teams + num_teams;
    Ok(FittedDixonColes {
        attack: slice[..num_teams].to_vec(),
        defense: slice[num_teams..ha_idx].to_vec(),
        home_advantage: slice[ha_idx],
        rho: slice[ha_idx + 1],
        log_likelihood: best_ll,
        iterations: iters_run,
        converged,
    })
}

/// Fits a time-decayed Dixon-Coles league model via [`Adam`] maximum likelihood estimation.
#[verified_engine::verified]
pub fn fit_dixon_coles_mle(
    matches: &[HistoricalMatch],
    num_teams: usize,
    config: &DixonColesMleConfig,
) -> Result<FittedDixonColes, MatchModelError> {
    let mut adam = Adam::<f64, usize>::new(config.learning_rate())
        .map_err(|_| MatchModelError::CalibrationFailed { iterations: 0 })?;
    fit_dixon_coles_with_optimizer(matches, num_teams, config, &mut adam)
}
