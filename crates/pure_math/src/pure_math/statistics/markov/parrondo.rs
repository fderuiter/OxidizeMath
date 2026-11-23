//! Parrondo's paradox discrete-time Markov ratchet and ensemble solvers.
//!
//! Models capital-dependent modulo-$M$ Brownian ratchets in which two individually
//! losing games ($A$ and $B$) generate positive expected wealth growth when alternated
//! randomly ($P_{\text{mix}} = \tfrac{1}{2}(P_A + P_B)$) or periodically (such as
//! $P_{ABB} = P_A P_B P_B$).
//!
//! # Example
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::statistics::markov::parrondo::{
//!     ParrondoParams, ParrondoSchedule, solve_parrondo_ratchet,
//! };
//!
//! let params = ParrondoParams::canonical();
//! let schedule = ParrondoSchedule::parse("ABB").unwrap();
//! let report = solve_parrondo_ratchet(&params, &schedule).unwrap();
//! assert!(report.gain_a < 0.0 && report.gain_b < 0.0 && report.is_paradox());
//! ```

use crate::error::MarkovError;
use crate::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
use nalgebra::{DMatrix, DVector};

mod ensemble;
pub use ensemble::{
    ParrondoEnsemble, ParrondoPhaseGridConfig, ParrondoPhasePoint, scan_parrondo_phase_region,
};

/// Result alias for Parrondo Markov ratchet operations.
pub type Result<T> = std::result::Result<T, MarkovError>;

/// Odd moduli supported by the modulo-$M$ ratchet to avoid bipartite parity traps.
pub const ALLOWED_MODULI: [usize; 4] = [3, 5, 7, 9];
/// Maximum supported modulus $M$ for fixed-size stack occupancy buffers.
pub const MAX_MODULUS: usize = 9;
/// Maximum periodic schedule length (bounded to fit URL and preset JSON budgets).
pub const MAX_SCHEDULE_LEN: usize = 16;

/// Elementary game branch in Parrondo's paradox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParrondoGame {
    /// State-independent biased coin with win probability $p_A = p_{A,0} - \epsilon$.
    GameA,
    /// Capital-dependent modulo-$M$ game with trap branch $W \equiv 0 \pmod M$.
    GameB,
}

/// Validated periodic game schedule over `{A, B}` with bounded length `1..=16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParrondoSchedule {
    steps: [ParrondoGame; MAX_SCHEDULE_LEN],
    len: usize,
}

impl Default for ParrondoSchedule {
    fn default() -> Self {
        Self::abb()
    }
}

impl ParrondoSchedule {
    /// Returns the canonical three-step `ABB` ratchet schedule.
    #[must_use]
    #[verified_engine::verified]
    pub fn abb() -> Self {
        let mut steps = [ParrondoGame::GameA; MAX_SCHEDULE_LEN];
        steps[1] = ParrondoGame::GameB;
        steps[2] = ParrondoGame::GameB;
        Self { steps, len: 3 }
    }

    /// Parses a non-empty schedule string of `'A'` and `'B'` characters up to length `16`.
    ///
    /// # Errors
    /// Returns [`MarkovError::InvalidState`] if `spec` is empty, exceeds `16`, or has non-`A`/`B` chars.
    #[verified_engine::verified]
    pub fn parse(spec: &str) -> Result<Self> {
        if spec.is_empty() || spec.len() > MAX_SCHEDULE_LEN {
            return Err(MarkovError::InvalidState {
                reason: format!(
                    "Schedule length must be in 1..={MAX_SCHEDULE_LEN}, got {}",
                    spec.len()
                ),
            });
        }
        let mut steps = [ParrondoGame::GameA; MAX_SCHEDULE_LEN];
        for (idx, ch) in spec.chars().enumerate() {
            steps[idx] = match ch {
                'A' => ParrondoGame::GameA,
                'B' => ParrondoGame::GameB,
                _ => {
                    return Err(MarkovError::InvalidState {
                        reason: format!("Schedule character '{ch}' must be 'A' or 'B'"),
                    });
                }
            };
        }
        Ok(Self {
            steps,
            len: spec.len(),
        })
    }

    /// Returns the active sequence of elementary games in one period.
    #[must_use]
    #[verified_engine::verified]
    pub fn steps(&self) -> &[ParrondoGame] {
        &self.steps[..self.len]
    }

    /// Returns the period length of the schedule.
    #[must_use]
    #[verified_engine::verified]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the schedule has zero steps (always `false` for constructed schedules).
    #[must_use]
    #[verified_engine::verified]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Formats the schedule back into its canonical `"ABB"` string representation.
    #[must_use]
    pub fn as_string(&self) -> String {
        let mut out = String::with_capacity(self.len);
        for &step in self.steps() {
            out.push(match step {
                ParrondoGame::GameA => 'A',
                ParrondoGame::GameB => 'B',
            });
        }
        out
    }
}

/// Validated parameters for a modulo-$M$ Parrondo Brownian ratchet.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ParrondoParams {
    modulus: usize,
    p_a_base: f64,
    p_b0_base: f64,
    p_b1_base: f64,
    epsilon: f64,
}

impl Default for ParrondoParams {
    fn default() -> Self {
        Self::canonical()
    }
}

impl ParrondoParams {
    /// Constructs validated Parrondo parameters.
    ///
    /// # Errors
    /// Returns [`MarkovError::InvalidState`] if `modulus` is not in `{3, 5, 7, 9}`, or
    /// [`MarkovError::InvalidProbability`] if any effective win probability falls outside `(0.0, 1.0)`.
    #[verified_engine::verified]
    pub fn new(
        modulus: usize,
        p_a_base: f64,
        p_b0_base: f64,
        p_b1_base: f64,
        epsilon: f64,
    ) -> Result<Self> {
        if !ALLOWED_MODULI.contains(&modulus) {
            return Err(MarkovError::InvalidState {
                reason: format!("Modulus M must be one of {ALLOWED_MODULI:?}, got {modulus}"),
            });
        }
        for raw in [p_a_base, p_b0_base, p_b1_base, epsilon] {
            if !raw.is_finite() {
                return Err(MarkovError::InvalidProbability { value: raw });
            }
        }
        for eff in [p_a_base - epsilon, p_b0_base - epsilon, p_b1_base - epsilon] {
            if !eff.is_finite() || eff <= 0.0 || eff >= 1.0 {
                return Err(MarkovError::InvalidProbability { value: eff });
            }
        }
        Ok(Self {
            modulus,
            p_a_base,
            p_b0_base,
            p_b1_base,
            epsilon,
        })
    }

    /// Returns the Harmer-Abbott canonical `M = 3, epsilon = 0.005` parameter set.
    #[must_use]
    #[verified_engine::verified]
    pub fn canonical() -> Self {
        Self {
            modulus: 3,
            p_a_base: 0.5,
            p_b0_base: 0.1,
            p_b1_base: 0.75,
            epsilon: 0.005,
        }
    }

    /// Returns the capital modulus $M \in \{3, 5, 7, 9\}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn modulus(&self) -> usize {
        self.modulus
    }

    /// Returns the unperturbed Game A win probability $p_{A,0}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn p_a_base(&self) -> f64 {
        self.p_a_base
    }

    /// Returns the unperturbed Game B trap-state ($W \equiv 0 \pmod M$) win probability $p_{B0,0}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn p_b0_base(&self) -> f64 {
        self.p_b0_base
    }

    /// Returns the unperturbed Game B non-trap ($W \not\equiv 0 \pmod M$) win probability $p_{B1,0}$.
    #[must_use]
    #[verified_engine::verified]
    pub fn p_b1_base(&self) -> f64 {
        self.p_b1_base
    }

    /// Returns the biasing perturbation $\epsilon$ subtracted from all coin branches.
    #[must_use]
    #[verified_engine::verified]
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the effective Game A win probability $p_A = p_{A,0} - \epsilon$.
    #[must_use]
    #[verified_engine::verified]
    pub fn p_a(&self) -> f64 {
        self.p_a_base - self.epsilon
    }

    /// Returns the effective Game B trap win probability $p_{B0} = p_{B0,0} - \epsilon$.
    #[must_use]
    #[verified_engine::verified]
    pub fn p_b0(&self) -> f64 {
        self.p_b0_base - self.epsilon
    }

    /// Returns the effective Game B non-trap win probability $p_{B1} = p_{B1,0} - \epsilon$.
    #[must_use]
    #[verified_engine::verified]
    pub fn p_b1(&self) -> f64 {
        self.p_b1_base - self.epsilon
    }

    /// Returns the branch win probability for `game` at capital residue `residue` (`0..M`).
    #[must_use]
    #[verified_engine::verified]
    pub fn win_probability(&self, game: ParrondoGame, residue: usize) -> f64 {
        debug_assert!(residue < self.modulus);
        match game {
            ParrondoGame::GameA => self.p_a(),
            ParrondoGame::GameB => {
                if residue == 0 {
                    self.p_b0()
                } else {
                    self.p_b1()
                }
            }
        }
    }
}

/// Exact Markov operators, stationary distributions, per-step drifts, and detailed-balance metrics.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParrondoReport {
    /// Transition matrix $P_A$ on $\mathbb{Z}/M\mathbb{Z}$.
    pub p_a: DMatrix<f64>,
    /// Transition matrix $P_B$ on $\mathbb{Z}/M\mathbb{Z}$.
    pub p_b: DMatrix<f64>,
    /// Equal-weight stochastic mixture matrix $P_{\text{mix}} = \tfrac{1}{2}(P_A + P_B)$.
    pub p_mix: DMatrix<f64>,
    /// Periodic schedule product matrix (e.g. $P_{ABB} = P_A P_B P_B$).
    pub p_schedule: DMatrix<f64>,
    /// Stationary distribution $\pi_A$ of Game A.
    pub pi_a: DVector<f64>,
    /// Stationary distribution $\pi_B$ of Game B.
    pub pi_b: DVector<f64>,
    /// Stationary distribution $\pi_{\text{mix}}$ of the 50/50 random mixture.
    pub pi_mix: DVector<f64>,
    /// Cycle-start stationary distribution $\pi_{\text{sched}}$ of `p_schedule`.
    pub pi_schedule: DVector<f64>,
    /// Time-averaged residue occupancy across all sub-steps of one schedule period.
    pub pi_schedule_mean: DVector<f64>,
    /// Asymptotic expected wealth change per step $E[\Delta W_A]$ under pure Game A.
    pub gain_a: f64,
    /// Asymptotic expected wealth change per step $E[\Delta W_B]$ under pure Game B.
    pub gain_b: f64,
    /// Asymptotic expected wealth change per step $E[\Delta W_{\text{mix}}]$ under 50/50 mixing.
    pub gain_mix: f64,
    /// Asymptotic expected wealth change per step $E[\Delta W_{\text{sched}}]$ under the schedule.
    pub gain_schedule: f64,
    /// Detailed-balance residual $\max_{i,j} |\pi_{B,i} P_{B,ij} - \pi_{B,j} P_{B,ji}|$.
    pub detailed_balance_b: f64,
    /// Detailed-balance residual $\max_{i,j} |\pi_{\text{mix},i} P_{\text{mix},ij} - \pi_{\text{mix},j} P_{\text{mix},ji}|$.
    pub detailed_balance_mix: f64,
    /// Detailed-balance residual $\max_{i,j} |\pi_{\text{sched},i} P_{\text{sched},ij} - \pi_{\text{sched},j} P_{\text{sched},ji}|$.
    pub detailed_balance_schedule: f64,
}

impl ParrondoReport {
    /// Returns `true` when pure games $A$ and $B$ lose while the schedule wins.
    #[must_use]
    #[verified_engine::verified]
    pub fn is_paradox(&self) -> bool {
        self.gain_a < 0.0 && self.gain_b < 0.0 && self.gain_schedule > 0.0
    }
}

/// Builds the $M \times M$ modulo-$M$ transition matrix $P_A$ for Game A.
#[must_use]
#[verified_engine::verified]
pub fn build_game_a_matrix(params: &ParrondoParams) -> DMatrix<f64> {
    let (m, p) = (params.modulus(), params.p_a());
    let mut mat = DMatrix::zeros(m, m);
    for s in 0..m {
        mat[(s, (s + 1) % m)] += p;
        mat[(s, (s + m - 1) % m)] += 1.0 - p;
    }
    mat
}

/// Builds the $M \times M$ modulo-$M$ transition matrix $P_B$ for Game B.
#[must_use]
#[verified_engine::verified]
pub fn build_game_b_matrix(params: &ParrondoParams) -> DMatrix<f64> {
    let m = params.modulus();
    let mut mat = DMatrix::zeros(m, m);
    for s in 0..m {
        let p = params.win_probability(ParrondoGame::GameB, s);
        mat[(s, (s + 1) % m)] += p;
        mat[(s, (s + m - 1) % m)] += 1.0 - p;
    }
    mat
}

/// Builds the equal-weight mixture transition matrix $P_{\text{mix}} = \tfrac{1}{2}(P_A + P_B)$.
#[verified_engine::verified]
pub fn build_mixture_matrix(p_a: &DMatrix<f64>, p_b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if p_a.nrows() != p_b.nrows() || p_a.ncols() != p_b.ncols() || p_a.nrows() != p_a.ncols() {
        use math_commons::{error::MathError::DimensionMismatch, math_kernel::types::Dimension};
        return Err(MarkovError::Math(DimensionMismatch {
            expected: Dimension(p_a.nrows()),
            actual: Dimension(p_b.ncols()),
        }));
    }
    Ok((p_a + p_b) * 0.5)
}

/// Multiplies elementary transition matrices in schedule order (e.g. $P_{ABB} = P_A P_B P_B$).
#[verified_engine::verified]
pub fn build_schedule_matrix(
    p_a: &DMatrix<f64>,
    p_b: &DMatrix<f64>,
    schedule: &ParrondoSchedule,
) -> Result<DMatrix<f64>> {
    let _ = build_mixture_matrix(p_a, p_b)?;
    let mut product = DMatrix::identity(p_a.nrows(), p_a.nrows());
    for &step in schedule.steps() {
        let factor = match step {
            ParrondoGame::GameA => p_a,
            ParrondoGame::GameB => p_b,
        };
        product = &product * factor;
    }
    Ok(product)
}

/// Computes the stationary distribution via lazy damping $\tfrac{1}{2}(I + P)$ and [`MarkovChain`].
#[verified_engine::verified]
pub fn lazy_stationary_distribution(transition: &DMatrix<f64>) -> Result<DVector<f64>> {
    let m = transition.nrows();
    let lazy = (DMatrix::identity(m, m) + transition) * 0.5;
    MarkovChain::new(lazy, vec![StateType::Transient; m])?
        .stationary_distribution()
        .ok_or_else(|| MarkovError::SingularMatrix {
            context: "Lazy-damped MarkovChain stationary_distribution did not converge".to_string(),
        })
}

/// Computes the maximum pairwise detailed-balance flux violation $\max_{i,j} |\pi_i P_{ij} - \pi_j P_{ji}|$.
#[must_use]
#[verified_engine::verified]
pub fn detailed_balance_violation(transition: &DMatrix<f64>, pi: &DVector<f64>) -> f64 {
    let m = transition.nrows().min(transition.ncols()).min(pi.len());
    let mut max_violation = 0.0_f64;
    for i in 0..m {
        for j in (i + 1)..m {
            let diff = (pi[i] * transition[(i, j)] - pi[j] * transition[(j, i)]).abs();
            max_violation = max_violation.max(diff);
        }
    }
    max_violation
}

#[verified_engine::verified]
fn evaluate_schedule_cycle(
    p_a: &DMatrix<f64>,
    p_b: &DMatrix<f64>,
    gain_vec_a: &DVector<f64>,
    gain_vec_b: &DVector<f64>,
    pi_cycle_start: &DVector<f64>,
    schedule: &ParrondoSchedule,
) -> (f64, DVector<f64>) {
    let (mut dist, mut accum_pi) = (pi_cycle_start.clone(), DVector::zeros(pi_cycle_start.len()));
    let (mut total_gain, p_a_t, p_b_t) = (0.0, p_a.transpose(), p_b.transpose());
    for &step in schedule.steps() {
        accum_pi += &dist;
        let (gain_v, p_t) = match step {
            ParrondoGame::GameA => (gain_vec_a, &p_a_t),
            ParrondoGame::GameB => (gain_vec_b, &p_b_t),
        };
        total_gain += dist.dot(gain_v);
        dist = p_t * &dist;
    }
    let inv_len = 1.0 / (schedule.len() as f64);
    (total_gain * inv_len, accum_pi * inv_len)
}

/// Solves exact stationary distributions, asymptotic drifts, and broken detailed-balance metrics.
#[verified_engine::verified]
pub fn solve_parrondo_ratchet(
    params: &ParrondoParams,
    schedule: &ParrondoSchedule,
) -> Result<ParrondoReport> {
    let (m, p_a, p_b) = (
        params.modulus(),
        build_game_a_matrix(params),
        build_game_b_matrix(params),
    );
    let p_mix = build_mixture_matrix(&p_a, &p_b)?;
    let p_schedule = build_schedule_matrix(&p_a, &p_b, schedule)?;
    let (pi_a, pi_b) = (
        lazy_stationary_distribution(&p_a)?,
        lazy_stationary_distribution(&p_b)?,
    );
    let (pi_mix, pi_schedule) = (
        lazy_stationary_distribution(&p_mix)?,
        lazy_stationary_distribution(&p_schedule)?,
    );
    let gain_vec_a = DVector::from_element(m, 2.0 * params.p_a() - 1.0);
    let gain_vec_b = DVector::from_fn(m, |s, _| {
        2.0 * params.win_probability(ParrondoGame::GameB, s) - 1.0
    });
    let gain_vec_mix = (&gain_vec_a + &gain_vec_b) * 0.5;
    let (gain_schedule, pi_schedule_mean) =
        evaluate_schedule_cycle(&p_a, &p_b, &gain_vec_a, &gain_vec_b, &pi_schedule, schedule);
    Ok(ParrondoReport {
        gain_a: pi_a.dot(&gain_vec_a),
        gain_b: pi_b.dot(&gain_vec_b),
        gain_mix: pi_mix.dot(&gain_vec_mix),
        gain_schedule,
        detailed_balance_b: detailed_balance_violation(&p_b, &pi_b),
        detailed_balance_mix: detailed_balance_violation(&p_mix, &pi_mix),
        detailed_balance_schedule: detailed_balance_violation(&p_schedule, &pi_schedule),
        p_a,
        p_b,
        p_mix,
        p_schedule,
        pi_a,
        pi_b,
        pi_mix,
        pi_schedule,
        pi_schedule_mean,
    })
}

#[cfg(test)]
mod tests;

// [cite:clinical_trials_statistics]
