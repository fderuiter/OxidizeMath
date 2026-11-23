use super::types::*;
use math_commons::constants;
use math_commons::registry::TOLERANCE_STANDARD;
use std::f64::consts::{LN_2, PI};

/// Maximum compile-time length supported by [`BatemanChain`] stack buffers.
pub const MAX_BATEMAN_CHAIN_LEN: usize = 8;

/// Calculates the remaining amount of a substance.
///
/// Formula: N(t) = N0 * e^(-lambda * t)
///
/// # Arguments
/// * `initial_quantity` - Initial quantity (N0).
/// * `half_life` - Half-life in seconds.
/// * `time` - Time elapsed in seconds.
#[verified_engine::verified]
pub fn calculate_remaining(
    initial_quantity: f64,
    half_life: f64,
    time: f64,
) -> Result<f64, NuclearError> {
    if !half_life.is_finite() || half_life <= 0.0 {
        return Err(NuclearError::InvalidHalfLife);
    }
    if !time.is_finite() || time < 0.0 {
        return Err(NuclearError::InvalidTime);
    }
    let lambda = LN_2 / half_life;
    Ok(initial_quantity * (-lambda * time).exp())
}

/// Calculates the Gamow factor for alpha decay.
///
/// # Arguments
/// * `z_daughter` - Atomic number of the daughter nucleus.
/// * `z_alpha` - Atomic number of the alpha particle (usually 2).
/// * `velocity` - Velocity of the alpha particle in fm/s.
#[verified_engine::verified]
pub fn gamow_factor(
    z_daughter: AtomicNumber,
    z_alpha: AtomicNumber,
    velocity: f64,
) -> Result<f64, NuclearError> {
    if !velocity.is_finite() || velocity <= 0.0 {
        return Err(NuclearError::InvalidVelocity);
    }
    let hbar = constants::HBAR_C / constants::LIGHT_SPEED;
    let numerator = PI * z_alpha.value() as f64 * z_daughter.value() as f64 * constants::E_SQUARED;
    let denominator = hbar * velocity;
    Ok(numerator / denominator)
}

/// Evaluates parent and daughter populations $(N_1(t), N_2(t))$ for a two-step decay $X_1 \to X_2$,
/// guarding equal/near-equal decay rates with [`TOLERANCE_STANDARD`].
///
/// Pass `half_life_daughter = None` for a stable daughter ($\lambda_2 = 0$).
#[verified_engine::verified]
pub fn bateman_two_step(
    initial_parent: f64,
    half_life_parent: f64,
    half_life_daughter: Option<f64>,
    time: f64,
) -> Result<(f64, f64), NuclearError> {
    if !initial_parent.is_finite() || initial_parent < 0.0 {
        return Err(NuclearError::InvalidDecayChain(
            "Initial population must be finite and non-negative",
        ));
    }
    if !half_life_parent.is_finite() || half_life_parent <= 0.0 {
        return Err(NuclearError::InvalidHalfLife);
    }
    if !time.is_finite() || time < 0.0 {
        return Err(NuclearError::InvalidTime);
    }
    let lambda1 = LN_2 / half_life_parent;
    let lambda2 = match half_life_daughter {
        Some(t_half) => {
            if !t_half.is_finite() || t_half <= 0.0 {
                return Err(NuclearError::InvalidHalfLife);
            }
            LN_2 / t_half
        }
        None => 0.0,
    };

    let n1 = initial_parent * (-lambda1 * time).exp();
    let n2 = if (lambda1 - lambda2).abs() < TOLERANCE_STANDARD {
        initial_parent * lambda1 * time * (-lambda1 * time).exp()
    } else {
        initial_parent
            * (lambda1 / (lambda2 - lambda1))
            * ((-lambda1 * time).exp() - (-lambda2 * time).exp())
    };
    Ok((n1.max(0.0), n2.max(0.0)))
}

/// Stack-allocated linear radioactive decay-chain solver ($X_0 \to X_1 \to \dots \to X_{N-1}$).
///
/// Supports $1 \le N \le 8$ (`MAX_BATEMAN_CHAIN_LEN`), arbitrary initial populations,
/// stable endpoints ($\lambda_k = 0$), secular/transient equilibrium, and degenerate decay constants
/// without heap allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatemanChain<const N: usize> {
    decay_constants: [f64; N],
    initial_populations: [f64; N],
}

impl<const N: usize> BatemanChain<N> {
    /// Constructs a decay chain from explicit decay constants $\lambda_k \ge 0$ ($\text{s}^{-1}$)
    /// and initial populations $N_k(0) \ge 0$.
    #[verified_engine::verified]
    pub fn from_decay_constants(
        decay_constants: [f64; N],
        initial_populations: [f64; N],
    ) -> Result<Self, NuclearError> {
        if N == 0 || N > MAX_BATEMAN_CHAIN_LEN {
            return Err(NuclearError::InvalidDecayChain(
                "Chain length N must satisfy 1 <= N <= MAX_BATEMAN_CHAIN_LEN",
            ));
        }
        let mut idx = 0;
        while idx < N {
            let lam = decay_constants[idx];
            let pop = initial_populations[idx];
            if !lam.is_finite() || lam < 0.0 {
                return Err(NuclearError::InvalidHalfLife);
            }
            if !pop.is_finite() || pop < 0.0 {
                return Err(NuclearError::InvalidDecayChain(
                    "Initial populations must be finite and non-negative",
                ));
            }
            idx += 1;
        }
        Ok(Self {
            decay_constants,
            initial_populations,
        })
    }

    /// Constructs a decay chain from half-lives (`None` denotes a stable nuclide with $\lambda = 0$).
    #[verified_engine::verified]
    pub fn from_half_lives(
        half_lives: [Option<f64>; N],
        initial_populations: [f64; N],
    ) -> Result<Self, NuclearError> {
        if N == 0 || N > MAX_BATEMAN_CHAIN_LEN {
            return Err(NuclearError::InvalidDecayChain(
                "Chain length N must satisfy 1 <= N <= MAX_BATEMAN_CHAIN_LEN",
            ));
        }
        let mut lambdas = [0.0; N];
        let mut idx = 0;
        while idx < N {
            if let Some(t_half) = half_lives[idx] {
                if !t_half.is_finite() || t_half <= 0.0 {
                    return Err(NuclearError::InvalidHalfLife);
                }
                lambdas[idx] = LN_2 / t_half;
            }
            idx += 1;
        }
        Self::from_decay_constants(lambdas, initial_populations)
    }

    /// Returns the decay constants $\lambda_k$ ($\text{s}^{-1}$) for each chain member.
    #[verified_engine::verified]
    pub fn decay_constants(&self) -> &[f64; N] {
        &self.decay_constants
    }

    /// Returns the initial populations $N_k(0)$ for each chain member.
    #[verified_engine::verified]
    pub fn initial_populations(&self) -> &[f64; N] {
        &self.initial_populations
    }

    /// Evaluates populations $N_0(t), \dots, N_{N-1}(t)$ at elapsed `time` ($t \ge 0$).
    #[verified_engine::verified]
    pub fn populations_at(&self, time: f64) -> Result<[f64; N], NuclearError> {
        if !time.is_finite() || time < 0.0 {
            return Err(NuclearError::InvalidTime);
        }
        if time == 0.0 {
            return Ok(self.initial_populations);
        }
        let mut out = [0.0; N];
        let mut src = 0;
        while src < N {
            let n0 = self.initial_populations[src];
            if n0 > 0.0 {
                let mut dst = src;
                while dst < N {
                    let g = chain_transfer_kernel(&self.decay_constants, src, dst, time);
                    out[dst] += n0 * g;
                    dst += 1;
                }
            }
            src += 1;
        }
        let mut k = 0;
        while k < N {
            if out[k] < 0.0 {
                out[k] = 0.0;
            }
            k += 1;
        }
        Ok(out)
    }

    /// Evaluates activities $A_k(t) = \lambda_k N_k(t)$ (decays per unit time) at elapsed `time`.
    #[verified_engine::verified]
    pub fn activities_at(&self, time: f64) -> Result<[f64; N], NuclearError> {
        let mut pops = self.populations_at(time)?;
        let mut k = 0;
        while k < N {
            pops[k] *= self.decay_constants[k];
            k += 1;
        }
        Ok(pops)
    }
}

#[verified_engine::verified]
fn chain_transfer_kernel(lambdas: &[f64], src: usize, dst: usize, time: f64) -> f64 {
    if src == dst {
        return (-lambdas[src] * time).exp();
    }
    let len = dst - src + 1;
    let mut max_lam = 0.0_f64;
    let mut prod_positive = true;
    let mut idx = src;
    while idx < dst {
        if lambdas[idx] <= 0.0 {
            prod_positive = false;
        }
        if lambdas[idx] > max_lam {
            max_lam = lambdas[idx];
        }
        idx += 1;
    }
    if !prod_positive {
        return 0.0;
    }
    if lambdas[dst] > max_lam {
        max_lam = lambdas[dst];
    }

    if max_lam * time <= 2.0 {
        return uniformized_bidiagonal_kernel(lambdas, src, len, max_lam, time);
    }
    clustered_partial_fraction_kernel(lambdas, src, dst, time)
}

/// Entrywise non-negative Jensen uniformization series for $\max_k(\lambda_k t) \le 2.0$.
#[verified_engine::verified]
fn uniformized_bidiagonal_kernel(
    lambdas: &[f64],
    src: usize,
    len: usize,
    max_lam: f64,
    time: f64,
) -> f64 {
    let mut vec = [0.0; MAX_BATEMAN_CHAIN_LEN];
    vec[0] = 1.0;
    let target = len - 1;
    let mut accum = 0.0;
    let mut step = 1usize;
    while step <= 40 {
        let inv_step = 1.0 / (step as f64);
        let mut r = target;
        loop {
            let diag = (max_lam - lambdas[src + r]) * time * vec[r];
            let sub = if r > 0 {
                lambdas[src + r - 1] * time * vec[r - 1]
            } else {
                0.0
            };
            vec[r] = (diag + sub) * inv_step;
            if r == 0 {
                break;
            }
            r -= 1;
        }
        if step >= target {
            accum += vec[target];
        }
        step += 1;
    }
    (-max_lam * time).exp() * accum
}

/// Exact confluent partial-fraction inverse Laplace evaluation with [`TOLERANCE_STANDARD`] rate clustering.
#[verified_engine::verified]
fn clustered_partial_fraction_kernel(lambdas: &[f64], src: usize, dst: usize, time: f64) -> f64 {
    let mut centers = [0.0; MAX_BATEMAN_CHAIN_LEN];
    let mut mults = [0usize; MAX_BATEMAN_CHAIN_LEN];
    let mut coeffs = [[0.0; MAX_BATEMAN_CHAIN_LEN]; MAX_BATEMAN_CHAIN_LEN];
    centers[0] = lambdas[src];
    mults[0] = 1;
    coeffs[0][0] = 1.0;
    let mut num_clusters = 1usize;

    let mut step = src;
    while step < dst {
        let rate_num = lambdas[step];
        let pole = lambdas[step + 1];
        num_clusters = multiply_partial_fraction_pole(
            &mut centers,
            &mut mults,
            &mut coeffs,
            num_clusters,
            rate_num,
            pole,
        );
        step += 1;
    }

    let mut sum = 0.0;
    let mut c = 0;
    while c < num_clusters {
        let exp_term = (-centers[c] * time).exp();
        let mut t_pow_over_fact = 1.0;
        let mut p = 0;
        while p < mults[c] {
            sum += coeffs[c][p] * t_pow_over_fact * exp_term;
            t_pow_over_fact *= time / ((p + 1) as f64);
            p += 1;
        }
        c += 1;
    }
    sum.max(0.0)
}

#[verified_engine::verified]
fn multiply_partial_fraction_pole(
    centers: &mut [f64; MAX_BATEMAN_CHAIN_LEN],
    mults: &mut [usize; MAX_BATEMAN_CHAIN_LEN],
    coeffs: &mut [[f64; MAX_BATEMAN_CHAIN_LEN]; MAX_BATEMAN_CHAIN_LEN],
    num_clusters: usize,
    rate_num: f64,
    pole: f64,
) -> usize {
    let mut match_idx = num_clusters;
    let mut c = 0;
    while c < num_clusters {
        let scale = 1.0 + centers[c].max(pole);
        if (centers[c] - pole).abs() < TOLERANCE_STANDARD * scale {
            match_idx = c;
            break;
        }
        c += 1;
    }

    let mut next_coeffs = [[0.0; MAX_BATEMAN_CHAIN_LEN]; MAX_BATEMAN_CHAIN_LEN];
    let mut next_mults = *mults;
    let mut next_count = num_clusters;
    if match_idx == num_clusters {
        centers[match_idx] = pole;
        next_mults[match_idx] = 1;
        next_count += 1;
    } else {
        next_mults[match_idx] = mults[match_idx] + 1;
    }

    let target_pole = centers[match_idx];
    let mut idx = 0;
    while idx < num_clusters {
        if idx == match_idx {
            let mut p = 0;
            while p < mults[idx] {
                next_coeffs[idx][p + 1] += rate_num * coeffs[idx][p];
                p += 1;
            }
        } else {
            let delta = target_pole - centers[idx];
            let mut p = 0;
            while p < mults[idx] {
                let q = p + 1;
                let weight = rate_num * coeffs[idx][p];
                let mut r = 1usize;
                while r <= q {
                    let pow = (q - r + 1) as i32;
                    let sign = if (q - r).is_multiple_of(2) { 1.0 } else { -1.0 };
                    next_coeffs[idx][r - 1] += weight * sign / delta.powi(pow);
                    r += 1;
                }
                let pole_sign = if q.is_multiple_of(2) { 1.0 } else { -1.0 };
                next_coeffs[match_idx][0] += weight * pole_sign / delta.powi(q as i32);
                p += 1;
            }
        }
        idx += 1;
    }

    *mults = next_mults;
    *coeffs = next_coeffs;
    next_count
}
