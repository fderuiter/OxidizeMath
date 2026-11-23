use super::special::{sample_uniform_open, validate_probability};
use super::traits::DiscreteDistribution;
use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;
use statrs::function::{beta, gamma};

/// Bernoulli distribution $\mathrm{Bernoulli}(p)$ on $\{0, 1\}$ with success probability $p \in [0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BernoulliDist {
    p: f64,
}

impl BernoulliDist {
    /// Constructs $\mathrm{Bernoulli}(p)$ with $p \in [0, 1]$.
    #[verified_engine::verified]
    pub fn new(p: f64) -> Result<Self, DistributionError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(DistributionError::InvalidParameter {
                name: "p",
                value: p,
            });
        }
        Ok(Self { p })
    }

    /// Returns success probability $p$.
    #[verified_engine::verified]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Maximum likelihood estimator $\hat{p} = \bar{k}$ on binary $\{0, 1\}$ data.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[u64]) -> Result<Self, DistributionError> {
        if data.is_empty() {
            return Err(DistributionError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        let mut ones = 0_u64;
        for &k in data {
            if k > 1 {
                return Err(DistributionError::FittingFailed {
                    reason: "Bernoulli observations must be 0 or 1",
                });
            }
            ones += k;
        }
        Self::new((ones as f64) / (data.len() as f64))
    }

    /// Method-of-moments estimator $\hat{p} = \bar{k}$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[u64]) -> Result<Self, DistributionError> {
        Self::fit_mle(data)
    }
}

impl DiscreteDistribution for BernoulliDist {
    #[verified_engine::verified]
    fn pmf(&self, k: u64) -> f64 {
        match k {
            0 => 1.0 - self.p,
            1 => self.p,
            _ => 0.0,
        }
    }

    #[verified_engine::verified]
    fn ln_pmf(&self, k: u64) -> f64 {
        let prob = self.pmf(k);
        if prob <= 0.0 {
            f64::NEG_INFINITY
        } else {
            prob.ln()
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, k: u64) -> f64 {
        if k == 0 { 1.0 - self.p } else { 1.0 }
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<u64, DistributionError> {
        validate_probability(p)?;
        if p <= 1.0 - self.p { Ok(0) } else { Ok(1) }
    }

    #[verified_engine::verified]
    fn mean(&self) -> f64 {
        self.p
    }

    #[verified_engine::verified]
    fn variance(&self) -> f64 {
        self.p * (1.0 - self.p)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> u64 {
        u64::from(sample_uniform_open(rng) < self.p)
    }
}

/// Binomial distribution $\mathrm{Binomial}(n, p)$ on $\{0, \dots, n\}$ with $n \ge 1$ and $p \in [0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinomialDist {
    n: u64,
    p: f64,
}

impl BinomialDist {
    /// Constructs $\mathrm{Binomial}(n, p)$ verifying `n >= 1` and $p \in [0, 1]$.
    #[verified_engine::verified]
    pub fn new(n: u64, p: f64) -> Result<Self, DistributionError> {
        if n == 0 {
            return Err(DistributionError::InvalidParameter {
                name: "n",
                value: 0.0,
            });
        }
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(DistributionError::InvalidParameter {
                name: "p",
                value: p,
            });
        }
        Ok(Self { n, p })
    }

    /// Returns trial count $n$.
    #[verified_engine::verified]
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Returns success probability $p$.
    #[verified_engine::verified]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Exact maximum likelihood estimator $\hat{p} = \bar{k} / n$ given known trial count `n`.
    #[verified_engine::verified]
    pub fn fit_mle_known_n(data: &[u64], n: u64) -> Result<Self, DistributionError> {
        if data.is_empty() {
            return Err(DistributionError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        if n == 0 {
            return Err(DistributionError::InvalidParameter {
                name: "n",
                value: 0.0,
            });
        }
        let mut sum = 0_u64;
        for &k in data {
            if k > n {
                return Err(DistributionError::FittingFailed {
                    reason: "observed count exceeds binomial trial count n",
                });
            }
            sum += k;
        }
        let mean = (sum as f64) / (data.len() as f64);
        Self::new(n, (mean / (n as f64)).clamp(0.0, 1.0))
    }

    /// Method-of-moments estimator for $(n, p)$ on underdispersed count data ($s^2 < \bar{k}$).
    #[verified_engine::verified]
    pub fn fit_mom(data: &[u64]) -> Result<Self, DistributionError> {
        if data.len() < 2 {
            return Err(DistributionError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }
        let len = data.len() as f64;
        let max_k = data.iter().copied().max().unwrap_or(1).max(1);
        let mean = data.iter().map(|&k| k as f64).sum::<f64>() / len;
        if mean <= 0.0 {
            return Self::new(max_k, 0.0);
        }
        let var = data
            .iter()
            .map(|&k| {
                let d = (k as f64) - mean;
                d * d
            })
            .sum::<f64>()
            / (len - 1.0);
        if var >= mean {
            return Err(DistributionError::FittingFailed {
                reason: "binomial MoM requires underdispersion (variance < mean)",
            });
        }
        let p_raw = (1.0 - var / mean).clamp(1e-6, 1.0);
        let n_est = (mean / p_raw).round().max(max_k as f64) as u64;
        Self::new(n_est, (mean / (n_est as f64)).clamp(0.0, 1.0))
    }

    /// Joint profile maximum likelihood estimator over $(n, p)$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[u64]) -> Result<Self, DistributionError> {
        let mom = Self::fit_mom(data)?;
        let max_k = data.iter().copied().max().unwrap_or(1).max(1);
        let mean = data.iter().map(|&k| k as f64).sum::<f64>() / (data.len() as f64);
        let n_lo = max_k;
        let n_hi = mom.n.saturating_mul(3).max(max_k + 32).min(max_k + 2048);
        let mut best = mom;
        let mut best_ll = f64::NEG_INFINITY;
        for cand_n in n_lo..=n_hi {
            let cand_p = (mean / (cand_n as f64)).clamp(0.0, 1.0);
            if let Ok(dist) = Self::new(cand_n, cand_p) {
                let ll: f64 = data.iter().map(|&k| dist.ln_pmf(k)).sum();
                if ll > best_ll {
                    best_ll = ll;
                    best = dist;
                }
            }
        }
        Ok(best)
    }
}

impl DiscreteDistribution for BinomialDist {
    #[verified_engine::verified]
    fn pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return 0.0;
        }
        if self.p == 0.0 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        if self.p == 1.0 {
            return if k == self.n { 1.0 } else { 0.0 };
        }
        self.ln_pmf(k).exp()
    }

    #[verified_engine::verified]
    fn ln_pmf(&self, k: u64) -> f64 {
        if k > self.n {
            return f64::NEG_INFINITY;
        }
        if self.p == 0.0 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        if self.p == 1.0 {
            return if k == self.n { 0.0 } else { f64::NEG_INFINITY };
        }
        let nf = self.n as f64;
        let kf = k as f64;
        gamma::ln_gamma(nf + 1.0) - gamma::ln_gamma(kf + 1.0) - gamma::ln_gamma(nf - kf + 1.0)
            + kf * self.p.ln()
            + (nf - kf) * (-self.p).ln_1p()
    }

    #[verified_engine::verified]
    fn cdf(&self, k: u64) -> f64 {
        if k >= self.n {
            return 1.0;
        }
        if self.p <= 0.0 {
            return 1.0;
        }
        if self.p >= 1.0 {
            return 0.0;
        }
        beta::beta_reg((self.n - k) as f64, (k + 1) as f64, 1.0 - self.p)
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<u64, DistributionError> {
        validate_probability(p)?;
        if p == 0.0 || self.p == 0.0 {
            return Ok(0);
        }
        if p == 1.0 || self.p == 1.0 {
            return Ok(self.n);
        }
        let mut lo = 0_u64;
        let mut hi = self.n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf(mid) >= p {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }

    #[verified_engine::verified]
    fn mean(&self) -> f64 {
        (self.n as f64) * self.p
    }

    #[verified_engine::verified]
    fn variance(&self) -> f64 {
        (self.n as f64) * self.p * (1.0 - self.p)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> u64 {
        if self.n <= 64 {
            let mut count = 0_u64;
            for _ in 0..self.n {
                if sample_uniform_open(rng) < self.p {
                    count += 1;
                }
            }
            count
        } else {
            let u = sample_uniform_open(rng);
            self.quantile(u).unwrap_or(0)
        }
    }
}

// [cite:clinical_trials_statistics]
