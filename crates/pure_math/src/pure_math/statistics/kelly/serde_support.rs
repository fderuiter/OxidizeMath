// [cite:pure_math]

use super::{
    DrawdownConstraint, MAX_MARKET_OUTCOMES, MAX_SLATE_WAGERS, MarketAllocation, Odds,
    PROBABILITY_SUM_TOLERANCE, PushWager, QuarterLineWager, SlateEvalMode, SlateSolution,
    SlateWager,
};
use crate::error::KellyError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl<'de> Deserialize<'de> for DrawdownConstraint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawDrawdownConstraint {
            alpha: f64,
            beta: f64,
        }
        let raw = RawDrawdownConstraint::deserialize(deserializer)?;
        Self::new(raw.alpha, raw.beta).map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for PushWager {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawPushWager {
            p_win: f64,
            p_push: f64,
            p_loss: f64,
            odds: Odds,
        }
        let raw = RawPushWager::deserialize(deserializer)?;
        Self::new(raw.p_win, raw.p_push, raw.p_loss, raw.odds).map_err(serde::de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for QuarterLineWager {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawQuarterLineWager {
            p_full_win: f64,
            p_half_win: f64,
            p_push: f64,
            p_half_loss: f64,
            p_full_loss: f64,
            odds: Odds,
        }
        let raw = RawQuarterLineWager::deserialize(deserializer)?;
        Self::new(
            raw.p_full_win,
            raw.p_half_win,
            raw.p_push,
            raw.p_half_loss,
            raw.p_full_loss,
            raw.odds,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl Serialize for MarketAllocation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct MarketAllocationRef<'a> {
            stakes: &'a [f64],
            reserve_rate: f64,
            total_fraction: f64,
            expected_value: f64,
            expected_growth_rate: f64,
            kkt_residual: f64,
        }
        MarketAllocationRef {
            stakes: self.stakes(),
            reserve_rate: self.reserve_rate,
            total_fraction: self.total_fraction,
            expected_value: self.expected_value,
            expected_growth_rate: self.expected_growth_rate,
            kkt_residual: self.kkt_residual,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MarketAllocation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawMarketAllocation {
            stakes: Vec<f64>,
            reserve_rate: f64,
            total_fraction: f64,
            expected_value: f64,
            expected_growth_rate: f64,
            kkt_residual: f64,
        }
        let raw = RawMarketAllocation::deserialize(deserializer)?;
        validate_market_allocation(
            raw.stakes,
            raw.reserve_rate,
            raw.total_fraction,
            raw.expected_value,
            raw.expected_growth_rate,
            raw.kkt_residual,
        )
        .map_err(serde::de::Error::custom)
    }
}

fn validate_market_allocation(
    raw_stakes: Vec<f64>,
    reserve_rate: f64,
    total_fraction: f64,
    expected_value: f64,
    expected_growth_rate: f64,
    kkt_residual: f64,
) -> Result<MarketAllocation, KellyError> {
    if raw_stakes.is_empty() {
        return Err(KellyError::EmptyPortfolio);
    }
    if raw_stakes.len() > MAX_MARKET_OUTCOMES {
        return Err(KellyError::PortfolioTooLarge {
            count: raw_stakes.len(),
            max: MAX_MARKET_OUTCOMES,
        });
    }
    if let Some(&bad) = raw_stakes
        .iter()
        .find(|&&f| !f.is_finite() || !(0.0..=1.0).contains(&f))
    {
        return Err(KellyError::InvalidFraction { value: bad });
    }
    for val in [
        reserve_rate,
        total_fraction,
        expected_value,
        expected_growth_rate,
        kkt_residual,
    ] {
        if !val.is_finite() {
            return Err(KellyError::InvalidFraction { value: val });
        }
    }
    let mut stakes = [0.0; MAX_MARKET_OUTCOMES];
    stakes[..raw_stakes.len()].copy_from_slice(&raw_stakes);
    Ok(MarketAllocation {
        len: raw_stakes.len(),
        stakes,
        reserve_rate,
        total_fraction,
        expected_value,
        expected_growth_rate,
        kkt_residual,
    })
}

impl Serialize for SlateWager {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct SlateWagerRef<'a> {
            probs: &'a [f64],
            net_returns: &'a [f64],
        }
        SlateWagerRef {
            probs: &self.probs[..self.state_count],
            net_returns: &self.net_returns[..self.state_count],
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SlateWager {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawSlateWager {
            probs: Vec<f64>,
            net_returns: Vec<f64>,
        }
        let raw = RawSlateWager::deserialize(deserializer)?;
        validate_slate_wager(raw.probs, raw.net_returns).map_err(serde::de::Error::custom)
    }
}

fn validate_slate_wager(
    raw_probs: Vec<f64>,
    raw_returns: Vec<f64>,
) -> Result<SlateWager, KellyError> {
    if raw_probs.len() != raw_returns.len() {
        return Err(KellyError::DimensionMismatch {
            expected: raw_probs.len(),
            actual: raw_returns.len(),
        });
    }
    if !matches!(raw_probs.len(), 2 | 3 | 5) {
        return Err(KellyError::DimensionMismatch {
            expected: 2,
            actual: raw_probs.len(),
        });
    }
    for &p in &raw_probs {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(KellyError::InvalidProbability { value: p });
        }
    }
    let sum: f64 = raw_probs.iter().sum();
    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(KellyError::InvalidProbabilityDistribution { sum });
    }
    if let Some(&bad) = raw_returns.iter().find(|&&r| !r.is_finite() || r < -1.0) {
        return Err(KellyError::InvalidNetOdds { value: bad });
    }
    let mut probs = [0.0; 5];
    let mut net_returns = [0.0; 5];
    probs[..raw_probs.len()].copy_from_slice(&raw_probs);
    net_returns[..raw_returns.len()].copy_from_slice(&raw_returns);
    Ok(SlateWager {
        state_count: raw_probs.len(),
        probs,
        net_returns,
    })
}

impl Serialize for SlateSolution {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct SlateSolutionRef<'a> {
            stakes: &'a [f64],
            total_fraction: f64,
            expected_value: f64,
            expected_growth_rate: f64,
            drawdown_expectation: f64,
            kkt_residual: f64,
            eval_mode_used: SlateEvalMode,
        }
        SlateSolutionRef {
            stakes: self.stakes(),
            total_fraction: self.total_fraction,
            expected_value: self.expected_value,
            expected_growth_rate: self.expected_growth_rate,
            drawdown_expectation: self.drawdown_expectation,
            kkt_residual: self.kkt_residual,
            eval_mode_used: self.eval_mode_used,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SlateSolution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawSlateSolution {
            stakes: Vec<f64>,
            total_fraction: f64,
            expected_value: f64,
            expected_growth_rate: f64,
            drawdown_expectation: f64,
            kkt_residual: f64,
            eval_mode_used: SlateEvalMode,
        }
        let raw = RawSlateSolution::deserialize(deserializer)?;
        validate_slate_solution(
            raw.stakes,
            raw.total_fraction,
            raw.expected_value,
            raw.expected_growth_rate,
            raw.drawdown_expectation,
            raw.kkt_residual,
            raw.eval_mode_used,
        )
        .map_err(serde::de::Error::custom)
    }
}

fn validate_slate_solution(
    raw_stakes: Vec<f64>,
    total_fraction: f64,
    expected_value: f64,
    expected_growth_rate: f64,
    drawdown_expectation: f64,
    kkt_residual: f64,
    eval_mode_used: SlateEvalMode,
) -> Result<SlateSolution, KellyError> {
    if raw_stakes.is_empty() {
        return Err(KellyError::EmptyPortfolio);
    }
    if raw_stakes.len() > MAX_SLATE_WAGERS {
        return Err(KellyError::PortfolioTooLarge {
            count: raw_stakes.len(),
            max: MAX_SLATE_WAGERS,
        });
    }
    if let Some(&bad) = raw_stakes
        .iter()
        .find(|&&f| !f.is_finite() || !(0.0..=1.0).contains(&f))
    {
        return Err(KellyError::InvalidFraction { value: bad });
    }
    for val in [
        total_fraction,
        expected_value,
        expected_growth_rate,
        drawdown_expectation,
        kkt_residual,
    ] {
        if !val.is_finite() {
            return Err(KellyError::InvalidFraction { value: val });
        }
    }
    let mut stakes = [0.0; MAX_SLATE_WAGERS];
    stakes[..raw_stakes.len()].copy_from_slice(&raw_stakes);
    Ok(SlateSolution {
        len: raw_stakes.len(),
        stakes,
        total_fraction,
        expected_value,
        expected_growth_rate,
        drawdown_expectation,
        kkt_residual,
        eval_mode_used,
    })
}

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    #[verified_engine::verified]
    fn test_kelly_multi_state_serde_roundtrip_and_invariant_rejection() {
        let dc = DrawdownConstraint::new(0.70, 0.10).unwrap();
        let dc_json = serde_json::to_string(&dc).unwrap();
        let dc_restored: DrawdownConstraint = serde_json::from_str(&dc_json).unwrap();
        assert_eq!(dc_restored, dc);
        assert!(serde_json::from_str::<DrawdownConstraint>(r#"{"alpha":1.1,"beta":0.1}"#).is_err());

        let odds = Odds::from_american(-110.0).unwrap();
        let push = PushWager::new(0.50, 0.10, 0.40, odds).unwrap();
        let push_restored: PushWager =
            serde_json::from_str(&serde_json::to_string(&push).unwrap()).unwrap();
        assert_eq!(push_restored, push);
        assert!(
            serde_json::from_str::<PushWager>(
                r#"{"p_win":0.6,"p_push":0.3,"p_loss":0.3,"odds":2.0}"#
            )
            .is_err()
        );

        let ql = QuarterLineWager::new(0.44, 0.12, 0.0, 0.0, 0.44, odds).unwrap();
        let ql_restored: QuarterLineWager =
            serde_json::from_str(&serde_json::to_string(&ql).unwrap()).unwrap();
        assert_eq!(ql_restored, ql);
        assert!(
            serde_json::from_str::<QuarterLineWager>(
                r#"{"p_full_win":0.5,"p_half_win":0.5,"p_push":0.2,"p_half_loss":0.0,"p_full_loss":0.0,"odds":2.0}"#
            )
            .is_err()
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_kelly_market_and_slate_serde_roundtrip() {
        let outcomes = [
            MarketOutcome::new(UnitInterval::new(0.50).unwrap(), Odds::new(2.20).unwrap()),
            MarketOutcome::new(UnitInterval::new(0.28).unwrap(), Odds::new(4.00).unwrap()),
        ];
        let outcomes_restored: [MarketOutcome; 2] =
            serde_json::from_str(&serde_json::to_string(&outcomes).unwrap()).unwrap();
        assert_eq!(outcomes_restored, outcomes);
        assert!(
            serde_json::from_str::<MarketOutcome>(r#"{"probability":0.5,"odds":0.9}"#).is_err()
        );

        let mut m_scratch = MarketScratch::new();
        let alloc = solve_mutually_exclusive_market(&outcomes, &mut m_scratch).unwrap();
        let alloc_restored: MarketAllocation =
            serde_json::from_str(&serde_json::to_string(&alloc).unwrap()).unwrap();
        assert_eq!(alloc_restored.stakes().len(), alloc.stakes().len());
        for (&a, &b) in alloc_restored.stakes().iter().zip(alloc.stakes()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!((alloc_restored.expected_growth_rate - alloc.expected_growth_rate).abs() < 1e-14);

        let wagers =
            [SlateWager::binary(UnitInterval::new(0.58).unwrap(), Odds::new(2.0).unwrap()); 4];
        let wagers_restored: [SlateWager; 4] =
            serde_json::from_str(&serde_json::to_string(&wagers).unwrap()).unwrap();
        for (a, b) in wagers_restored.iter().zip(&wagers) {
            assert_eq!(a.state_count, b.state_count);
            for (&pa, &pb) in a.probs[..a.state_count]
                .iter()
                .zip(&b.probs[..b.state_count])
            {
                assert!((pa - pb).abs() < 1e-14);
            }
        }
        assert!(
            serde_json::from_str::<SlateWager>(r#"{"probs":[0.7,0.7],"net_returns":[1.0,-1.0]}"#)
                .is_err()
        );

        let mut s_scratch = SlateScratch::new();
        let sol = solve_simultaneous_slate(&wagers, &mut s_scratch).unwrap();
        let sol_restored: SlateSolution =
            serde_json::from_str(&serde_json::to_string(&sol).unwrap()).unwrap();
        assert_eq!(sol_restored.eval_mode_used, sol.eval_mode_used);
        for (&a, &b) in sol_restored.stakes().iter().zip(sol.stakes()) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!((sol_restored.expected_growth_rate - sol.expected_growth_rate).abs() < 1e-14);
    }
}
