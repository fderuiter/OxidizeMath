//! Zero-allocation composite `WinRatioAnalysis`, `BivariateWeibullModel` `PR(c)`, and `TTestIndependent` engine.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::cohort::{ComparatorKind, PairingMode, DEFAULT_PRC_SAMPLES, MAX_PRC_SAMPLES};
use super::WinRatioTool;
use math_commons::registry::TOLERANCE_FAST;
use math_explorer::applied::clinical_trials::hypothesis_testing::{
    HypothesisTest, TTestIndependent, TestResult,
};
use math_explorer::applied::clinical_trials::types::GroupData;
use math_explorer::applied::win_ratio::bmi::calculate_bmi;
use math_explorer::applied::win_ratio::pair_comparison::{
    calculate_statistics, ComparisonResult, HigherIsBetter, LowerIsBetter, MatchedPairing,
    OutcomeComparator, ThresholdComparator, UnmatchedPairing, WinRatioAnalysis, WinRatioStats,
};
use math_explorer::applied::win_ratio::probability_win_ratio::{
    calculate_probability_win_ratio, ProbabilityWinRatioContext,
};
use math_explorer::applied::win_ratio::sample_win_ratio::{
    calculate_confidence_interval, calculate_sample_win_ratio,
    calculate_significance_test_statistic, calculate_win_proportion, WinLossCounts,
};
use math_explorer::applied::win_ratio::simulation::{
    conditional_survival_x_given_t_control, conditional_survival_x_given_t_treatment,
    marginal_survival_t_control, marginal_survival_t_treatment, pdf_t_control, pdf_t_treatment,
    pdf_x_given_t_control, pdf_x_given_t_treatment, win_ratio_parameter, SimulationParams,
};

/// Per-tier composite hierarchy win/loss/tie resolution summary.
#[derive(Debug, Clone, Copy, Default)]
pub struct TierResolution {
    /// 1-based hierarchy tier index (`1..=3`).
    pub tier_index: usize,
    /// Pairs resolved as Treatment wins on this tier.
    pub tier_wins: i32,
    /// Pairs resolved as Treatment losses on this tier.
    pub tier_losses: i32,
    /// Pairs remaining tied after evaluating through this tier.
    pub remaining_ties: i32,
    /// Cumulative Win Ratio (`wins / losses`) through this tier.
    pub cumulative_wr: f64,
    /// Cumulative 95% CI lower bound through this tier.
    pub cumulative_ci_low: f64,
    /// Cumulative 95% CI upper bound through this tier.
    pub cumulative_ci_high: f64,
}

/// Pocock two-tier sample win-ratio summary backed by [`WinLossCounts`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PocockTwoTierSummary {
    /// Categorized win/loss counts `(Na, Nb, Nc, Nd)`.
    pub counts: Option<WinLossCounts>,
    /// Sample win ratio `R = (Nb + Nd) / (Na + Nc)`.
    pub sample_wr: f64,
    /// Untied win proportion `pw = Nw / (Nw + Nl)`.
    pub win_proportion: f64,
    /// 95% confidence interval `(ci_low, ci_high)`.
    pub ci: (f64, f64),
    /// Standardized asymptotic normal Z-statistic.
    pub z_statistic: f64,
}

/// Computed empirical and Weibull `PR(c)` outputs cached on [`WinRatioTool`].
#[derive(Debug, Clone, Default)]
pub struct WinRatioEvaluation {
    /// Total Treatment vs Control subject pairs evaluated.
    pub total_pairs: usize,
    /// Total wins across all active hierarchy tiers.
    pub wins: i32,
    /// Total losses across all active hierarchy tiers.
    pub losses: i32,
    /// Total ties remaining after the final tier.
    pub ties: i32,
    /// Overall `WinRatioStats` (`win_ratio`, `ci_low`, `ci_high`, `p_value`).
    pub stats: Option<WinRatioStats>,
    /// Per-tier resolution breakdown (up to 3 tiers).
    pub tiers: [TierResolution; 3],
    /// Number of active tiers (`2` or `3`).
    pub active_tier_count: usize,
    /// Pocock 2-tier `WinLossCounts` breakdown.
    pub pocock: PocockTwoTierSummary,
    /// Secondary continuous endpoint independent t-test result.
    pub ttest_result: Option<TestResult>,
    /// Informative status when secondary `TTestIndependent` is unavailable.
    pub ttest_note: Option<String>,
    /// Reference Treatment BMI (`kg/m^2`) from `calculate_bmi`.
    pub treat_ref_bmi: f64,
    /// Reference Control BMI (`kg/m^2`) from `calculate_bmi`.
    pub ctrl_ref_bmi: f64,
    /// Uncensored theoretical Weibull win ratio `PR_W = 1 / theta^alpha`.
    pub weibull_pr_w: f64,
    /// Terminal `PR(c_max)` at the configured follow-up horizon.
    pub terminal_pr_c: f64,
    /// Terminal win probability `W(c_max)`.
    pub terminal_w_c: f64,
    /// Terminal loss probability `L(c_max)`.
    pub terminal_l_c: f64,
}

pub(crate) fn build_comparator(kind: ComparatorKind) -> Box<dyn OutcomeComparator<f64>> {
    match kind {
        ComparatorKind::HigherIsBetter => Box::new(HigherIsBetter),
        ComparatorKind::LowerIsBetter => Box::new(LowerIsBetter),
        ComparatorKind::Threshold { threshold } => Box::new(ThresholdComparator::new(threshold)),
    }
}

impl WinRatioTool {
    /// Compiles `WinRatioAnalysis`, validates cohort and `GroupData`, and reserves curve buffers.
    pub(crate) fn prepare_workspace(&mut self) {
        self.error_message = None;
        self.compiled_analysis = None;
        self.compiled_tier_comparators.clear();
        self.ttest_groups = None;
        self.eval.ttest_note = None;

        let parsed =
            match super::cohort::parse_composite_cohort(&self.cohort_text, self.derive_bmi_tier3) {
                Ok(c) => c,
                Err(err) => {
                    self.error_message = Some(err);
                    return;
                }
            };

        if self.pairing_mode == PairingMode::Matched
            && parsed.treatment.len() != parsed.control.len()
        {
            self.error_message = Some(format!(
                "Matched pairing requires equal arm sizes (Treatment N={}, Control N={})",
                parsed.treatment.len(),
                parsed.control.len()
            ));
            return;
        }

        let tier_count = parsed.treatment[0].len().min(3);
        let kinds = [self.tier1_kind, self.tier2_kind, self.tier3_kind];
        let mut analysis = WinRatioAnalysis::new();
        for &kind in kinds.iter().take(tier_count) {
            analysis = analysis.add_strategy(build_comparator(kind));
            self.compiled_tier_comparators.push(build_comparator(kind));
        }
        self.prepare_ttest_groups(&parsed.treatment, &parsed.control, tier_count);
        self.parsed_cohort = parsed;
        self.compiled_analysis = Some(analysis);
        self.reserve_curve_buffers();
    }

    fn prepare_ttest_groups(
        &mut self,
        treatment: &[Vec<f64>],
        control: &[Vec<f64>],
        tier_count: usize,
    ) {
        let idx = self.ttest_tier_index.min(tier_count.saturating_sub(1));
        let t_vals: Vec<f64> = treatment.iter().map(|r| r[idx]).collect();
        let c_vals: Vec<f64> = control.iter().map(|r| r[idx]).collect();
        match (GroupData::new(t_vals), GroupData::new(c_vals)) {
            (Ok(g1), Ok(g2)) => {
                self.ttest_groups = Some((g1, g2));
            }
            (Err(e), _) | (_, Err(e)) => {
                self.eval.ttest_note = Some(e.to_string());
            }
        }
    }

    fn reserve_curve_buffers(&mut self) {
        let cap = self.prc_samples.clamp(16, MAX_PRC_SAMPLES);
        if self.prc_points.capacity() < cap {
            self.prc_points.reserve(cap - self.prc_points.capacity());
        }
        if self.win_prob_points.capacity() < cap {
            self.win_prob_points
                .reserve(cap - self.win_prob_points.capacity());
        }
        if self.loss_prob_points.capacity() < cap {
            self.loss_prob_points
                .reserve(cap - self.loss_prob_points.capacity());
        }
        if self.surv_treat_points.capacity() < cap {
            self.surv_treat_points
                .reserve(cap - self.surv_treat_points.capacity());
        }
        if self.surv_ctrl_points.capacity() < cap {
            self.surv_ctrl_points
                .reserve(cap - self.surv_ctrl_points.capacity());
        }
    }

    /// Executes the zero-allocation statistical and Clenshaw-Curtis integration hot path.
    pub fn evaluate_hot_path(&mut self) {
        self.eval.treat_ref_bmi = calculate_bmi(self.treat_weight_kg, self.treat_height_m);
        self.eval.ctrl_ref_bmi = calculate_bmi(self.ctrl_weight_kg, self.ctrl_height_m);
        self.evaluate_weibull_prc_curves();
        if self.error_message.is_some() {
            return;
        }
        self.evaluate_empirical_hierarchy();
        self.evaluate_secondary_ttest();
    }

    fn evaluate_empirical_hierarchy(&mut self) {
        let Some(ref analysis) = self.compiled_analysis else {
            return;
        };
        let treat = &self.parsed_cohort.treatment;
        let ctrl = &self.parsed_cohort.control;
        let (wins, losses) = match self.pairing_mode {
            PairingMode::Unmatched => analysis.evaluate_pairs(treat, ctrl, &UnmatchedPairing),
            PairingMode::Matched => analysis.evaluate_pairs(treat, ctrl, &MatchedPairing),
        };
        let total_pairs = match self.pairing_mode {
            PairingMode::Unmatched => treat.len() * ctrl.len(),
            PairingMode::Matched => treat.len(),
        };
        self.eval.total_pairs = total_pairs;
        self.eval.wins = wins;
        self.eval.losses = losses;
        self.eval.ties = (total_pairs as i32) - wins - losses;
        self.eval.stats = calculate_statistics(wins, losses);
        self.compute_tier_breakdown(total_pairs);
    }

    fn compute_tier_breakdown(&mut self, total_pairs: usize) {
        let tier_count = self.compiled_tier_comparators.len().min(3);
        self.eval.active_tier_count = tier_count;
        let mut tier_w = [0_i32; 3];
        let mut tier_l = [0_i32; 3];

        let mut tally_pair = |s1: &[f64], s2: &[f64]| {
            for k in 0..tier_count {
                match self.compiled_tier_comparators[k].compare(&s1[k], &s2[k]) {
                    ComparisonResult::Win => {
                        tier_w[k] += 1;
                        break;
                    }
                    ComparisonResult::Loss => {
                        tier_l[k] += 1;
                        break;
                    }
                    ComparisonResult::Tie => {}
                }
            }
        };

        match self.pairing_mode {
            PairingMode::Unmatched => {
                for s1 in &self.parsed_cohort.treatment {
                    for s2 in &self.parsed_cohort.control {
                        tally_pair(s1, s2);
                    }
                }
            }
            PairingMode::Matched => {
                for (s1, s2) in self
                    .parsed_cohort
                    .treatment
                    .iter()
                    .zip(self.parsed_cohort.control.iter())
                {
                    tally_pair(s1, s2);
                }
            }
        }

        let mut cum_w = 0_i32;
        let mut cum_l = 0_i32;
        for k in 0..tier_count {
            cum_w += tier_w[k];
            cum_l += tier_l[k];
            let stats = calculate_statistics(cum_w, cum_l);
            self.eval.tiers[k] = TierResolution {
                tier_index: k + 1,
                tier_wins: tier_w[k],
                tier_losses: tier_l[k],
                remaining_ties: (total_pairs as i32) - cum_w - cum_l,
                cumulative_wr: stats.map_or(1.0, |s| s.win_ratio),
                cumulative_ci_low: stats.map_or(1.0, |s| s.ci_low),
                cumulative_ci_high: stats.map_or(1.0, |s| s.ci_high),
            };
        }
        self.update_pocock_two_tier_summary(tier_w, tier_l);
    }

    fn update_pocock_two_tier_summary(&mut self, tier_w: [i32; 3], tier_l: [i32; 3]) {
        let counts = WinLossCounts::new(
            tier_l[0].max(0) as u32,
            tier_w[0].max(0) as u32,
            tier_l[1].max(0) as u32,
            tier_w[1].max(0) as u32,
        );
        let nw = counts.n_wins();
        let nl = counts.n_losses();
        self.eval.pocock = PocockTwoTierSummary {
            counts: Some(counts),
            sample_wr: calculate_sample_win_ratio(&counts),
            win_proportion: calculate_win_proportion(nw, nl),
            ci: calculate_confidence_interval(nw, nl),
            z_statistic: calculate_significance_test_statistic(nw, nl),
        };
    }

    fn evaluate_secondary_ttest(&mut self) {
        self.eval.ttest_result = None;
        if let Some((ref g1, ref g2)) = self.ttest_groups {
            if let Ok(res) = TTestIndependent::new(self.ttest_alpha).test((g1, g2)) {
                self.eval.ttest_result = Some(res);
            }
        }
    }

    fn evaluate_weibull_prc_curves(&mut self) {
        self.prc_points.clear();
        self.win_prob_points.clear();
        self.loss_prob_points.clear();
        self.surv_treat_points.clear();
        self.surv_ctrl_points.clear();

        let params = SimulationParams::new(
            self.weibull_lambda1,
            self.weibull_lambda2,
            self.weibull_alpha,
            self.weibull_theta,
        );
        let ctrl_model = params.control_model();
        let treat_model = params.treatment_model();
        self.eval.weibull_pr_w = win_ratio_parameter(&params);

        let n = self
            .prc_samples
            .clamp(16, MAX_PRC_SAMPLES)
            .max(DEFAULT_PRC_SAMPLES.min(16));
        let c_min = (self.horizon_c * 0.04).clamp(0.05, 0.5);
        let span = (self.horizon_c - c_min).max(0.25);

        for i in 0..n {
            let frac = (i as f64) / ((n - 1) as f64);
            let c = c_min + frac * span;
            let s0_c = ctrl_model.marginal_survival_t(c);
            let s1_c = treat_model.marginal_survival_t(c);
            self.surv_ctrl_points.push([c, s0_c]);
            self.surv_treat_points.push([c, s1_c]);

            let ctx = ProbabilityWinRatioContext::new(s0_c, s1_c, c, TOLERANCE_FAST);
            let w_c = ctx.calculate_win_probability(
                |t| marginal_survival_t_treatment(t, &params),
                |t| pdf_t_control(t, &params),
                |x| conditional_survival_x_given_t_treatment(x, c, &params),
                |x| pdf_x_given_t_control(x, c, &params),
            );
            let l_c = ctx.calculate_loss_probability(
                |t| marginal_survival_t_control(t, &params),
                |t| pdf_t_treatment(t, &params),
                |x| conditional_survival_x_given_t_control(x, c, &params),
                |x| pdf_x_given_t_treatment(x, c, &params),
            );
            let pr_c = calculate_probability_win_ratio(w_c, l_c);
            self.win_prob_points.push([c, w_c.max(0.0)]);
            self.loss_prob_points.push([c, l_c.max(0.0)]);
            if pr_c.is_finite() {
                self.prc_points.push([c, pr_c]);
                self.eval.terminal_pr_c = pr_c;
            }
            self.eval.terminal_w_c = w_c;
            self.eval.terminal_l_c = l_c;
        }
    }
}

// [cite:win_ratio]
// [cite:clinical_trials_statistics]
