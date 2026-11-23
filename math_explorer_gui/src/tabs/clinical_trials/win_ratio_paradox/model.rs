//! Statistical cohort generation and zero-allocation regime evaluation for `WinRatioParadoxLab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    WinRatioParadoxConfig, WinRatioParadoxLab, WinRatioParadoxMetrics, BASE_SEED,
    MAX_SUBJECTS_PER_ARM, REGIME_GRID_DIM,
};
use crate::async_sim::atlas::atlas_regime_color;
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::applied::clinical_trials::hypothesis_testing::{
    HypothesisTest, TTestIndependent,
};
use math_explorer::applied::clinical_trials::survival_analysis::{
    kaplan_meier, try_estimate_hazard_ratio, Observation, TimePoint,
};
use math_explorer::applied::clinical_trials::types::{GroupData, SurvivalTime};
use math_explorer::applied::win_ratio::pair_comparison::{
    calculate_statistics, calculate_win_ratio, MatchedPairing, UnmatchedPairing,
};
use math_explorer::applied::win_ratio::sample_win_ratio::{
    calculate_sample_win_ratio, WinLossCounts,
};
use math_explorer::applied::win_ratio::WinRatioScratch;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

impl WinRatioParadoxLab {
    /// Clamps configuration parameters to valid finite domain bounds and active challenge locks.
    pub fn clamp_config_in_place(&mut self) {
        let c = &mut self.config;
        c.sample_size_per_arm = c.sample_size_per_arm.clamp(10, MAX_SUBJECTS_PER_ARM);
        c.horizon_months = sanitize_f64(c.horizon_months, 36.0, 12.0, 60.0);
        c.control_mortality_rate = sanitize_f64(c.control_mortality_rate, 0.60, 0.20, 0.85);
        c.mortality_reduction = sanitize_f64(c.mortality_reduction, 5.0 / 6.0, 0.0, 0.90);
        c.early_hosp_rate = sanitize_f64(c.early_hosp_rate, 0.90, 0.0, 0.95);
        c.early_hosp_mean_months = sanitize_f64(c.early_hosp_mean_months, 4.487_285, 1.0, 24.0);
        if self.active_challenge.is_some() {
            self.enforce_challenge_locks();
        }
    }

    /// Advances the deterministic `OxidizeRng` stream and recalculates cohort statistics.
    pub fn resample_seeded_cohort(&mut self) {
        self.config.seed_counter = self.config.seed_counter.wrapping_add(1).max(1);
        self.recalculate();
    }

    /// Resets cohort parameters to the canonical analytical severity-reversal preset.
    pub fn reset_starter_cohort(&mut self) {
        self.config = WinRatioParadoxConfig::default();
        if self.active_challenge.is_some() {
            self.enforce_challenge_locks();
        }
        self.recalculate();
    }

    /// Recomputes two-arm survival, `try_estimate_hazard_ratio`, `TTestIndependent`,
    /// `WinRatioAnalysis`, and the 2D `(mortality_reduction, early_hosp_rate)` regime map.
    pub fn recalculate(&mut self) {
        self.clamp_config_in_place();
        let n = self.config.sample_size_per_arm;
        populate_cohort_into_scratch(&self.config, &mut self.scratch);

        let treat_vecs = flat_to_subject_vecs(self.scratch.treatment_two_tier(n));
        let ctrl_vecs = flat_to_subject_vecs(self.scratch.control_two_tier(n));
        let unmatched_wl = self
            .analysis
            .evaluate_pairs(&treat_vecs, &ctrl_vecs, &UnmatchedPairing);
        let matched_wl = self
            .analysis
            .evaluate_pairs(&treat_vecs, &ctrl_vecs, &MatchedPairing);
        let breakdown = self
            .scratch
            .evaluate_staged_two_tier(n, n, self.config.matched_pairing);

        let (active_w, active_l) = if self.config.matched_pairing {
            matched_wl
        } else {
            unmatched_wl
        };
        let wr = calculate_win_ratio(active_w, active_l);
        let wr_stats = calculate_statistics(active_w, active_l);
        let matched_wr = calculate_win_ratio(matched_wl.0, matched_wl.1);
        let wl_counts = WinLossCounts::new(
            breakdown.tier1_losses.max(0) as u32,
            breakdown.tier1_wins.max(0) as u32,
            breakdown.tier2_losses.max(0) as u32,
            breakdown.tier2_wins.max(0) as u32,
        );
        let sample_wr = calculate_sample_win_ratio(&wl_counts);

        let (hr, t_stat, t_pval) = self.compute_survival_and_ttest(n);
        let (ci_low, ci_high, wr_p) = wr_stats
            .map(|s| (s.ci_low, s.ci_high, s.p_value))
            .unwrap_or((1.0, 1.0, 1.0));

        self.metrics = WinRatioParadoxMetrics {
            pooled_hazard_ratio: hr,
            win_ratio: wr,
            win_ratio_ci: (ci_low, ci_high),
            win_ratio_p_value: wr_p,
            matched_win_ratio: matched_wr,
            sample_win_ratio: sample_wr,
            t_statistic: t_stat,
            t_p_value: t_pval,
            wins: active_w,
            losses: active_l,
            ties: breakdown.ties,
            tier1_wins: breakdown.tier1_wins,
            tier1_losses: breakdown.tier1_losses,
            tier2_wins: breakdown.tier2_wins,
            tier2_losses: breakdown.tier2_losses,
            is_severity_reversal: hr > 1.0 && wr > 1.0,
        };

        self.populate_regime_map(false);
        self.sync_callouts();
    }

    fn compute_survival_and_ttest(&mut self, n: usize) -> (f64, f64, f64) {
        let horizon = self.config.horizon_months;
        let treat_flat = self.scratch.treatment_two_tier(n);
        let ctrl_flat = self.scratch.control_two_tier(n);

        let mut comp_treat_obs = Vec::with_capacity(n);
        let mut comp_ctrl_obs = Vec::with_capacity(n);
        let mut fatal_treat_obs = Vec::with_capacity(n);
        let mut fatal_ctrl_obs = Vec::with_capacity(n);
        let mut fatal_treat_vals = Vec::with_capacity(n);
        let mut fatal_ctrl_vals = Vec::with_capacity(n);

        for subj in treat_flat.as_chunks::<2>().0 {
            let first_t = subj[0].min(subj[1]);
            comp_treat_obs.push(make_obs(first_t, first_t < horizon - 1e-9));
            fatal_treat_obs.push(make_obs(subj[0], subj[0] < horizon - 1e-9));
            fatal_treat_vals.push(subj[0]);
        }
        for subj in ctrl_flat.as_chunks::<2>().0 {
            let first_t = subj[0].min(subj[1]);
            comp_ctrl_obs.push(make_obs(first_t, first_t < horizon - 1e-9));
            fatal_ctrl_obs.push(make_obs(subj[0], subj[0] < horizon - 1e-9));
            fatal_ctrl_vals.push(subj[0]);
        }

        self.km_comp_treat = km_to_step_points(&kaplan_meier(&comp_treat_obs), horizon);
        self.km_comp_ctrl = km_to_step_points(&kaplan_meier(&comp_ctrl_obs), horizon);
        self.km_fatal_treat = km_to_step_points(&kaplan_meier(&fatal_treat_obs), horizon);
        self.km_fatal_ctrl = km_to_step_points(&kaplan_meier(&fatal_ctrl_obs), horizon);

        let hr = try_estimate_hazard_ratio(&comp_treat_obs, &comp_ctrl_obs).unwrap_or(1.0);
        let (t_stat, t_pval) = match (
            GroupData::new(fatal_treat_vals),
            GroupData::new(fatal_ctrl_vals),
        ) {
            (Ok(g_t), Ok(g_c)) => TTestIndependent::new(0.05)
                .test((&g_t, &g_c))
                .map(|r| (r.statistic, r.p_value))
                .unwrap_or((0.0, 1.0)),
            _ => (0.0, 1.0),
        };
        (hr, t_stat, t_pval)
    }

    /// Evaluates the 2D `(mortality_reduction, early_hosp_rate)` regime grid in place with zero allocations.
    pub fn populate_regime_map(&mut self, high_contrast: bool) {
        evaluate_regime_grid_zero_alloc(&self.config, &mut self.scratch);
        for row in 0..REGIME_GRID_DIM {
            // Row 0 of ColorImage corresponds to top of plot (`y = max`).
            let data_row = REGIME_GRID_DIM - 1 - row;
            for col in 0..REGIME_GRID_DIM {
                let (wr, hr) = self.scratch.regime_cell(col, data_row);
                let slot = classify_regime_slot(wr, hr);
                let intensity = ((wr - 0.6) / 1.6).clamp(0.0, 1.0);
                let px_idx = flatten_2d_index(col, row, REGIME_GRID_DIM);
                self.regime_pixels[px_idx] = atlas_regime_color(slot, 4, intensity, high_contrast);
            }
        }
        self.texture_dirty = true;
    }

    fn sync_callouts(&mut self) {
        self.callout_manager.clear_auto();
        let m = &self.metrics;
        let label = if m.is_severity_reversal {
            format!(
                "Severity Reversal: HR={:.2} vs WR={:.2}",
                m.pooled_hazard_ratio, m.win_ratio
            )
        } else {
            format!(
                "Concordant: HR={:.2}, WR={:.2}",
                m.pooled_hazard_ratio, m.win_ratio
            )
        };
        self.callout_manager.set_auto_callout(
            "wr_regime_cursor",
            [self.config.mortality_reduction, self.config.early_hosp_rate],
            label,
            [18.0, -20.0],
        );
    }
}

/// Zero-allocation 2D grid sweep suitable for `VerifiedEngine::run_verified` verification.
pub fn evaluate_regime_grid_zero_alloc(
    base: &WinRatioParadoxConfig,
    scratch: &mut WinRatioScratch,
) {
    let n = base.sample_size_per_arm.clamp(10, MAX_SUBJECTS_PER_ARM);
    let horizon = base.horizon_months;
    let mut probe = base.clone();
    probe.seed_counter = 0;

    for row in 0..REGIME_GRID_DIM {
        let hosp_rate = 0.95 * (row as f64) / ((REGIME_GRID_DIM - 1) as f64);
        probe.early_hosp_rate = hosp_rate;
        for col in 0..REGIME_GRID_DIM {
            let mort_red = 0.90 * (col as f64) / ((REGIME_GRID_DIM - 1) as f64);
            probe.mortality_reduction = mort_red;
            populate_deterministic_cohort(&probe, scratch);
            let bd = scratch.evaluate_staged_two_tier(n, n, probe.matched_pairing);
            let wr = calculate_win_ratio(bd.total_wins(), bd.total_losses());
            let hr = fast_composite_hr(
                scratch.treatment_two_tier(n),
                scratch.control_two_tier(n),
                horizon,
            );
            scratch.set_regime_cell(col, row, wr, hr);
        }
    }
    populate_cohort_into_scratch(base, scratch);
}

fn populate_cohort_into_scratch(cfg: &WinRatioParadoxConfig, scratch: &mut WinRatioScratch) {
    populate_deterministic_cohort(cfg, scratch);
    if cfg.seed_counter == 0 {
        return;
    }
    let n = cfg.sample_size_per_arm.clamp(10, MAX_SUBJECTS_PER_ARM);
    let horizon = cfg.horizon_months;
    let mut rng = OxidizeRng::from_seed(BASE_SEED ^ cfg.seed_counter);
    {
        let treat = scratch.treatment_two_tier_mut(n);
        for v in treat.iter_mut() {
            if *v < horizon - 1e-9 {
                let jitter = (rng.r#gen::<f64>() - 0.5) * 0.6;
                *v = (*v + jitter).clamp(0.4, horizon - 0.2);
            }
        }
    }
    {
        let ctrl = scratch.control_two_tier_mut(n);
        for pair in ctrl.as_chunks_mut::<2>().0 {
            if pair[0] < horizon - 1e-9 {
                let jitter = (rng.r#gen::<f64>() - 0.5) * 0.6;
                let d = (pair[0] + jitter).clamp(0.4, horizon - 0.2);
                pair[0] = d;
                pair[1] = d;
            }
        }
    }
}

fn populate_deterministic_cohort(cfg: &WinRatioParadoxConfig, scratch: &mut WinRatioScratch) {
    let n = cfg.sample_size_per_arm.clamp(10, MAX_SUBJECTS_PER_ARM);
    let horizon = cfg.horizon_months;
    let ctrl_deaths = ((n as f64) * cfg.control_mortality_rate).round() as usize;
    let ctrl_deaths = ctrl_deaths.clamp(1, n - 1);
    let treat_deaths = ((ctrl_deaths as f64) * (1.0 - cfg.mortality_reduction)).round() as usize;
    let treat_deaths = treat_deaths.min(n - 1);
    let treat_hosps = ((n as f64) * cfg.early_hosp_rate).round() as usize;
    let treat_hosps = treat_hosps.min(n);

    {
        let ctrl = scratch.control_two_tier_mut(n);
        let scale = horizon / 36.0;
        for i in 0..n {
            let d = if i < ctrl_deaths {
                ((6.580_42 + 0.4 * (i as f64)) * scale).clamp(0.5, horizon - 0.2)
            } else {
                horizon
            };
            ctrl[2 * i] = d;
            ctrl[2 * i + 1] = d;
        }
    }
    {
        let treat = scratch.treatment_two_tier_mut(n);
        let scale = horizon / 36.0;
        let center = 0.5 * (treat_hosps.saturating_sub(1) as f64);
        for i in 0..n {
            let d = if i < treat_deaths {
                let late_step = 4.0 / (treat_deaths.saturating_sub(1).max(1) as f64);
                ((28.0 + late_step * (i as f64)) * scale).clamp(1.0, horizon - 0.2)
            } else {
                horizon
            };
            let h = if i < treat_hosps {
                let offset = 0.08 * ((i as f64) - center);
                (cfg.early_hosp_mean_months + offset).clamp(0.4, d.min(horizon - 0.2))
            } else {
                horizon
            };
            treat[2 * i] = d;
            treat[2 * i + 1] = h;
        }
    }
}

fn fast_composite_hr(treat_flat: &[f64], ctrl_flat: &[f64], horizon: f64) -> f64 {
    let mut ev_t = 0.0;
    let mut sum_t = 0.0;
    for subj in treat_flat.as_chunks::<2>().0 {
        let first = subj[0].min(subj[1]);
        if first < horizon - 1e-9 {
            ev_t += 1.0;
        }
        sum_t += first;
    }
    let mut ev_c = 0.0;
    let mut sum_c = 0.0;
    for subj in ctrl_flat.as_chunks::<2>().0 {
        let first = subj[0].min(subj[1]);
        if first < horizon - 1e-9 {
            ev_c += 1.0;
        }
        sum_c += first;
    }
    if sum_t <= 0.0 || sum_c <= 0.0 || ev_c <= 0.0 {
        return 1.0;
    }
    (ev_t / sum_t) / (ev_c / sum_c)
}

fn classify_regime_slot(wr: f64, hr: f64) -> u8 {
    if hr > 1.0 && wr > 1.0 {
        // Paradoxical severity reversal!
        2
    } else if hr <= 1.0 && wr > 1.0 {
        // Concordant treatment benefit
        1
    } else if hr > 1.0 && wr <= 1.0 {
        // Concordant treatment harm
        3
    } else {
        0
    }
}

fn sanitize_f64(val: f64, fallback: f64, min: f64, max: f64) -> f64 {
    if val.is_finite() {
        val.clamp(min, max)
    } else {
        fallback
    }
}

fn make_obs(time: f64, event: bool) -> Observation {
    let st = SurvivalTime::new(time.max(0.0)).unwrap_or_else(|_| SurvivalTime::new(0.0).unwrap());
    Observation {
        time: st,
        event_occurred: event,
    }
}

fn flat_to_subject_vecs(flat: &[f64]) -> Vec<Vec<f64>> {
    flat.as_chunks::<2>()
        .0
        .iter()
        .map(|c| vec![c[0], c[1]])
        .collect()
}

fn km_to_step_points(curve: &[TimePoint], horizon: f64) -> Vec<[f64; 2]> {
    let mut pts = Vec::with_capacity(curve.len() * 2 + 2);
    pts.push([0.0, 1.0]);
    let mut prev_s = 1.0;
    for pt in curve {
        pts.push([pt.time, prev_s]);
        pts.push([pt.time, pt.survival_probability]);
        prev_s = pt.survival_probability;
    }
    pts.push([horizon, prev_s]);
    pts
}

// [cite:win_ratio]
