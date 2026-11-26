//! Three-state push settlement and drawdown-constrained ensemble helpers for [`super::BankrollEnsembleLab`].

use super::{
    sorted_percentile, BankrollEnsembleLab, MAX_ENSEMBLE_BETS, MIN_POSITIVE_WEALTH,
    PUSH_PREVIEW_PATH_COUNT,
};
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_push_kelly, push_expected_growth_rate, push_expected_value,
    push_kelly_fraction, BankrollFraction, DrawdownConstraint, PushOutcome, PushWager,
    UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::{
    wilson_interval_95, DrawdownQuantiles, EnsembleSolvencyReport,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

impl BankrollEnsembleLab {
    /// Evaluates three-state push settlements and drawdown-constrained ensemble paths with zero allocations.
    pub fn evaluate_push_and_drawdown_in_place(
        &mut self,
        initial_bankroll: f64,
        wager: &PushWager,
        constraint: Option<&DrawdownConstraint>,
        num_bets: usize,
        seed_counter: u64,
    ) {
        if wager.p_push() <= 1e-12 {
            if let Ok(p) = UnitInterval::new(wager.p_win()) {
                self.eval_binary(
                    initial_bankroll,
                    &p,
                    &wager.odds(),
                    constraint,
                    num_bets,
                    seed_counter,
                );
                return;
            }
        }
        self.clamp_fraction();
        let (steps, w0) = (num_bets.min(MAX_ENSEMBLE_BETS), initial_bankroll.max(1e-6));
        self.last_initial_bankroll = w0;
        let Ok(frac) = BankrollFraction::new(self.custom_fraction) else {
            return;
        };
        let f_star = push_kelly_fraction(wager).ok().map(|fk| fk.value());
        self.last_growth_rate = push_expected_growth_rate(wager, &frac);
        self.last_f_star = f_star;
        self.last_drawdown_f = constraint
            .and_then(|dc| drawdown_constrained_push_kelly(wager, dc).ok())
            .map(|fr| fr.value());
        self.is_ruin_zone = f_star.is_some_and(|fs| self.custom_fraction > 2.0 * fs);
        self.last_barrier_risk = None;
        self.run_push_series(w0, wager, push_expected_value(wager), steps, seed_counter);
    }

    fn run_push_series(
        &mut self,
        w0: f64,
        wager: &PushWager,
        ev: f64,
        steps: usize,
        seed_counter: u64,
    ) {
        let (f_clamped, g, rc_frac, barrier_floor) = (
            self.custom_fraction,
            self.last_growth_rate,
            self.last_drawdown_f.unwrap_or(0.0),
            w0 * self.barrier_floor_ratio,
        );
        self.reset_step_zero(w0, barrier_floor, steps);
        self.peak_wealths.fill(w0);
        self.max_drawdowns.fill(0.0);
        self.barrier_hits.fill(0);
        self.drawdown_path_wealths.fill(w0);

        let mut rng = OxidizeRng::new(BASE_SEED).with_stream(seed_counter);
        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;
        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let w_med = if g.is_finite() {
                (w0 * ((t as f64) * g).exp()).max(0.0)
            } else {
                0.0
            };
            let (p05, p25, emp_med, p75, p95, rc_med) =
                self.step_push_quantiles(&mut rng, wager, f_clamped, rc_frac, barrier_floor);
            let x = t as f64;
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.empirical_median_points
                .push(PlotPoint::new(x, emp_med));
            self.drawdown_median_points.push(PlotPoint::new(x, rc_med));
            self.p05_points.push(PlotPoint::new(x, p05));
            self.p25_points.push(PlotPoint::new(x, p25));
            self.p75_points.push(PlotPoint::new(x, p75));
            self.p95_points.push(PlotPoint::new(x, p95));
            self.median_wealth_buf[t] = emp_med.max(MIN_POSITIVE_WEALTH);
        }
        self.rebuild_fan_polygons_and_tail_risk(steps, w0, PUSH_PREVIEW_PATH_COUNT);
        self.last_solvency_report = self.build_push_solvency_report(steps);
        self.update_terminal_readouts(w0);
        self.update_terminal_median_bca(true, &mut rng);
    }

    fn step_push_quantiles(
        &mut self,
        rng: &mut OxidizeRng,
        wager: &PushWager,
        f_custom: f64,
        f_rc: f64,
        barrier_floor: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let net_b = wager.odds().net_profit_multiplier();
        let n = PUSH_PREVIEW_PATH_COUNT;
        for idx in 0..n {
            let outcome = wager.sample_outcome(rng.r#gen::<f64>());
            let w_new = settle_raw_push(self.path_wealths[idx], f_custom, net_b, outcome);
            self.path_wealths[idx] = w_new;
            if w_new > self.peak_wealths[idx] {
                self.peak_wealths[idx] = w_new;
            }
            let peak = self.peak_wealths[idx];
            if peak > 0.0 {
                let dd = (peak - w_new) / peak;
                if dd > self.max_drawdowns[idx] {
                    self.max_drawdowns[idx] = dd;
                }
            }
            if w_new <= barrier_floor {
                self.barrier_hits[idx] = 1;
            }
            self.drawdown_path_wealths[idx] =
                settle_raw_push(self.drawdown_path_wealths[idx], f_rc, net_b, outcome);
        }
        let sort_slice = &mut self.scratch_sort[..n];
        sort_slice.copy_from_slice(&self.path_wealths[..n]);
        sort_slice.sort_unstable_by(f64::total_cmp);
        let mid = n / 2;
        let emp_med = 0.5 * (sort_slice[mid - 1] + sort_slice[mid]);
        let p05 = sorted_percentile(sort_slice, 0.05);
        let p25 = sorted_percentile(sort_slice, 0.25);
        let p75 = sorted_percentile(sort_slice, 0.75);
        let p95 = sorted_percentile(sort_slice, 0.95);
        let rc_med = fast_even_median(&self.drawdown_path_wealths[..n], sort_slice);
        (p05, p25, emp_med, p75, p95, rc_med)
    }

    fn build_push_solvency_report(&mut self, steps: usize) -> Option<EnsembleSolvencyReport> {
        let n = PUSH_PREVIEW_PATH_COUNT;
        let ruin_count = self.barrier_hits.iter().filter(|&&h| h != 0).count();
        let wilson_95 = wilson_interval_95(ruin_count, n).ok()?;
        let terminal_mean_wealth = self.path_wealths[..n].iter().sum::<f64>() / (n as f64);
        let sort_slice = &mut self.scratch_sort[..n];
        let terminal_median_wealth = fast_even_median(&self.path_wealths[..n], sort_slice);
        sort_slice.copy_from_slice(&self.max_drawdowns[..n]);
        sort_slice.sort_unstable_by(f64::total_cmp);
        Some(EnsembleSolvencyReport {
            num_paths: n,
            num_steps: steps,
            ruin_count,
            ruin_probability: wilson_95.estimate,
            wilson_95,
            terminal_median_wealth,
            terminal_mean_wealth,
            drawdown_quantiles: DrawdownQuantiles {
                p50: sorted_percentile(sort_slice, 0.50),
                p90: sorted_percentile(sort_slice, 0.90),
                p95: sorted_percentile(sort_slice, 0.95),
                p99: sorted_percentile(sort_slice, 0.99),
            },
        })
    }
}

fn settle_raw_push(wealth: f64, frac: f64, net_b: f64, outcome: PushOutcome) -> f64 {
    let stake = wealth * frac;
    match outcome {
        PushOutcome::Win => (wealth + stake * net_b).max(0.0),
        PushOutcome::Push => wealth.max(0.0),
        PushOutcome::Loss => (wealth - stake).max(0.0),
    }
}

fn fast_even_median(values: &[f64], scratch: &mut [f64]) -> f64 {
    scratch.copy_from_slice(values);
    let mid = scratch.len() / 2;
    let (left, mid_val, _) = scratch.select_nth_unstable_by(mid, f64::total_cmp);
    let max_left = left.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    0.5 * (max_left + *mid_val)
}

// [cite:clinical_trials_statistics]
