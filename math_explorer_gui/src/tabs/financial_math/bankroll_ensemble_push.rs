//! Three-state push settlement and drawdown-constrained ensemble helpers for [`super::BankrollEnsembleLab`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    interp_band_point, BankrollEnsembleLab, ENSEMBLE_BOOT_RESAMPLES, MAX_BOOTSTRAP_KNOTS,
    MAX_ENSEMBLE_BETS, PUSH_PREVIEW_PATH_COUNT,
};
use egui_plot::PlotPoint;
use math_explorer::prob::{
    bootstrap::{bca_bootstrap_median, percentile_bootstrap_median},
    SolverWorkspace,
};
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_push_kelly, push_expected_growth_rate, push_expected_value,
    push_kelly_fraction, BankrollFraction, DrawdownConstraint, PushOutcome, PushWager,
    UnitInterval,
};
use math_explorer::pure_math::statistics::ruin::{
    wilson_interval_95, BootstrapInterval, BootstrapMethod, DrawdownQuantiles,
    EnsembleSolvencyReport,
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
        self.reset_push_buffers(w0, barrier_floor, steps);
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ seed_counter.wrapping_mul(0x9E37_79B9));
        let mut boot_rng = OxidizeRng::from_seed(
            BASE_SEED ^ 0xB007_57AB ^ seed_counter.wrapping_mul(0x517C_C1B7_2722_0A95),
        );
        let stride = (steps / MAX_BOOTSTRAP_KNOTS).max(1);
        let mut knot_state = (0usize, w0, w0);
        let step_ev_factor = (1.0 + f_clamped * ev).max(0.0);
        let mut ew = w0;
        for t in 1..=steps {
            ew = (ew * step_ev_factor).max(0.0);
            let w_med = if g.is_finite() {
                (w0 * ((t as f64) * g).exp()).max(0.0)
            } else {
                0.0
            };
            let (emp_med, rc_med) =
                self.step_push_medians(&mut rng, wager, f_clamped, rc_frac, barrier_floor);
            let x = t as f64;
            self.expected_points.push(PlotPoint::new(x, ew));
            self.median_points.push(PlotPoint::new(x, w_med));
            self.empirical_median_points
                .push(PlotPoint::new(x, emp_med));
            self.drawdown_median_points.push(PlotPoint::new(x, rc_med));
            if t % stride == 0 || t == steps {
                self.record_push_bootstrap_knot(t, steps, emp_med, &mut knot_state, &mut boot_rng);
            }
        }
        self.last_solvency_report = self.build_push_solvency_report(steps);
        self.update_terminal_readouts(w0);
    }

    fn record_push_bootstrap_knot(
        &mut self,
        t: usize,
        steps: usize,
        emp_med: f64,
        knot_state: &mut (usize, f64, f64),
        boot_rng: &mut OxidizeRng,
    ) {
        let mut ws = SolverWorkspace::new(&mut self.boot_floats, &mut self.boot_indices);
        let res = match self.bootstrap_method {
            BootstrapMethod::Percentile => percentile_bootstrap_median(
                &self.path_wealths,
                ENSEMBLE_BOOT_RESAMPLES,
                0.95,
                boot_rng,
                &mut ws,
            ),
            BootstrapMethod::BCa => bca_bootstrap_median(
                &self.path_wealths,
                ENSEMBLE_BOOT_RESAMPLES,
                0.95,
                boot_rng,
                &mut ws,
            ),
        };
        let ci = res.unwrap_or(BootstrapInterval {
            point_estimate: emp_med,
            lower: emp_med,
            upper: emp_med,
            std_error: 0.0,
            confidence: 0.95,
            bias_z0: 0.0,
            acceleration: 0.0,
        });
        let (prev_t, prev_lo, prev_hi) = *knot_state;
        let span = (t - prev_t) as f64;
        for s in (prev_t + 1)..=t {
            let alpha = ((s - prev_t) as f64) / span;
            let lo = interp_band_point(prev_lo, ci.lower.max(0.0), alpha);
            let hi = interp_band_point(prev_hi, ci.upper.max(lo), alpha);
            self.median_ci_lower_points
                .push(PlotPoint::new(s as f64, lo));
            self.median_ci_upper_points
                .push(PlotPoint::new(s as f64, hi));
        }
        *knot_state = (t, ci.lower.max(0.0), ci.upper.max(ci.lower.max(0.0)));
        if t == steps {
            self.last_terminal_median_ci = Some(ci);
        }
    }

    fn reset_push_buffers(&mut self, w0: f64, barrier_floor: f64, steps: usize) {
        self.expected_points.clear();
        self.median_points.clear();
        self.empirical_median_points.clear();
        self.median_ci_lower_points.clear();
        self.median_ci_upper_points.clear();
        self.drawdown_median_points.clear();
        self.barrier_points.clear();
        self.last_terminal_median_ci = None;
        self.path_wealths.fill(w0);
        self.peak_wealths.fill(w0);
        self.max_drawdowns.fill(0.0);
        self.barrier_hits.fill(0);
        self.drawdown_path_wealths.fill(w0);

        self.expected_points.push(PlotPoint::new(0.0, w0));
        self.median_points.push(PlotPoint::new(0.0, w0));
        self.empirical_median_points.push(PlotPoint::new(0.0, w0));
        self.median_ci_lower_points.push(PlotPoint::new(0.0, w0));
        self.median_ci_upper_points.push(PlotPoint::new(0.0, w0));
        self.drawdown_median_points.push(PlotPoint::new(0.0, w0));
        self.barrier_points.push(PlotPoint::new(0.0, barrier_floor));
        self.barrier_points
            .push(PlotPoint::new(steps as f64, barrier_floor));
    }

    fn step_push_medians(
        &mut self,
        rng: &mut OxidizeRng,
        wager: &PushWager,
        f_custom: f64,
        f_rc: f64,
        barrier_floor: f64,
    ) -> (f64, f64) {
        let net_b = wager.odds().net_profit_multiplier();
        for idx in 0..PUSH_PREVIEW_PATH_COUNT {
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
        let emp_med = fast_even_median(&self.path_wealths, &mut self.scratch_sort);
        let rc_med = fast_even_median(&self.drawdown_path_wealths, &mut self.scratch_sort);
        (emp_med, rc_med)
    }

    fn build_push_solvency_report(&mut self, steps: usize) -> Option<EnsembleSolvencyReport> {
        let n = self.path_wealths.len();
        let ruin_count = self.barrier_hits.iter().filter(|&&h| h != 0).count();
        let wilson_95 = wilson_interval_95(ruin_count, n).ok()?;
        let terminal_mean_wealth = self.path_wealths.iter().sum::<f64>() / (n as f64);
        let terminal_median_wealth = fast_even_median(&self.path_wealths, &mut self.scratch_sort);
        self.scratch_sort.copy_from_slice(&self.max_drawdowns);
        self.scratch_sort.sort_unstable_by(f64::total_cmp);
        Some(EnsembleSolvencyReport {
            num_paths: n,
            num_steps: steps,
            ruin_count,
            ruin_probability: wilson_95.estimate,
            wilson_95,
            terminal_median_wealth,
            terminal_mean_wealth,
            drawdown_quantiles: DrawdownQuantiles {
                p50: sorted_quantile(&self.scratch_sort, 0.50),
                p90: sorted_quantile(&self.scratch_sort, 0.90),
                p95: sorted_quantile(&self.scratch_sort, 0.95),
                p99: sorted_quantile(&self.scratch_sort, 0.99),
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

fn sorted_quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// [cite:clinical_trials_statistics]
