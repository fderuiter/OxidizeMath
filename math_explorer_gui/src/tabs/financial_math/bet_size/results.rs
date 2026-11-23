//! Cached multi-state solver evaluation, sizing rows, and growth-curve series for [`super::BetSizeCalculatorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::bankroll_growth::homogeneous_slate_growth_rate;
use super::state::{BetSizingMode, UI_MAX_MARKET_ROWS};
use super::{BetSizeCalculatorTool, KellyStrategyRow};
use crate::web_io::PlatformInstant;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_kelly, drawdown_constrained_push_kelly,
    drawdown_constrained_quarter_line_kelly, expected_growth_rate, expected_value, kelly_fraction,
    push_expected_growth_rate, push_expected_value, push_kelly_fraction,
    quarter_line_expected_growth_rate, quarter_line_expected_value, quarter_line_kelly_fraction,
    solve_drawdown_constrained_slate, solve_mutually_exclusive_market, solve_simultaneous_slate,
    BankrollFraction, MarketOutcome, Odds, PushWager, QuarterLineWager, SlateWager, UnitInterval,
};

const CURVE_STEPS: usize = 120;

enum MarkerSlot {
    Quarter,
    Half,
    Full,
    Drawdown,
}

impl BetSizeCalculatorTool {
    pub(crate) fn recalculate(&mut self) {
        self.clear_cached_outputs();
        match self.mode {
            BetSizingMode::Binary => self.recalculate_binary(),
            BetSizingMode::PushSpread => self.recalculate_push(),
            BetSizingMode::QuarterLine => self.recalculate_quarter_line(),
            BetSizingMode::MutuallyExclusive => self.recalculate_market(),
            BetSizingMode::SimultaneousSlate => self.recalculate_slate(),
        }
    }

    fn clear_cached_outputs(&mut self) {
        self.growth_curve_points.clear();
        self.quarter_marker.clear();
        self.half_marker.clear();
        self.full_marker.clear();
        self.drawdown_marker.clear();
        self.strategy_rows.clear();
        self.ev_per_unit = None;
        self.summary_line = None;
        self.calc_error = None;
    }

    fn push_marker_and_row(
        &mut self,
        label: &str,
        frac: f64,
        g: f64,
        ev: f64,
        slot: Option<MarkerSlot>,
    ) {
        if g.is_finite() {
            if let Some(s) = slot {
                let pt = PlotPoint::new(frac, g);
                match s {
                    MarkerSlot::Quarter => self.quarter_marker.push(pt),
                    MarkerSlot::Half => self.half_marker.push(pt),
                    MarkerSlot::Full => self.full_marker.push(pt),
                    MarkerSlot::Drawdown => self.drawdown_marker.push(pt),
                }
            }
        }
        self.strategy_rows.push(KellyStrategyRow {
            label: label.to_string(),
            fraction: frac,
            bet_amount: (frac * self.bankroll).max(0.0),
            growth_rate: g,
            ev_per_unit: ev,
        });
    }

    fn populate_curve(&mut self, f_max: f64, mut eval_g: impl FnMut(f64) -> f64) {
        let cap = f_max.clamp(1e-4, 0.98);
        for idx in 0..=CURVE_STEPS {
            let f = cap * (idx as f64 / CURVE_STEPS as f64);
            let g = eval_g(f);
            if g.is_finite() {
                self.growth_curve_points.push(PlotPoint::new(f, g));
            }
        }
        self.zero_line_points = [PlotPoint::new(0.0, 0.0), PlotPoint::new(cap, 0.0)];
    }

    fn push_no_edge_row(&mut self, ev: f64) {
        self.strategy_rows.push(KellyStrategyRow {
            label: "No Wager (No Edge)".to_string(),
            fraction: 0.0,
            bet_amount: 0.0,
            growth_rate: 0.0,
            ev_per_unit: ev,
        });
    }

    fn recalculate_binary(&mut self) {
        let prob = match UnitInterval::new(self.probability_input / 100.0) {
            Ok(p) => p,
            Err(e) => {
                self.calc_error = Some(format!("Invalid Probability: {e}"));
                return;
            }
        };
        let odds = match self.active_odds() {
            Ok(o) => o,
            Err(e) => {
                self.calc_error = Some(format!("Invalid Odds: {e}"));
                return;
            }
        };
        let ev = expected_value(&prob, &odds);
        let fk = (ev > 0.0)
            .then(|| kelly_fraction(&prob, &odds).ok())
            .flatten();
        let rc = self
            .active_drawdown_constraint()
            .and_then(|dc| drawdown_constrained_kelly(&prob, &odds, &dc).ok());
        self.finish_single_wager(ev, odds.net_profit_multiplier(), fk, rc, |bf| {
            expected_growth_rate(&prob, &odds, bf)
        });
    }

    fn recalculate_push(&mut self) {
        let odds = match self.active_odds() {
            Ok(o) => o,
            Err(e) => {
                self.calc_error = Some(format!("Invalid Odds: {e}"));
                return;
            }
        };
        let wager = match PushWager::from_win_and_push(
            self.probability_input / 100.0,
            self.push_probability_pct / 100.0,
            odds,
        ) {
            Ok(w) => w,
            Err(e) => {
                self.calc_error = Some(format!("Invalid push distribution: {e}"));
                return;
            }
        };
        let rc = self
            .active_drawdown_constraint()
            .and_then(|dc| drawdown_constrained_push_kelly(&wager, &dc).ok());
        self.finish_single_wager(
            push_expected_value(&wager),
            odds.net_profit_multiplier(),
            push_kelly_fraction(&wager).ok(),
            rc,
            |bf| push_expected_growth_rate(&wager, bf),
        );
    }

    fn recalculate_quarter_line(&mut self) {
        let odds = match self.active_odds() {
            Ok(o) => o,
            Err(e) => {
                self.calc_error = Some(format!("Invalid Odds: {e}"));
                return;
            }
        };
        let wager = match QuarterLineWager::new(
            self.ql_full_win_pct / 100.0,
            self.ql_half_win_pct / 100.0,
            self.ql_push_pct / 100.0,
            self.ql_half_loss_pct / 100.0,
            self.ql_full_loss_pct / 100.0,
            odds,
        ) {
            Ok(w) => w,
            Err(e) => {
                self.calc_error = Some(format!("Quarter-line probabilities must sum to 100%: {e}"));
                return;
            }
        };
        let rc = self
            .active_drawdown_constraint()
            .and_then(|dc| drawdown_constrained_quarter_line_kelly(&wager, &dc).ok());
        self.finish_single_wager(
            quarter_line_expected_value(&wager),
            odds.net_profit_multiplier(),
            quarter_line_kelly_fraction(&wager).ok(),
            rc,
            |bf| quarter_line_expected_growth_rate(&wager, bf),
        );
    }

    fn finish_single_wager(
        &mut self,
        ev: f64,
        net_b: f64,
        fk_opt: Option<BankrollFraction>,
        rc: Option<BankrollFraction>,
        eval_g: impl Fn(&BankrollFraction) -> f64,
    ) {
        self.ev_per_unit = Some(ev);
        let f_cap = (0.98 / net_b).clamp(1e-4, 0.98);
        if let Some(fk) = fk_opt {
            let f_max = (fk.value() * 2.25).clamp(0.05_f64.min(f_cap), f_cap);
            self.populate_curve(f_max, |f| {
                BankrollFraction::new(f)
                    .map(|bf| eval_g(&bf))
                    .unwrap_or(f64::NAN)
            });
            self.push_single_tiers(
                fk,
                BankrollFraction::new(0.5 * fk.value()).ok(),
                BankrollFraction::new(0.25 * fk.value()).ok(),
                rc,
                ev,
                eval_g,
            );
        } else {
            self.populate_curve(f_cap.min(0.50), |f| {
                BankrollFraction::new(f)
                    .map(|bf| eval_g(&bf))
                    .unwrap_or(f64::NAN)
            });
            self.push_no_edge_row(ev);
        }
    }

    fn push_single_tiers(
        &mut self,
        fk: BankrollFraction,
        hk: Option<BankrollFraction>,
        qk: Option<BankrollFraction>,
        rc: Option<BankrollFraction>,
        ev: f64,
        eval_g: impl Fn(&BankrollFraction) -> f64,
    ) {
        if let Some(q) = qk {
            self.push_marker_and_row(
                "Quarter Kelly",
                q.value(),
                eval_g(&q),
                ev,
                Some(MarkerSlot::Quarter),
            );
        }
        if let Some(h) = hk {
            self.push_marker_and_row(
                "Half Kelly",
                h.value(),
                eval_g(&h),
                ev,
                Some(MarkerSlot::Half),
            );
        }
        self.push_marker_and_row(
            "Full Kelly",
            fk.value(),
            eval_g(&fk),
            ev,
            Some(MarkerSlot::Full),
        );
        if let Some(r) = rc {
            self.push_marker_and_row(
                "Drawdown-Constrained Kelly",
                r.value(),
                eval_g(&r),
                ev,
                Some(MarkerSlot::Drawdown),
            );
        }
    }

    fn recalculate_market(&mut self) {
        let n = self.market_outcome_count.min(UI_MAX_MARKET_ROWS);
        let default_mo =
            MarketOutcome::new(UnitInterval::new(0.5).unwrap(), Odds::new(2.0).unwrap());
        let mut outcomes = [default_mo; UI_MAX_MARKET_ROWS];
        for (slot, row) in outcomes.iter_mut().zip(self.market_rows.iter()).take(n) {
            let Ok(p) = UnitInterval::new(row.probability_pct / 100.0) else {
                self.calc_error = Some("Invalid market outcome probability".into());
                return;
            };
            let Ok(o) = Odds::new(row.decimal_odds) else {
                self.calc_error = Some("Invalid market outcome decimal odds".into());
                return;
            };
            *slot = MarketOutcome::new(p, o);
        }
        let alloc = match solve_mutually_exclusive_market(&outcomes[..n], &mut self.market_scratch)
        {
            Ok(a) => a,
            Err(e) => {
                self.calc_error = Some(format!("Market probabilities must sum to 100%: {e}"));
                return;
            }
        };
        let total_f = alloc.total_fraction();
        let cash_pct = (1.0 - total_f) * 100.0;
        let g_opt = alloc.expected_growth_rate();
        self.summary_line = Some(format!(
            "Reserve Rate r(S): {:.4} | Total Staked: {:.2}% | Cash Reserve: {cash_pct:.2}% | Log-Growth: {g_opt:+.6}",
            alloc.reserve_rate(),
            total_f * 100.0
        ));
        let mut stakes_copy = [0.0; UI_MAX_MARKET_ROWS];
        stakes_copy[..n].copy_from_slice(alloc.stakes());
        let portfolio_ev: f64 = (0..n)
            .map(|i| {
                stakes_copy[i]
                    * (outcomes[i].probability().value() * outcomes[i].odds().value() - 1.0)
            })
            .sum();
        self.ev_per_unit = Some(if total_f > 1e-12 {
            portfolio_ev / total_f
        } else {
            0.0
        });
        for i in 0..n {
            let f_i = stakes_copy[i];
            let ev_i = outcomes[i].probability().value() * outcomes[i].odds().value() - 1.0;
            self.push_marker_and_row(&format!("Outcome #{}", i + 1), f_i, g_opt, ev_i, None);
        }
        self.populate_market_ray(&outcomes[..n], &stakes_copy[..n], total_f);
    }

    fn populate_market_ray(&mut self, outcomes: &[MarketOutcome], stakes: &[f64], total_f: f64) {
        let eval_ray = |f_tot: f64| -> f64 {
            if total_f <= 1e-12 || f_tot <= 0.0 {
                return 0.0;
            }
            let scale = f_tot / total_f;
            let mut sum = 0.0;
            for (mo, &fi) in outcomes.iter().zip(stakes.iter()) {
                let w = 1.0 - f_tot + scale * fi * mo.odds().value();
                if w <= 0.0 {
                    return f64::NAN;
                }
                sum += mo.probability().value() * w.ln();
            }
            sum
        };
        let f_max = if total_f > 1e-9 {
            (total_f * 1.85).clamp(0.05, 0.98)
        } else {
            0.25
        };
        self.populate_curve(f_max, eval_ray);
        if total_f > 1e-9 {
            let q = 0.25 * total_f;
            let h = 0.50 * total_f;
            self.quarter_marker.push(PlotPoint::new(q, eval_ray(q)));
            self.half_marker.push(PlotPoint::new(h, eval_ray(h)));
            self.full_marker
                .push(PlotPoint::new(total_f, eval_ray(total_f)));
        }
    }

    fn recalculate_slate(&mut self) {
        let k = self.slate_count.clamp(1, 64);
        let pw = match Odds::from_net_odds(self.slate_net_odds).and_then(|o| {
            PushWager::from_win_and_push(self.slate_win_pct / 100.0, self.slate_push_pct / 100.0, o)
        }) {
            Ok(pw) => pw,
            Err(e) => {
                self.calc_error = Some(format!("Invalid slate wager: {e}"));
                return;
            }
        };
        let ev = push_expected_value(&pw);
        self.ev_per_unit = Some(ev);
        let wagers = [SlateWager::from_push(pw); 64];
        let t0 = PlatformInstant::now();
        let sol = match solve_simultaneous_slate(&wagers[..k], &mut self.slate_scratch) {
            Ok(s) => s,
            Err(e) => {
                self.calc_error = Some(format!("Slate solver error: {e}"));
                return;
            }
        };
        let per_game = sol.stakes().first().copied().unwrap_or(0.0);
        let total_f = sol.total_fraction();
        let g_opt = sol.expected_growth_rate();
        let kkt = sol.kkt_residual();
        let rc_pair = self.active_drawdown_constraint().and_then(|dc| {
            solve_drawdown_constrained_slate(&wagers[..k], &dc, &mut self.slate_scratch)
                .ok()
                .map(|rc| (rc.total_fraction(), rc.expected_growth_rate()))
        });
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.summary_line = Some(format!(
            "Solved {k}-Game Slate in {elapsed_ms:.3} ms (KKT {kkt:.2e}) | Per-Game: {:.4}% | Joint g(f*) = {g_opt:+.6}",
            per_game * 100.0
        ));
        self.populate_slate_rows_and_curve(
            &pw,
            k,
            SlateDisplayMetrics {
                per_game,
                total_f,
                g_opt,
                ev,
                rc_pair,
            },
        );
    }

    fn populate_slate_rows_and_curve(&mut self, pw: &PushWager, k: usize, m: SlateDisplayMetrics) {
        let eval_tot = |f_tot: f64| -> f64 {
            let g = homogeneous_slate_growth_rate(pw, k, f_tot / (k as f64));
            if g > -19.5 {
                g
            } else {
                f64::NAN
            }
        };
        let f_max = if m.total_f > 1e-9 {
            (m.total_f * 1.5).clamp(0.05, 0.98)
        } else {
            0.25
        };
        self.populate_curve(f_max, eval_tot);
        if m.total_f <= 1e-9 {
            self.push_no_edge_row(m.ev);
            return;
        }
        let q_tot = 0.25 * m.total_f;
        let h_tot = 0.50 * m.total_f;
        self.push_marker_and_row(
            "Per-Game Unconstrained Stake",
            m.per_game,
            m.g_opt,
            m.ev,
            None,
        );
        self.push_marker_and_row(
            "Quarter Slate Total",
            q_tot,
            eval_tot(q_tot),
            m.ev,
            Some(MarkerSlot::Quarter),
        );
        self.push_marker_and_row(
            "Half Slate Total",
            h_tot,
            eval_tot(h_tot),
            m.ev,
            Some(MarkerSlot::Half),
        );
        self.push_marker_and_row(
            &format!("Full Kelly Slate ({k} Games)"),
            m.total_f,
            m.g_opt,
            m.ev,
            Some(MarkerSlot::Full),
        );
        if let Some((rc_tot, rc_g)) = m.rc_pair {
            self.push_marker_and_row(
                "Drawdown-Constrained Slate",
                rc_tot,
                rc_g,
                m.ev,
                Some(MarkerSlot::Drawdown),
            );
        }
    }
}

struct SlateDisplayMetrics {
    per_game: f64,
    total_f: f64,
    g_opt: f64,
    ev: f64,
    rc_pair: Option<(f64, f64)>,
}

// [cite:clinical_trials_statistics]
