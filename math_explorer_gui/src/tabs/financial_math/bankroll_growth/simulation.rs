//! Multi-state, slate, and drawdown-constrained compounding simulation routines.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::BankrollWagerModel;
use super::BankrollGrowthTool;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_kelly, drawdown_constrained_push_kelly,
    drawdown_constrained_quarter_line_kelly, expected_growth_rate, expected_value, kelly_fraction,
    push_expected_growth_rate, push_expected_value, push_kelly_fraction,
    quarter_line_expected_growth_rate, quarter_line_expected_value, quarter_line_kelly_fraction,
    solve_drawdown_constrained_slate, solve_simultaneous_slate, variants, BankrollFraction,
    DrawdownConstraint, Odds, PushOutcome, PushWager, QuarterLineOutcome, QuarterLineWager,
    SlateWager, UnitInterval,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

#[derive(Clone, Copy)]
struct PathWealths {
    fk: f64,
    hk: f64,
    qk: f64,
    rc: f64,
    act: f64,
}

impl PathWealths {
    const fn splat(w0: f64) -> Self {
        Self {
            fk: w0,
            hk: w0,
            qk: w0,
            rc: w0,
            act: w0,
        }
    }
}

/// Settles a binary wager at net profit multiplier `net_b`.
pub(super) fn settle_bet(bankroll: f64, fraction: &BankrollFraction, net_b: f64, win: bool) -> f64 {
    settle_raw_fraction(bankroll, fraction.value(), net_b, win)
}

fn settle_raw_fraction(bankroll: f64, stake_fraction: f64, net_b: f64, win: bool) -> f64 {
    let bet_amount = (bankroll * stake_fraction).max(0.0);
    if win {
        (bankroll + bet_amount * net_b).max(0.0)
    } else {
        (bankroll - bet_amount).max(0.0)
    }
}

fn finite_binary_growth_rate(p: &UnitInterval, o: &Odds, stake_fraction: f64) -> f64 {
    let clamped = stake_fraction.clamp(0.0, 0.9999);
    if let Ok(bf) = BankrollFraction::new(clamped) {
        let g = expected_growth_rate(p, o, &bf);
        if g.is_finite() {
            return g;
        }
    }
    let b = o.net_profit_multiplier();
    if clamped <= 0.0 {
        return 0.0;
    }
    if clamped >= 1.0 || 1.0 + b * clamped <= 0.0 {
        return -20.0;
    }
    let fallback = p.value() * (1.0 + b * clamped).ln() + p.complement() * (1.0 - clamped).ln();
    if fallback.is_finite() {
        fallback
    } else {
        -20.0
    }
}

fn finite_push_growth_rate(wager: &PushWager, stake_fraction: f64) -> f64 {
    let clamped = stake_fraction.clamp(0.0, 0.9999);
    if let Ok(bf) = BankrollFraction::new(clamped) {
        let g = push_expected_growth_rate(wager, &bf);
        if g.is_finite() {
            return g;
        }
    }
    -20.0
}

fn finite_quarter_growth_rate(wager: &QuarterLineWager, stake_fraction: f64) -> f64 {
    let clamped = stake_fraction.clamp(0.0, 0.9999);
    if let Ok(bf) = BankrollFraction::new(clamped) {
        let g = quarter_line_expected_growth_rate(wager, &bf);
        if g.is_finite() {
            return g;
        }
    }
    -20.0
}

fn ln_factorial(n: usize) -> f64 {
    let mut acc = 0.0;
    for k in 2..=n {
        acc += (k as f64).ln();
    }
    acc
}

fn finite_slate_growth_rate(wager: &PushWager, k: usize, game_f: f64) -> f64 {
    if game_f <= 0.0 {
        return 0.0;
    }
    let (pw, pp, pl) = (wager.p_win(), wager.p_push(), wager.p_loss());
    let net_b = wager.odds().net_profit_multiplier();
    let lnf_k = ln_factorial(k);
    let mut sum = 0.0;
    for nw in 0..=k {
        if nw > 0 && pw <= 0.0 {
            continue;
        }
        for np in 0..=(k - nw) {
            let nl = k - nw - np;
            if (np > 0 && pp <= 0.0) || (nl > 0 && pl <= 0.0) {
                continue;
            }
            let ln_c = lnf_k - ln_factorial(nw) - ln_factorial(np) - ln_factorial(nl);
            let ln_p = (if nw > 0 { (nw as f64) * pw.ln() } else { 0.0 })
                + (if np > 0 { (np as f64) * pp.ln() } else { 0.0 })
                + (if nl > 0 { (nl as f64) * pl.ln() } else { 0.0 });
            let prob = (ln_c + ln_p).exp();
            let wealth = 1.0 + game_f * ((nw as f64) * net_b - (nl as f64));
            if wealth <= 0.0 && prob > 0.0 {
                return -20.0;
            }
            if prob > 0.0 {
                sum += prob * wealth.ln();
            }
        }
    }
    if sum.is_finite() {
        sum
    } else {
        -20.0
    }
}

fn settle_push_step(bankroll: f64, frac: f64, net_b: f64, outcome: PushOutcome) -> f64 {
    let stake = (bankroll * frac).max(0.0);
    match outcome {
        PushOutcome::Win => (bankroll + stake * net_b).max(0.0),
        PushOutcome::Push => bankroll.max(0.0),
        PushOutcome::Loss => (bankroll - stake).max(0.0),
    }
}

fn settle_quarter_step(bankroll: f64, frac: f64, net_b: f64, outcome: QuarterLineOutcome) -> f64 {
    let stake = (bankroll * frac).max(0.0);
    match outcome {
        QuarterLineOutcome::FullWin => (bankroll + stake * net_b).max(0.0),
        QuarterLineOutcome::HalfWin => (bankroll + 0.5 * stake * net_b).max(0.0),
        QuarterLineOutcome::Push => bankroll.max(0.0),
        QuarterLineOutcome::HalfLoss => (bankroll - 0.5 * stake).max(0.0),
        QuarterLineOutcome::FullLoss => (bankroll - stake).max(0.0),
    }
}

impl BankrollGrowthTool {
    pub(super) fn recalculate(&mut self) {
        self.clear_trajectory_buffers();
        let o = match Odds::from_net_odds(self.odds) {
            Ok(o) => o,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        let dc = DrawdownConstraint::new(self.drawdown_alpha, self.drawdown_beta).ok();
        self.sync_ensemble_lab(&o, dc.as_ref());

        match self.wager_model {
            BankrollWagerModel::BinaryOrPush => self.simulate_binary_or_push(&o, dc.as_ref()),
            BankrollWagerModel::QuarterLine => self.simulate_quarter_line(&o, dc.as_ref()),
            BankrollWagerModel::SimultaneousSlate => self.simulate_slate(&o, dc.as_ref()),
        }
    }

    fn clear_trajectory_buffers(&mut self) {
        self.full_kelly_points.clear();
        self.half_kelly_points.clear();
        self.quarter_kelly_points.clear();
        self.risk_constrained_points.clear();
        self.active_stake_points.clear();
        self.kelly_f_star = 0.0;
        self.active_fraction = 0.0;
        self.active_growth_rate = 0.0;
        self.unit_expected_value = 0.0;
        self.error_msg = None;
    }

    fn sync_ensemble_lab(&mut self, o: &Odds, dc: Option<&DrawdownConstraint>) {
        let p_push = if self.wager_model == BankrollWagerModel::BinaryOrPush {
            self.push_probability
                .clamp(0.0, (0.99 - self.probability).max(0.0))
        } else {
            0.0
        };
        let default_dc =
            (self.drawdown_alpha - 0.70).abs() < 1e-12 && (self.drawdown_beta - 0.10).abs() < 1e-12;
        if p_push <= 1e-12 && default_dc {
            if let Ok(p) = UnitInterval::new(self.probability) {
                self.ensemble_lab.evaluate_in_place(
                    self.initial_bankroll,
                    &p,
                    o,
                    self.num_bets,
                    self.seed_counter,
                );
                self.ensemble_lab.sync_callouts(self.num_bets);
            }
            return;
        }
        if let Ok(wager) = PushWager::from_win_and_push(self.probability, p_push, *o) {
            self.ensemble_lab.evaluate_push_and_drawdown_in_place(
                self.initial_bankroll,
                &wager,
                dc,
                self.num_bets,
                self.seed_counter,
            );
            self.ensemble_lab.sync_callouts(self.num_bets);
        }
    }

    fn simulate_binary_or_push(&mut self, o: &Odds, dc: Option<&DrawdownConstraint>) {
        if self.push_probability <= 1e-12 {
            self.simulate_classic_binary(o, dc);
            return;
        }
        let wager = match PushWager::from_win_and_push(self.probability, self.push_probability, *o)
        {
            Ok(w) => w,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        self.unit_expected_value = push_expected_value(&wager);
        let fk = match push_kelly_fraction(&wager) {
            Ok(f) => f.value(),
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        let rc = dc
            .and_then(|c| drawdown_constrained_push_kelly(&wager, c).ok())
            .map_or(fk, |f| f.value());
        let active_f = (fk * self.stake_multiplier).clamp(0.0, 0.9999);
        self.kelly_f_star = fk;
        self.active_fraction = active_f;
        self.active_growth_rate = finite_push_growth_rate(&wager, active_f);
        self.run_push_paths(&wager, fk, rc, active_f);
    }

    fn simulate_classic_binary(&mut self, o: &Odds, dc: Option<&DrawdownConstraint>) {
        let p = match UnitInterval::new(self.probability) {
            Ok(p) => p,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        self.unit_expected_value = expected_value(&p, o);
        let tiers = (
            kelly_fraction(&p, o),
            variants::half_kelly(&p, o),
            variants::quarter_kelly(&p, o),
        );
        let (fk, hk, qk) = match tiers {
            (Ok(fk), Ok(hk), Ok(qk)) => (fk, hk, qk),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        let rc = dc
            .and_then(|c| drawdown_constrained_kelly(&p, o, c).ok())
            .unwrap_or(fk);
        let active_f = (fk.value() * self.stake_multiplier).clamp(0.0, 0.9999);
        self.kelly_f_star = fk.value();
        self.active_fraction = active_f;
        self.active_growth_rate = finite_binary_growth_rate(&p, o, active_f);

        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_b = o.net_profit_multiplier();
        let mut w = PathWealths::splat(self.initial_bankroll.max(0.0));
        self.push_trajectory_point(0.0, w);

        for i in 1..=self.num_bets {
            let win = rng.r#gen::<f64>() < p.value();
            w.fk = settle_bet(w.fk, &fk, net_b, win);
            w.hk = settle_bet(w.hk, &hk, net_b, win);
            w.qk = settle_bet(w.qk, &qk, net_b, win);
            w.rc = settle_bet(w.rc, &rc, net_b, win);
            w.act = settle_raw_fraction(w.act, active_f, net_b, win);
            self.push_trajectory_point(i as f64, w);
        }
    }

    fn run_push_paths(&mut self, wager: &PushWager, fk: f64, rc: f64, active_f: f64) {
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_b = wager.odds().net_profit_multiplier();
        let mut w = PathWealths::splat(self.initial_bankroll.max(0.0));
        self.push_trajectory_point(0.0, w);

        for i in 1..=self.num_bets {
            let outcome = wager.sample_outcome(rng.r#gen::<f64>());
            w.fk = settle_push_step(w.fk, fk, net_b, outcome);
            w.hk = settle_push_step(w.hk, 0.5 * fk, net_b, outcome);
            w.qk = settle_push_step(w.qk, 0.25 * fk, net_b, outcome);
            w.rc = settle_push_step(w.rc, rc, net_b, outcome);
            w.act = settle_push_step(w.act, active_f, net_b, outcome);
            self.push_trajectory_point(i as f64, w);
        }
    }

    fn simulate_quarter_line(&mut self, o: &Odds, dc: Option<&DrawdownConstraint>) {
        let p_fl = 1.0
            - self.probability
            - self.half_win_probability
            - self.push_probability
            - self.half_loss_probability;
        let wager = match QuarterLineWager::new(
            self.probability,
            self.half_win_probability,
            self.push_probability,
            self.half_loss_probability,
            p_fl,
            *o,
        ) {
            Ok(w) => w,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        self.unit_expected_value = quarter_line_expected_value(&wager);
        let fk = match quarter_line_kelly_fraction(&wager) {
            Ok(f) => f.value(),
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        let rc = dc
            .and_then(|c| drawdown_constrained_quarter_line_kelly(&wager, c).ok())
            .map_or(fk, |f| f.value());
        let active_f = (fk * self.stake_multiplier).clamp(0.0, 0.9999);
        self.kelly_f_star = fk;
        self.active_fraction = active_f;
        self.active_growth_rate = finite_quarter_growth_rate(&wager, active_f);
        self.run_quarter_paths(&wager, fk, rc, active_f);
    }

    fn run_quarter_paths(&mut self, wager: &QuarterLineWager, fk: f64, rc: f64, active_f: f64) {
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_b = wager.odds().net_profit_multiplier();
        let mut w = PathWealths::splat(self.initial_bankroll.max(0.0));
        self.push_trajectory_point(0.0, w);

        for i in 1..=self.num_bets {
            let outcome = wager.sample_outcome(rng.r#gen::<f64>());
            w.fk = settle_quarter_step(w.fk, fk, net_b, outcome);
            w.hk = settle_quarter_step(w.hk, 0.5 * fk, net_b, outcome);
            w.qk = settle_quarter_step(w.qk, 0.25 * fk, net_b, outcome);
            w.rc = settle_quarter_step(w.rc, rc, net_b, outcome);
            w.act = settle_quarter_step(w.act, active_f, net_b, outcome);
            self.push_trajectory_point(i as f64, w);
        }
    }

    fn simulate_slate(&mut self, o: &Odds, dc: Option<&DrawdownConstraint>) {
        let k = self.slate_size.clamp(2, 32);
        let pw = match PushWager::from_win_and_push(self.probability, self.push_probability, *o) {
            Ok(pw) => pw,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        self.unit_expected_value = push_expected_value(&pw);
        let sw = SlateWager::from_push(pw);
        let wagers = [sw; 32];
        let sol = match solve_simultaneous_slate(&wagers[..k], &mut self.slate_scratch) {
            Ok(s) => s,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
        let fk_per_game = sol.stakes().first().copied().unwrap_or(0.0);
        if fk_per_game <= 0.0 {
            self.error_msg = Some("Error: No positive edge on simultaneous slate".into());
            return;
        }
        let total_fk = sol.total_fraction();
        let active_total = (total_fk * self.stake_multiplier).clamp(0.0, 0.9999);
        let act_per_game = active_total / (k as f64);
        self.kelly_f_star = total_fk;
        self.active_fraction = active_total;
        self.active_growth_rate = finite_slate_growth_rate(&pw, k, act_per_game);
        let rc_per_game = dc
            .and_then(|c| {
                solve_drawdown_constrained_slate(&wagers[..k], c, &mut self.slate_scratch).ok()
            })
            .and_then(|s| s.stakes().first().copied())
            .unwrap_or(fk_per_game);
        self.run_slate_paths(&sw, k, fk_per_game, rc_per_game, act_per_game);
    }

    fn run_slate_paths(
        &mut self,
        sw: &SlateWager,
        k: usize,
        fk_game: f64,
        rc_game: f64,
        act_game: f64,
    ) {
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let mut w = PathWealths::splat(self.initial_bankroll.max(0.0));
        self.push_trajectory_point(0.0, w);

        for i in 1..=self.num_bets {
            let mut sum_x = 0.0;
            for _ in 0..k {
                sum_x += sw.sample_net_return(rng.r#gen::<f64>());
            }
            w.fk = (w.fk * (1.0 + fk_game * sum_x)).max(0.0);
            w.hk = (w.hk * (1.0 + 0.5 * fk_game * sum_x)).max(0.0);
            w.qk = (w.qk * (1.0 + 0.25 * fk_game * sum_x)).max(0.0);
            w.rc = (w.rc * (1.0 + rc_game * sum_x)).max(0.0);
            w.act = (w.act * (1.0 + act_game * sum_x)).max(0.0);
            self.push_trajectory_point(i as f64, w);
        }
    }

    fn push_trajectory_point(&mut self, x: f64, w: PathWealths) {
        self.full_kelly_points.push(PlotPoint::new(x, w.fk));
        self.half_kelly_points.push(PlotPoint::new(x, w.hk));
        self.quarter_kelly_points.push(PlotPoint::new(x, w.qk));
        self.risk_constrained_points.push(PlotPoint::new(x, w.rc));
        self.active_stake_points.push(PlotPoint::new(x, w.act));
    }
}

// [cite:clinical_trials_statistics]
