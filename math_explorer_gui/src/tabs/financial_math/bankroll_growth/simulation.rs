//! Multi-state, slate, and drawdown-constrained compounding simulation routines.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::state::BankrollWagerModel;
use super::BankrollGrowthTool;
use egui_plot::PlotPoint;
use math_explorer::pure_math::statistics::kelly::{
    drawdown_constrained_kelly, drawdown_constrained_push_kelly,
    drawdown_constrained_quarter_line_kelly, kelly_fraction, push_kelly_fraction,
    quarter_line_kelly_fraction, solve_drawdown_constrained_slate, solve_simultaneous_slate,
    variants, BankrollFraction, DrawdownConstraint, Odds, PushOutcome, PushWager,
    QuarterLineOutcome, QuarterLineWager, SlateWager, UnitInterval,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

/// Settles a binary wager at net profit multiplier `net_b`.
pub(super) fn settle_bet(bankroll: f64, fraction: &BankrollFraction, net_b: f64, win: bool) -> f64 {
    let bet_amount = bankroll * fraction.value();
    if win {
        (bankroll + bet_amount * net_b).max(0.0)
    } else {
        (bankroll - bet_amount).max(0.0)
    }
}

fn settle_push_step(bankroll: f64, frac: f64, net_b: f64, outcome: PushOutcome) -> f64 {
    let stake = bankroll * frac;
    match outcome {
        PushOutcome::Win => (bankroll + stake * net_b).max(0.0),
        PushOutcome::Push => bankroll.max(0.0),
        PushOutcome::Loss => (bankroll - stake).max(0.0),
    }
}

fn settle_quarter_step(bankroll: f64, frac: f64, net_b: f64, outcome: QuarterLineOutcome) -> f64 {
    let stake = bankroll * frac;
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
        self.run_push_paths(&wager, fk, rc);
    }

    fn simulate_classic_binary(&mut self, o: &Odds, dc: Option<&DrawdownConstraint>) {
        let p = match UnitInterval::new(self.probability) {
            Ok(p) => p,
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
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

        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_b = o.net_profit_multiplier();
        let w0 = self.initial_bankroll.max(0.0);
        let (mut fk_w, mut hk_w, mut qk_w, mut rc_w) = (w0, w0, w0, w0);
        self.push_trajectory_point(0.0, fk_w, hk_w, qk_w, rc_w);

        for i in 1..=self.num_bets {
            let win = rng.r#gen::<f64>() < p.value();
            fk_w = settle_bet(fk_w, &fk, net_b, win);
            hk_w = settle_bet(hk_w, &hk, net_b, win);
            qk_w = settle_bet(qk_w, &qk, net_b, win);
            rc_w = settle_bet(rc_w, &rc, net_b, win);
            self.push_trajectory_point(i as f64, fk_w, hk_w, qk_w, rc_w);
        }
    }

    fn run_push_paths(&mut self, wager: &PushWager, fk: f64, rc: f64) {
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_b = wager.odds().net_profit_multiplier();
        let w0 = self.initial_bankroll.max(0.0);
        let (mut fk_w, mut hk_w, mut qk_w, mut rc_w) = (w0, w0, w0, w0);
        self.push_trajectory_point(0.0, fk_w, hk_w, qk_w, rc_w);

        for i in 1..=self.num_bets {
            let outcome = wager.sample_outcome(rng.r#gen::<f64>());
            fk_w = settle_push_step(fk_w, fk, net_b, outcome);
            hk_w = settle_push_step(hk_w, 0.5 * fk, net_b, outcome);
            qk_w = settle_push_step(qk_w, 0.25 * fk, net_b, outcome);
            rc_w = settle_push_step(rc_w, rc, net_b, outcome);
            self.push_trajectory_point(i as f64, fk_w, hk_w, qk_w, rc_w);
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

        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let net_b = o.net_profit_multiplier();
        let w0 = self.initial_bankroll.max(0.0);
        let (mut fk_w, mut hk_w, mut qk_w, mut rc_w) = (w0, w0, w0, w0);
        self.push_trajectory_point(0.0, fk_w, hk_w, qk_w, rc_w);

        for i in 1..=self.num_bets {
            let outcome = wager.sample_outcome(rng.r#gen::<f64>());
            fk_w = settle_quarter_step(fk_w, fk, net_b, outcome);
            hk_w = settle_quarter_step(hk_w, 0.5 * fk, net_b, outcome);
            qk_w = settle_quarter_step(qk_w, 0.25 * fk, net_b, outcome);
            rc_w = settle_quarter_step(rc_w, rc, net_b, outcome);
            self.push_trajectory_point(i as f64, fk_w, hk_w, qk_w, rc_w);
        }
    }

    fn simulate_slate(&mut self, o: &Odds, dc: Option<&DrawdownConstraint>) {
        let k = self.slate_size.clamp(2, 32);
        let sw = match PushWager::from_win_and_push(self.probability, self.push_probability, *o) {
            Ok(pw) => SlateWager::from_push(pw),
            Err(e) => {
                self.error_msg = Some(format!("Error: {e}"));
                return;
            }
        };
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
        let rc_per_game = dc
            .and_then(|c| {
                solve_drawdown_constrained_slate(&wagers[..k], c, &mut self.slate_scratch).ok()
            })
            .and_then(|s| s.stakes().first().copied())
            .unwrap_or(fk_per_game);
        self.run_slate_paths(&sw, k, fk_per_game, rc_per_game);
    }

    fn run_slate_paths(&mut self, sw: &SlateWager, k: usize, fk_game: f64, rc_game: f64) {
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ self.seed_counter);
        let w0 = self.initial_bankroll.max(0.0);
        let (mut fk_w, mut hk_w, mut qk_w, mut rc_w) = (w0, w0, w0, w0);
        self.push_trajectory_point(0.0, fk_w, hk_w, qk_w, rc_w);

        for i in 1..=self.num_bets {
            let mut sum_x = 0.0;
            for _ in 0..k {
                sum_x += sw.sample_net_return(rng.r#gen::<f64>());
            }
            fk_w = (fk_w * (1.0 + fk_game * sum_x)).max(0.0);
            hk_w = (hk_w * (1.0 + 0.5 * fk_game * sum_x)).max(0.0);
            qk_w = (qk_w * (1.0 + 0.25 * fk_game * sum_x)).max(0.0);
            rc_w = (rc_w * (1.0 + rc_game * sum_x)).max(0.0);
            self.push_trajectory_point(i as f64, fk_w, hk_w, qk_w, rc_w);
        }
    }

    fn push_trajectory_point(&mut self, x: f64, fk: f64, hk: f64, qk: f64, rc: f64) {
        self.full_kelly_points.push(PlotPoint::new(x, fk));
        self.half_kelly_points.push(PlotPoint::new(x, hk));
        self.quarter_kelly_points.push(PlotPoint::new(x, qk));
        self.risk_constrained_points.push(PlotPoint::new(x, rc));
    }
}

// [cite:clinical_trials_statistics]
