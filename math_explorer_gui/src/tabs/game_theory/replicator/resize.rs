//! Dynamic strategy dimension resizing (`2..=8`) for [`super::ReplicatorDynamicsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ReplicatorDynamicsTool, MAX_REPLICATOR_STRATEGIES, MIN_REPLICATOR_STRATEGIES};
use crate::framework::math_input::MathExpressionInputBar;
use math_commons::ast::Expr;
use nalgebra::{DMatrix, DVector};

impl ReplicatorDynamicsTool {
    pub fn strategy_count(&self) -> usize {
        self.payoff_matrix.nrows()
    }

    pub fn resize_strategies(&mut self, requested: usize) {
        let new_n = requested.clamp(MIN_REPLICATOR_STRATEGIES, MAX_REPLICATOR_STRATEGIES);
        let old_n = self.payoff_matrix.nrows();
        if new_n == old_n {
            return;
        }
        let keep = old_n.min(new_n);
        let mut next_payoff = DMatrix::zeros(new_n, new_n);
        for i in 0..keep {
            for j in 0..keep {
                next_payoff[(i, j)] = self.payoff_matrix[(i, j)];
            }
        }
        self.payoff_matrix = next_payoff;

        self.strategy_names.truncate(keep);
        for idx in keep..new_n {
            self.strategy_names.push(format!("Strategy {}", idx + 1));
        }

        let mut next_pop = vec![1.0 / (new_n as f64); new_n];
        for (i, slot) in next_pop.iter_mut().enumerate().take(keep) {
            *slot = self.initial_population[i];
        }
        self.initial_population = DVector::from_vec(next_pop);
        self.resize_symbolic_grids(keep, new_n);
        self.recompute_trajectory_preserving_playback();
    }

    fn resize_symbolic_grids(&mut self, keep: usize, new_n: usize) {
        self.symbolic_bars.truncate(keep);
        self.last_valid_exprs.truncate(keep);
        for i in 0..keep {
            self.symbolic_bars[i].truncate(keep);
            self.last_valid_exprs[i].truncate(keep);
            for _ in keep..new_n {
                self.symbolic_bars[i].push(
                    MathExpressionInputBar::new("", "0", Vec::new())
                        .with_axes(&[])
                        .with_inline_sliders(false),
                );
                self.last_valid_exprs[i].push(Expr::Number(0.0));
            }
        }
        for _ in keep..new_n {
            let row_bars = (0..new_n)
                .map(|_| {
                    MathExpressionInputBar::new("", "0", Vec::new())
                        .with_axes(&[])
                        .with_inline_sliders(false)
                })
                .collect();
            self.symbolic_bars.push(row_bars);
            self.last_valid_exprs.push(vec![Expr::Number(0.0); new_n]);
        }
        self.symbolic_compiled = vec![vec![None; new_n]; new_n];
        if self.symbolic_mode {
            self.sync_and_eval_symbolic_payoffs();
        }
    }
}
