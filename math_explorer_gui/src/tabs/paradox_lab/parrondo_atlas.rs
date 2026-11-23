//! Zero-allocation 2D parameter regime atlas for [`super::parrondo::ParrondoJumpRatchetTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::parrondo::{ParrondoParadoxTool, ParrondoVariant, next_markov_state, win_probs};
use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent, RegimeCartographer,
    atlas_regime_color,
};
use eframe::egui;
use math_explorer::nalgebra::DMatrix;
use math_explorer::pure_math::statistics::markov::dtmc::{MarkovChain, StateType};
use math_explorer::pure_math::statistics::markov::tensor::{TimeIndex, TransitionTensor};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Discrete dynamical drift regime for `ParrondoJumpRatchetTool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ParrondoJumpRegime {
    /// Combined mixture or SSA jump-ratchet drift is strictly negative.
    LosingDrift = 0,
    /// Paradox active: both pure games lose ($\mu_A < 0, \mu_B < 0$) while coupled ratchet wins.
    ParadoxRatchet = 1,
    /// Near-zero fair drift boundary ($|\mu| \le 10^{-4}$).
    NeutralBoundary = 2,
    /// Positive drift where at least one pure game already has non-negative expectation.
    DirectPositiveDrift = 3,
}

impl ParrondoJumpRegime {
    /// Converts a raw `u8` regime index into [`ParrondoJumpRegime`].
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::LosingDrift,
            1 => Self::ParadoxRatchet,
            2 => Self::NeutralBoundary,
            _ => Self::DirectPositiveDrift,
        }
    }

    /// Human-readable regime label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LosingDrift => "Losing Drift (μ < 0)",
            Self::ParadoxRatchet => "Paradox Ratchet (μ_A<0, μ_B<0, μ>0)",
            Self::NeutralBoundary => "Critical Fair Boundary (μ ≈ 0)",
            Self::DirectPositiveDrift => "Unbiased / Direct Positive Drift",
        }
    }
}

pub(super) fn compute_abb_tensor_metrics(
    p_a: &DMatrix<f64>,
    p_b: &DMatrix<f64>,
    win_a: &[f64; 4],
    win_b: &[f64; 4],
) -> (f64, f64, f64) {
    let n = p_a.nrows();
    let (t_min, t_mid, t_one, t_max) = (
        TimeIndex::new(0.0).unwrap(),
        TimeIndex::new(0.5).unwrap(),
        TimeIndex::new(1.0).unwrap(),
        TimeIndex::new(2.0).unwrap(),
    );
    let mut tensor = TransitionTensor::new(n, t_min, t_max);
    tensor.add_time_slice(t_min, p_a.clone()).unwrap();
    tensor.add_time_slice(t_one, p_b.clone()).unwrap();
    tensor.add_time_slice(t_max, p_b.clone()).unwrap();

    let s0 = tensor.transition_matrix_at(t_min).unwrap();
    let s1 = tensor.transition_matrix_at(t_one).unwrap();
    let s2 = tensor.transition_matrix_at(t_max).unwrap();
    let chain_abb = MarkovChain::new(&s0 * &s1 * &s2, vec![StateType::Transient; n]).unwrap();
    let pi0 = chain_abb.stationary_distribution().unwrap();
    let pi1 = s0.transpose() * &pi0;
    let pi2 = s1.transpose() * &pi1;
    let gain = |pi: &math_explorer::nalgebra::DVector<f64>, w: &[f64; 4]| -> f64 {
        (0..n).map(|s| pi[s] * (2.0 * w[s] - 1.0)).sum()
    };
    let drift_abb = (gain(&pi0, win_a) + gain(&pi1, win_b) + gain(&pi2, win_b)) / 3.0;
    let avg = tensor.average_transition(t_min, t_max, 3).unwrap();
    let mid = tensor.transition_matrix_at(t_mid).unwrap();
    (drift_abb, avg[(0, 1)], mid[(0, 1)])
}

/// Stack-allocated 2D cartographer for discrete convex mixtures and continuous-time jump ratchets.
#[derive(Clone, Debug)]
pub struct ParrondoJumpCartographer {
    axes: AtlasAxisPair,
    variant: ParrondoVariant,
    base_eps: f64,
    base_gamma: f64,
    base_lambda: f64,
}

impl ParrondoJumpCartographer {
    /// Constructs a [`ParrondoJumpCartographer`] for `(x_key, y_key)`.
    #[must_use]
    pub fn new(
        theory: &HashMap<String, ParameterConstraint>,
        keys: (&'static str, &'static str),
        variant: ParrondoVariant,
        base_eps: f64,
        base_gamma: f64,
        base_lambda: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: jump_axis_spec(theory, keys.0),
                y: jump_axis_spec(theory, keys.1),
            },
            variant,
            base_eps,
            base_gamma,
            base_lambda,
        }
    }
}

fn jump_axis_spec(
    theory: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    match key {
        "mix_gamma" => AtlasAxisSpec::from_theory(
            theory,
            "mix_gamma",
            "Mixture Weight γ (Game A)",
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        ),
        "jump_switch_rate" => AtlasAxisSpec::from_theory(
            theory,
            "jump_switch_rate",
            "SSA Jump Rate λ",
            ParameterConstraint {
                min: 0.1,
                max: 5.0,
                step: 0.05,
            },
        ),
        _ => AtlasAxisSpec::from_theory(
            theory,
            "bias_epsilon",
            "Coin Bias Disadvantage (ε)",
            ParameterConstraint {
                min: 0.0,
                max: 0.05,
                step: 0.0005,
            },
        ),
    }
}

/// Evaluates `(drift_a, drift_b, drift_mix, drift_abb, drift_ssa)` strictly on `[f64; 8]` stack arrays.
#[must_use]
pub fn evaluate_jump_ratchet_stack(
    variant: ParrondoVariant,
    eps: f64,
    gamma: f64,
    lambda: f64,
) -> (f64, f64, f64, f64, f64) {
    let (win_a, win_b, n) = win_probs(variant, eps.clamp(0.0, 0.20));
    let g = gamma.clamp(0.0, 1.0);
    let mut win_mix = [0.0; 4];
    for s in 0..n {
        win_mix[s] = g * win_a[s] + (1.0 - g) * win_b[s];
    }
    let pa = build_dtmc_4(variant, &win_a, n);
    let pb = build_dtmc_4(variant, &win_b, n);
    let pmix = build_dtmc_4(variant, &win_mix, n);
    let drift_a = 2.0 * win_a[0] - 1.0;
    let pi_b = solve_linear_8(&dtmc_to_generator_8(&pb, n), n);
    let pi_mix = solve_linear_8(&dtmc_to_generator_8(&pmix, n), n);
    let drift_b: f64 = (0..n).map(|s| pi_b[s] * (2.0 * win_b[s] - 1.0)).sum();
    let drift_mix: f64 = (0..n).map(|s| pi_mix[s] * (2.0 * win_mix[s] - 1.0)).sum();

    let p_abb = mul_4(&mul_4(&pa, &pb, n), &pb, n);
    let pi0 = solve_linear_8(&dtmc_to_generator_8(&p_abb, n), n);
    let pi1 = step_4(&pa, &pi0, n);
    let pi2 = step_4(&pb, &pi1, n);
    let g0: f64 = (0..n).map(|s| pi0[s] * (2.0 * win_a[s] - 1.0)).sum();
    let g1: f64 = (0..n).map(|s| pi1[s] * (2.0 * win_b[s] - 1.0)).sum();
    let g2: f64 = (0..n).map(|s| pi2[s] * (2.0 * win_b[s] - 1.0)).sum();
    let drift_abb = (g0 + g1 + g2) / 3.0;

    let q_ssa = build_ssa_generator_8(variant, &win_a, &win_b, n, lambda.clamp(0.05, 10.0));
    let pi_ssa = solve_linear_8(&q_ssa, 2 * n);
    let mut drift_ssa = 0.0;
    for s in 0..n {
        drift_ssa += pi_ssa[s] * (2.0 * win_a[s] - 1.0);
        drift_ssa += pi_ssa[n + s] * (2.0 * win_b[s] - 1.0);
    }
    (drift_a, drift_b, drift_mix, drift_abb, drift_ssa)
}

fn build_dtmc_4(variant: ParrondoVariant, probs: &[f64; 4], n: usize) -> [[f64; 4]; 4] {
    let mut mat = [[0.0; 4]; 4];
    for s in 0..n {
        mat[s][next_markov_state(variant, s, true)] += probs[s];
        mat[s][next_markov_state(variant, s, false)] += 1.0 - probs[s];
    }
    mat
}

fn mul_4(lhs: &[[f64; 4]; 4], rhs: &[[f64; 4]; 4], n: usize) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for i in 0..n {
        for k in 0..n {
            for j in 0..n {
                out[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }
    out
}

fn step_4(mat: &[[f64; 4]; 4], dist: &[f64; 8], n: usize) -> [f64; 8] {
    let mut next = [0.0; 8];
    for i in 0..n {
        for j in 0..n {
            next[j] += dist[i] * mat[i][j];
        }
    }
    next
}

fn dtmc_to_generator_8(p: &[[f64; 4]; 4], n: usize) -> [[f64; 8]; 8] {
    let mut q = [[0.0; 8]; 8];
    for i in 0..n {
        for j in 0..n {
            q[i][j] = p[i][j] - if i == j { 1.0 } else { 0.0 };
        }
    }
    q
}

fn build_ssa_generator_8(
    variant: ParrondoVariant,
    win_a: &[f64; 4],
    win_b: &[f64; 4],
    n: usize,
    lambda: f64,
) -> [[f64; 8]; 8] {
    let mut q = [[0.0; 8]; 8];
    let sw_ab = lambda * 1.6;
    let sw_ba = lambda * 0.8;
    for s in 0..n {
        let win_s = next_markov_state(variant, s, true);
        let loss_s = next_markov_state(variant, s, false);
        q[s][s] -= 1.0 + sw_ab;
        q[s][win_s] += win_a[s];
        q[s][loss_s] += 1.0 - win_a[s];
        q[s][n + s] += sw_ab;

        let b_idx = n + s;
        q[b_idx][b_idx] -= 1.0 + sw_ba;
        q[b_idx][n + win_s] += win_b[s];
        q[b_idx][n + loss_s] += 1.0 - win_b[s];
        q[b_idx][s] += sw_ba;
    }
    q
}

fn solve_linear_8(q: &[[f64; 8]; 8], m: usize) -> [f64; 8] {
    let mut aug = [[0.0_f64; 9]; 8];
    for r in 0..(m - 1) {
        for c in 0..m {
            aug[r][c] = q[c][r];
        }
    }
    for val in aug[m - 1].iter_mut().take(m) {
        *val = 1.0;
    }
    aug[m - 1][m] = 1.0;

    for col in 0..m {
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for (row, candidate) in aug.iter().enumerate().take(m).skip(col + 1) {
            let mag = candidate[col].abs();
            if mag > best {
                best = mag;
                pivot = row;
            }
        }
        if best <= 1e-14 {
            let mut uni = [0.0; 8];
            for v in uni.iter_mut().take(m) {
                *v = 1.0 / (m as f64);
            }
            return uni;
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        let inv = 1.0 / aug[col][col];
        let pivot_row = aug[col];
        for row_vals in aug.iter_mut().take(m).skip(col + 1) {
            let factor = row_vals[col] * inv;
            for (dst, &src) in row_vals[col..=m].iter_mut().zip(&pivot_row[col..=m]) {
                *dst -= factor * src;
            }
        }
    }

    let mut pi = [0.0; 8];
    for r in (0..m).rev() {
        let mut rhs = aug[r][m];
        for c in (r + 1)..m {
            rhs -= aug[r][c] * pi[c];
        }
        pi[r] = (rhs / aug[r][r]).max(0.0);
    }
    let sum: f64 = pi.iter().take(m).sum();
    if sum > 1e-12 && sum.is_finite() {
        for val in pi.iter_mut().take(m) {
            *val /= sum;
        }
    }
    pi
}

impl RegimeCartographer for ParrondoJumpCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let mut eps = self.base_eps;
        let mut gamma = self.base_gamma;
        let mut lambda = self.base_lambda;
        let mut uses_ssa = false;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "bias_epsilon" => eps = v,
                "mix_gamma" => gamma = v,
                "jump_switch_rate" => {
                    lambda = v;
                    uses_ssa = true;
                }
                _ => {}
            }
        }
        let (da, db, dmix, dabb, dssa) =
            evaluate_jump_ratchet_stack(self.variant, eps, gamma, lambda);
        let primary = if uses_ssa { dssa } else { dmix };
        let regime = if primary.abs() <= 1e-4 {
            ParrondoJumpRegime::NeutralBoundary
        } else if primary < 0.0 {
            ParrondoJumpRegime::LosingDrift
        } else if da < -1e-6 && db < -1e-6 {
            ParrondoJumpRegime::ParadoxRatchet
        } else {
            ParrondoJumpRegime::DirectPositiveDrift
        };
        AtlasCell {
            regime: regime as u8,
            primary,
            secondary: dabb,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (cell.primary.abs() / 0.04).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        ParrondoJumpRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        if self.axes.y.key == "jump_switch_rate" || self.axes.x.key == "jump_switch_rate" {
            "SSA Jump Drift μ(SSA)"
        } else {
            "Mixture Drift μ(Mix)"
        }
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {:.4}, {} = {:.4}\nRegime: {}\n{} = {:+.5}  |  Periodic μ(ABB) = {:+.5}",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            self.primary_metric_label(),
            cell.primary,
            cell.secondary
        )
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            ParrondoJumpRegime::ParadoxRatchet,
            ParrondoJumpRegime::DirectPositiveDrift,
            ParrondoJumpRegime::NeutralBoundary,
            ParrondoJumpRegime::LosingDrift,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 4, 0.75, high_contrast),
            )
        })
        .collect()
    }
}

impl ParrondoParadoxTool {
    /// Applies a 2D atlas teleport across `100%` of slider bounds and synchronizes `CalloutManager`.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "bias_epsilon" => self.state.bias_epsilon = val.clamp(0.0, 0.05),
                "mix_gamma" => self.state.mix_gamma = val.clamp(0.0, 1.0),
                "jump_switch_rate" => self.state.jump_switch_rate = val.clamp(0.1, 5.0),
                _ => {}
            }
        }
        self.rebuild_full_simulation();
        let x = self.state.steps as f64;
        let y = x * self.metrics.drift_mix;
        self.callouts.set_auto_callout(
            "atlas_teleport",
            [x, y],
            format!("Atlas: {}", ev.regime_label),
            [-110.0, -26.0],
        );
    }

    pub(super) fn render_jump_atlas_panel(&mut self, ui: &mut egui::Ui) {
        if !self.atlas.is_visible() {
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let val_of = |k: &str| match k {
            "bias_epsilon" => self.state.bias_epsilon,
            "mix_gamma" => self.state.mix_gamma,
            "jump_switch_rate" => self.state.jump_switch_rate,
            _ => 0.0,
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.state.variant as u8).hash(&mut hasher);
        for (k, v) in [
            ("bias_epsilon", self.state.bias_epsilon),
            ("mix_gamma", self.state.mix_gamma),
            ("jump_switch_rate", self.state.jump_switch_rate),
        ] {
            if k != xk && k != yk {
                v.to_bits().hash(&mut hasher);
            }
        }
        let theory = self.theory_parameters();
        let variant = self.state.variant;
        let (eps, gamma, lambda) = (
            self.state.bias_epsilon,
            self.state.mix_gamma,
            self.state.jump_switch_rate,
        );
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "parrondo_jump_atlas",
            val_of(xk),
            val_of(yk),
            hasher.finish(),
            move |keys| {
                Arc::new(ParrondoJumpCartographer::new(
                    &theory, keys, variant, eps, gamma, lambda,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}

// [cite:paradox_lab]
