//! Zero-allocation 2D parameter regime atlas for [`super::ParrondoParadoxTool`] (`Parrondo's Kelly Ratchet`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ParrondoParadoxTool;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use eframe::egui;
use math_explorer::pure_math::statistics::markov::parrondo::{
    ParrondoGame, ParrondoSchedule, MAX_MODULUS,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

const MAX_SCHEDULE_STEPS: usize = 16;
const PROB_FLOOR: f64 = 1e-4;

/// Discrete drift and Kelly-edge regime across the Parrondo parameter plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ParrondoKellyRegime {
    /// Periodic schedule has negative expected step drift ($E[\Delta W_{\text{sched}}] < 0$).
    LosingTrap = 0,
    /// Periodic schedule wins while both pure games lose ($E[\Delta W_A] < 0, E[\Delta W_B] < 0$).
    ParadoxRatchet = 1,
    /// Near-fair critical boundary ($|E[\Delta W_{\text{sched}}]| \le 10^{-4}$).
    NeutralBoundary = 2,
    /// Schedule wins and at least one standalone game already has non-negative edge.
    DirectPositiveEdge = 3,
}

impl ParrondoKellyRegime {
    /// Converts a raw `u8` regime index into [`ParrondoKellyRegime`].
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::LosingTrap,
            1 => Self::ParadoxRatchet,
            2 => Self::NeutralBoundary,
            _ => Self::DirectPositiveEdge,
        }
    }

    /// Human-readable regime title for tooltips and legends.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::LosingTrap => "Losing Drift (NoEdge)",
            Self::ParadoxRatchet => "Paradox Wedge (A<0, B<0, Sched>0)",
            Self::NeutralBoundary => "Critical Fair Boundary",
            Self::DirectPositiveEdge => "Direct Positive Edge",
        }
    }
}

/// Stack-allocated modulo-$M$ ($M \le 9$) Markov ratchet evaluator (`zero` heap allocations).
#[derive(Clone, Debug)]
pub struct ParrondoKellyCartographer {
    axes: AtlasAxisPair,
    modulus: usize,
    schedule_steps: [ParrondoGame; MAX_SCHEDULE_STEPS],
    schedule_len: usize,
    base_p_a: f64,
    base_p_b0: f64,
    base_p_b1: f64,
    base_eps: f64,
}

impl ParrondoKellyCartographer {
    /// Constructs a [`ParrondoKellyCartographer`] for `(x_key, y_key)`.
    #[must_use]
    pub fn new(
        theory: &HashMap<String, ParameterConstraint>,
        keys: (&'static str, &'static str),
        modulus: usize,
        schedule: &ParrondoSchedule,
        base_probs: [f64; 4],
    ) -> Self {
        let mut schedule_steps = [ParrondoGame::GameA; MAX_SCHEDULE_STEPS];
        let raw_steps = schedule.steps();
        let schedule_len = raw_steps.len().clamp(1, MAX_SCHEDULE_STEPS);
        for (idx, &game) in raw_steps.iter().take(schedule_len).enumerate() {
            schedule_steps[idx] = game;
        }
        Self {
            axes: AtlasAxisPair {
                x: kelly_axis_spec(theory, keys.0),
                y: kelly_axis_spec(theory, keys.1),
            },
            modulus: modulus.clamp(3, MAX_MODULUS),
            schedule_steps,
            schedule_len,
            base_p_a: base_probs[0],
            base_p_b0: base_probs[1],
            base_p_b1: base_probs[2],
            base_eps: base_probs[3],
        }
    }
}

fn kelly_axis_spec(
    theory: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (k, label, min, max, step) = match key {
        "epsilon" => ("epsilon", "Bias Perturbation (ε)", -0.02, 0.045, 0.0005),
        "p_b0_base" => ("p_b0_base", "Game B Trap Prob (p_B0)", 0.03, 0.50, 0.005),
        "p_b1_base" => (
            "p_b1_base",
            "Game B Non-Trap Prob (p_B1)",
            0.50,
            0.95,
            0.005,
        ),
        _ => (
            "p_a_base",
            "Game A Base Win Prob (p_A,0)",
            0.35,
            0.65,
            0.005,
        ),
    };
    AtlasAxisSpec::from_theory(theory, k, label, ParameterConstraint { min, max, step })
}

/// Evaluates `(gain_a, gain_b, gain_schedule)` on fixed `[f64; 9]` stack buffers with zero heap allocations.
#[must_use]
pub fn evaluate_modulo_ratchet_stack(
    modulus: usize,
    p_a: f64,
    p_b0: f64,
    p_b1: f64,
    schedule_steps: &[ParrondoGame],
) -> (f64, f64, f64) {
    let m = modulus.clamp(3, MAX_MODULUS);
    let pa = p_a.clamp(PROB_FLOOR, 1.0 - PROB_FLOOR);
    let pb0 = p_b0.clamp(PROB_FLOOR, 1.0 - PROB_FLOOR);
    let pb1 = p_b1.clamp(PROB_FLOOR, 1.0 - PROB_FLOOR);

    let (mat_a, mat_b) = build_modulo_matrices(m, pa, pb0, pb1);
    let pi_b = solve_stationary_9(&mat_b, m);
    let gain_a = 2.0 * pa - 1.0;
    let mut gain_b = 0.0;
    for (s, &prob_s) in pi_b.iter().enumerate().take(m) {
        let p_s = if s == 0 { pb0 } else { pb1 };
        gain_b += prob_s * (2.0 * p_s - 1.0);
    }

    let mut prod = identity_9(m);
    for &step in schedule_steps {
        let rhs = match step {
            ParrondoGame::GameA => &mat_a,
            ParrondoGame::GameB => &mat_b,
        };
        prod = mul_9(&prod, rhs, m);
    }
    let mut dist = solve_stationary_9(&prod, m);
    let mut total_gain = 0.0;
    for &step in schedule_steps {
        match step {
            ParrondoGame::GameA => {
                total_gain += gain_a;
                dist = step_dist_9(&mat_a, &dist, m);
            }
            ParrondoGame::GameB => {
                for (s, &prob_s) in dist.iter().enumerate().take(m) {
                    let p_s = if s == 0 { pb0 } else { pb1 };
                    total_gain += prob_s * (2.0 * p_s - 1.0);
                }
                dist = step_dist_9(&mat_b, &dist, m);
            }
        }
    }
    let len = schedule_steps.len().max(1) as f64;
    (gain_a, gain_b, total_gain / len)
}

fn identity_9(m: usize) -> [[f64; MAX_MODULUS]; MAX_MODULUS] {
    let mut mat = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for (idx, row) in mat.iter_mut().enumerate().take(m) {
        row[idx] = 1.0;
    }
    mat
}

fn build_modulo_matrices(
    m: usize,
    pa: f64,
    pb0: f64,
    pb1: f64,
) -> (
    [[f64; MAX_MODULUS]; MAX_MODULUS],
    [[f64; MAX_MODULUS]; MAX_MODULUS],
) {
    let mut mat_a = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    let mut mat_b = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for s in 0..m {
        let next = (s + 1) % m;
        let prev = (s + m - 1) % m;
        mat_a[s][next] += pa;
        mat_a[s][prev] += 1.0 - pa;
        let pb = if s == 0 { pb0 } else { pb1 };
        mat_b[s][next] += pb;
        mat_b[s][prev] += 1.0 - pb;
    }
    (mat_a, mat_b)
}

fn mul_9(
    lhs: &[[f64; MAX_MODULUS]; MAX_MODULUS],
    rhs: &[[f64; MAX_MODULUS]; MAX_MODULUS],
    m: usize,
) -> [[f64; MAX_MODULUS]; MAX_MODULUS] {
    let mut out = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for i in 0..m {
        for k in 0..m {
            let aik = lhs[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..m {
                out[i][j] += aik * rhs[k][j];
            }
        }
    }
    out
}

fn step_dist_9(
    mat: &[[f64; MAX_MODULUS]; MAX_MODULUS],
    dist: &[f64; MAX_MODULUS],
    m: usize,
) -> [f64; MAX_MODULUS] {
    let mut next = [0.0; MAX_MODULUS];
    for i in 0..m {
        let di = dist[i];
        for j in 0..m {
            next[j] += di * mat[i][j];
        }
    }
    next
}

fn solve_stationary_9(p: &[[f64; MAX_MODULUS]; MAX_MODULUS], m: usize) -> [f64; MAX_MODULUS] {
    let mut aug = [[0.0_f64; MAX_MODULUS + 1]; MAX_MODULUS];
    for r in 0..(m - 1) {
        for c in 0..m {
            aug[r][c] = p[c][r] - if c == r { 1.0 } else { 0.0 };
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
            return uniform_9(m);
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        let inv = 1.0 / aug[col][col];
        let pivot_row = aug[col];
        for row_vals in aug.iter_mut().take(m).skip(col + 1) {
            let factor = row_vals[col] * inv;
            if factor == 0.0 {
                continue;
            }
            for (dst, &src) in row_vals[col..=m].iter_mut().zip(&pivot_row[col..=m]) {
                *dst -= factor * src;
            }
        }
    }

    let mut pi = [0.0; MAX_MODULUS];
    for r in (0..m).rev() {
        let mut rhs = aug[r][m];
        for c in (r + 1)..m {
            rhs -= aug[r][c] * pi[c];
        }
        pi[r] = (rhs / aug[r][r]).max(0.0);
    }
    let sum: f64 = pi.iter().take(m).sum();
    if sum <= 1e-12 || !sum.is_finite() {
        return uniform_9(m);
    }
    for val in pi.iter_mut().take(m) {
        *val /= sum;
    }
    pi
}

fn uniform_9(m: usize) -> [f64; MAX_MODULUS] {
    let mut pi = [0.0; MAX_MODULUS];
    let inv = 1.0 / (m.max(1) as f64);
    for val in pi.iter_mut().take(m) {
        *val = inv;
    }
    pi
}

impl RegimeCartographer for ParrondoKellyCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let mut pa_base = self.base_p_a;
        let mut pb0_base = self.base_p_b0;
        let mut pb1_base = self.base_p_b1;
        let mut eps = self.base_eps;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "p_a_base" => pa_base = v,
                "epsilon" => eps = v,
                "p_b0_base" => pb0_base = v,
                "p_b1_base" => pb1_base = v,
                _ => {}
            }
        }
        let (gain_a, gain_b, gain_sched) = evaluate_modulo_ratchet_stack(
            self.modulus,
            pa_base - eps,
            pb0_base - eps,
            pb1_base - eps,
            &self.schedule_steps[..self.schedule_len],
        );
        let regime = if gain_sched.abs() <= 1e-4 {
            ParrondoKellyRegime::NeutralBoundary
        } else if gain_sched < 0.0 {
            ParrondoKellyRegime::LosingTrap
        } else if gain_a < 0.0 && gain_b < 0.0 {
            ParrondoKellyRegime::ParadoxRatchet
        } else {
            ParrondoKellyRegime::DirectPositiveEdge
        };
        AtlasCell {
            regime: regime as u8,
            primary: gain_sched,
            secondary: gain_a.max(gain_b),
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (cell.primary.abs() / 0.12).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        ParrondoKellyRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Schedule Drift E[ΔW_sched]"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {:.4}, {} = {:.4}\nRegime: {}\nSchedule Drift E[ΔW_sched] = {:+.5}  |  max(E[ΔW_A], E[ΔW_B]) = {:+.5}",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            ParrondoKellyRegime::ParadoxRatchet,
            ParrondoKellyRegime::DirectPositiveEdge,
            ParrondoKellyRegime::NeutralBoundary,
            ParrondoKellyRegime::LosingTrap,
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
    /// Applies a 2D atlas click or keyboard teleport across `100%` of slider bounds and updates plot callouts.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "p_a_base" => self.p_a_base = val.clamp(0.35, 0.65),
                "epsilon" => self.epsilon = val.clamp(-0.02, 0.045),
                "p_b0_base" => self.p_b0_base = val.clamp(0.03, 0.50),
                "p_b1_base" => self.p_b1_base = val.clamp(0.50, 0.95),
                _ => {}
            }
        }
        let min_safe = (self.epsilon + 1e-3).max(0.03);
        let max_safe = (0.999 + self.epsilon).min(0.95);
        self.p_a_base = self.p_a_base.clamp(min_safe.max(0.35), max_safe.min(0.65));
        self.p_b0_base = self.p_b0_base.clamp(min_safe, 0.50);
        self.p_b1_base = self.p_b1_base.clamp(0.50, max_safe);
        self.recompute_all();
        let target_x = self.num_steps as f64;
        let target_y = self
            .report
            .as_ref()
            .map_or(0.0, |r| r.gain_schedule * target_x);
        let msg = format!("Atlas: {}", ev.regime_label);
        self.callouts.set_auto_callout(
            "atlas_teleport",
            [target_x, target_y],
            msg,
            [-110.0, -28.0],
        );
    }

    pub(super) fn render_phase_atlas_card(&mut self, ui: &mut egui::Ui, _height: f32) {
        if !self.atlas.is_visible() {
            ui.small("2D Parameter Atlas hidden — toggle '2D Parameter Atlas' in the side panel.");
            return;
        }
        let (xk, yk) = self.atlas.axis_keys();
        let val_of = |k: &str| match k {
            "p_a_base" => self.p_a_base,
            "epsilon" => self.epsilon,
            "p_b0_base" => self.p_b0_base,
            "p_b1_base" => self.p_b1_base,
            _ => 0.0,
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.modulus.hash(&mut hasher);
        self.active_schedule.as_string().hash(&mut hasher);
        for (k, v) in [
            ("p_a_base", self.p_a_base),
            ("epsilon", self.epsilon),
            ("p_b0_base", self.p_b0_base),
            ("p_b1_base", self.p_b1_base),
        ] {
            if k != xk && k != yk {
                v.to_bits().hash(&mut hasher);
            }
        }
        let theory = self.theory_parameters();
        let modulus = self.modulus;
        let sched = self.active_schedule;
        let base_probs = [self.p_a_base, self.p_b0_base, self.p_b1_base, self.epsilon];
        if let Some(ev) = self.atlas.show_panel(
            ui,
            "parrondo_kelly_atlas",
            val_of(xk),
            val_of(yk),
            hasher.finish(),
            move |keys| {
                Arc::new(ParrondoKellyCartographer::new(
                    &theory, keys, modulus, &sched, base_probs,
                ))
            },
        ) {
            self.apply_atlas_teleport(ev);
        }
    }
}

// [cite:clinical_trials_statistics]
