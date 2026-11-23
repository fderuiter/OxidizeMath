//! Zero-allocation 9-state stack Markov solver and 2D [`RegimeCartographer`]
//! for [`super::ParrondoParadoxTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::ParrondoParadoxTool;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
use math_explorer::pure_math::statistics::markov::parrondo::{
    ParrondoGame, ALLOWED_MODULI, MAX_MODULUS, MAX_SCHEDULE_LEN,
};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};

/// Supported 2D parameter atlas axis pairs for [`ParrondoParadoxTool`].
pub const PARRONDO_ATLAS_AXIS_PAIRS: &[(&str, &str)] =
    &[("p_a", "epsilon"), ("p_b0_base", "p_b1_base")];

pub(super) const WEDGE_PA_MIN: f64 = 0.42;
pub(super) const WEDGE_PA_MAX: f64 = 0.51;
pub(super) const WEDGE_EPS_MIN: f64 = -0.02;
pub(super) const WEDGE_EPS_MAX: f64 = 0.05;

/// Stack-resident stationary drift and trap-occupancy summary matching `solve_parrondo_ratchet`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackParrondoSummary {
    /// Stationary expected step gain of isolated Game A.
    pub gain_a: f64,
    /// Stationary expected step gain of isolated Game B.
    pub gain_b: f64,
    /// Stationary expected step gain of 50/50 random mixture.
    pub gain_mix: f64,
    /// Cycle-averaged stationary step gain of the periodic schedule.
    pub gain_schedule: f64,
    /// Stationary trap residue `r = 0` probability under isolated Game B.
    pub pi_b_trap: f64,
    /// Cycle-averaged trap residue `r = 0` probability under the schedule.
    pub pi_sched_trap: f64,
    /// `true` iff `gain_a < 0`, `gain_b < 0`, and `gain_schedule > 0`.
    pub is_paradox: bool,
}

type StackMatrix = [[f64; MAX_MODULUS]; MAX_MODULUS];
type StackVec = [f64; MAX_MODULUS];

fn build_transition_stack(m: usize, p_zero: f64, p_other: f64) -> StackMatrix {
    let mut mat = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for (s, row) in mat.iter_mut().enumerate().take(m) {
        let p_win = if s == 0 { p_zero } else { p_other };
        row[(s + 1) % m] += p_win;
        row[(s + m - 1) % m] += 1.0 - p_win;
    }
    mat
}

fn mat_mul_stack(lhs: &StackMatrix, rhs: &StackMatrix, m: usize) -> StackMatrix {
    let mut out = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for i in 0..m {
        for k in 0..m {
            let a = lhs[i][k];
            if a != 0.0 {
                for j in 0..m {
                    out[i][j] += a * rhs[k][j];
                }
            }
        }
    }
    out
}

fn row_vec_mul_stack(vec: &StackVec, mat: &StackMatrix, m: usize) -> StackVec {
    let mut out = [0.0; MAX_MODULUS];
    for i in 0..m {
        let v = vec[i];
        if v != 0.0 {
            for j in 0..m {
                out[j] += v * mat[i][j];
            }
        }
    }
    out
}

fn power_iter_stationary_stack(mat: &StackMatrix, m: usize) -> StackVec {
    let mut pi = [0.0; MAX_MODULUS];
    let inv = 1.0 / (m as f64);
    for v in pi.iter_mut().take(m) {
        *v = inv;
    }
    for _ in 0..256 {
        let stepped = row_vec_mul_stack(&pi, mat, m);
        let mut diff = 0.0_f64;
        for s in 0..m {
            let next = 0.5 * pi[s] + 0.5 * stepped[s];
            diff = diff.max((next - pi[s]).abs());
            pi[s] = next;
        }
        if diff < 1e-14 {
            break;
        }
    }
    pi
}

fn solve_stationary_stack(mat: &StackMatrix, m: usize) -> StackVec {
    let mut aug = [[0.0; MAX_MODULUS + 1]; MAX_MODULUS];
    for r in 0..(m - 1) {
        for c in 0..m {
            aug[r][c] = mat[c][r] - if r == c { 1.0 } else { 0.0 };
        }
    }
    aug[m - 1][..m].fill(1.0);
    aug[m - 1][m] = 1.0;

    for col in 0..m {
        let mut pivot = col;
        let mut best = aug[col][col].abs();
        for (r, row) in aug.iter().enumerate().take(m).skip(col + 1) {
            let cand = row[col].abs();
            if cand > best {
                best = cand;
                pivot = r;
            }
        }
        if best < 1e-13 {
            return power_iter_stationary_stack(mat, m);
        }
        if pivot != col {
            aug.swap(col, pivot);
        }
        let pivot_row = aug[col];
        let inv_diag = 1.0 / pivot_row[col];
        for row in aug.iter_mut().take(m).skip(col + 1) {
            let factor = row[col] * inv_diag;
            if factor != 0.0 {
                for (dst, &piv) in row[col..=m].iter_mut().zip(&pivot_row[col..=m]) {
                    *dst -= factor * piv;
                }
            }
        }
    }

    let mut sol = [0.0; MAX_MODULUS];
    for r in (0..m).rev() {
        let mut rhs = aug[r][m];
        for c in (r + 1)..m {
            rhs -= aug[r][c] * sol[c];
        }
        sol[r] = (rhs / aug[r][r]).max(0.0);
    }
    let sum: f64 = sol.iter().take(m).sum();
    if sum > 0.0 {
        for v in sol.iter_mut().take(m) {
            *v /= sum;
        }
    }
    sol
}

#[inline]
fn step_gain_b_stack(pi: &StackVec, m: usize, p_b0: f64, p_b1: f64) -> f64 {
    let mut p_win = 0.0;
    for (s, &weight) in pi.iter().enumerate().take(m) {
        p_win += weight * if s == 0 { p_b0 } else { p_b1 };
    }
    2.0 * p_win - 1.0
}

fn evaluate_cycle_schedule_stack(
    mat_a: &StackMatrix,
    mat_b: &StackMatrix,
    m: usize,
    gain_a: f64,
    p_b0: f64,
    p_b1: f64,
    schedule: &[ParrondoGame],
) -> (f64, f64) {
    let mut mat_cycle = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for (i, row) in mat_cycle.iter_mut().enumerate().take(m) {
        row[i] = 1.0;
    }
    for &g in schedule {
        let step_mat = match g {
            ParrondoGame::GameA => mat_a,
            ParrondoGame::GameB => mat_b,
        };
        mat_cycle = mat_mul_stack(&mat_cycle, step_mat, m);
    }

    let mut pi_step = solve_stationary_stack(&mat_cycle, m);
    let mut pi_sched_trap = 0.0;
    let mut gain_sum = 0.0;
    for &g in schedule {
        pi_sched_trap += pi_step[0];
        match g {
            ParrondoGame::GameA => {
                gain_sum += gain_a;
                pi_step = row_vec_mul_stack(&pi_step, mat_a, m);
            }
            ParrondoGame::GameB => {
                gain_sum += step_gain_b_stack(&pi_step, m, p_b0, p_b1);
                pi_step = row_vec_mul_stack(&pi_step, mat_b, m);
            }
        }
    }
    let inv_len = 1.0 / (schedule.len() as f64);
    (gain_sum * inv_len, pi_sched_trap * inv_len)
}

/// Solves the modulo-`M` (`M <= 9`) Parrondo Markov chain entirely on the stack with zero heap allocations.
#[must_use]
pub fn solve_parrondo_ratchet_stack(
    modulus: usize,
    p_a_base: f64,
    p_b0_base: f64,
    p_b1_base: f64,
    epsilon: f64,
    schedule: &[ParrondoGame],
) -> Option<StackParrondoSummary> {
    if !ALLOWED_MODULI.contains(&modulus) || schedule.is_empty() {
        return None;
    }
    let (p_a, p_b0, p_b1) = (p_a_base - epsilon, p_b0_base - epsilon, p_b1_base - epsilon);
    if ![p_a, p_b0, p_b1]
        .into_iter()
        .all(|p| p.is_finite() && p > 0.0 && p < 1.0)
    {
        return None;
    }

    let m = modulus;
    let mat_a = build_transition_stack(m, p_a, p_a);
    let mat_b = build_transition_stack(m, p_b0, p_b1);
    let mut mat_mix = [[0.0; MAX_MODULUS]; MAX_MODULUS];
    for r in 0..m {
        for c in 0..m {
            mat_mix[r][c] = 0.5 * (mat_a[r][c] + mat_b[r][c]);
        }
    }

    let gain_a = 2.0 * p_a - 1.0;
    let pi_b = solve_stationary_stack(&mat_b, m);
    let gain_b = step_gain_b_stack(&pi_b, m, p_b0, p_b1);
    let pi_mix = solve_stationary_stack(&mat_mix, m);
    let gain_mix = 0.5 * gain_a + 0.5 * step_gain_b_stack(&pi_mix, m, p_b0, p_b1);
    let (gain_schedule, pi_sched_trap) =
        evaluate_cycle_schedule_stack(&mat_a, &mat_b, m, gain_a, p_b0, p_b1, schedule);

    Some(StackParrondoSummary {
        gain_a,
        gain_b,
        gain_mix,
        gain_schedule,
        pi_b_trap: pi_b[0],
        pi_sched_trap,
        is_paradox: gain_a < 0.0 && gain_b < 0.0 && gain_schedule > 0.0,
    })
}

/// Zero-allocation 2D parameter cartographer for [`ParrondoParadoxTool`].
pub struct ParrondoCartographer {
    axes: AtlasAxisPair,
    modulus: usize,
    p_a_base: f64,
    p_b0_base: f64,
    p_b1_base: f64,
    epsilon: f64,
    schedule: [ParrondoGame; MAX_SCHEDULE_LEN],
    schedule_len: usize,
}

fn parrondo_axis_pair(
    theory: &std::collections::HashMap<String, ParameterConstraint>,
    axis_keys: (&'static str, &'static str),
) -> Option<AtlasAxisPair> {
    let c = |min, max, step| ParameterConstraint {
        min,
        max,
        step,
        default: None,
    };
    match axis_keys {
        ("p_a", "epsilon") => Some(AtlasAxisPair {
            x: AtlasAxisSpec::from_theory(
                theory,
                "p_a",
                "Effective Game A Win Prob p_A",
                c(WEDGE_PA_MIN, WEDGE_PA_MAX, 0.002),
            ),
            y: AtlasAxisSpec::from_theory(
                theory,
                "epsilon",
                "Bias Perturbation ε",
                c(WEDGE_EPS_MIN, WEDGE_EPS_MAX, 0.001),
            ),
        }),
        ("p_b0_base", "p_b1_base") => Some(AtlasAxisPair {
            x: AtlasAxisSpec::from_theory(
                theory,
                "p_b0_base",
                "Trap Win Prob p_B0,0 (r = 0)",
                c(0.05, 0.45, 0.005),
            ),
            y: AtlasAxisSpec::from_theory(
                theory,
                "p_b1_base",
                "Non-Trap Win Prob p_B1,0 (r ≠ 0)",
                c(0.55, 0.95, 0.005),
            ),
        }),
        _ => None,
    }
}

impl ParrondoCartographer {
    /// Constructs a stack-resident [`ParrondoCartographer`] for `("p_a", "epsilon")` or `("p_b0_base", "p_b1_base")`.
    #[must_use]
    pub fn new(
        tool: &ParrondoParadoxTool,
        axis_keys: (&'static str, &'static str),
    ) -> Option<Self> {
        let theory = tool.theory_parameters();
        let axes = parrondo_axis_pair(&theory, axis_keys)?;
        let mut schedule = [ParrondoGame::GameA; MAX_SCHEDULE_LEN];
        let steps = tool.active_schedule.steps();
        let schedule_len = steps.len().min(MAX_SCHEDULE_LEN);
        schedule[..schedule_len].copy_from_slice(&steps[..schedule_len]);
        Some(Self {
            axes,
            modulus: tool.modulus,
            p_a_base: tool.p_a_base,
            p_b0_base: tool.p_b0_base,
            p_b1_base: tool.p_b1_base,
            epsilon: tool.epsilon,
            schedule,
            schedule_len,
        })
    }
}

impl RegimeCartographer for ParrondoCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (p_a_base, p_b0_base, p_b1_base, eps) = match (self.axes.x.key, self.axes.y.key) {
            ("p_a", "epsilon") => (p1 + p2, self.p_b0_base, self.p_b1_base, p2),
            _ => (self.p_a_base, p1, p2, self.epsilon),
        };
        let Some(summary) = solve_parrondo_ratchet_stack(
            self.modulus,
            p_a_base,
            p_b0_base,
            p_b1_base,
            eps,
            &self.schedule[..self.schedule_len],
        ) else {
            return AtlasCell {
                regime: 3,
                primary: -1.0,
                secondary: 0.0,
            };
        };
        let regime = if summary.is_paradox {
            1
        } else if summary.gain_schedule.abs() <= 1e-4 {
            2
        } else if summary.gain_schedule > 0.0 {
            0
        } else {
            3
        };
        AtlasCell {
            regime,
            primary: summary.gain_schedule,
            secondary: summary.pi_sched_trap,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (0.5 + 15.0 * cell.primary).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        match regime {
            0 => "Positive Drift (Non-Paradox)",
            1 => "Paradox Wedge (A<0, B<0, Sched>0)",
            2 => "Fair Ratchet Boundary (|g| ≤ 1e-4)",
            _ => "Negative Ratchet Drift (E[ΔW] < 0)",
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Schedule Drift E[ΔW]"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "{} = {p1:.4}, {} = {p2:.4}\nRegime: {}\nE[ΔW_sched] = {:+.5} | Trap π(0) = {:.4}",
            self.axes.x.label,
            self.axes.y.label,
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if (self.axes.x.key, self.axes.y.key) == ("p_a", "epsilon") {
            Some((
                "Fair Game A (p_A = 0.5)",
                vec![
                    [0.5, self.axes.y.constraint.min],
                    [0.5, self.axes.y.constraint.max],
                ],
            ))
        } else {
            None
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        (0..4_u8)
            .map(|r| {
                (
                    self.regime_label(r),
                    atlas_regime_color(r, 4, 0.75, high_contrast),
                )
            })
            .collect()
    }
}
