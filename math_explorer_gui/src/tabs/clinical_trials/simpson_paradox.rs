//! Interactive Simpson's Paradox and Stratified Clinical Trial Laboratory.
//!
//! Wires `ContingencyTable`, `calculate_risk_metrics`, `ChiSquare2x2`, and
//! `StratifiedRandomizer` across Mild, Severe, and Pooled cohorts so users can
//! predict and test confounded relative-risk reversals ($RR_1 < 1, RR_2 < 1$
//! yet $RR_{\text{pooled}} > 1$).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::CalloutManager;
use math_explorer::applied::clinical_trials::{
    analysis::{calculate_risk_metrics, RiskMetrics},
    design::{BlockRandomizer, Group, Patient, StratifiedRandomizer},
    hypothesis_testing::{ChiSquare2x2, HypothesisTest, TestResult},
    types::ContingencyTable,
};
use oxidize_core::rng::{OxidizeRng, BASE_SEED};
use rand::Rng;

const ALPHA: f64 = 0.05;
const MILD_TREAT_RATE: f64 = 0.10;
const MILD_CTRL_RATE: f64 = 0.15;
const SEVERE_TREAT_RATE: f64 = 0.30;
const SEVERE_CTRL_RATE: f64 = 0.40;

/// User hypothesis choice for the pooled trial winner before inspecting pooled RR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WinnerPrediction {
    /// User predicts Treatment wins pooled (`RR_pooled < 1.0`).
    TreatmentWins,
    /// User predicts confounding flips pooled result (`RR_pooled > 1.0`).
    ControlWinsReversal,
    /// User predicts equal pooled risk (`RR_pooled == 1.0`).
    Tie,
}

/// Non-zero 2x2 contingency cell counts for a single clinical stratum or pooled table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TableCounts {
    /// Adverse events in Treatment arm (`a >= 1`).
    pub treat_event: u32,
    /// Event-free subjects in Treatment arm (`b >= 1`).
    pub treat_no_event: u32,
    /// Adverse events in Control arm (`c >= 1`).
    pub ctrl_event: u32,
    /// Event-free subjects in Control arm (`d >= 1`).
    pub ctrl_no_event: u32,
}

impl TableCounts {
    /// Checks that all four cells are strictly positive (`>= 1`).
    pub fn has_valid_nonzero_cells(self) -> bool {
        self.treat_event >= 1
            && self.treat_no_event >= 1
            && self.ctrl_event >= 1
            && self.ctrl_no_event >= 1
    }

    /// Sums two stratum tables cell-by-cell to form the pooled 2x2 table.
    pub fn pooled_with(self, other: Self) -> Self {
        Self {
            treat_event: self.treat_event.saturating_add(other.treat_event),
            treat_no_event: self.treat_no_event.saturating_add(other.treat_no_event),
            ctrl_event: self.ctrl_event.saturating_add(other.ctrl_event),
            ctrl_no_event: self.ctrl_no_event.saturating_add(other.ctrl_no_event),
        }
    }

    fn from_share_and_rates(total_n: u32, treat_share: f64, p_treat: f64, p_ctrl: f64) -> Self {
        let n = total_n.max(4) as f64;
        let share = treat_share.clamp(0.10, 0.90);
        let n_treat = (n * share).round().max(2.0);
        let n_ctrl = (n - n_treat).max(2.0);
        let a = ((n_treat * p_treat).round() as u32)
            .max(1)
            .min(n_treat as u32 - 1);
        let b = (n_treat as u32).saturating_sub(a).max(1);
        let c = ((n_ctrl * p_ctrl).round() as u32)
            .max(1)
            .min(n_ctrl as u32 - 1);
        let d = (n_ctrl as u32).saturating_sub(c).max(1);
        Self {
            treat_event: a,
            treat_no_event: b,
            ctrl_event: c,
            ctrl_no_event: d,
        }
    }
}

/// Serializable configuration state for [`SimpsonParadoxLab`].
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SimpsonParadoxConfig {
    /// Fraction of Mild stratum allocated to Treatment (`0.10..=0.90`).
    pub mild_treatment_share: f64,
    /// Fraction of Severe stratum allocated to Treatment (`0.10..=0.90`).
    pub severe_treatment_share: f64,
    /// Cohort size per severity stratum (`20..=800`).
    pub stratum_size: u32,
    /// Mild stratum 2x2 cell counts.
    pub mild_counts: TableCounts,
    /// Severe stratum 2x2 cell counts.
    pub severe_counts: TableCounts,
    /// Optional `"Predict Winner"` selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction: Option<WinnerPrediction>,
}

impl Default for SimpsonParadoxConfig {
    fn default() -> Self {
        let stratum_size = 400;
        let mild_treatment_share = 0.20;
        let severe_treatment_share = 0.80;
        Self {
            mild_treatment_share,
            severe_treatment_share,
            stratum_size,
            mild_counts: TableCounts::from_share_and_rates(
                stratum_size,
                mild_treatment_share,
                MILD_TREAT_RATE,
                MILD_CTRL_RATE,
            ),
            severe_counts: TableCounts::from_share_and_rates(
                stratum_size,
                severe_treatment_share,
                SEVERE_TREAT_RATE,
                SEVERE_CTRL_RATE,
            ),
            prediction: None,
        }
    }
}

impl SimpsonParadoxConfig {
    /// Returns true if configuration matches baseline default for compact serialization.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Validates numeric bounds and minimum cell counts (`>= 1`).
    pub fn validate(&self) -> Result<(), String> {
        if !self.mild_treatment_share.is_finite()
            || !self.severe_treatment_share.is_finite()
            || !(0.10..=0.90).contains(&self.mild_treatment_share)
            || !(0.10..=0.90).contains(&self.severe_treatment_share)
            || !(20..=800).contains(&self.stratum_size)
        {
            return Err("Simpson allocation shares or stratum_size out of bounds".into());
        }
        if !self.mild_counts.has_valid_nonzero_cells()
            || !self.severe_counts.has_valid_nonzero_cells()
        {
            return Err(
                "All 2x2 cell counts must be >= 1 for RR and Chi-Square evaluation".into(),
            );
        }
        Ok(())
    }
}

/// Computed risk and Pearson Chi-Square metrics for one 2x2 table.
#[derive(Clone, Debug)]
pub struct TableEvaluation {
    /// Validated `ContingencyTable`.
    pub table: ContingencyTable,
    /// Relative risk, odds ratio, and 95% confidence intervals.
    pub risk: RiskMetrics,
    /// 2x2 Pearson Chi-Square hypothesis test output.
    pub chi_sq: TestResult,
}

/// Interactive laboratory demonstrating Simpson's Paradox and stratified block balancing.
pub struct SimpsonParadoxLab {
    /// Editable trial allocation and cell-count configuration.
    pub config: SimpsonParadoxConfig,
    /// Plot callout manager for RR confidence intervals and Simpson reversal alerts.
    pub callout_manager: CalloutManager,
    /// Latest Mild stratum evaluation.
    pub mild_eval: Option<TableEvaluation>,
    /// Latest Severe stratum evaluation.
    pub severe_eval: Option<TableEvaluation>,
    /// Latest Pooled (`Mild + Severe`) evaluation.
    pub pooled_eval: Option<TableEvaluation>,
    /// Whether pooled relative risk reverses the consistent stratum direction.
    pub is_simpson_reversal: bool,
    /// Validation or statistical error message when counts are invalid.
    pub validation_error: Option<String>,
}

impl Default for SimpsonParadoxLab {
    fn default() -> Self {
        let mut lab = Self {
            config: SimpsonParadoxConfig::default(),
            callout_manager: CalloutManager::new(),
            mild_eval: None,
            severe_eval: None,
            pooled_eval: None,
            is_simpson_reversal: false,
            validation_error: None,
        };
        lab.recalculate();
        lab
    }
}

impl SimpsonParadoxLab {
    /// Resets the lab to the canonical confounded Simpson's paradox configuration.
    pub fn reset_confounded_preset(&mut self) {
        self.config = SimpsonParadoxConfig::default();
        self.callout_manager.clear_auto();
        self.recalculate();
    }

    /// Recomputes Mild and Severe cell counts from the allocation sliders and refreshes stats.
    pub fn apply_allocation_sliders(&mut self) {
        self.config.mild_counts = TableCounts::from_share_and_rates(
            self.config.stratum_size,
            self.config.mild_treatment_share,
            MILD_TREAT_RATE,
            MILD_CTRL_RATE,
        );
        self.config.severe_counts = TableCounts::from_share_and_rates(
            self.config.stratum_size,
            self.config.severe_treatment_share,
            SEVERE_TREAT_RATE,
            SEVERE_CTRL_RATE,
        );
        self.recalculate();
    }

    /// Balances Mild and Severe cohorts via `StratifiedRandomizer<BlockRandomizer>` and seeded RNG.
    pub fn balance_via_stratified_randomizer(&mut self, seed_counter: &mut u64) {
        *seed_counter = seed_counter.wrapping_add(1);
        let mut rng = OxidizeRng::from_seed(BASE_SEED ^ *seed_counter);
        let Ok(block) = BlockRandomizer::new(4) else {
            return;
        };
        let strat_rand = StratifiedRandomizer::new(block);
        let per_stratum = self.config.stratum_size.clamp(20, 800) as usize;
        let mut patients = Vec::with_capacity(per_stratum * 2);
        for i in 0..per_stratum {
            patients.push(Patient {
                id: format!("M-{i}"),
                stratum: "Mild".to_string(),
            });
            patients.push(Patient {
                id: format!("S-{i}"),
                stratum: "Severe".to_string(),
            });
        }
        let Ok(assigned) = strat_rand.assign_stratified(&mut rng, &patients) else {
            return;
        };
        self.populate_counts_from_assignments(&patients, &assigned, &mut rng);
        self.recalculate();
    }

    fn populate_counts_from_assignments(
        &mut self,
        patients: &[Patient],
        assigned: &std::collections::HashMap<String, Group>,
        rng: &mut OxidizeRng,
    ) {
        let (mut m_t, mut m_c, mut s_t, mut s_c) = (0_u32, 0_u32, 0_u32, 0_u32);
        let (mut m_te, mut m_ce, mut s_te, mut s_ce) = (0_u32, 0_u32, 0_u32, 0_u32);
        for p in patients {
            let grp = assigned.get(&p.id).copied().unwrap_or(Group::Control);
            if p.stratum == "Mild" {
                if grp == Group::Treatment {
                    m_t += 1;
                    if rng.r#gen::<f64>() < MILD_TREAT_RATE {
                        m_te += 1;
                    }
                } else {
                    m_c += 1;
                    if rng.r#gen::<f64>() < MILD_CTRL_RATE {
                        m_ce += 1;
                    }
                }
            } else if grp == Group::Treatment {
                s_t += 1;
                if rng.r#gen::<f64>() < SEVERE_TREAT_RATE {
                    s_te += 1;
                }
            } else {
                s_c += 1;
                if rng.r#gen::<f64>() < SEVERE_CTRL_RATE {
                    s_ce += 1;
                }
            }
        }
        self.config.mild_treatment_share =
            (m_t as f64 / (m_t + m_c).max(1) as f64).clamp(0.10, 0.90);
        self.config.severe_treatment_share =
            (s_t as f64 / (s_t + s_c).max(1) as f64).clamp(0.10, 0.90);
        self.config.mild_counts = clamp_drawn_counts(m_t, m_te, m_c, m_ce);
        self.config.severe_counts = clamp_drawn_counts(s_t, s_te, s_c, s_ce);
    }

    /// Evaluates Mild, Severe, and Pooled `ContingencyTable`s, `RiskMetrics`, and `ChiSquare2x2`.
    pub fn recalculate(&mut self) {
        if !self.config.mild_counts.has_valid_nonzero_cells()
            || !self.config.severe_counts.has_valid_nonzero_cells()
        {
            self.validation_error = Some(
                "Validation Error: Every 2x2 cell count must be at least 1 to prevent division-by-zero in RR and Chi-Square.".into(),
            );
            self.mild_eval = None;
            self.severe_eval = None;
            self.pooled_eval = None;
            self.is_simpson_reversal = false;
            self.callout_manager.clear_auto();
            return;
        }

        let pooled_counts = self.config.mild_counts.pooled_with(self.config.severe_counts);
        match (
            evaluate_single_table(self.config.mild_counts),
            evaluate_single_table(self.config.severe_counts),
            evaluate_single_table(pooled_counts),
        ) {
            (Ok(mild), Ok(sev), Ok(pool)) => {
                let rr1 = mild.risk.relative_risk;
                let rr2 = sev.risk.relative_risk;
                let rrp = pool.risk.relative_risk;
                self.is_simpson_reversal =
                    (rr1 < 1.0 && rr2 < 1.0 && rrp > 1.0) || (rr1 > 1.0 && rr2 > 1.0 && rrp < 1.0);
                self.validation_error = None;
                self.sync_callouts(&mild, &sev, &pool);
                self.mild_eval = Some(mild);
                self.severe_eval = Some(sev);
                self.pooled_eval = Some(pool);
            }
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
                self.validation_error = Some(format!("Statistical Error: {e}"));
                self.is_simpson_reversal = false;
                self.callout_manager.clear_auto();
            }
        }
    }

    fn sync_callouts(
        &mut self,
        mild: &TableEvaluation,
        sev: &TableEvaluation,
        pool: &TableEvaluation,
    ) {
        self.callout_manager.set_auto_callout(
            "mild_rr",
            [mild.risk.relative_risk, 3.0],
            format!("Mild RR = {:.2}", mild.risk.relative_risk),
            [20.0, -18.0],
        );
        self.callout_manager.set_auto_callout(
            "severe_rr",
            [sev.risk.relative_risk, 2.0],
            format!("Severe RR = {:.2}", sev.risk.relative_risk),
            [20.0, -18.0],
        );
        let tag_text = if self.is_simpson_reversal {
            format!(
                "Simpson Reversal! Pooled RR = {:.2} (p = {:.3})",
                pool.risk.relative_risk, pool.chi_sq.p_value
            )
        } else {
            format!(
                "Balanced Pooled RR = {:.2} (p = {:.3})",
                pool.risk.relative_risk, pool.chi_sq.p_value
            )
        };
        self.callout_manager.set_auto_callout(
            "simpson_reversal",
            [pool.risk.relative_risk, 1.0],
            tag_text,
            [24.0, -22.0],
        );
    }
}

fn clamp_drawn_counts(n_t: u32, ev_t: u32, n_c: u32, ev_c: u32) -> TableCounts {
    let total_t = n_t.max(2);
    let total_c = n_c.max(2);
    let a = ev_t.clamp(1, total_t - 1);
    let b = (total_t - a).max(1);
    let c = ev_c.clamp(1, total_c - 1);
    let d = (total_c - c).max(1);
    TableCounts {
        treat_event: a,
        treat_no_event: b,
        ctrl_event: c,
        ctrl_no_event: d,
    }
}

fn evaluate_single_table(
    counts: TableCounts,
) -> Result<TableEvaluation, math_explorer::applied::clinical_trials::types::ClinicalTrialError> {
    let table = ContingencyTable::new(
        counts.treat_event,
        counts.treat_no_event,
        counts.ctrl_event,
        counts.ctrl_no_event,
    )?;
    let risk = calculate_risk_metrics(&table, ALPHA)?;
    let chi_sq = ChiSquare2x2::new(ALPHA).test(&table)?;
    Ok(TableEvaluation {
        table,
        risk,
        chi_sq,
    })
}

#[path = "simpson_paradox_ui.rs"]
mod ui;

#[cfg(test)]
#[path = "simpson_paradox_tests.rs"]
mod tests;

// [cite:clinical_trials_statistics]
