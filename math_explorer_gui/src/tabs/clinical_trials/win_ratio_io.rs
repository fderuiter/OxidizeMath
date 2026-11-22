//! Cohort CSV/TSV parsing, hierarchical tier evaluation, table formatting, and local file I/O
//! helpers for [`super::win_ratio::WinRatioWorkbenchTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use std::sync::mpsc::{channel, Receiver};

use crate::web_io::{
    spawn_open_file_dialog, spawn_save_as_file_dialog, strip_utf8_bom, OpenDialogConfig,
    SaveAsDialogConfig,
};
use math_explorer::applied::win_ratio::pair_comparison::{
    calculate_statistics, HigherIsBetter, LowerIsBetter, MatchedPairing, OutcomeComparator,
    ThresholdComparator, TierOutcomeCounts, UnmatchedPairing, WinRatioAnalysis, WinRatioStats,
};
use math_explorer::applied::win_ratio::sample_win_ratio::{
    calculate_sample_win_ratio, calculate_win_proportion, WinLossCounts,
};

pub use super::win_ratio_weibull::{
    evaluate_weibull_simulator, WeibullEvaluation, WeibullSimConfig,
};

/// Maximum number of prioritized composite endpoint tiers supported in the workbench.
pub const MAX_WIN_RATIO_TIERS: usize = 4;

const COHORT_OPEN_CONFIG: OpenDialogConfig = OpenDialogConfig {
    filter_name: "Clinical Cohort (*.csv, *.tsv, *.txt)",
    extensions: &["csv", "tsv", "txt"],
    accept: ".csv,.tsv,.txt,text/csv,text/tab-separated-values,text/plain",
    read_error_prefix: "Failed to read cohort file",
};

const CSV_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "CSV Table (*.csv)",
    extensions: &["csv"],
    fallback_filename: "win_ratio_waterfall.csv",
    write_error_prefix: "Failed to save CSV summary",
};

const JSON_SAVE_CONFIG: SaveAsDialogConfig = SaveAsDialogConfig {
    filter_name: "JSON Summary (*.json)",
    extensions: &["json"],
    fallback_filename: "win_ratio_waterfall.json",
    write_error_prefix: "Failed to save JSON summary",
};

/// Comparator direction or threshold rule for a single endpoint tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TierComparatorKind {
    /// Larger numerical values represent better clinical outcomes (e.g., survival days).
    HigherIsBetter,
    /// Smaller numerical values represent better clinical outcomes (e.g., hospitalizations).
    LowerIsBetter,
    /// Higher values win only when exceeding the comparator's clinical margin `threshold`.
    Threshold,
}

impl TierComparatorKind {
    /// Human-readable label for UI combo boxes and summary exports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HigherIsBetter => "Higher is Better",
            Self::LowerIsBetter => "Lower is Better",
            Self::Threshold => "Threshold Margin",
        }
    }
}

/// Persisted configuration for one hierarchical endpoint tier.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TierSpec {
    /// Display name of this clinical tier (e.g., `"Survival Days"`).
    pub label: String,
    /// Direction or threshold rule applied at this tier.
    pub kind: TierComparatorKind,
    /// Minimum positive difference required when `kind == TierComparatorKind::Threshold`.
    pub threshold: f64,
}

impl TierSpec {
    /// Constructs a new tier specification.
    #[must_use]
    pub fn new(label: &str, kind: TierComparatorKind, threshold: f64) -> Self {
        Self {
            label: label.to_string(),
            kind,
            threshold,
        }
    }

    pub(crate) fn is_valid(&self) -> bool {
        !self.label.trim().is_empty()
            && self.label.len() <= 64
            && self.threshold.is_finite()
            && (0.0..=10_000.0).contains(&self.threshold)
    }

    fn to_comparator(&self) -> Box<dyn OutcomeComparator<f64>> {
        match self.kind {
            TierComparatorKind::HigherIsBetter => Box::new(HigherIsBetter),
            TierComparatorKind::LowerIsBetter => Box::new(LowerIsBetter),
            TierComparatorKind::Threshold => Box::new(ThresholdComparator::new(self.threshold)),
        }
    }
}

/// Subject pairing scheme across treatment and control arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PairingMode {
    /// All-pairs Cartesian comparison ($N_T \times N_C$).
    Unmatched,
    /// 1:1 matched-pair comparison requiring $N_T = N_C$.
    Matched,
}

/// Active visualization mode inside [`super::win_ratio::WinRatioWorkbenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WinRatioView {
    /// Empirical cohort tier waterfall `BarChart`.
    CohortWaterfall,
    /// Analytical `BivariateWeibullModel` survival and probability win-ratio curves.
    WeibullSimulator,
}

/// Validated two-arm cohort parsed from CSV or TSV text.
#[derive(Debug, Clone, Default)]
pub struct ParsedCohort {
    /// Treatment-arm outcome vectors (`len == tier_count`).
    pub treatment: Vec<Vec<f64>>,
    /// Control-arm outcome vectors (`len == tier_count`).
    pub control: Vec<Vec<f64>>,
    /// Number of outcome columns detected in the cohort text (`1..=4`).
    pub tier_count: usize,
}

/// Summary metrics and waterfall counts for an evaluated cohort.
#[derive(Debug, Clone)]
pub struct CohortEvaluation {
    /// Per-tier win, loss, and remaining tie counts.
    pub tiers: Vec<TierOutcomeCounts>,
    /// Asymptotic normal-approximation win ratio statistics (`WR`, 95% CI, $p$-value).
    pub stats: Option<WinRatioStats>,
    /// Categorized Tier-1 vs Tier-2+ sample counts (`Na`, `Nb`, `Nc`, `Nd`).
    pub win_loss_counts: WinLossCounts,
    /// Sample win ratio computed via `calculate_sample_win_ratio`.
    pub sample_wr: f64,
    /// Untied win proportion $p_W = N_W / (N_W + N_L)$.
    pub win_proportion: f64,
    /// Total pairs won across all active tiers.
    pub total_wins: usize,
    /// Total pairs lost across all active tiers.
    pub total_losses: usize,
    /// Pairs remaining tied after the final active tier.
    pub total_ties: usize,
    /// Subject count in the treatment arm.
    pub n_treatment: usize,
    /// Subject count in the control arm.
    pub n_control: usize,
}

fn is_header_token(first_col: &str) -> bool {
    let t = first_col.trim().trim_matches('"');
    t.eq_ignore_ascii_case("arm")
        || t.eq_ignore_ascii_case("group")
        || t.eq_ignore_ascii_case("cohort")
}

fn parse_arm_is_treatment(line_no: usize, token: &str) -> Result<bool, String> {
    let c = token.trim().trim_matches('"');
    if c.eq_ignore_ascii_case("treatment")
        || c.eq_ignore_ascii_case("treated")
        || c.eq_ignore_ascii_case("t")
        || c == "1"
    {
        Ok(true)
    } else if c.eq_ignore_ascii_case("control")
        || c.eq_ignore_ascii_case("placebo")
        || c.eq_ignore_ascii_case("c")
        || c == "0"
    {
        Ok(false)
    } else {
        Err(format!(
            "Line {line_no}: Unknown arm '{c}' (expected Treatment/1 or Control/0)"
        ))
    }
}

fn parse_cohort_row(line_no: usize, line: &str) -> Result<(bool, Vec<f64>), String> {
    let delimiter = if line.contains('\t') { '\t' } else { ',' };
    let cols: Vec<&str> = line.split(delimiter).collect();
    if cols.len() < 2 || cols.len() > 1 + MAX_WIN_RATIO_TIERS {
        return Err(format!(
            "Line {line_no}: Expected arm plus 1..={MAX_WIN_RATIO_TIERS} outcome columns, got {}",
            cols.len()
        ));
    }
    let is_treatment = parse_arm_is_treatment(line_no, cols[0])?;
    let mut values = Vec::with_capacity(cols.len() - 1);
    for (col_idx, raw_val) in cols.iter().skip(1).enumerate() {
        let trimmed = raw_val.trim().trim_matches('"');
        let val = trimmed.parse::<f64>().map_err(|_| {
            format!(
                "Line {line_no}: Invalid numeric value '{trimmed}' in tier {}",
                col_idx + 1
            )
        })?;
        if !val.is_finite() {
            return Err(format!(
                "Line {line_no}: Tier {} value must be finite, got '{trimmed}'",
                col_idx + 1
            ));
        }
        values.push(val);
    }
    Ok((is_treatment, values))
}

/// Parses two-arm multi-tier CSV or TSV text into a [`ParsedCohort`].
pub fn parse_win_ratio_cohort(raw: &str) -> Result<ParsedCohort, String> {
    let cleaned = strip_utf8_bom(raw);
    let mut treatment = Vec::new();
    let mut control = Vec::new();
    let mut expected_tiers: Option<usize> = None;
    let mut saw_non_empty = false;

    for (idx, raw_line) in cleaned.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let line_no = idx + 1;
        if !saw_non_empty {
            saw_non_empty = true;
            let delimiter = if line.contains('\t') { '\t' } else { ',' };
            if line.split(delimiter).next().is_some_and(is_header_token) {
                continue;
            }
        }
        if treatment.len() + control.len() >= math_commons::registry::MAX_ITERATIONS {
            return Err(format!(
                "Line {line_no}: Maximum of {} cohort rows exceeded",
                math_commons::registry::MAX_ITERATIONS
            ));
        }
        let (is_trt, values) = parse_cohort_row(line_no, line)?;
        if let Some(expected) = expected_tiers {
            if values.len() != expected {
                return Err(format!(
                    "Line {line_no}: Expected {expected} tier columns, got {}",
                    values.len()
                ));
            }
        } else {
            expected_tiers = Some(values.len());
        }
        if is_trt {
            treatment.push(values);
        } else {
            control.push(values);
        }
    }

    let tier_count = expected_tiers.ok_or_else(|| "Cohort data cannot be empty".to_string())?;
    if treatment.is_empty() || control.is_empty() {
        return Err("Cohort must include at least one Treatment and one Control subject".into());
    }
    Ok(ParsedCohort {
        treatment,
        control,
        tier_count,
    })
}

/// Evaluates hierarchical tier counts, sample win-loss counts, and 95% CI statistics.
pub fn evaluate_cohort(
    cohort: &ParsedCohort,
    tier_specs: &[TierSpec],
    pairing: PairingMode,
) -> Result<CohortEvaluation, String> {
    let active_tiers = cohort.tier_count.min(tier_specs.len());
    if active_tiers == 0 {
        return Err("At least one active tier comparator is required".into());
    }
    let mut analysis = WinRatioAnalysis::new();
    for spec in tier_specs.iter().take(active_tiers) {
        analysis = analysis.add_strategy(spec.to_comparator());
    }
    let tiers = match pairing {
        PairingMode::Unmatched => analysis
            .evaluate_tiers(&cohort.treatment, &cohort.control, &UnmatchedPairing)
            .map_err(|e| e.to_string())?,
        PairingMode::Matched => analysis
            .evaluate_tiers(&cohort.treatment, &cohort.control, &MatchedPairing)
            .map_err(|e| e.to_string())?,
    };

    let total_wins: usize = tiers.iter().map(|t| t.wins).sum();
    let total_losses: usize = tiers.iter().map(|t| t.losses).sum();
    let total_ties = tiers.last().map_or(0, |t| t.ties);
    let n_a = tiers.first().map_or(0, |t| t.losses as u32);
    let n_b = tiers.first().map_or(0, |t| t.wins as u32);
    let n_c: u32 = tiers.iter().skip(1).map(|t| t.losses as u32).sum();
    let n_d: u32 = tiers.iter().skip(1).map(|t| t.wins as u32).sum();
    let win_loss_counts = WinLossCounts::new(n_a, n_b, n_c, n_d);
    let sample_wr = calculate_sample_win_ratio(&win_loss_counts);
    let win_proportion =
        calculate_win_proportion(win_loss_counts.n_wins(), win_loss_counts.n_losses());
    let stats = calculate_statistics(total_wins as i32, total_losses as i32);

    Ok(CohortEvaluation {
        tiers,
        stats,
        win_loss_counts,
        sample_wr,
        win_proportion,
        total_wins,
        total_losses,
        total_ties,
        n_treatment: cohort.treatment.len(),
        n_control: cohort.control.len(),
    })
}

/// Formats the hierarchical waterfall summary table as CSV.
#[must_use]
pub fn format_waterfall_csv(eval: &CohortEvaluation, specs: &[TierSpec]) -> String {
    use std::fmt::Write;
    let mut out = String::from("tier_index,tier_label,comparator,wins,losses,ties,decided\n");
    for (idx, counts) in eval.tiers.iter().enumerate() {
        let spec = specs.get(idx);
        let label = spec.map_or("Tier", |s| s.label.as_str()).replace(',', ";");
        let comp = spec.map_or("HigherIsBetter", |s| s.kind.label());
        let _ = writeln!(
            &mut out,
            "{},{},{},{},{},{},{}",
            idx + 1,
            label,
            comp,
            counts.wins,
            counts.losses,
            counts.ties,
            counts.decided()
        );
    }
    out
}

/// Formats the hierarchical waterfall summary and overall statistics as JSON.
#[must_use]
pub fn format_waterfall_json(eval: &CohortEvaluation, specs: &[TierSpec]) -> String {
    let tiers_json: Vec<serde_json::Value> = eval
        .tiers
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            let s = specs.get(idx);
            serde_json::json!({
                "tier_index": idx + 1,
                "label": s.map_or("Tier", |sp| sp.label.as_str()),
                "comparator": s.map_or("Higher is Better", |sp| sp.kind.label()),
                "wins": c.wins, "losses": c.losses, "ties": c.ties, "decided": c.decided(),
            })
        })
        .collect();
    let stats_json = eval.stats.map(|s| {
        serde_json::json!({
            "win_ratio": s.win_ratio.is_finite().then_some(s.win_ratio),
            "ci_low": s.ci_low.is_finite().then_some(s.ci_low),
            "ci_high": s.ci_high.is_finite().then_some(s.ci_high),
            "p_value": s.p_value,
        })
    });
    serde_json::to_string_pretty(&serde_json::json!({
        "n_treatment": eval.n_treatment, "n_control": eval.n_control,
        "total_wins": eval.total_wins, "total_losses": eval.total_losses,
        "total_ties": eval.total_ties, "win_proportion": eval.win_proportion,
        "statistics": stats_json, "tiers": tiers_json,
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

/// Non-blocking file open/save channel state for cohort imports and table downloads.
#[derive(Default)]
pub struct FileDialogChannels {
    open_rx: Option<Receiver<Result<(String, String), String>>>,
    save_rx: Option<Receiver<Result<String, String>>>,
    /// Most recent file I/O status banner message (`is_error`, `message`).
    pub status: Option<(bool, String)>,
}

impl FileDialogChannels {
    /// Opens a local CSV or TSV cohort file dialog.
    pub fn trigger_open_cohort(&mut self) {
        let (tx, rx) = channel();
        self.open_rx = Some(rx);
        spawn_open_file_dialog(tx, COHORT_OPEN_CONFIG, |path| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "cohort.csv".to_string())
        });
    }

    /// Triggers a CSV download/save-as dialog.
    pub fn trigger_save_csv(&mut self, default_filename: &str, content: String) {
        self.trigger_save(CSV_SAVE_CONFIG, default_filename, content);
    }

    /// Triggers a JSON download/save-as dialog.
    pub fn trigger_save_json(&mut self, default_filename: &str, content: String) {
        self.trigger_save(JSON_SAVE_CONFIG, default_filename, content);
    }

    fn trigger_save(&mut self, cfg: SaveAsDialogConfig, default_filename: &str, content: String) {
        let (tx, rx) = channel();
        self.save_rx = Some(rx);
        spawn_save_as_file_dialog(tx, cfg, default_filename, content, |path, fallback| {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| fallback.to_string())
        });
    }

    /// Polls pending open/save channels and returns newly loaded file `(filename, text)` if ready.
    pub fn poll(&mut self) -> Option<(String, String)> {
        let mut loaded = None;
        if let Some(Ok(res)) = self.open_rx.as_ref().map(Receiver::try_recv) {
            self.open_rx = None;
            match res {
                Ok((name, content)) => {
                    self.status = Some((false, format!("Loaded {name}")));
                    loaded = Some((name, content));
                }
                Err(err) => self.status = Some((true, err)),
            }
        }
        if let Some(Ok(res)) = self.save_rx.as_ref().map(Receiver::try_recv) {
            self.save_rx = None;
            match res {
                Ok(name) => self.status = Some((false, format!("Saved {name}"))),
                Err(err) => self.status = Some((true, err)),
            }
        }
        loaded
    }
}

// [cite:win_ratio]
