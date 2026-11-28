//! State, benchmark generation, and evaluation pipeline for `CalibrationWorkbenchTool`.

use math_explorer::primitives::{PositiveFloat, UnitInterval};
use math_explorer::pure_math::statistics::calibration::{
    binary_log_loss, brier_score, dataset_gaussian_crps, ensemble_crps_sorted,
    expected_calibration_error, gaussian_crps, maximum_calibration_error,
    murphy_decomposition_binned, pr_auc, precision_recall_curve, reliability_curve, roc_auc,
    roc_curve, BinningStrategy, BrierDecomposition, IsotonicCalibrator, PlattScaler, PrIntegration,
    PrPoint, ReliabilityBin, RocPoint, TemperatureScaler, DEFAULT_LOG_LOSS_EPSILON,
};
use oxidize_core::rng::OxidizeRng;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Synthetic benchmark forecast scenario used in [`super::CalibrationWorkbenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastScenario {
    /// Overconfident and bias-shifted neural classifier logits.
    OverconfidentMlp,
    /// Underconfident ensemble logits compressed toward `0.5`.
    UnderconfidentEnsemble,
    /// Low-prevalence clinical cohort with nonlinear S-curve distortion.
    SkewedClinicalCohort,
    /// Near-ideal calibrated reference classifier.
    CalibratedBaseline,
}

impl ForecastScenario {
    /// All selectable synthetic forecast scenarios.
    pub const ALL: [Self; 4] = [
        Self::OverconfidentMlp,
        Self::UnderconfidentEnsemble,
        Self::SkewedClinicalCohort,
        Self::CalibratedBaseline,
    ];

    /// Display label for combo-box controls.
    pub const fn label(self) -> &'static str {
        match self {
            Self::OverconfidentMlp => "Overconfident MLP",
            Self::UnderconfidentEnsemble => "Underconfident Ensemble",
            Self::SkewedClinicalCohort => "Skewed Clinical Cohort",
            Self::CalibratedBaseline => "Calibrated Baseline",
        }
    }
}

/// Post-hoc recalibration method applied to held-out calibration logits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecalibratorKind {
    /// Uncalibrated raw sigmoid probabilities.
    None,
    /// Damped Newton-Raphson logistic Platt scaling $\sigma(a z + b)$.
    Platt,
    /// Convex single-parameter temperature scaling $\sigma(z / T)$.
    Temperature,
    /// Monotonic `PAVA` isotonic regression with piecewise-linear interpolation.
    IsotonicPava,
}

impl RecalibratorKind {
    /// All selectable post-hoc recalibrators.
    pub const ALL: [Self; 4] = [
        Self::None,
        Self::Platt,
        Self::Temperature,
        Self::IsotonicPava,
    ];

    /// Display label for combo-box controls.
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "None (Raw)",
            Self::Platt => "Platt Scaling",
            Self::Temperature => "Temperature Scaling",
            Self::IsotonicPava => "PAVA Isotonic",
        }
    }
}

/// Primary visualization pane rendered in [`super::CalibrationWorkbenchTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkbenchView {
    /// Reliability diagram comparing confidence vs empirical accuracy across bins.
    Reliability,
    /// Grouped bar chart of Murphy's `Reliability`, `Resolution`, `Uncertainty`, and `Brier`.
    MurphyBars,
    /// Monotonic threshold-sweep `ROC` curve with `ROC-AUC`.
    RocCurve,
    /// Monotonic threshold-sweep Precision-Recall curve with `Step` and `Trapezoidal` `PR-AUC`.
    PrCurve,
    /// Analytical Gaussian `CRPS` and empirical sorted ensemble `CRPS` across forecast errors.
    CrpsForecast,
}

impl WorkbenchView {
    /// All selectable diagnostic plot views.
    pub const ALL: [Self; 5] = [
        Self::Reliability,
        Self::MurphyBars,
        Self::RocCurve,
        Self::PrCurve,
        Self::CrpsForecast,
    ];

    /// Display label for combo-box controls and plot headers.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Reliability => "Reliability Diagram",
            Self::MurphyBars => "Murphy Decomposition",
            Self::RocCurve => "ROC Curve",
            Self::PrCurve => "Precision-Recall",
            Self::CrpsForecast => "CRPS Forecast",
        }
    }
}

/// Bounded serializable configuration for URL sharing and starter presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationWorkbenchConfig {
    /// Multiplicative distortion slope applied to ground-truth logits (`[0.1, 8.0]`).
    pub logit_scale: f64,
    /// Additive distortion intercept applied to ground-truth logits (`[-4.0, 4.0]`).
    pub logit_bias: f64,
    /// Uncalibrated predictive Gaussian standard deviation $\sigma$ (`[0.1, 5.0]`).
    pub crps_spread: f64,
    /// Number of reliability diagram bins (`2..=40`).
    pub num_bins: usize,
    /// Number of synthetic samples per calibration and evaluation split (`40..=1200`).
    pub sample_count: usize,
    /// Deterministic PRNG seed for synthetic cohort sampling.
    pub seed: u64,
    /// Active synthetic miscalibration benchmark.
    pub scenario: ForecastScenario,
    /// Active post-hoc recalibration method.
    pub recalibrator: RecalibratorKind,
    /// Active diagnostic plot view.
    pub view: WorkbenchView,
    /// Reliability curve binning strategy (`Uniform` or `Quantile`).
    pub binning: BinningStrategy,
}

impl Default for CalibrationWorkbenchConfig {
    fn default() -> Self {
        Self {
            logit_scale: 2.8,
            logit_bias: 0.65,
            crps_spread: 1.8,
            num_bins: 10,
            sample_count: 240,
            seed: 42,
            scenario: ForecastScenario::OverconfidentMlp,
            recalibrator: RecalibratorKind::Temperature,
            view: WorkbenchView::Reliability,
            binning: BinningStrategy::Uniform,
        }
    }
}

impl CalibrationWorkbenchConfig {
    /// Validates finite numeric ranges for safe URL and preset deserialization.
    pub fn validate(&self) -> Result<(), String> {
        if !self.logit_scale.is_finite() || !(0.1..=8.0).contains(&self.logit_scale) {
            return Err("logit_scale must be finite and within [0.1, 8.0]".into());
        }
        if !self.logit_bias.is_finite() || !(-4.0..=4.0).contains(&self.logit_bias) {
            return Err("logit_bias must be finite and within [-4.0, 4.0]".into());
        }
        if !self.crps_spread.is_finite() || !(0.1..=5.0).contains(&self.crps_spread) {
            return Err("crps_spread must be finite and within [0.1, 5.0]".into());
        }
        if !(2..=40).contains(&self.num_bins) {
            return Err("num_bins must be within 2..=40".into());
        }
        if !(40..=1200).contains(&self.sample_count) {
            return Err("sample_count must be within 40..=1200".into());
        }
        Ok(())
    }

    /// Clamps out-of-range or non-finite parameters into valid bounds, returning `true` if modified.
    pub fn clamp_in_place(&mut self) -> bool {
        let orig = self.clone();
        self.logit_scale = if self.logit_scale.is_finite() {
            self.logit_scale.clamp(0.1, 8.0)
        } else {
            2.8
        };
        self.logit_bias = if self.logit_bias.is_finite() {
            self.logit_bias.clamp(-4.0, 4.0)
        } else {
            0.65
        };
        self.crps_spread = if self.crps_spread.is_finite() {
            self.crps_spread.clamp(0.1, 5.0)
        } else {
            1.8
        };
        self.num_bins = self.num_bins.clamp(2, 40);
        self.sample_count = self.sample_count.clamp(40, 1200);
        *self != orig
    }
}

/// Held-out scoring, calibration, and ranking summary for a single forecast stream.
#[derive(Debug, Clone, Copy)]
pub struct MetricSnapshot {
    /// Exact quadratic Brier score.
    pub brier: f64,
    /// Symmetrically clipped binary dataset log-loss.
    pub log_loss: f64,
    /// Expected Calibration Error (`ECE`).
    pub ece: f64,
    /// Maximum Calibration Error (`MCE`).
    pub mce: f64,
    /// Receiver Operating Characteristic area under the curve (`ROC-AUC`).
    pub roc_auc: f64,
    /// Average-precision (`Step`) `PR-AUC`.
    pub pr_auc_step: f64,
    /// Piecewise-linear trapezoidal `PR-AUC`.
    pub pr_auc_trap: f64,
    /// Binned Murphy (`REL`, `RES`, `UNC`, `Brier`) decomposition.
    pub murphy: BrierDecomposition,
}

/// Cached evaluation artifacts rendered across [`WorkbenchView`] modes.
#[derive(Debug, Clone)]
pub struct WorkbenchEvaluation {
    /// Metrics on raw uncalibrated probabilities.
    pub raw_metrics: MetricSnapshot,
    /// Metrics after applying the selected post-hoc recalibrator.
    pub cal_metrics: MetricSnapshot,
    /// Raw reliability curve bins.
    pub raw_bins: Vec<ReliabilityBin>,
    /// Recalibrated reliability curve bins.
    pub cal_bins: Vec<ReliabilityBin>,
    /// Raw `ROC` sweep points.
    pub raw_roc: Vec<RocPoint>,
    /// Recalibrated `ROC` sweep points.
    pub cal_roc: Vec<RocPoint>,
    /// Raw Precision-Recall sweep points.
    pub raw_pr: Vec<PrPoint>,
    /// Recalibrated Precision-Recall sweep points.
    pub cal_pr: Vec<PrPoint>,
    /// Empirical positive class prevalence on the evaluation split.
    pub prevalence: f64,
    /// Uncalibrated analytical Gaussian `CRPS` curve `[obs_offset, crps]`.
    pub crps_raw_curve: Vec<[f64; 2]>,
    /// Calibrated analytical Gaussian `CRPS` curve `[obs_offset, crps]`.
    pub crps_cal_curve: Vec<[f64; 2]>,
    /// Empirical sorted ensemble `CRPS` curve `[obs_offset, crps]`.
    pub crps_ens_curve: Vec<[f64; 2]>,
    /// Dataset mean Gaussian `CRPS` before spread calibration.
    pub mean_raw_crps: f64,
    /// Dataset mean Gaussian `CRPS` at calibrated unit spread.
    pub mean_cal_crps: f64,
    /// Human-readable summary of fitted recalibrator parameters.
    pub fit_summary: String,
}

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn distort_logit(z_true: f64, cfg: &CalibrationWorkbenchConfig) -> f64 {
    match cfg.scenario {
        ForecastScenario::OverconfidentMlp => cfg.logit_scale * z_true + cfg.logit_bias,
        ForecastScenario::UnderconfidentEnsemble => {
            (0.32 * cfg.logit_scale).clamp(0.12, 3.0) * z_true + 0.25 * cfg.logit_bias
        }
        ForecastScenario::SkewedClinicalCohort => {
            cfg.logit_scale * z_true + 0.45 * z_true * z_true.abs().sqrt() + cfg.logit_bias
        }
        ForecastScenario::CalibratedBaseline => z_true + 0.1 * cfg.logit_bias,
    }
}

fn sample_cohort(
    rng: &mut OxidizeRng,
    n: usize,
    cfg: &CalibrationWorkbenchConfig,
) -> (Vec<f64>, Vec<bool>) {
    let shift = if cfg.scenario == ForecastScenario::SkewedClinicalCohort {
        -1.15
    } else {
        0.0
    };
    let mut logits = Vec::with_capacity(n);
    let mut outcomes = Vec::with_capacity(n);
    for i in 0..n {
        let grid = -2.4 + 4.8 * ((i as f64) / ((n.saturating_sub(1)).max(1) as f64));
        let jitter = (rng.gen::<f64>() - 0.5) * 0.35;
        let z_true = grid + shift + jitter;
        let p_true = sigmoid(z_true);
        outcomes.push(rng.gen::<f64>() < p_true);
        logits.push(distort_logit(z_true, cfg));
    }
    if outcomes.iter().all(|&y| y) {
        outcomes[0] = false;
    } else if outcomes.iter().all(|&y| !y) {
        outcomes[n - 1] = true;
    }
    (logits, outcomes)
}

fn compute_snapshot(
    probs: &[UnitInterval],
    scores: &[f64],
    outcomes: &[bool],
    cfg: &CalibrationWorkbenchConfig,
) -> (
    MetricSnapshot,
    Vec<ReliabilityBin>,
    Vec<RocPoint>,
    Vec<PrPoint>,
) {
    let bins = reliability_curve(probs, outcomes, cfg.num_bins, cfg.binning).unwrap_or_default();
    let murphy = murphy_decomposition_binned(probs, outcomes, cfg.num_bins, cfg.binning).unwrap_or(
        BrierDecomposition {
            brier_score: 0.0,
            reliability: 0.0,
            resolution: 0.0,
            uncertainty: 0.0,
        },
    );
    let snap = MetricSnapshot {
        brier: brier_score(probs, outcomes).unwrap_or(0.0),
        log_loss: binary_log_loss(probs, outcomes, DEFAULT_LOG_LOSS_EPSILON).unwrap_or(0.0),
        ece: expected_calibration_error(probs, outcomes, cfg.num_bins, cfg.binning).unwrap_or(0.0),
        mce: maximum_calibration_error(probs, outcomes, cfg.num_bins, cfg.binning).unwrap_or(0.0),
        roc_auc: roc_auc(scores, outcomes).unwrap_or(0.5),
        pr_auc_step: pr_auc(scores, outcomes, PrIntegration::Step).unwrap_or(0.5),
        pr_auc_trap: pr_auc(scores, outcomes, PrIntegration::Trapezoidal).unwrap_or(0.5),
        murphy,
    };
    let roc = roc_curve(scores, outcomes).unwrap_or_default();
    let pr = precision_recall_curve(scores, outcomes).unwrap_or_default();
    (snap, bins, roc, pr)
}

/// Samples deterministic calibration and held-out splits, fits the active recalibrator,
/// and evaluates all proper scoring rules and diagnostic curves.
pub fn evaluate_workbench(cfg: &CalibrationWorkbenchConfig) -> WorkbenchEvaluation {
    let mut rng = OxidizeRng::new(cfg.seed);
    let (cal_logits, cal_outcomes) = sample_cohort(&mut rng, cfg.sample_count, cfg);
    let (eval_logits, eval_outcomes) = sample_cohort(&mut rng, cfg.sample_count, cfg);

    let raw_probs: Vec<UnitInterval> = eval_logits
        .iter()
        .map(|&z| UnitInterval::new(sigmoid(z).clamp(0.0, 1.0)).unwrap())
        .collect();
    let raw_scores: Vec<f64> = raw_probs.iter().map(|p| p.value()).collect();

    let (cal_probs, fit_summary) =
        apply_recalibrator(&cal_logits, &cal_outcomes, &eval_logits, cfg);
    let cal_scores: Vec<f64> = cal_probs.iter().map(|p| p.value()).collect();

    let (raw_metrics, raw_bins, raw_roc, raw_pr) =
        compute_snapshot(&raw_probs, &raw_scores, &eval_outcomes, cfg);
    let (cal_metrics, cal_bins, cal_roc, cal_pr) =
        compute_snapshot(&cal_probs, &cal_scores, &eval_outcomes, cfg);

    let prevalence =
        (eval_outcomes.iter().filter(|&&y| y).count() as f64) / (eval_outcomes.len().max(1) as f64);
    let crps = compute_crps_series(&mut rng, cfg);

    WorkbenchEvaluation {
        raw_metrics,
        cal_metrics,
        raw_bins,
        cal_bins,
        raw_roc,
        cal_roc,
        raw_pr,
        cal_pr,
        prevalence,
        crps_raw_curve: crps.raw_curve,
        crps_cal_curve: crps.cal_curve,
        crps_ens_curve: crps.ens_curve,
        mean_raw_crps: crps.mean_raw,
        mean_cal_crps: crps.mean_cal,
        fit_summary,
    }
}

fn apply_recalibrator(
    cal_logits: &[f64],
    cal_outcomes: &[bool],
    eval_logits: &[f64],
    cfg: &CalibrationWorkbenchConfig,
) -> (Vec<UnitInterval>, String) {
    match cfg.recalibrator {
        RecalibratorKind::None => {
            let probs = eval_logits
                .iter()
                .map(|&z| UnitInterval::new(sigmoid(z).clamp(0.0, 1.0)).unwrap())
                .collect();
            (probs, "Identity p = σ(z)".to_string())
        }
        RecalibratorKind::Platt => {
            let scaler = PlattScaler::fit(cal_logits, cal_outcomes).unwrap_or(PlattScaler {
                scale: 1.0,
                bias: 0.0,
            });
            let mut out = vec![UnitInterval::new(0.5).unwrap(); eval_logits.len()];
            let _ = scaler.predict_into(eval_logits, &mut out);
            let msg = format!("Platt a = {:.3}, b = {:.3}", scaler.scale, scaler.bias);
            (out, msg)
        }
        RecalibratorKind::Temperature => {
            let scaler = TemperatureScaler::fit_binary(cal_logits, cal_outcomes)
                .unwrap_or_else(|_| TemperatureScaler::new(PositiveFloat::new(1.0).unwrap()));
            let probs = eval_logits
                .iter()
                .map(|&z| {
                    scaler
                        .predict_binary(z)
                        .unwrap_or_else(|_| UnitInterval::new(0.5).unwrap())
                })
                .collect();
            let msg = format!("Temperature T = {:.3}", scaler.temperature.value());
            (probs, msg)
        }
        RecalibratorKind::IsotonicPava => {
            let iso = IsotonicCalibrator::fit(cal_logits, cal_outcomes).unwrap();
            let mut out = vec![UnitInterval::new(0.5).unwrap(); eval_logits.len()];
            let _ = iso.predict_into(eval_logits, &mut out);
            let msg = format!("PAVA {} monotonic knots", iso.blocks().len());
            (out, msg)
        }
    }
}

struct CrpsSeries {
    raw_curve: Vec<[f64; 2]>,
    cal_curve: Vec<[f64; 2]>,
    ens_curve: Vec<[f64; 2]>,
    mean_raw: f64,
    mean_cal: f64,
}

fn compute_crps_series(rng: &mut OxidizeRng, cfg: &CalibrationWorkbenchConfig) -> CrpsSeries {
    let raw_sigma = PositiveFloat::new(cfg.crps_spread.clamp(0.1, 5.0)).unwrap();
    let cal_sigma = PositiveFloat::new(1.0).unwrap();
    let mut ensemble = Vec::with_capacity(31);
    for m in 0..31 {
        let q = -2.2 + 4.4 * ((m as f64) / 30.0);
        ensemble.push(q);
    }
    ensemble.sort_unstable_by(|a, b| a.total_cmp(b));

    let mut raw_curve = Vec::with_capacity(61);
    let mut cal_curve = Vec::with_capacity(61);
    let mut ens_curve = Vec::with_capacity(61);
    for k in 0..=60 {
        let obs = -3.0 + 6.0 * ((k as f64) / 60.0);
        raw_curve.push([obs, gaussian_crps(0.0, raw_sigma, obs).unwrap_or(0.0)]);
        cal_curve.push([obs, gaussian_crps(0.0, cal_sigma, obs).unwrap_or(0.0)]);
        ens_curve.push([obs, ensemble_crps_sorted(&ensemble, obs).unwrap_or(0.0)]);
    }

    let n = 64;
    let means = vec![0.0; n];
    let raw_stds = vec![raw_sigma; n];
    let cal_stds = vec![cal_sigma; n];
    let obs: Vec<f64> = (0..n)
        .map(|_| (rng.gen::<f64>() + rng.gen::<f64>() + rng.gen::<f64>() - 1.5) * 1.4)
        .collect();
    let mean_raw = dataset_gaussian_crps(&means, &raw_stds, &obs).unwrap_or(0.0);
    let mean_cal = dataset_gaussian_crps(&means, &cal_stds, &obs).unwrap_or(0.0);
    CrpsSeries {
        raw_curve,
        cal_curve,
        ens_curve,
        mean_raw,
        mean_cal,
    }
}

// [cite:clinical_trials_statistics]
