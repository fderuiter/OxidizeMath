//! Starter gallery presets for `CalibrationWorkbenchTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// Starter preset JSON for overconfident neural logits with post-hoc temperature scaling.
pub const PRESET_OVERCONFIDENT_MLP: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Calibration Workbench",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "logit_scale": 3.1,
    "logit_bias": 0.75,
    "crps_spread": 1.8,
    "num_bins": 10,
    "sample_count": 260,
    "seed": 42,
    "scenario": "OverconfidentMlp",
    "recalibrator": "Temperature",
    "view": "Reliability",
    "binning": "Uniform"
  }
}"#;

/// Starter preset JSON for a skewed clinical risk cohort with monotonic `PAVA` recalibration.
pub const PRESET_CLINICAL_ISOTONIC: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Calibration Workbench",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "logit_scale": 2.4,
    "logit_bias": 0.55,
    "crps_spread": 1.5,
    "num_bins": 12,
    "sample_count": 300,
    "seed": 108,
    "scenario": "SkewedClinicalCohort",
    "recalibrator": "IsotonicPava",
    "view": "MurphyBars",
    "binning": "Quantile"
  }
}"#;

/// Starter preset JSON for Gaussian and empirical ensemble `CRPS` diagnostics.
pub const PRESET_ENSEMBLE_CRPS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Calibration Workbench",
    "domain": "ai",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "logit_scale": 1.9,
    "logit_bias": -0.35,
    "crps_spread": 2.4,
    "num_bins": 10,
    "sample_count": 220,
    "seed": 77,
    "scenario": "UnderconfidentEnsemble",
    "recalibrator": "Platt",
    "view": "CrpsForecast",
    "binning": "Uniform"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "calibration_overconfident_mlp",
        title: "Overconfident Neural Recalibration",
        description: "Diagnose overconfident neural classifier logits on reliability diagrams and apply post-hoc temperature scaling.",
        category: "Probabilistic Calibration",
        icon: "🎯",
        domain: "ai",
        tool_name: Some("CalibrationWorkbenchTool"),
        preset_generator: || PRESET_OVERCONFIDENT_MLP,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "calibration_clinical_isotonic",
        title: "Clinical Risk Murphy & PAVA",
        description: "Inspect Murphy reliability-resolution-uncertainty bars on a skewed clinical risk cohort recalibrated via monotonic PAVA isotonic regression.",
        category: "Probabilistic Calibration",
        icon: "🏥",
        domain: "ai",
        tool_name: Some("CalibrationWorkbenchTool"),
        preset_generator: || PRESET_CLINICAL_ISOTONIC,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "calibration_ensemble_crps",
        title: "Gaussian & Ensemble CRPS",
        description: "Compare analytical Gaussian CRPS and sorted empirical ensemble CRPS alongside Platt-scaled forecast diagnostics.",
        category: "Probabilistic Calibration",
        icon: "📊",
        domain: "ai",
        tool_name: Some("CalibrationWorkbenchTool"),
        preset_generator: || PRESET_ENSEMBLE_CRPS,
    }
}

// [cite:clinical_trials_statistics]
