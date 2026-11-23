#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Federated `ToolMetadata` and `StarterPresetMetadata` registrations for [`ConditioningWorkbenchTool`].

use super::state::ConditioningWorkbenchTool;

/// Starter preset demonstrating explicit Normal-Equation rank collapse on Läuchli ($\mu = 10^{-8}$).
pub const PRESET_LINALG_LAUCHLI: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Conditioning Workbench - Lauchli Normal-Equation Breakdown",
    "domain": "linear_algebra",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "perturbation_log10": -8.5,
    "noise_std": 0.0,
    "ridge_lambda_log10": -10.0,
    "rows": 7,
    "cols": 6,
    "truncation_rank": 4,
    "family": "Lauchli"
  }
}"#;

/// Starter preset exploring exponential singular-value decay on a $16 \times 10$ Vandermonde system.
pub const PRESET_LINALG_VANDERMONDE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Conditioning Workbench - Vandermonde Spectral Decay",
    "domain": "linear_algebra",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "perturbation_log10": -6.0,
    "noise_std": 0.0,
    "ridge_lambda_log10": -9.0,
    "rows": 16,
    "cols": 10,
    "truncation_rank": 5,
    "family": "Vandermonde"
  }
}"#;

/// Starter preset showing Augmented-QR Tikhonov Ridge stabilization on a noisy Hilbert matrix.
pub const PRESET_LINALG_HILBERT_RIDGE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Conditioning Workbench - Noisy Hilbert & Ridge Regularization",
    "domain": "linear_algebra",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "perturbation_log10": -5.0,
    "noise_std": 0.001,
    "ridge_lambda_log10": -5.0,
    "rows": 8,
    "cols": 8,
    "truncation_rank": 4,
    "family": "Hilbert"
  }
}"#;

/// Starter preset highlighting Lawson-Hanson active-set NNLS on a noisy near-collinear system.
pub const PRESET_LINALG_NNLS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Conditioning Workbench - Lawson-Hanson Non-Negative Least Squares",
    "domain": "linear_algebra",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "perturbation_log10": -10.5,
    "noise_std": 0.05,
    "ridge_lambda_log10": -4.0,
    "rows": 12,
    "cols": 6,
    "truncation_rank": 3,
    "family": "RankDeficient"
  }
}"#;

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LinearAlgebraConditioningTool",
        display_name: "Conditioning & Least-Squares Workbench",
        domain: "linear_algebra",
        tags: &["qr", "svd", "cholesky", "conditioning", "ridge", "nnls"],
        build: || Box::new(ConditioningWorkbenchTool::default()),
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "linalg_lauchli_breakdown",
        title: "Läuchli Normal-Equation Breakdown",
        description: "kappa_2(A^T A) = kappa_2(A)^2: Explicit normal equations fail at mu = 10^-8.5 while Householder QR and thin SVD stay exact.",
        category: "Linear Algebra",
        icon: "📐",
        domain: "linear_algebra",
        tool_name: Some("LinearAlgebraConditioningTool"),
        preset_generator: || PRESET_LINALG_LAUCHLI,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "linalg_vandermonde_conditioning",
        title: "Vandermonde Spectral Decay",
        description: "V_{ij} = t_i^j: Exponential singular-value decay and Householder QR polynomial fitting without forming V^T V.",
        category: "Linear Algebra",
        icon: "📉",
        domain: "linear_algebra",
        tool_name: Some("LinearAlgebraConditioningTool"),
        preset_generator: || PRESET_LINALG_VANDERMONDE,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "linalg_hilbert_ridge",
        title: "Noisy Hilbert & Ridge Regularization",
        description: "H_{ij} = 1/(i+j+1): Compare unregularized noise amplification against Augmented-QR Tikhonov Ridge shrinkage.",
        category: "Linear Algebra",
        icon: "🛡",
        domain: "linear_algebra",
        tool_name: Some("LinearAlgebraConditioningTool"),
        preset_generator: || PRESET_LINALG_HILBERT_RIDGE,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "linalg_nnls_active_set",
        title: "Lawson-Hanson Non-Negative Least Squares",
        description: "min_{x >= 0} ||Ax - b||_2^2: Active-set NNLS enforces physical non-negativity on collinear noisy observations.",
        category: "Linear Algebra",
        icon: "⚖",
        domain: "linear_algebra",
        tool_name: Some("LinearAlgebraConditioningTool"),
        preset_generator: || PRESET_LINALG_NNLS,
    }
}

// [cite:advanced_linear_algebra]
