#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Predictor formula mode, scientific theory metadata, and starter gallery presets
//! for [`LossLandscapeTool`].

use super::surfaces::LandscapeSurface;
use super::LossLandscapeTool;

pub(crate) const DEFAULT_LOSS_EXPR: &str = "w * sin(b * x)";

/// Predictor formula mode when evaluating MSE over the regression dataset.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PredictorMode {
    #[default]
    Linear,
    Custom,
}

impl PredictorMode {
    #[inline]
    #[must_use]
    pub const fn is_linear(&self) -> bool {
        matches!(self, Self::Linear)
    }
}

impl scientific_metadata::theory::TheoryDescribable for LossLandscapeTool {
    fn theory_description(&self) -> String {
        if self.surface == LandscapeSurface::LinearMse {
            let eq = if self.loss_metric.is_mse() {
                "$L(w, b) = \\frac{1}{n}\\sum_{i=1}^n (f(x_i, w, b) - y_i)^2$"
            } else {
                "$L(w, b) = \\frac{1}{n}\\sum_{i=1}^n \\ell(f(x_i, w, b) - y_i)$"
            };
            format!(
                "Empirical risk landscape {eq} using {} objective over {} target dataset, raced in 3D/2D by Hamiltonian and Langevin dynamics.",
                self.loss_metric.label(),
                self.target_mode.label()
            )
        } else {
            "3D and 2D non-convex optimization arena comparing first-order optimizers ($\\text{SGD}$, $\\text{Adam}$) against symplectic/conformal Hamiltonian dynamics $H(q, p) = \\frac{\\|p\\|^2}{2m} + L(q)$ and underdamped BAOAB Langevin diffusion $dp = -\\nabla L(q)\\,dt - \\gamma p\\,dt + \\sqrt{2\\gamma m k_B T}\\,dW_t$ across sculpted Gaussian potentials.".into()
        }
    }

    fn phonetic_description(&self) -> String {
        "Hamiltonian and Langevin momentum optimization dynamics on 3D MSE, MAE, and Huber loss landscapes"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Visualizing the Loss Landscape of Neural Nets (Li et al., 2018); Huber (1964), Annals of Mathematical Statistics; Variational Perspectives on Accelerated Optimization (Wibisono, Wilson & Jordan, 2016); Molecular Dynamics (Leimkuhler & Matthews, 2015)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

pub(crate) const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Loss Landscape","domain":"ai","timestamp":"2026-11-16T00:00:00Z"},"config":{"w_min":-1.0,"w_max":5.0,"b_min":-2.0,"b_max":4.0,"resolution":30}}"#;

pub(crate) const NONCONVEX_SINUSOIDAL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Non-Convex Sinusoidal Loss",
    "domain": "ai",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "w_min": -4.0,
    "w_max": 4.0,
    "b_min": -3.0,
    "b_max": 3.0,
    "resolution": 32,
    "predictor_mode": "Custom",
    "custom_expr": "w * sin(b * x) + a",
    "custom_params": {
      "a": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "loss_landscape_starter",
        title: "Loss Landscape",
        description: "3D and 2D Hamiltonian & Langevin loss landscape arena over Linear MSE, Rosenbrock, Rastrigin, and sculpted Gaussian surfaces.",
        category: "Optimization",
        icon: "⛰️",
        domain: "ai",
        tool_name: Some("LossLandscapeTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "loss_landscape_nonconvex_sinusoid",
        title: "Non-Convex Sinusoidal Loss",
        description: "f(x, w, b) = w*sin(b*x) + a: Multi-minima non-convex MSE landscape with live offset slider a.",
        category: "Optimization",
        icon: "🌄",
        domain: "ai",
        tool_name: Some("LossLandscapeTool"),
        preset_generator: || NONCONVEX_SINUSOIDAL_PRESET_JSON,
    }
}
