#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::RiemannIntegrationTool;

impl scientific_metadata::theory::TheoryDescribable for RiemannIntegrationTool {
    fn theory_description(&self) -> String {
        format!(
            "Numerical approximation of the definite integral $\\int_a^b f(x) dx$ for {} using {} over interval [{:.2}, {:.2}] (estimated error {:.2e}).",
            self.preset.name(),
            self.quadrature_method.name(),
            self.min_x,
            self.max_x,
            self.error_estimate
        )
    }

    fn phonetic_description(&self) -> String {
        "Riemann integration, composite trapezoidal, composite Simpson, Clenshaw-Curtis, and adaptive Gauss-Kronrod quadrature theory"
            .into()
    }

    fn theory_citation(&self) -> String {
        "Calculus, Vol. 1 (Apostol, 1967); Clenshaw & Curtis (1960); Piessens et al., QUADPACK (1983)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "SinX".into(),
            "Integral of sin(x) showing oscillatory area cancellation".into(),
        );
        map.insert(
            "XSquared".into(),
            "Integral of x^2 demonstrating exact parabolic Simpson quadrature".into(),
        );
        map.insert(
            "EToX".into(),
            "Integral of e^x demonstrating exponential growth integration".into(),
        );
        map.insert(
            "Custom".into(),
            "Composite Newton-Cotes, Clenshaw-Curtis, and adaptive Gauss-Kronrod quadrature over compiled formulas".into(),
        );
        map
    }
}

pub(super) const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Composite Trapezoidal Integration",
    "domain": "analysis",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "preset": "SinX",
    "min_x": 0.0,
    "max_x": 3.141592653589793,
    "steps": 20
  }
}"#;

pub(super) const CUSTOM_PIECEWISE_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Piecewise Pulse Quadrature",
    "domain": "analysis",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "preset": "Custom",
    "quadrature_method": "GaussKronrod15",
    "cc_tolerance": 1e-8,
    "min_x": -3.0,
    "max_x": 3.0,
    "steps": 36,
    "custom_expr": "clamp(sin(k * x), -0.5, 0.8)",
    "custom_params": {
      "k": 3.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemann_integration_starter",
        title: "Composite Trapezoidal Integration",
        description: "Definite integral numerical quadrature via composite trapezoidal rule partitions.",
        category: "Integral Calculus",
        icon: "📐",
        domain: "analysis",
        tool_name: Some("RiemannIntegrationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "riemann_custom_piecewise_quadrature",
        title: "Piecewise Pulse Quadrature",
        description: "Adaptive Gauss-Kronrod 15 quadrature of clamp(sin(k*x), -0.5, 0.8) with live frequency slider k.",
        category: "Integral Calculus",
        icon: "∫",
        domain: "analysis",
        tool_name: Some("RiemannIntegrationTool"),
        preset_generator: || CUSTOM_PIECEWISE_PRESET_JSON,
    }
}
