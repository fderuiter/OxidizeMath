#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{RootDimension, RootFinderTool};

impl scientific_metadata::theory::TheoryDescribable for RootFinderTool {
    fn theory_description(&self) -> String {
        match self.dimension {
            RootDimension::Scalar1D => format!(
                "Scalar nonlinear root-finding for f(x) = 0 using {} over [{:.3}, {:.3}] (tolerance {:.1e}).",
                self.scalar_method.name(),
                self.bracket_min,
                self.bracket_max,
                self.tolerance
            ),
            RootDimension::System2D => format!(
                "Coupled 2D nonlinear system F(x, y) = 0 solved via damped Multivariate Newton-Raphson from ({:.2}, {:.2}) with zero-level contour overlays.",
                self.init_2d[0], self.init_2d[1]
            ),
        }
    }

    fn phonetic_description(&self) -> String {
        "Nonlinear root finding theory with bisection, Brent inverse quadratic interpolation, secant chords, Newton tangents, and multivariate Jacobian Newton steps".into()
    }

    fn theory_citation(&self) -> String {
        "Algorithms for Minimization without Derivatives (Brent, 1973); Numerical Analysis (Burden & Faires, 2010)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Bisection".into(),
            "Guaranteed binary bracket halving under intermediate value theorem sign changes".into(),
        );
        map.insert(
            "Brent".into(),
            "Hybrid superlinear root-finder combining inverse quadratic interpolation, secant chords, and bisection safeguards".into(),
        );
        map.insert(
            "NewtonRaphson".into(),
            "Quadratic-order open tangent iteration x_{k+1} = x_k - f(x_k)/f'(x_k)".into(),
        );
        map.insert(
            "Secant".into(),
            "Derivative-free superlinear chord method of order (1 + sqrt(5))/2".into(),
        );
        map.insert(
            "MultivariateNewton".into(),
            "Damped 2D Jacobian linearizations J(x_k) delta = -F(x_k) across F_1=0 and F_2=0 contours".into(),
        );
        map
    }
}

pub(super) const BRENT_TRANSCENDENTAL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Brent Kepler & Dottie Root",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "dimension": "Scalar1D",
    "scalar_method": "Brent",
    "bracket_min": 0.0,
    "bracket_max": 2.0,
    "tolerance": 1e-8,
    "max_iterations": 60,
    "scalar_expr": "cos(x) - k * x",
    "scalar_params": {
      "k": 1.0
    }
  }
}"#;

pub(super) const NEWTON_2D_CONTOUR_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "2D Circle-Parabola Newton Intersection",
    "domain": "analysis",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "dimension": "System2D",
    "init_2d": [1.6, 1.4],
    "domain_span_2d": 3.0,
    "tolerance": 1e-8,
    "max_iterations": 40,
    "coupled_exprs": ["x^2 + y^2 - r^2", "x^2 - y - c"],
    "coupled_params": {
      "r": 2.0,
      "c": 0.5
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "root_finder_brent_kepler",
        title: "Brent Transcendental Root",
        description: "Superlinear Brent bracketing on cos(x) - k*x = 0 with interactive bracket & chord scrubbing.",
        category: "Numerical Analysis",
        icon: "🎯",
        domain: "analysis",
        tool_name: Some("RootFinderTool"),
        preset_generator: || BRENT_TRANSCENDENTAL_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "root_finder_newton_2d_contours",
        title: "2D Multivariate Newton Contours",
        description: "Coupled circle-parabola zero-contour intersection solved via damped 2D Multivariate Newton.",
        category: "Numerical Analysis",
        icon: "🧭",
        domain: "analysis",
        tool_name: Some("RootFinderTool"),
        preset_generator: || NEWTON_2D_CONTOUR_PRESET_JSON,
    }
}
