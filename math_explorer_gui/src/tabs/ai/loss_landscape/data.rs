#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::MathExpressionInputBar;

pub const DEFAULT_TARGET_EXPR: &str = "2 * x + 1";
pub const DEFAULT_TARGET_X_BOUNDS: [f64; 2] = [0.0, 4.0];
pub const DEFAULT_TARGET_SAMPLES: usize = 9;
pub const DEFAULT_HUBER_DELTA: f64 = 1.0;
pub const MAX_TABLE_ROWS: usize = 128;
pub const DEFAULT_TABLE_TEXT: &str = "0.0, 1.0\n1.0, 3.0\n2.0, 5.0\n3.0, 7.0\n4.0, 9.0";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LossMetric {
    #[default]
    Mse,
    Mae,
    Huber,
}

impl LossMetric {
    #[must_use]
    pub const fn is_mse(&self) -> bool {
        matches!(self, Self::Mse)
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mse => "MSE (Mean Squared)",
            Self::Mae => "MAE (Mean Absolute)",
            Self::Huber => "Huber (Smooth L1)",
        }
    }

    #[inline]
    #[must_use]
    pub fn point_loss(self, residual: f64, huber_delta: f64) -> f64 {
        match self {
            Self::Mse => residual * residual,
            Self::Mae => residual.abs(),
            Self::Huber => {
                let delta = huber_delta.max(1e-6);
                let abs_r = residual.abs();
                if abs_r <= delta {
                    0.5 * residual * residual
                } else {
                    delta * (abs_r - 0.5 * delta)
                }
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetDataMode {
    #[default]
    PresetLinear,
    Formula,
    Table,
}

impl TargetDataMode {
    #[must_use]
    pub const fn is_preset_linear(&self) -> bool {
        matches!(self, Self::PresetLinear)
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::PresetLinear => "Preset (y = 2x + 1)",
            Self::Formula => "Formula y(x)",
            Self::Table => "Table (x, y)",
        }
    }
}

#[must_use]
pub fn default_preset_targets() -> Vec<[f64; 2]> {
    vec![[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0], [4.0, 9.0]]
}

#[must_use]
pub fn default_target_formula_bar() -> MathExpressionInputBar {
    MathExpressionInputBar::new(
        "y(x)",
        DEFAULT_TARGET_EXPR,
        vec![
            ("Linear y = 2*x + 1", "2 * x + 1"),
            ("Harmonic Target", "a * sin(k * x) + c"),
            ("Quadratic Target", "a * x^2 + b0"),
            ("Outlier Spike", "2 * x + 1 + 8 * rect((x - 2) * 2)"),
        ],
    )
    .with_axes(&["x"])
}

pub fn sample_formula_targets(
    bar: &MathExpressionInputBar,
    x_bounds: [f64; 2],
    sample_count: usize,
) -> Option<Vec<[f64; 2]>> {
    let compiled = bar.active_compiled()?;
    let [x0, x1] = x_bounds;
    if !x0.is_finite()
        || !x1.is_finite()
        || x0 >= x1
        || !(2..=MAX_TABLE_ROWS).contains(&sample_count)
    {
        return None;
    }
    let params = bar.param_slice();
    let denom = (sample_count - 1) as f64;
    let mut pts = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let x = x0 + (x1 - x0) * (i as f64 / denom);
        let y = compiled.eval_1d(x, params);
        if !y.is_finite() {
            return None;
        }
        pts.push([x, y.clamp(-1e6, 1e6)]);
    }
    Some(pts)
}

pub fn parse_target_table(text: &str) -> Result<Vec<[f64; 2]>, String> {
    let mut rows = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line
            .split([',', ';', '\t', ' '])
            .filter(|tok| !tok.is_empty());
        let Some(xs) = parts.next() else {
            continue;
        };
        let Some(ys) = parts.next() else {
            return Err(format!("Row {} missing y value", idx + 1));
        };
        if parts.next().is_some() {
            return Err(format!("Row {} has more than 2 columns", idx + 1));
        }
        let x: f64 = xs
            .parse()
            .map_err(|_| format!("Row {} invalid x '{xs}'", idx + 1))?;
        let y: f64 = ys
            .parse()
            .map_err(|_| format!("Row {} invalid y '{ys}'", idx + 1))?;
        if !x.is_finite() || !y.is_finite() || x.abs() > 1e6 || y.abs() > 1e6 {
            return Err(format!("Row {} values out of finite range", idx + 1));
        }
        rows.push([x, y]);
        if rows.len() > MAX_TABLE_ROWS {
            return Err(format!(
                "Target table exceeds maximum {MAX_TABLE_ROWS} rows"
            ));
        }
    }
    if rows.is_empty() {
        return Err("Target table requires at least one (x, y) row".into());
    }
    Ok(rows)
}

#[must_use]
pub fn format_target_table(rows: &[[f64; 2]]) -> String {
    rows.iter()
        .map(|[x, y]| format!("{x}, {y}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Loss Landscape",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "w_min": -1.0,
    "w_max": 5.0,
    "b_min": -2.0,
    "b_max": 4.0,
    "resolution": 30
  }
}"#;

pub const NONCONVEX_SINUSOIDAL_PRESET_JSON: &str = r#"{
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
