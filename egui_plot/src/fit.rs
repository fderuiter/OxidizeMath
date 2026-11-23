//! Curve fitting and nonlinear regression analysis for discrete 2D plot datasets.

mod custom;
mod lm;
mod nonlinear;
mod poly;
mod seeds;
#[cfg(test)]
mod tests;

pub use custom::{
    CustomEvaluatorHandle, CustomFitCallback, CustomFitEvaluator, fit_custom_evaluator,
};

thread_local! {
    static FIT_MODEL_WITH_AXES_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Returns the total number of [`fit_model_with_axes`] solver executions on the current thread for regression diagnostics.
#[doc(hidden)]
#[must_use]
pub fn fit_model_with_axes_call_count() -> usize {
    FIT_MODEL_WITH_AXES_CALL_COUNT.with(std::cell::Cell::get)
}

/// Resets the [`fit_model_with_axes`] solver invocation counter on the current thread to zero.
#[doc(hidden)]
pub fn reset_fit_model_with_axes_call_count() {
    FIT_MODEL_WITH_AXES_CALL_COUNT.with(|c| c.set(0));
}

/// Maximum Levenberg-Marquardt iterations per nonlinear fit.
pub const MAX_LM_ITERATIONS: usize = 50;

/// Maximum number of free parameters supported by Levenberg-Marquardt nonlinear regression.
pub const MAX_FIT_PARAMS: usize = 6;

/// Supported curve fitting models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitType {
    /// Linear regression ($y = a x + b$)
    Linear,
    /// Polynomial regression of specified degree ($1 \le \text{degree} \le 4$)
    Polynomial(usize),
    /// Exponential model ($y = a \exp(b x)$)
    Exponential,
    /// Power-law model ($y = a x^b$)
    PowerLaw,
    /// Natural logarithmic model ($y = a \ln(x) + b$)
    Logarithmic,
    /// Three-parameter logistic growth/decay model ($y = L / (1 + a \exp(-k x))$)
    Logistic,
    /// Four-parameter sinusoidal model ($y = a \sin(b x + c) + d$)
    Sinusoidal,
    /// Callback-compiled custom formula ($1 \le k \le 6$ parameters)
    Custom,
}

impl FitType {
    /// Preset list exposed in interactive `Fit Curve` menus.
    pub const MENU_PRESETS: [Self; 9] = [
        Self::Linear,
        Self::Polynomial(2),
        Self::Polynomial(3),
        Self::Polynomial(4),
        Self::Exponential,
        Self::PowerLaw,
        Self::Logarithmic,
        Self::Logistic,
        Self::Sinusoidal,
    ];

    /// User-friendly menu label for the fit type.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear (Degree 1)",
            Self::Polynomial(1) => "Polynomial (Degree 1)",
            Self::Polynomial(2) => "Polynomial (Degree 2)",
            Self::Polynomial(3) => "Polynomial (Degree 3)",
            Self::Polynomial(4) => "Polynomial (Degree 4)",
            Self::Polynomial(_) => "Polynomial",
            Self::Exponential => "Exponential (a·exp(bx))",
            Self::PowerLaw => "Power Law (a·x^b)",
            Self::Logarithmic => "Logarithmic (a·ln(x)+b)",
            Self::Logistic => "Logistic (L/(1+a·exp(-kx)))",
            Self::Sinusoidal => "Sinusoidal (a·sin(bx+c)+d)",
            Self::Custom => "Custom Formula f(x)",
        }
    }

    /// Concise model name used in plot series legends and badges.
    #[must_use]
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Polynomial(2) => "Quad",
            Self::Polynomial(3) => "Cubic",
            Self::Polynomial(4) => "Quartic",
            Self::Polynomial(_) => "Poly",
            Self::Exponential => "Exp",
            Self::PowerLaw => "Power",
            Self::Logarithmic => "Log",
            Self::Logistic => "Logistic",
            Self::Sinusoidal => "Sinusoidal",
            Self::Custom => "Custom",
        }
    }

    /// Returns `true` when transformed linearized fitting (Desmos Log Mode) applies.
    #[must_use]
    pub const fn supports_log_mode(self) -> bool {
        matches!(self, Self::Exponential | Self::PowerLaw)
    }
}

/// Result of a fitted curve regression model.
#[derive(Debug, Clone, PartialEq)]
pub struct FitResult {
    /// Selected regression model family.
    pub fit_type: FitType,
    /// Fitted model parameters (polynomial `[c0..cd]` or nonlinear model parameters).
    pub coefficients: Vec<f64>,
    /// Coefficient of determination ($R^2 \in [0, 1]$).
    pub r_squared: f64,
    /// Root-mean-square vertical error in data coordinates.
    pub rmse: f64,
    /// Vertical residuals `[x_i, y_i - y_hat(x_i)]` for valid fitted points.
    pub residuals: Vec<[f64; 2]>,
    /// Polynomial degree (`0` for nonlinear models).
    pub degree: usize,
    /// Mathematical equation string representation.
    pub formula: String,
    /// Name of the fitted dataset.
    pub dataset_name: String,
    /// Whether the independent variable was transformed via $\log_{10}(x)$.
    pub log_x: bool,
    /// Whether the dependent variable was transformed via $\log_{10}(y)$.
    pub log_y: bool,
    /// Whether transformed linearized fitting (Desmos Log Mode) was used.
    pub log_mode: bool,
    /// Optional compiled callback evaluator when `fit_type == FitType::Custom`.
    pub custom_evaluator: Option<CustomEvaluatorHandle>,
    pub(crate) poly_centered: Option<(f64, f64, Vec<f64>)>,
}

impl FitResult {
    /// Evaluates the fitted function at data-space coordinate `x`.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        match self.fit_type {
            FitType::Linear | FitType::Polynomial(_) => {
                if let Some((u_mean, u_scale, ref centered)) = self.poly_centered {
                    poly::eval_centered_polynomial(
                        u_mean, u_scale, centered, x, self.log_x, self.log_y,
                    )
                } else {
                    poly::eval_scaled_polynomial(&self.coefficients, x, self.log_x, self.log_y)
                }
            }
            FitType::Exponential
            | FitType::PowerLaw
            | FitType::Logarithmic
            | FitType::Logistic
            | FitType::Sinusoidal => {
                nonlinear::eval_nonlinear_model(self.fit_type, &self.coefficients, x)
            }
            FitType::Custom => self
                .custom_evaluator
                .as_ref()
                .and_then(|h| h.0.eval(x, &self.coefficients))
                .unwrap_or(f64::NAN),
        }
    }
}

/// Fits a preset [`FitType`] respecting axis log flags for polynomial models and `log_mode` for nonlinear models.
#[must_use]
pub fn fit_model_with_axes(
    points: &[[f64; 2]],
    fit_type: FitType,
    dataset_name: &str,
    log_x: bool,
    log_y: bool,
    log_mode: bool,
) -> Option<FitResult> {
    FIT_MODEL_WITH_AXES_CALL_COUNT.with(|c| c.set(c.get() + 1));
    match fit_type {
        FitType::Linear => poly::fit_polynomial_scaled(points, 1, dataset_name, log_x, log_y),
        FitType::Polynomial(deg) => {
            poly::fit_polynomial_scaled(points, deg.clamp(1, 4), dataset_name, log_x, log_y)
        }
        FitType::Custom => None,
        _ => nonlinear::fit_nonlinear(points, fit_type, dataset_name, log_mode),
    }
}
