//! Curve fitting and nonlinear regression analysis for discrete 2D plot datasets.

mod lm;
mod nonlinear;
mod poly;
mod seeds;
#[cfg(test)]
mod tests;

/// Maximum Levenberg-Marquardt iterations per nonlinear fit.
pub const MAX_LM_ITERATIONS: usize = 50;

/// Supported preset curve fitting models.
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
                poly::eval_scaled_polynomial(&self.coefficients, x, self.log_x, self.log_y)
            }
            FitType::Exponential
            | FitType::PowerLaw
            | FitType::Logarithmic
            | FitType::Logistic
            | FitType::Sinusoidal => {
                nonlinear::eval_nonlinear_model(self.fit_type, &self.coefficients, x)
            }
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
    match fit_type {
        FitType::Linear => poly::fit_polynomial_scaled(points, 1, dataset_name, log_x, log_y),
        FitType::Polynomial(deg) => {
            poly::fit_polynomial_scaled(points, deg.clamp(1, 4), dataset_name, log_x, log_y)
        }
        _ => nonlinear::fit_nonlinear(points, fit_type, dataset_name, log_mode),
    }
}
