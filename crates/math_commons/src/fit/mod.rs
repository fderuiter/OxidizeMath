//! Zero-allocation polynomial and nonlinear least-squares curve-fitting solvers (`fit`).
//!
//! Exposes canonical [`FitModel`] presets (degrees 1–4 polynomial OLS plus Exponential,
//! Power-Law, Logarithmic, Logistic, and Sinusoidal damped Levenberg-Marquardt regression)
//! backed by fixed stack buffers and typed [`FitError`] diagnostics.
// [cite:math_commons]

mod lm;
mod nonlinear;
mod poly;
mod seeds;

#[cfg(test)]
mod tests;

use crate::error::FitError;
use crate::series::PointSeries;

pub use lm::{
    MAX_FIT_PARAMS, MAX_LM_ITERATIONS, levenberg_marquardt, levenberg_marquardt_filtered,
    levenberg_marquardt_numeric, sum_squared_errors_numeric,
};
pub use nonlinear::{eval_nonlinear, fit_nonlinear, format_nonlinear_formula};
pub use poly::{
    eval_polynomial_horner, fit_polynomial, fit_polynomial_scaled, format_polynomial_formula,
    solve_dense_small,
};
pub use seeds::{
    WeightedOlsAccum, seed_exponential, seed_logarithmic, seed_logistic, seed_power_law,
    seed_sinusoidal, weighted_ols_line,
};

/// Maximum polynomial regression degree supported on stack buffers without heap allocation.
pub const MAX_POLYNOMIAL_DEGREE: usize = 4;

/// Canonical parametric regression model presets without degree clamping or hash collisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FitModel {
    /// Degree-1 polynomial regression ($y = c_0 + c_1 x$).
    Linear,
    /// Degree-2 polynomial regression ($y = c_0 + c_1 x + c_2 x^2$).
    Quadratic,
    /// Degree-3 polynomial regression ($y = c_0 + c_1 x + c_2 x^2 + c_3 x^3$).
    Cubic,
    /// Degree-4 polynomial regression ($y = c_0 + c_1 x + c_2 x^2 + c_3 x^3 + c_4 x^4$).
    Quartic,
    /// Exponential model ($y = a e^{b x}$).
    Exponential,
    /// Power-law model ($y = a x^b$, $x > 0$).
    PowerLaw,
    /// Logarithmic model ($y = a + b \ln(x)$, $x > 0$).
    Logarithmic,
    /// 3-parameter logistic Carrying-Capacity curve ($y = L / (1 + e^{-k(x - x_0)})$).
    Logistic,
    /// 4-parameter sinusoidal wave ($y = A \sin(B x + C) + D$).
    Sinusoidal,
}

impl FitModel {
    /// Ordered list of canonical regression models for UI menus and batch sweeps.
    pub const MENU_PRESETS: [Self; 9] = [
        Self::Linear,
        Self::Quadratic,
        Self::Cubic,
        Self::Quartic,
        Self::Exponential,
        Self::PowerLaw,
        Self::Logarithmic,
        Self::Logistic,
        Self::Sinusoidal,
    ];

    /// Maps a polynomial degree (`1..=4`) to its canonical [`FitModel`] variant.
    #[verified_engine::verified]
    pub fn from_degree(degree: usize) -> Result<Self, FitError> {
        match degree {
            1 => Ok(Self::Linear),
            2 => Ok(Self::Quadratic),
            3 => Ok(Self::Cubic),
            4 => Ok(Self::Quartic),
            _ => Err(FitError::UnsupportedDegree { degree }),
        }
    }

    /// Returns the polynomial degree (`1..=4`) if this model is a polynomial preset.
    #[must_use]
    pub const fn degree(self) -> Option<usize> {
        match self {
            Self::Linear => Some(1),
            Self::Quadratic => Some(2),
            Self::Cubic => Some(3),
            Self::Quartic => Some(4),
            Self::Exponential
            | Self::PowerLaw
            | Self::Logarithmic
            | Self::Logistic
            | Self::Sinusoidal => None,
        }
    }

    /// Human-readable menu label describing the model equation.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear (deg 1)",
            Self::Quadratic => "Quadratic (deg 2)",
            Self::Cubic => "Cubic (deg 3)",
            Self::Quartic => "Quartic (deg 4)",
            Self::Exponential => "Exponential (a·exp(b·x))",
            Self::PowerLaw => "Power-Law (a·x^b)",
            Self::Logarithmic => "Logarithmic (a + b·ln(x))",
            Self::Logistic => "Logistic (L / (1 + exp(-k·(x - x₀))))",
            Self::Sinusoidal => "Sinusoidal (A·sin(B·x + C) + D)",
        }
    }

    /// Compact badge label for legends and plot callouts.
    #[must_use]
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Quadratic => "Quadratic",
            Self::Cubic => "Cubic",
            Self::Quartic => "Quartic",
            Self::Exponential => "Exp",
            Self::PowerLaw => "Power",
            Self::Logarithmic => "Log",
            Self::Logistic => "Logistic",
            Self::Sinusoidal => "Sin",
        }
    }

    /// Whether this preset supports linearized log-space fitting (`Exponential`, `PowerLaw`).
    #[must_use]
    pub const fn supports_log_mode(self) -> bool {
        matches!(self, Self::Exponential | Self::PowerLaw)
    }
}

impl std::fmt::Display for FitModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Stack-backed result of a polynomial or nonlinear least-squares fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveFit {
    /// Fitted canonical regression model.
    pub fit_model: FitModel,
    coeffs: [f64; 5],
    coeff_len: usize,
    /// Coefficient of determination ($R^2$).
    pub r_squared: f64,
    /// Root-mean-square error in fit space.
    pub rmse: f64,
    /// Polynomial degree (`1..=4`) for polynomial fits, or `0` for nonlinear models.
    pub degree: usize,
    /// Number of valid points used to estimate parameters.
    pub valid_points: usize,
    /// Whether horizontal coordinates were $\log_{10}$-transformed.
    pub log_x: bool,
    /// Whether vertical coordinates were $\log_{10}$-transformed.
    pub log_y: bool,
    /// Whether linearized log-space regression was used.
    pub log_mode: bool,
    poly_center_scale: Option<(f64, f64, [f64; 5])>,
}

impl CurveFit {
    /// Constructs a new stack-backed [`CurveFit`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn new(
        fit_model: FitModel,
        coeffs: [f64; 5],
        coeff_len: usize,
        r_squared: f64,
        rmse: f64,
        degree: usize,
        valid_points: usize,
        log_x: bool,
        log_y: bool,
        log_mode: bool,
    ) -> Self {
        Self {
            fit_model,
            coeffs,
            coeff_len,
            r_squared,
            rmse,
            degree,
            valid_points,
            log_x,
            log_y,
            log_mode,
            poly_center_scale: None,
        }
    }

    pub(super) fn with_centered_poly(
        mut self,
        u_mean: f64,
        u_scale: f64,
        centered: [f64; 5],
    ) -> Self {
        self.poly_center_scale = Some((u_mean, u_scale, centered));
        self
    }

    /// Returns the active fitted parameter slice (`c_0..=c_d` or nonlinear parameters).
    #[inline]
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs[..self.coeff_len]
    }

    /// Evaluates the fitted curve at original data coordinate `x` with zero heap allocations.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn eval(&self, x: f64) -> f64 {
        self.eval_raw(x)
    }

    #[inline]
    fn eval_raw(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return f64::NAN;
        }
        match self.fit_model {
            FitModel::Linear | FitModel::Quadratic | FitModel::Cubic | FitModel::Quartic => {
                if self.log_x && x <= 0.0 {
                    return f64::NAN;
                }
                let tx = if self.log_x { x.log10() } else { x };
                let ty = if let Some((u_mean, u_scale, ref centered)) = self.poly_center_scale {
                    let z = (tx - u_mean) / u_scale;
                    poly::eval_polynomial_horner_raw(&centered[..self.coeff_len], z)
                } else {
                    poly::eval_polynomial_horner_raw(&self.coeffs[..self.coeff_len], tx)
                };
                let out = if self.log_y {
                    10.0_f64.powf(ty.clamp(-300.0, 300.0)).clamp(1e-300, 1e300)
                } else {
                    ty
                };
                if out.is_finite() { out } else { f64::NAN }
            }
            FitModel::Exponential
            | FitModel::PowerLaw
            | FitModel::Logarithmic
            | FitModel::Logistic
            | FitModel::Sinusoidal => {
                nonlinear::eval_nonlinear_raw(self.fit_model, self.coefficients(), x)
            }
        }
    }

    /// Writes `[x, y - y_hat]` residuals in original data space into `out` with zero allocations.
    #[verified_engine::verified]
    pub fn residuals_into<P: PointSeries + ?Sized>(
        &self,
        points: &P,
        out: &mut [[f64; 2]],
    ) -> usize {
        self.residuals_into_raw(points, out)
    }

    fn residuals_into_raw<P: PointSeries + ?Sized>(
        &self,
        points: &P,
        out: &mut [[f64; 2]],
    ) -> usize {
        let mut written = 0usize;
        for i in 0..points.len() {
            if written == out.len() {
                break;
            }
            let p = points.point_at(i);
            if !p[0].is_finite() || !p[1].is_finite() {
                continue;
            }
            let y_hat = self.eval_raw(p[0]);
            if y_hat.is_finite() {
                out[written] = [p[0], p[1] - y_hat];
                written += 1;
            }
        }
        written
    }

    /// Convenience helper collecting `[x, y - y_hat]` residuals into a `Vec<[f64; 2]>`.
    #[must_use]
    pub fn residuals<P: PointSeries + ?Sized>(&self, points: &P) -> Vec<[f64; 2]> {
        let mut out = vec![[0.0_f64; 2]; points.len()];
        let count = self.residuals_into(points, &mut out);
        out.truncate(count);
        out
    }

    /// Formats a human-readable equation string for UI legends and export captions.
    #[must_use]
    pub fn format_formula(&self) -> String {
        match self.fit_model {
            FitModel::Linear | FitModel::Quadratic | FitModel::Cubic | FitModel::Quartic => {
                format_polynomial_formula(self.coefficients(), self.log_x, self.log_y)
            }
            FitModel::Exponential
            | FitModel::PowerLaw
            | FitModel::Logarithmic
            | FitModel::Logistic
            | FitModel::Sinusoidal => format_nonlinear_formula(self.fit_model, self.coefficients()),
        }
    }
}

/// Fits a canonical [`FitModel`] to `points` with zero heap allocations.
#[verified_engine::verified]
pub fn fit_model<P: PointSeries + ?Sized>(
    points: &P,
    model: FitModel,
    log_mode: bool,
) -> Result<CurveFit, FitError> {
    fit_model_with_axes(points, model, log_mode, false, false)
}

/// Fits a canonical [`FitModel`] respecting optional logarithmic plot axes for polynomials.
#[verified_engine::verified]
pub fn fit_model_with_axes<P: PointSeries + ?Sized>(
    points: &P,
    model: FitModel,
    log_mode: bool,
    plot_log_x: bool,
    plot_log_y: bool,
) -> Result<CurveFit, FitError> {
    if let Some(deg) = model.degree() {
        fit_polynomial_scaled(points, deg, plot_log_x, plot_log_y)
    } else {
        fit_nonlinear(points, model, log_mode)
    }
}
