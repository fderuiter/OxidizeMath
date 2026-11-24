#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Bounded CSV parameter calibration via numeric Levenberg-Marquardt and `FiniteStateGuard` rollback.

use math_commons::fit::{levenberg_marquardt_numeric, sum_squared_errors_numeric, MAX_FIT_PARAMS};

use super::{
    inspect_shared_datasets_for_divergence, persistence, ExportStatusSeverity, SimulationFramework,
};

/// Bounded scalar parameter specification for nonlinear CSV curve calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratableParamSpec {
    /// Canonical parameter name.
    pub name: &'static str,
    /// Initial parameter guess prior to optimization.
    pub initial: f64,
    /// Lower physical parameter bound.
    pub min: f64,
    /// Upper physical parameter bound.
    pub max: f64,
}

impl CalibratableParamSpec {
    /// Constructs a bounded parameter specification.
    #[must_use]
    pub const fn new(name: &'static str, initial: f64, min: f64, max: f64) -> Self {
        Self {
            name,
            initial,
            min,
            max,
        }
    }
}

#[inline]
fn map_logistic_to_bounds(u: f64, min: f64, max: f64) -> f64 {
    let span = (max - min).max(0.0);
    if span <= f64::EPSILON {
        return min;
    }
    let clamped_u = u.clamp(-40.0, 40.0);
    let s = 1.0 / (1.0 + (-clamped_u).exp());
    (min + span * s).clamp(min, max)
}

#[inline]
fn unmap_bounds_to_logistic(val: f64, min: f64, max: f64) -> f64 {
    let span = (max - min).max(0.0);
    if span <= f64::EPSILON {
        return 0.0;
    }
    let frac = ((val - min) / span).clamp(1.0e-4, 1.0 - 1.0e-4);
    (frac / (1.0 - frac)).ln()
}

/// Extracts the most recently imported non-empty `[x, y]` CSV series after finiteness validation.
pub fn extract_calibration_series(
    datasets: &[egui_plot::ImportedDataset],
) -> Result<&[[f64; 2]], String> {
    let ds = datasets
        .iter()
        .rev()
        .find(|d| !d.points.is_empty())
        .ok_or_else(|| "No non-empty imported CSV dataset available".to_owned())?;
    if ds
        .points
        .iter()
        .any(|pt| !pt[0].is_finite() || !pt[1].is_finite())
    {
        return Err(format!(
            "Imported dataset '{}' contains non-finite coordinates",
            ds.name
        ));
    }
    Ok(&ds.points)
}

fn validate_calibration_inputs(
    points: &[[f64; 2]],
    specs: &[CalibratableParamSpec],
) -> Result<usize, String> {
    let p = specs.len();
    if p == 0 || p > MAX_FIT_PARAMS {
        return Err(format!(
            "Parameter count {p} outside supported [1, {MAX_FIT_PARAMS}] range"
        ));
    }
    if points.len() < p.max(2) {
        return Err(format!(
            "Need at least {} data points to calibrate {p} parameter(s)",
            p.max(2)
        ));
    }
    if points
        .iter()
        .any(|pt| !pt[0].is_finite() || !pt[1].is_finite())
    {
        return Err("Calibration dataset contains non-finite coordinates".to_owned());
    }
    for spec in specs {
        if !spec.initial.is_finite()
            || !spec.min.is_finite()
            || !spec.max.is_finite()
            || spec.max < spec.min
        {
            return Err(format!("Invalid bounds for parameter '{}'", spec.name));
        }
    }
    Ok(p)
}

/// Fits up to `MAX_FIT_PARAMS` (`6`) bounded tool parameters against `[x, y]` observations
/// using [`levenberg_marquardt_numeric`] with logistic box constraints.
pub fn fit_bounded_parameters<F>(
    points: &[[f64; 2]],
    specs: &[CalibratableParamSpec],
    eval_model: F,
) -> Result<Vec<f64>, String>
where
    F: Fn(f64, &[f64]) -> Option<f64>,
{
    let p = validate_calibration_inputs(points, specs)?;
    let mut u0 = [0.0_f64; MAX_FIT_PARAMS];
    let mut theta0 = [0.0_f64; MAX_FIT_PARAMS];
    for (j, spec) in specs.iter().enumerate() {
        let clamped_init = spec.initial.clamp(spec.min, spec.max);
        theta0[j] = clamped_init;
        u0[j] = unmap_bounds_to_logistic(clamped_init, spec.min, spec.max);
    }

    let sse_before = sum_squared_errors_numeric(points, &theta0[..p], |x, th| eval_model(x, th))
        .ok_or_else(|| "Initial parameter guess produced non-finite model output".to_owned())?;

    let u_fit = levenberg_marquardt_numeric(points, &u0[..p], |x, u_slice| {
        let mut bounded = [0.0_f64; MAX_FIT_PARAMS];
        for (j, spec) in specs.iter().enumerate() {
            if !u_slice[j].is_finite() {
                return None;
            }
            bounded[j] = map_logistic_to_bounds(u_slice[j], spec.min, spec.max);
        }
        eval_model(x, &bounded[..p]).filter(|y| y.is_finite())
    })
    .map_err(|e| format!("Levenberg-Marquardt calibration failed: {e}"))?;

    let mut fitted = Vec::with_capacity(p);
    for (j, spec) in specs.iter().enumerate() {
        let val = map_logistic_to_bounds(u_fit[j], spec.min, spec.max);
        if !val.is_finite() {
            return Err(format!(
                "Fitted parameter '{}' became non-finite",
                spec.name
            ));
        }
        fitted.push(val);
    }

    let sse_after = sum_squared_errors_numeric(points, &fitted, |x, th| eval_model(x, th))
        .ok_or_else(|| "Fitted parameters produced non-finite model output".to_owned())?;
    if sse_after > sse_before + 1.0e-9 {
        return Ok(theta0[..p].to_vec());
    }
    Ok(fitted)
}

impl SimulationFramework {
    /// Calibrates the active tool against imported CSV datasets with automatic `FiniteStateGuard`
    /// snapshot rollback if calibration fails or yields non-finite parameters/trajectories.
    pub fn calibrate_active_tool_from_datasets(
        &mut self,
        datasets: &[egui_plot::ImportedDataset],
    ) -> Result<bool, String> {
        if self.is_diverged || datasets.is_empty() {
            return Ok(false);
        }
        let Some(tool) = &mut self.active_tool else {
            return Ok(false);
        };
        let pre_recovery = tool.save_recovery_state();
        let outcome = tool.calibrate_from_datasets(datasets);

        let params_finite = tool.has_finite_parameters();
        let refreshed = persistence::capture_isolated_tool_datasets(tool.as_ref());
        let dataset_fault =
            inspect_shared_datasets_for_divergence(&refreshed, &self.current_valid_plot_datasets)
                .err();

        let rollback_err = match outcome {
            Err(err) => Some(err),
            Ok(_) if !params_finite => {
                Some("Calibrated parameters failed finiteness check".to_owned())
            }
            Ok(_) if dataset_fault.is_some() => Some(
                dataset_fault
                    .map(|r| r.message)
                    .unwrap_or_else(|| "Non-finite trajectory after calibration".to_owned()),
            ),
            Ok(applied) => {
                if applied {
                    self.current_valid_plot_datasets = refreshed;
                    self.export_status_message =
                        Some("Calibrated parameters from imported CSV".to_owned());
                    self.export_status_severity = ExportStatusSeverity::Success;
                    self.bump_framework_revision();
                }
                return Ok(applied);
            }
        };

        let reason = rollback_err.unwrap_or_else(|| "Calibration rejected".to_owned());
        if let Some(ref snap) = pre_recovery {
            let _ = tool.try_load_recovery_state(snap);
        }
        if !tool.has_finite_parameters() {
            let _ = tool.clamp_parameters();
        }
        self.is_diverged = false;
        self.diagnostic_report = None;
        self.export_status_message = Some(format!("Calibration rolled back: {reason}"));
        self.export_status_severity = ExportStatusSeverity::Error;
        Err(reason)
    }
}
