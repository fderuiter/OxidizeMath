//! Interactive biochemical enzyme kinetics explorer (`MichaelisMenten` and `HillKinetics`)
//! with empirical CSV rate calibration, AIC model comparison, and residual CSV export.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::InteractiveTool;
use crate::framework::empirical_csv::{CalibrationReport, EmpiricalCsvState, EmpiricalCsvTable};
use crate::framework::recovery::FiniteStateGuard;
use crate::web_io::PlatformInstant;
use eframe::egui;
use math_explorer::biology::kinetics::{HillKinetics, KineticsModel, MichaelisMenten};
use math_explorer::pure_math::analysis::optimization::{
    CalibrationSummary, ClosureLeastSquaresObjective, LevenbergMarquardtSolver,
    LevenbergMarquardtWorkspace, ParameterBounds,
};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[path = "kinetics_ui.rs"]
mod ui;

pub(super) const MIN_VMAX: f64 = 0.05;
pub(super) const MAX_VMAX: f64 = 10_000.0;
pub(super) const MIN_KM: f64 = 0.01;
pub(super) const MAX_KM: f64 = 5_000.0;
pub(super) const MIN_HILL_N: f64 = 0.2;
pub(super) const MAX_HILL_N: f64 = 8.0;
pub(super) const MIN_S_MAX: f64 = 1.0;
pub(super) const MAX_S_MAX: f64 = 10_000.0;

/// Selectable enzymatic rate law in [`EnzymeKineticsTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnzymeKineticsVariant {
    /// Classical hyperbolic Michaelis-Menten rate law $v = V_{\max}\[S\] / (K_m + \[S\])$.
    #[default]
    MichaelisMenten,
    /// Cooperative Hill rate law $v = V_{\max}\[S\]^n / (K_m^n + \[S\]^n)$.
    Hill,
}

impl EnzymeKineticsVariant {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::MichaelisMenten => "Michaelis-Menten (n = 1)",
            Self::Hill => "Hill Cooperative Kinetics",
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EnzymeKineticsState {
    variant: EnzymeKineticsVariant,
    v_max: f64,
    k_m: f64,
    hill_n: f64,
    substrate_max: f64,
}

/// Interactive tool for simulating and calibrating [`MichaelisMenten`] and [`HillKinetics`].
pub struct EnzymeKineticsTool {
    /// Active kinetic rate law (`MichaelisMenten` or `Hill`).
    pub variant: EnzymeKineticsVariant,
    /// Maximum reaction velocity $V_{\max}$.
    pub v_max: f64,
    /// Substrate concentration at half-maximal velocity $K_m$.
    pub k_m: f64,
    /// Hill cooperativity exponent $n$.
    pub hill_n: f64,
    /// Upper substrate axis range $\[S\]_{\max}$.
    pub substrate_max: f64,
    /// Empirical CSV table, column mapping, diagnostics, and residual export state.
    pub(crate) empirical_csv: EmpiricalCsvState,
    pub(super) sensitivity_engine: crate::sensitivity::SensitivityEngine,
    pub(super) revision: u64,
}

impl Default for EnzymeKineticsTool {
    fn default() -> Self {
        Self {
            variant: EnzymeKineticsVariant::MichaelisMenten,
            v_max: 100.0,
            k_m: 25.0,
            hill_n: 2.0,
            substrate_max: 200.0,
            empirical_csv: EmpiricalCsvState::default(),
            sensitivity_engine: crate::sensitivity::SensitivityEngine::new(),
            revision: 1,
        }
    }
}

impl FiniteStateGuard for EnzymeKineticsTool {
    fn is_finite_state(&self) -> bool {
        [self.v_max, self.k_m, self.hill_n, self.substrate_max]
            .into_iter()
            .all(f64::is_finite)
            && self.v_max > 0.0
            && self.k_m > 0.0
            && self.hill_n > 0.0
            && self.substrate_max > 0.0
    }

    fn clamp_parameters(&mut self) {
        self.v_max = if self.v_max.is_finite() {
            self.v_max.clamp(MIN_VMAX, MAX_VMAX)
        } else {
            100.0
        };
        self.k_m = if self.k_m.is_finite() {
            self.k_m.clamp(MIN_KM, MAX_KM)
        } else {
            25.0
        };
        self.hill_n = if self.hill_n.is_finite() {
            self.hill_n.clamp(MIN_HILL_N, MAX_HILL_N)
        } else {
            2.0
        };
        self.substrate_max = if self.substrate_max.is_finite() {
            self.substrate_max.clamp(MIN_S_MAX, MAX_S_MAX)
        } else {
            200.0
        };
    }

    fn reset_parameters(&mut self) {
        self.v_max = 100.0;
        self.k_m = 25.0;
        self.hill_n = 2.0;
        self.substrate_max = 200.0;
    }
}

impl EnzymeKineticsTool {
    /// Evaluates reaction velocity $v(\[S\])$ under the currently selected kinetic model.
    pub fn evaluate_velocity(&self, substrate: f64) -> f64 {
        let s = substrate.max(0.0);
        match self.variant {
            EnzymeKineticsVariant::MichaelisMenten => MichaelisMenten::new(self.v_max, self.k_m)
                .and_then(|m| m.reaction_velocity(s))
                .unwrap_or(0.0),
            EnzymeKineticsVariant::Hill => HillKinetics::new(self.v_max, self.k_m, self.hill_n)
                .and_then(|m| m.reaction_velocity(s))
                .unwrap_or(0.0),
        }
    }

    /// Fits $V_{\max}$, $K_m$, and (for Hill kinetics) $n$ to the loaded empirical CSV dataset.
    pub fn calibrate_from_loaded_csv(&mut self, ctx: Option<&egui::Context>) -> Result<(), String> {
        let Some(ref table) = self.empirical_csv.table else {
            return Err("Load an enzyme kinetics CSV dataset before fitting parameters".into());
        };
        let (substrates, velocities) =
            table.extract_xy_sorted(self.empirical_csv.x_col, self.empirical_csv.y_col)?;
        if substrates.iter().any(|&s| s < 0.0) || velocities.iter().any(|&v| v < 0.0) {
            let err = "Substrate concentrations [S] and reaction rates v must be non-negative"
                .to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        if self.variant == EnzymeKineticsVariant::Hill && substrates.len() < 3 {
            let err =
                "Hill kinetics requires at least 3 data rows to fit Vmax, Km, and n".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        let start = PlatformInstant::now();
        let outcome = match self.variant {
            EnzymeKineticsVariant::MichaelisMenten => {
                self.fit_michaelis_menten(&substrates, &velocities)
            }
            EnzymeKineticsVariant::Hill => self.fit_hill_kinetics(&substrates, &velocities),
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match outcome {
            Ok((names, summary)) => {
                self.commit_kinetics_fit(ctx, &names, &summary, &substrates, &velocities, elapsed_ms)
            }
            Err(err) => {
                self.empirical_csv.record_error(ctx, err.clone());
                Err(err)
            }
        }
    }

    fn fit_michaelis_menten(
        &self,
        substrates: &[f64],
        velocities: &[f64],
    ) -> Result<(Vec<&'static str>, CalibrationSummary), String> {
        let mut obj = ClosureLeastSquaresObjective::new(substrates, velocities, |s, p| {
            MichaelisMenten::new(p[0], p[1])
                .and_then(|m| m.reaction_velocity(s))
                .unwrap_or(f64::NAN)
        })
        .map_err(|e| e.to_string())?;
        let v_peak = velocities.iter().copied().fold(1.0_f64, f64::max);
        let s_mid = substrates[substrates.len() / 2].max(0.1);
        let bounds = [
            ParameterBounds::new(MIN_VMAX, MAX_VMAX).unwrap(),
            ParameterBounds::new(MIN_KM, MAX_KM).unwrap(),
        ];
        let init = [
            (v_peak * 1.2).clamp(MIN_VMAX, MAX_VMAX),
            s_mid.clamp(MIN_KM, MAX_KM),
        ];
        let mut ws =
            LevenbergMarquardtWorkspace::new(substrates.len(), 2).map_err(|e| e.to_string())?;
        let summary = LevenbergMarquardtSolver::default()
            .fit(&mut obj, &init, &bounds, &mut ws)
            .map_err(|e| e.to_string())?;
        Ok((vec!["Vmax", "Km"], summary))
    }

    fn fit_hill_kinetics(
        &self,
        substrates: &[f64],
        velocities: &[f64],
    ) -> Result<(Vec<&'static str>, CalibrationSummary), String> {
        let mut obj = ClosureLeastSquaresObjective::new(substrates, velocities, |s, p| {
            HillKinetics::new(p[0], p[1], p[2])
                .and_then(|m| m.reaction_velocity(s))
                .unwrap_or(f64::NAN)
        })
        .map_err(|e| e.to_string())?;
        let v_peak = velocities.iter().copied().fold(1.0_f64, f64::max);
        let s_mid = substrates[substrates.len() / 2].max(0.1);
        let bounds = [
            ParameterBounds::new(MIN_VMAX, MAX_VMAX).unwrap(),
            ParameterBounds::new(MIN_KM, MAX_KM).unwrap(),
            ParameterBounds::new(MIN_HILL_N, MAX_HILL_N).unwrap(),
        ];
        let init = [
            (v_peak * 1.2).clamp(MIN_VMAX, MAX_VMAX),
            s_mid.clamp(MIN_KM, MAX_KM),
            self.hill_n.clamp(MIN_HILL_N, MAX_HILL_N),
        ];
        let mut ws =
            LevenbergMarquardtWorkspace::new(substrates.len(), 3).map_err(|e| e.to_string())?;
        let summary = LevenbergMarquardtSolver::default()
            .fit(&mut obj, &init, &bounds, &mut ws)
            .map_err(|e| e.to_string())?;
        Ok((vec!["Vmax", "Km", "n"], summary))
    }

    fn commit_kinetics_fit(
        &mut self,
        ctx: Option<&egui::Context>,
        names: &[&str],
        summary: &CalibrationSummary,
        substrates: &[f64],
        velocities: &[f64],
        elapsed_ms: f64,
    ) -> Result<(), String> {
        let prev = (self.v_max, self.k_m, self.hill_n);
        self.v_max = summary.parameters[0];
        self.k_m = summary.parameters[1];
        if summary.parameters.len() >= 3 {
            self.hill_n = summary.parameters[2];
        }
        if !FiniteStateGuard::is_finite_state(self) {
            (self.v_max, self.k_m, self.hill_n) = prev;
            let err = "Rejected non-finite enzyme kinetics fit via FiniteStateGuard".to_string();
            self.empirical_csv.record_error(ctx, err.clone());
            return Err(err);
        }
        FiniteStateGuard::clamp_parameters(self);
        if let Some(&s_last) = substrates.last() {
            if s_last > self.substrate_max {
                self.substrate_max = (s_last * 1.1).clamp(MIN_S_MAX, MAX_S_MAX);
            }
        }
        self.revision = self.revision.wrapping_add(1).max(1);

        let mut report =
            CalibrationReport::from_summary(names, summary, substrates, velocities, elapsed_ms)?;
        let kcat_over_km = self.v_max / self.k_m.max(1e-12);
        report.derived_metrics.push((
            "Specificity Ratio (Vmax / Km)".to_string(),
            format!("{kcat_over_km:.4}"),
        ));
        let announce = format!(
            "Calibrated {}: Vmax {:.3}, Km {:.3}, AIC {:.2}",
            self.variant.label(),
            self.v_max,
            self.k_m,
            report.aic
        );
        self.empirical_csv.record_fit_success(ctx, report, announce);
        Ok(())
    }
}

impl InteractiveTool for EnzymeKineticsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Enzyme Kinetics"
    }
    fn domain(&self) -> &'static str {
        "biology"
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.revision.hash(&mut h);
        (self.variant as u8).hash(&mut h);
        self.v_max.to_bits().hash(&mut h);
        self.k_m.to_bits().hash(&mut h);
        self.hill_n.to_bits().hash(&mut h);
        self.substrate_max.to_bits().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn supports_empirical_csv(&self) -> bool {
        true
    }

    fn ingest_empirical_dataset(&mut self, dataset: EmpiricalCsvTable) -> Result<String, String> {
        self.revision = self.revision.wrapping_add(1).max(1);
        Ok(self.empirical_csv.set_table(dataset))
    }

    fn notify_empirical_csv_error(&mut self, error: String) {
        self.empirical_csv.record_error(None, error);
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut ds = egui_plot::get_last_plot_datasets();
        if ds.is_empty() {
            ds.push((self.variant.label().to_string(), self.sample_curve()));
        }
        ds
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&EnzymeKineticsState {
            variant: self.variant,
            v_max: self.v_max,
            k_m: self.k_m,
            hill_n: self.hill_n,
            substrate_max: self.substrate_max,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: EnzymeKineticsState =
            serde_json::from_str(state).map_err(|e| format!("Invalid enzyme kinetics state: {e}"))?;
        if ![s.v_max, s.k_m, s.hill_n, s.substrate_max]
            .into_iter()
            .all(f64::is_finite)
            || !(MIN_VMAX..=MAX_VMAX).contains(&s.v_max)
            || !(MIN_KM..=MAX_KM).contains(&s.k_m)
            || !(MIN_HILL_N..=MAX_HILL_N).contains(&s.hill_n)
            || !(MIN_S_MAX..=MAX_S_MAX).contains(&s.substrate_max)
        {
            return Err("Enzyme kinetics parameters out of bounds".into());
        }
        self.variant = s.variant;
        self.v_max = s.v_max;
        self.k_m = s.k_m;
        self.hill_n = s.hill_n;
        self.substrate_max = s.substrate_max;
        self.revision = self.revision.wrapping_add(1).max(1);
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        FiniteStateGuard::clamp_parameters(self);
        let panel_id = crate::framework::scoped_panel_id(ui, "enzyme_kinetics_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.render_sidebar(ui);
            });
        });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.render_plot(ui);
        });
    }
}
