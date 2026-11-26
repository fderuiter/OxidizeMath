//! Interactive ADME multi-dose and missed-dose pharmacokinetic regimen planner.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod export;
mod plot;
pub mod presets;
pub mod state;
mod ui;

#[cfg(test)]
mod tests;

use std::sync::mpsc::Receiver;

use eframe::egui;
use egui_plot::ExportPrimitive;
use math_explorer::applied::pharmacokinetics::{
    summarize_exposure, BatemanModel, EnantiomerModel, ExposureSummary, PKParameters,
    PharmacokineticModel, RegimenSuperposition, SuperpositionModel, TherapeuticWindow,
};

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use state::{PkModelKind, RegimenState};

/// Default uniform grid resolution (`961` points -> `dt = 0.05 h` over `48 h`).
pub const DEFAULT_SAMPLE_POINTS: usize = 961;

/// Preallocated plasma concentration buffers and summary exposure metrics.
#[derive(Clone, Debug, Default)]
pub struct ComputedCurves {
    /// Sampled time grid in hours.
    pub times: Vec<f64>,
    /// Total plasma concentration profile (mg/L).
    pub total: Vec<f64>,
    /// Uniform maintenance baseline profile (mg/L) via `SuperpositionModel`.
    pub baseline: Vec<f64>,
    /// Chiral `d`-enantiomer plasma profile (mg/L) when in `ChiralEnantiomer` mode.
    pub d_isomer: Vec<f64>,
    /// Chiral `l`-enantiomer plasma profile (mg/L) when in `ChiralEnantiomer` mode.
    pub l_isomer: Vec<f64>,
    /// Latest computed exposure and therapeutic-window summary.
    pub summary: Option<ExposureSummary>,
    /// Validation or calculation error message if invalid parameters were supplied.
    pub error: Option<String>,
}

/// Interactive ADME multi-dose regimen, loading-dose, and missed-dose recovery planner.
pub struct PharmacokineticRegimenTool {
    /// Serializable regimen configuration and schedule.
    pub state: RegimenState,
    /// Cached curves and exposure metrics evaluated into preallocated buffers.
    pub curves: ComputedCurves,
    pub(crate) theme_mode: crate::app::ThemeMode,
    pub(crate) calibration_status: Option<String>,
    pub(crate) io_rx: Option<Receiver<Result<String, String>>>,
    pub(crate) io_status: Option<String>,
}

impl Default for PharmacokineticRegimenTool {
    fn default() -> Self {
        Self::with_state(RegimenState::default())
    }
}

impl PharmacokineticRegimenTool {
    /// Constructs a [`PharmacokineticRegimenTool`] initialized with `state`.
    pub fn with_state(state: RegimenState) -> Self {
        let mut tool = Self {
            state,
            curves: ComputedCurves {
                times: vec![0.0; DEFAULT_SAMPLE_POINTS],
                total: vec![0.0; DEFAULT_SAMPLE_POINTS],
                baseline: vec![0.0; DEFAULT_SAMPLE_POINTS],
                d_isomer: vec![0.0; DEFAULT_SAMPLE_POINTS],
                l_isomer: vec![0.0; DEFAULT_SAMPLE_POINTS],
                summary: None,
                error: None,
            },
            theme_mode: crate::app::ThemeMode::Dark,
            calibration_status: None,
            io_rx: None,
            io_status: None,
        };
        tool.recalculate();
        tool
    }
    /// Recomputes sampled concentration curves and exposure metrics into preallocated buffers.
    pub fn recalculate(&mut self) {
        self.curves.error = None;
        let n = DEFAULT_SAMPLE_POINTS;
        self.curves.times.resize(n, 0.0);
        self.curves.total.resize(n, 0.0);
        self.curves.baseline.resize(n, 0.0);

        let dt = self.state.horizon_h / (n - 1) as f64;
        for (i, slot) in self.curves.times.iter_mut().enumerate() {
            *slot = i as f64 * dt;
        }

        let events = match self.state.to_domain_events() {
            Ok(ev) => ev,
            Err(err) => {
                self.curves.summary = None;
                self.curves.error = Some(err);
                return;
            }
        };

        if let Err(err) = self.evaluate_active_model(&events) {
            self.curves.summary = None;
            self.curves.error = Some(err);
            return;
        }

        self.evaluate_uniform_baseline();
        match TherapeuticWindow::new(self.state.mec, self.state.mtc).and_then(|win| {
            summarize_exposure(
                &self.curves.times,
                &self.curves.total,
                win,
                self.state.tau_h,
                self.state.ke,
            )
        }) {
            Ok(summary) => self.curves.summary = Some(summary),
            Err(err) => {
                self.curves.summary = None;
                self.curves.error = Some(err.to_string());
            }
        }
    }

    fn evaluate_active_model(
        &mut self,
        events: &[math_explorer::applied::pharmacokinetics::DoseEvent],
    ) -> Result<(), String> {
        let base_params = PKParameters::new(
            self.state.f,
            self.state.maintenance_dose_mg,
            self.state.ka,
            self.state.ke,
            self.state.v,
        )
        .map_err(|e| e.to_string())?;

        match self.state.model {
            PkModelKind::ImmediateRelease | PkModelKind::BiphasicXr => {
                self.curves.d_isomer.clear();
                self.curves.l_isomer.clear();
                let regimen = RegimenSuperposition::new(BatemanModel::new(base_params), events)
                    .map_err(|e| e.to_string())?;
                regimen
                    .sample_into(&self.curves.times, &mut self.curves.total)
                    .map_err(|e| e.to_string())
            }
            PkModelKind::ChiralEnantiomer => {
                let l_params = PKParameters::new(
                    self.state.f,
                    self.state.maintenance_dose_mg,
                    self.state.ka,
                    self.state.chiral_l_ke,
                    self.state.v,
                )
                .map_err(|e| e.to_string())?;
                let f_d = self.state.chiral_f_d;
                let chiral = EnantiomerModel::new(base_params, l_params, f_d, (1.0 - f_d).max(0.0))
                    .map_err(|e| e.to_string())?;
                let regimen =
                    RegimenSuperposition::new(chiral, events).map_err(|e| e.to_string())?;
                let n = self.curves.times.len();
                self.curves.d_isomer.resize(n, 0.0);
                self.curves.l_isomer.resize(n, 0.0);
                regimen
                    .sample_isomers_into(
                        &self.curves.times,
                        &mut self.curves.d_isomer,
                        &mut self.curves.l_isomer,
                        &mut self.curves.total,
                    )
                    .map_err(|e| e.to_string())
            }
        }
    }

    fn evaluate_uniform_baseline(&mut self) {
        let Ok(params) = PKParameters::new(
            self.state.f,
            self.state.maintenance_dose_mg,
            self.state.ka,
            self.state.ke,
            self.state.v,
        ) else {
            return;
        };
        let mut dose_times = Vec::new();
        let mut t = 0.0;
        while t < self.state.horizon_h - 1e-9 && dose_times.len() < 64 {
            dose_times.push(t);
            t += self.state.tau_h;
        }
        let uniform = SuperpositionModel::new(BatemanModel::new(params), dose_times);
        for (t_val, out) in self
            .curves
            .times
            .iter()
            .copied()
            .zip(self.curves.baseline.iter_mut())
        {
            *out = uniform.concentration(t_val);
        }
    }

    pub(super) fn poll_io_status(&mut self) {
        let polled = self.io_rx.as_ref().and_then(|rx| rx.try_recv().ok());
        if let Some(res) = polled {
            self.io_status = Some(match res {
                Ok(msg) => msg,
                Err(err) => err,
            });
            self.io_rx = None;
        }
    }
}

impl InteractiveTool for PharmacokineticRegimenTool {
    fn name(&self) -> &'static str {
        "Pharmacokinetic Regimen"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn has_finite_parameters(&self) -> bool {
        self.state.is_finite()
    }

    fn clamp_parameters(&mut self) -> bool {
        let clamped = self.state.clamp();
        self.recalculate();
        clamped
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.state).ok()
    }

    fn try_load_state(&mut self, raw: &str) -> Result<(), String> {
        self.state_bounds().validate_json(raw)?;
        let loaded: RegimenState = serde_json::from_str(raw)
            .map_err(|e| format!("Invalid PharmacokineticRegimen state: {e}"))?;
        loaded.validate()?;
        self.state = loaded;
        self.recalculate();
        Ok(())
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        export::build_export_datasets(&self.state, &self.curves)
    }

    fn export_plot_primitives(&self) -> Vec<ExportPrimitive> {
        export::build_export_primitives(&self.state, &self.curves, self.theme_mode)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_inside(ui);
    }
}

// [cite:pharmacokinetics]
