//! Cooperative sensitivity hooks, CSV curve calibration, and `ScenarioKpi` builders for [`super::PharmacokineticsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    AbsorptionMode, PharmacokineticsTool, MAX_DOSE_MG, MAX_KA, MAX_KE, MAX_VOLUME_L, MIN_RATE,
    MIN_VOLUME_L,
};
use crate::framework::{
    extract_calibration_series, fit_bounded_parameters, CalibratableParamSpec, ScenarioKpi,
    SensitivityParamSpec,
};
use math_explorer::applied::pharmacokinetics::{
    t_max, BatemanModel, PKParameters, PharmacokineticModel, SuperpositionModel,
};

impl PharmacokineticsTool {
    /// Builds scalar `ScenarioKpi` summary entries for `TraceSnapshot` baseline comparison & CSV export.
    #[must_use]
    pub fn scenario_kpis(&self) -> Vec<ScenarioKpi> {
        let m = &self.metrics;
        vec![
            ScenarioKpi::new("Peak C_max", m.c_max, "mg/L"),
            ScenarioKpi::new("Trough C_min", m.c_min, "mg/L"),
            ScenarioKpi::new("Trapezoidal AUC", m.auc_trapezoidal, "mg·h/L"),
            ScenarioKpi::new("Therapeutic Window", m.therapeutic_window_pct, "%"),
        ]
    }

    pub(super) fn pk_sensitivity_parameters(&self) -> Vec<SensitivityParamSpec> {
        vec![
            SensitivityParamSpec::new(
                "dose_mg",
                "Maintenance Dose D (mg)",
                self.dose_mg.max(1.0),
                5.0,
                MAX_DOSE_MG,
            ),
            SensitivityParamSpec::new(
                "ka",
                "Absorption Rate k_a (1/h)",
                self.effective_ka.clamp(MIN_RATE, MAX_KA),
                MIN_RATE,
                MAX_KA,
            ),
            SensitivityParamSpec::new(
                "ke",
                "Elimination Rate k_e (1/h)",
                self.ke.clamp(MIN_RATE, MAX_KE),
                MIN_RATE,
                MAX_KE,
            ),
            SensitivityParamSpec::new(
                "volume_l",
                "Volume V_d (L)",
                self.volume_l.clamp(MIN_VOLUME_L, MAX_VOLUME_L),
                MIN_VOLUME_L,
                MAX_VOLUME_L,
            ),
        ]
    }

    pub(super) fn eval_pk_sensitivity_observable(
        &self,
        params: &[(&str, f64)],
    ) -> Option<(&'static str, f64)> {
        let mut dose = self.dose_mg;
        let mut ka = self.effective_ka;
        let mut ke = self.ke;
        let mut vol = self.volume_l;
        for &(k, v) in params {
            if !v.is_finite() {
                return None;
            }
            match k {
                "dose_mg" => dose = v.clamp(0.0, MAX_DOSE_MG),
                "ka" => ka = v.clamp(MIN_RATE, MAX_KA),
                "ke" => ke = v.clamp(MIN_RATE, MAX_KE),
                "volume_l" => vol = v.clamp(MIN_VOLUME_L, MAX_VOLUME_L),
                _ => {}
            }
        }
        let pk = PKParameters::new(self.bioavailability.clamp(0.01, 1.0), dose, ka, ke, vol).ok()?;
        let (maint_times, has_loading, _) = self.build_schedule_times();
        let last_dose_t = maint_times.last().copied().unwrap_or(0.0);
        let sup = SuperpositionModel::new(BatemanModel::new(pk), maint_times);
        let loading = if has_loading {
            Some(BatemanModel::new(pk.with_dose(self.loading_dose_mg).ok()?))
        } else {
            None
        };
        let tp = t_max(ka, ke).max(0.0);
        let mut c_peak = 0.0_f64;
        for cand_t in [tp, last_dose_t + tp] {
            let c = sup.concentration(cand_t)
                + loading.as_ref().map_or(0.0, |m| m.concentration(cand_t));
            if c.is_finite() {
                c_peak = c_peak.max(c);
            }
        }
        Some(("Peak Concentration C_max (mg/L)", c_peak))
    }

    pub(super) fn calibrate_pk_from_datasets(
        &mut self,
        datasets: &[egui_plot::ImportedDataset],
    ) -> Result<bool, String> {
        if self.active_challenge.is_some() {
            return Err("Cannot calibrate locked parameters during an active challenge".into());
        }
        let points = extract_calibration_series(datasets)?;
        let f_bio = self.bioavailability.clamp(0.01, 1.0);
        let dose = self.dose_mg.clamp(1.0, MAX_DOSE_MG);
        let (maint_times, has_loading, _) = self.build_schedule_times();
        let load_dose = self.loading_dose_mg;
        let specs = [
            CalibratableParamSpec::new("ka", self.effective_ka.clamp(0.05, 5.0), MIN_RATE, MAX_KA),
            CalibratableParamSpec::new("ke", self.ke.clamp(0.02, 1.0), MIN_RATE, MAX_KE),
            CalibratableParamSpec::new(
                "volume_l",
                self.volume_l.clamp(MIN_VOLUME_L, MAX_VOLUME_L),
                MIN_VOLUME_L,
                MAX_VOLUME_L,
            ),
        ];
        let fitted = fit_bounded_parameters(points, &specs, |t_h, theta| {
            let pk = PKParameters::new(f_bio, dose, theta[0], theta[1], theta[2]).ok()?;
            let sup = SuperpositionModel::new(BatemanModel::new(pk), maint_times.clone());
            let mut conc = sup.concentration(t_h);
            if has_loading && t_h >= 0.0 {
                let load_m = BatemanModel::new(pk.with_dose(load_dose).ok()?);
                conc += load_m.concentration(t_h);
            }
            conc.is_finite().then_some(conc)
        })?;
        self.absorption_mode = AbsorptionMode::DirectKa;
        self.ka = fitted[0].clamp(MIN_RATE, MAX_KA);
        self.ke = fitted[1].clamp(MIN_RATE, MAX_KE);
        self.volume_l = fitted[2].clamp(MIN_VOLUME_L, MAX_VOLUME_L);
        self.recalculate();
        Ok(true)
    }
}
