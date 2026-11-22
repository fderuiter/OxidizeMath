//! Compact URL-bounded state and clinical scenario builders for [`super::PharmacokineticRegimenTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::applied::pharmacokinetics::{
    DoseEvent, MAX_REGIMEN_EVENTS, ReleaseFormulation, TherapeuticWindow,
};
use serde::{Deserialize, Serialize};

fn is_false(flag: &bool) -> bool {
    !*flag
}

/// Pharmacokinetic structural model family selected in the planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PkModelKind {
    /// One-compartment Bateman model with immediate-release events by default.
    ImmediateRelease,
    /// Biphasic two-pulse extended-release formulation.
    BiphasicXr,
    /// Chiral mixture (`d`- and `l`-enantiomers) with distinct elimination rates.
    ChiralEnantiomer,
}

impl PkModelKind {
    /// Human-readable label shown in the model selector.
    pub fn label(self) -> &'static str {
        match self {
            Self::ImmediateRelease => "1-Compartment IR (Bateman)",
            Self::BiphasicXr => "Biphasic Extended-Release (XR)",
            Self::ChiralEnantiomer => "Chiral Mixture (d / l Isomers)",
        }
    }
}

/// Compact per-event representation keeping 64-dose share links well below `4,096` JSON bytes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactDoseEvent {
    /// Administration time in hours ($\ge 0$).
    pub t: f64,
    /// Administered dose in milligrams ($\ge 0$).
    pub d: f64,
    /// When `true`, uses the extended-release biphasic formulation (`xr_lag_h`, `xr_f1`).
    #[serde(default, skip_serializing_if = "is_false")]
    pub xr: bool,
}

/// Serializable state for [`super::PharmacokineticRegimenTool`].
///
/// Places `maintenance_dose_mg` first so automated URL/preset perturbation contract tests scale a
/// wide-range positive parameter without violating `[0, 1]` bioavailability bounds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegimenState {
    /// Reference maintenance dose $D$ (mg) used for uniform baseline and scenario builders.
    pub maintenance_dose_mg: f64,
    /// Nominal dosing interval $\tau$ (hours).
    pub tau_h: f64,
    /// Total simulation horizon (hours).
    pub horizon_h: f64,
    /// Oral bioavailability fraction $F \in (0, 1]$.
    pub f: f64,
    /// First-order absorption rate constant $k_a$ ($\text{h}^{-1}$).
    pub ka: f64,
    /// First-order elimination rate constant $k_e$ ($\text{h}^{-1}$, or $d$-isomer $k_e$ in chiral mode).
    pub ke: f64,
    /// Apparent volume of distribution $V_d$ (liters).
    pub v: f64,
    /// Minimum Effective Concentration (MEC, mg/L).
    pub mec: f64,
    /// Minimum Toxic Concentration (MTC, mg/L).
    pub mtc: f64,
    /// Active pharmacokinetic model family.
    pub model: PkModelKind,
    /// Delayed-pulse lag time $t_{\text{lag}}$ (hours) for extended-release events.
    pub xr_lag_h: f64,
    /// Immediate pulse fraction $f_1 \in [0, 1]$ for extended-release events ($f_2 = 1 - f_1$).
    pub xr_f1: f64,
    /// Fraction $f_d \in [0, 1]$ of $d$-enantiomer in chiral mode ($f_l = 1 - f_d$).
    pub chiral_f_d: f64,
    /// Elimination rate constant $k_{e,l}$ ($\text{h}^{-1}$) of the $l$-enantiomer in chiral mode.
    pub chiral_l_ke: f64,
    /// Target $T_{\max}$ (hours) input for `solve_ka` calibration.
    pub cal_t_max_h: f64,
    /// Target elimination half-life $t_{1/2}$ (hours) input for `solve_ka` calibration.
    pub cal_half_life_h: f64,
    /// Whether to overlay the uniform maintenance `SuperpositionModel` baseline curve.
    pub show_baseline: bool,
    /// Scheduled dose events (`1..=MAX_REGIMEN_EVENTS`).
    pub events: Vec<CompactDoseEvent>,
}

impl Default for RegimenState {
    fn default() -> Self {
        Self::uniform_maintenance_scenario()
    }
}

impl RegimenState {
    /// Uniform `q12h` `100 mg` immediate-release maintenance schedule (`66.49%` TTR).
    pub fn uniform_maintenance_scenario() -> Self {
        Self {
            maintenance_dose_mg: 100.0,
            tau_h: 12.0,
            horizon_h: 48.0,
            f: 1.0,
            ka: 1.2,
            ke: 0.1,
            v: 20.0,
            mec: 3.0,
            mtc: 8.5,
            model: PkModelKind::ImmediateRelease,
            xr_lag_h: 4.0,
            xr_f1: 0.5,
            chiral_f_d: 0.75,
            chiral_l_ke: 0.055,
            cal_t_max_h: 2.26,
            cal_half_life_h: 6.93,
            show_baseline: true,
            events: vec![
                CompactDoseEvent { t: 0.0, d: 100.0, xr: false },
                CompactDoseEvent { t: 12.0, d: 100.0, xr: false },
                CompactDoseEvent { t: 24.0, d: 100.0, xr: false },
                CompactDoseEvent { t: 36.0, d: 100.0, xr: false },
            ],
        }
    }

    /// `2D` (`200 mg`) initial loading dose followed by `100 mg` `q12h` maintenance (`86.89%` TTR).
    pub fn loading_dose_scenario() -> Self {
        let mut s = Self::uniform_maintenance_scenario();
        s.events[0].d = 200.0;
        s
    }

    /// Missed `12 h` dose recovered via a `1.5D` (`150 mg`) catch-up dose at `18 h` (`0.00%` toxic).
    pub fn missed_dose_catchup_scenario() -> Self {
        let mut s = Self::uniform_maintenance_scenario();
        s.events = vec![
            CompactDoseEvent { t: 0.0, d: 100.0, xr: false },
            CompactDoseEvent { t: 18.0, d: 150.0, xr: false },
            CompactDoseEvent { t: 24.0, d: 100.0, xr: false },
            CompactDoseEvent { t: 36.0, d: 100.0, xr: false },
        ];
        s
    }

    /// Missed `12 h` dose followed by an unsafe `2D` (`200 mg`) double dose at `18 h` (`8.32%` toxic).
    pub fn missed_dose_double_scenario() -> Self {
        let mut s = Self::missed_dose_catchup_scenario();
        s.events[1].d = 200.0;
        s
    }

    /// Chiral `3:1` (`d:l`) biphasic extended-release morning regimen with afternoon IR booster.
    pub fn chiral_xr_scenario() -> Self {
        Self {
            maintenance_dose_mg: 20.0,
            tau_h: 24.0,
            horizon_h: 48.0,
            f: 1.0,
            ka: 1.1,
            ke: 0.0693,
            v: 25.0,
            mec: 0.35,
            mtc: 1.15,
            model: PkModelKind::ChiralEnantiomer,
            xr_lag_h: 4.0,
            xr_f1: 0.5,
            chiral_f_d: 0.75,
            chiral_l_ke: 0.0533,
            cal_t_max_h: 3.0,
            cal_half_life_h: 10.0,
            show_baseline: false,
            events: vec![
                CompactDoseEvent { t: 0.0, d: 25.0, xr: true },
                CompactDoseEvent { t: 10.0, d: 7.5, xr: false },
                CompactDoseEvent { t: 24.0, d: 20.0, xr: true },
                CompactDoseEvent { t: 34.0, d: 7.5, xr: false },
            ],
        }
    }

    /// Checks that all numeric fields are finite.
    pub fn is_finite(&self) -> bool {
        let scalars_ok = [
            self.maintenance_dose_mg,
            self.tau_h,
            self.horizon_h,
            self.f,
            self.ka,
            self.ke,
            self.v,
            self.mec,
            self.mtc,
            self.xr_lag_h,
            self.xr_f1,
            self.chiral_f_d,
            self.chiral_l_ke,
            self.cal_t_max_h,
            self.cal_half_life_h,
        ]
        .into_iter()
        .all(f64::is_finite);
        scalars_ok
            && self
                .events
                .iter()
                .all(|ev| ev.t.is_finite() && ev.d.is_finite())
    }

    /// Clamps interactive fields to safe physical bounds.
    pub fn clamp(&mut self) -> bool {
        self.maintenance_dose_mg = self.maintenance_dose_mg.clamp(1.0, 5000.0);
        self.tau_h = self.tau_h.clamp(1.0, 72.0);
        self.horizon_h = self.horizon_h.clamp(6.0, 168.0);
        self.f = self.f.clamp(0.05, 1.0);
        self.ka = self.ka.clamp(0.05, 10.0);
        self.ke = self.ke.clamp(0.01, 2.0);
        self.v = self.v.clamp(0.5, 500.0);
        self.mec = self.mec.clamp(0.0, 99.0);
        self.mtc = self.mtc.clamp(self.mec + 0.05, 200.0);
        self.xr_lag_h = self.xr_lag_h.clamp(0.0, 24.0);
        self.xr_f1 = self.xr_f1.clamp(0.0, 1.0);
        self.chiral_f_d = self.chiral_f_d.clamp(0.0, 1.0);
        self.chiral_l_ke = self.chiral_l_ke.clamp(0.01, 2.0);
        self.cal_t_max_h = self.cal_t_max_h.clamp(0.1, 24.0);
        self.cal_half_life_h = self.cal_half_life_h.clamp(0.5, 72.0);
        self.events.truncate(MAX_REGIMEN_EVENTS);
        for ev in &mut self.events {
            ev.t = ev.t.clamp(0.0, 768.0);
            ev.d = ev.d.clamp(0.0, 5000.0);
        }
        if self.events.is_empty() {
            self.events.push(CompactDoseEvent {
                t: 0.0,
                d: self.maintenance_dose_mg,
                xr: self.model == PkModelKind::BiphasicXr,
            });
        }
        true
    }

    /// Validates all scalar and schedule bounds for preset/URL loading.
    pub fn validate(&self) -> Result<(), String> {
        if !self.is_finite() {
            return Err("All pharmacokinetic regimen parameters must be finite".to_string());
        }
        if !(1.0..=5000.0).contains(&self.maintenance_dose_mg) {
            return Err("maintenance_dose_mg must be in [1.0, 5000.0]".to_string());
        }
        if !(1.0..=72.0).contains(&self.tau_h) || !(6.0..=168.0).contains(&self.horizon_h) {
            return Err("tau_h must be in [1, 72] and horizon_h in [6, 168]".to_string());
        }
        if !(0.05..=1.0).contains(&self.f)
            || !(0.05..=10.0).contains(&self.ka)
            || !(0.01..=2.0).contains(&self.ke)
            || !(0.5..=500.0).contains(&self.v)
        {
            return Err("PK parameters (f, ka, ke, v) out of valid physical range".to_string());
        }
        if !(0.0..=24.0).contains(&self.xr_lag_h)
            || !(0.0..=1.0).contains(&self.xr_f1)
            || !(0.0..=1.0).contains(&self.chiral_f_d)
            || !(0.01..=2.0).contains(&self.chiral_l_ke)
        {
            return Err("XR or chiral parameters out of valid physical range".to_string());
        }
        if !(0.1..=24.0).contains(&self.cal_t_max_h)
            || !(0.5..=72.0).contains(&self.cal_half_life_h)
        {
            return Err("Calibration T_max or half_life out of range".to_string());
        }
        TherapeuticWindow::new(self.mec, self.mtc).map_err(|e| e.to_string())?;
        if self.events.is_empty() || self.events.len() > MAX_REGIMEN_EVENTS {
            return Err(format!(
                "Regimen must contain 1..={MAX_REGIMEN_EVENTS} dose events"
            ));
        }
        let _ = self.to_domain_events()?;
        Ok(())
    }

    /// Converts compact GUI events into validated domain [`DoseEvent`]s.
    pub fn to_domain_events(&self) -> Result<Vec<DoseEvent>, String> {
        let mut out = Vec::with_capacity(self.events.len());
        let f1 = self.xr_f1;
        let f2 = (1.0 - f1).max(0.0);
        for ev in &self.events {
            if ev.d > 5000.0 || ev.t > 768.0 {
                return Err("DoseEvent time or mass exceeds planner bounds".to_string());
            }
            let use_xr = ev.xr || self.model == PkModelKind::BiphasicXr;
            let formulation = if use_xr {
                ReleaseFormulation::ExtendedRelease {
                    lag_time_h: self.xr_lag_h,
                    f1,
                    f2,
                }
            } else {
                ReleaseFormulation::ImmediateRelease
            };
            let domain_ev =
                DoseEvent::new(ev.t, ev.d, formulation).map_err(|e| e.to_string())?;
            out.push(domain_ev);
        }
        Ok(out)
    }
}
