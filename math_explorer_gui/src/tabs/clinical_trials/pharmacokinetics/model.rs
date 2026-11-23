//! Pharmacokinetic model compilation and zero-allocation curve evaluation.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{
    metrics, AbsorptionMode, FormulationKind, PharmacokineticsTool, LOG10_CONC_FLOOR, MAX_DOSES,
    MAX_SAMPLE_POINTS, MIN_SAMPLE_POINTS, SOLVE_KA_MAX_ITER,
};
use egui_plot::PlotPoint;
use math_commons::registry::TOLERANCE_FAST;
use math_explorer::applied::pharmacokinetics::{
    solve_ka, BatemanModel, EnantiomerModel, PKParameters, PharmacokineticModel,
    SuperpositionModel, TwoPulseModel,
};
use math_explorer::error::PharmacokineticsError;

/// Pre-built superposition + optional `t = 0` loading dose wrapper for any `PharmacokineticModel`.
#[derive(Debug, Clone)]
pub(crate) struct CompiledRegimen<M> {
    pub(crate) maintenance: SuperpositionModel<M>,
    pub(crate) loading: Option<M>,
}

impl<M: PharmacokineticModel> CompiledRegimen<M> {
    #[inline]
    pub(crate) fn concentration(&self, t: f64) -> f64 {
        let base = self.maintenance.concentration(t);
        let load = self
            .loading
            .as_ref()
            .map_or(0.0, |m| if t >= 0.0 { m.concentration(t) } else { 0.0 });
        base + load
    }
}

/// Pre-compiled formulation models built once per parameter update outside sample loops.
#[derive(Debug, Clone)]
pub(crate) enum CompiledFormulation {
    SingleImmediate(CompiledRegimen<BatemanModel>),
    SingleTwoPulse(CompiledRegimen<TwoPulseModel<BatemanModel>>),
    ChiralImmediate {
        total: CompiledRegimen<EnantiomerModel>,
        d_isomer: CompiledRegimen<BatemanModel>,
        l_isomer: CompiledRegimen<BatemanModel>,
    },
    ChiralTwoPulse {
        total: CompiledRegimen<TwoPulseModel<EnantiomerModel>>,
        d_isomer: CompiledRegimen<TwoPulseModel<BatemanModel>>,
        l_isomer: CompiledRegimen<TwoPulseModel<BatemanModel>>,
    },
}

impl CompiledFormulation {
    #[inline]
    pub(crate) fn sample(&self, t: f64) -> (f64, Option<(f64, f64)>) {
        match self {
            Self::SingleImmediate(reg) => (reg.concentration(t), None),
            Self::SingleTwoPulse(reg) => (reg.concentration(t), None),
            Self::ChiralImmediate {
                total,
                d_isomer,
                l_isomer,
            } => (
                total.concentration(t),
                Some((d_isomer.concentration(t), l_isomer.concentration(t))),
            ),
            Self::ChiralTwoPulse {
                total,
                d_isomer,
                l_isomer,
            } => (
                total.concentration(t),
                Some((d_isomer.concentration(t), l_isomer.concentration(t))),
            ),
        }
    }
}

/// Computes a convexity-safe positive initial guess `0 < guess <= ka*` and invokes `solve_ka`.
pub(crate) fn solve_ka_calibrated(
    target_t_max: f64,
    ke: f64,
) -> Result<f64, PharmacokineticsError> {
    if !target_t_max.is_finite() || !ke.is_finite() || target_t_max <= 0.0 || ke <= 0.0 {
        return Err(PharmacokineticsError::InvalidParameter(
            "Target T_max and elimination rate ke must be strictly positive".into(),
        ));
    }
    let equal_rate_tmax = 1.0 / ke;
    let guess = if target_t_max <= equal_rate_tmax {
        ke
    } else {
        (ke * (-ke * target_t_max).exp()).max(1.0e-12)
    };
    let solved = solve_ka(target_t_max, ke, guess, SOLVE_KA_MAX_ITER, TOLERANCE_FAST)?;
    if !solved.is_finite() || solved <= 0.0 {
        return Err(PharmacokineticsError::InvalidParameter(format!(
            "Solved absorption rate ka={solved} is outside physical positive bounds"
        )));
    }
    Ok(solved)
}

#[inline]
fn project_plot_y(c: f64, log10_y: bool) -> f64 {
    let clamped = c.max(0.0);
    if log10_y {
        clamped.max(LOG10_CONC_FLOOR).log10()
    } else {
        clamped
    }
}

impl PharmacokineticsTool {
    /// Returns `(maintenance_dose_times, has_active_loading_at_zero, first_administered_time)`.
    pub(crate) fn build_schedule_times(&self) -> (Vec<f64>, bool, Option<f64>) {
        let count = self.n_doses.clamp(1, MAX_DOSES);
        let mut maint_times = Vec::with_capacity(count);
        let mut has_loading = false;
        let mut first_time = None;

        for slot in 0..count {
            if self.missed_slots.binary_search(&slot).is_ok() {
                continue;
            }
            let t = (slot as f64) * self.interval_hours;
            if first_time.is_none() {
                first_time = Some(t);
            }
            if slot == 0 && self.use_loading_dose {
                has_loading = true;
            } else {
                maint_times.push(t);
            }
        }
        (maint_times, has_loading, first_time)
    }

    fn resolve_effective_ka(&self) -> Result<f64, PharmacokineticsError> {
        match self.absorption_mode {
            AbsorptionMode::DirectKa => Ok(self.ka),
            AbsorptionMode::SolveFromTMax => solve_ka_calibrated(self.target_t_max, self.ke),
        }
    }

    fn build_bateman_regimen(
        params: PKParameters,
        loading_dose: f64,
        has_loading: bool,
        maint_times: Vec<f64>,
    ) -> Result<CompiledRegimen<BatemanModel>, PharmacokineticsError> {
        let loading = if has_loading {
            Some(BatemanModel::new(params.with_dose(loading_dose)?))
        } else {
            None
        };
        let maintenance = SuperpositionModel::new(BatemanModel::new(params), maint_times);
        Ok(CompiledRegimen {
            maintenance,
            loading,
        })
    }

    fn build_two_pulse_bateman_regimen(
        params: PKParameters,
        loading_dose: f64,
        has_loading: bool,
        maint_times: Vec<f64>,
        lag: f64,
        f1: f64,
        f2: f64,
    ) -> Result<CompiledRegimen<TwoPulseModel<BatemanModel>>, PharmacokineticsError> {
        let loading = if has_loading {
            let base = BatemanModel::new(params.with_dose(loading_dose)?);
            Some(TwoPulseModel::new(base, lag, f1, f2))
        } else {
            None
        };
        let maint_base = TwoPulseModel::new(BatemanModel::new(params), lag, f1, f2);
        Ok(CompiledRegimen {
            maintenance: SuperpositionModel::new(maint_base, maint_times),
            loading,
        })
    }

    fn build_chiral_immediate(
        &self,
        d_params: PKParameters,
        l_params: PKParameters,
        has_loading: bool,
        maint_times: Vec<f64>,
    ) -> Result<CompiledFormulation, PharmacokineticsError> {
        let total_maint = EnantiomerModel::new(d_params, l_params, self.f_d, self.f_l)?;
        let total_load = if has_loading {
            Some(EnantiomerModel::new(
                d_params.with_dose(self.loading_dose_mg)?,
                l_params.with_dose(self.loading_dose_mg)?,
                self.f_d,
                self.f_l,
            )?)
        } else {
            None
        };
        let d_reg = Self::build_bateman_regimen(
            d_params.with_dose(self.dose_mg * self.f_d)?,
            self.loading_dose_mg * self.f_d,
            has_loading,
            maint_times.clone(),
        )?;
        let l_reg = Self::build_bateman_regimen(
            l_params.with_dose(self.dose_mg * self.f_l)?,
            self.loading_dose_mg * self.f_l,
            has_loading,
            maint_times.clone(),
        )?;
        Ok(CompiledFormulation::ChiralImmediate {
            total: CompiledRegimen {
                maintenance: SuperpositionModel::new(total_maint, maint_times),
                loading: total_load,
            },
            d_isomer: d_reg,
            l_isomer: l_reg,
        })
    }

    fn build_chiral_two_pulse(
        &self,
        d_params: PKParameters,
        l_params: PKParameters,
        has_loading: bool,
        maint_times: Vec<f64>,
    ) -> Result<CompiledFormulation, PharmacokineticsError> {
        let lag = self.lag_time_hours;
        let total_maint = TwoPulseModel::new(
            EnantiomerModel::new(d_params, l_params, self.f_d, self.f_l)?,
            lag,
            self.f1,
            self.f2,
        );
        let total_load = if has_loading {
            let load_enantiomer = EnantiomerModel::new(
                d_params.with_dose(self.loading_dose_mg)?,
                l_params.with_dose(self.loading_dose_mg)?,
                self.f_d,
                self.f_l,
            )?;
            Some(TwoPulseModel::new(load_enantiomer, lag, self.f1, self.f2))
        } else {
            None
        };
        let d_reg = Self::build_two_pulse_bateman_regimen(
            d_params.with_dose(self.dose_mg * self.f_d)?,
            self.loading_dose_mg * self.f_d,
            has_loading,
            maint_times.clone(),
            lag,
            self.f1,
            self.f2,
        )?;
        let l_reg = Self::build_two_pulse_bateman_regimen(
            l_params.with_dose(self.dose_mg * self.f_l)?,
            self.loading_dose_mg * self.f_l,
            has_loading,
            maint_times.clone(),
            lag,
            self.f1,
            self.f2,
        )?;
        Ok(CompiledFormulation::ChiralTwoPulse {
            total: CompiledRegimen {
                maintenance: SuperpositionModel::new(total_maint, maint_times),
                loading: total_load,
            },
            d_isomer: d_reg,
            l_isomer: l_reg,
        })
    }

    /// Validates parameters, builds `SuperpositionModel` instances once, and reserves point buffers.
    pub(crate) fn compile_models(&mut self) {
        let cap = self.sample_points.clamp(MIN_SAMPLE_POINTS, MAX_SAMPLE_POINTS);
        for buf in [
            &mut self.total_points,
            &mut self.d_points,
            &mut self.l_points,
            &mut self.plot_total_points,
            &mut self.plot_d_points,
            &mut self.plot_l_points,
        ] {
            if buf.capacity() < cap {
                buf.reserve(cap - buf.capacity());
            }
        }

        let res = self.try_build_formulation();
        match res {
            Ok((ka_eff, compiled)) => {
                self.effective_ka = ka_eff;
                self.compiled = Some(compiled);
                self.solver_error = None;
            }
            Err(err) => {
                self.compiled = None;
                self.solver_error = Some(err.to_string());
            }
        }
    }

    fn build_l_isomer_params(&self) -> Result<PKParameters, PharmacokineticsError> {
        PKParameters::new(
            self.bioavailability,
            self.dose_mg,
            self.ka_l,
            self.ke_l,
            self.volume_l,
        )
    }

    fn try_build_formulation(
        &self,
    ) -> Result<(f64, CompiledFormulation), PharmacokineticsError> {
        if !(0.0..=1.0).contains(&self.f1) || !(0.0..=1.0).contains(&self.f2) {
            return Err(PharmacokineticsError::InvalidParameter(
                "Pulse fractions f1 and f2 must be between 0.0 and 1.0".into(),
            ));
        }
        if self.lag_time_hours < 0.0 {
            return Err(PharmacokineticsError::InvalidParameter(
                "Lag time must be non-negative".into(),
            ));
        }
        let ka_eff = self.resolve_effective_ka()?;
        let base_params = PKParameters::new(
            self.bioavailability,
            self.dose_mg,
            ka_eff,
            self.ke,
            self.volume_l,
        )?;
        let (maint_times, has_loading, _) = self.build_schedule_times();

        let compiled = match self.formulation() {
            FormulationKind::SingleImmediate => CompiledFormulation::SingleImmediate(
                Self::build_bateman_regimen(
                    base_params,
                    self.loading_dose_mg,
                    has_loading,
                    maint_times,
                )?,
            ),
            FormulationKind::SingleTwoPulse => CompiledFormulation::SingleTwoPulse(
                Self::build_two_pulse_bateman_regimen(
                    base_params,
                    self.loading_dose_mg,
                    has_loading,
                    maint_times,
                    self.lag_time_hours,
                    self.f1,
                    self.f2,
                )?,
            ),
            FormulationKind::ChiralImmediate => {
                let l_params = self.build_l_isomer_params()?;
                self.build_chiral_immediate(base_params, l_params, has_loading, maint_times)?
            }
            FormulationKind::ChiralTwoPulse => {
                let l_params = self.build_l_isomer_params()?;
                self.build_chiral_two_pulse(base_params, l_params, has_loading, maint_times)?
            }
        };
        Ok((ka_eff, compiled))
    }

    /// Evaluates plasma curves and PK metrics in-place with zero dynamic heap allocations.
    pub fn evaluate_sampled_regimen(&mut self) {
        self.total_points.clear();
        self.d_points.clear();
        self.l_points.clear();
        self.plot_total_points.clear();
        self.plot_d_points.clear();
        self.plot_l_points.clear();

        let Some(ref compiled) = self.compiled else {
            self.metrics = metrics::PkMetrics::default();
            return;
        };

        let count = self.sample_points.clamp(MIN_SAMPLE_POINTS, MAX_SAMPLE_POINTS);
        let dt = self.horizon_hours / ((count - 1) as f64);
        let log10_y = self.log10_y;

        for idx in 0..count {
            let t = (idx as f64) * dt;
            let (c_tot, chiral_pair) = compiled.sample(t);
            self.total_points.push(PlotPoint::new(t, c_tot));
            self.plot_total_points
                .push(PlotPoint::new(t, project_plot_y(c_tot, log10_y)));

            if let Some((c_d, c_l)) = chiral_pair {
                self.d_points.push(PlotPoint::new(t, c_d));
                self.l_points.push(PlotPoint::new(t, c_l));
                self.plot_d_points
                    .push(PlotPoint::new(t, project_plot_y(c_d, log10_y)));
                self.plot_l_points
                    .push(PlotPoint::new(t, project_plot_y(c_l, log10_y)));
            }
        }

        let trough_end_h = if self.n_doses > 1 {
            ((self.n_doses as f64) * self.interval_hours).min(self.horizon_hours)
        } else {
            self.horizon_hours
        };
        self.metrics = metrics::evaluate_pk_metrics(
            &self.total_points,
            metrics::MetricContext {
                ka: self.effective_ka,
                ke: self.ke,
                volume_l: self.volume_l,
                interval_hours: self.interval_hours,
                trough_end_hours: trough_end_h,
                mec: self.mec,
                mtc: self.mtc,
            },
        );
    }
}

// [cite:clinical_trials_statistics]
