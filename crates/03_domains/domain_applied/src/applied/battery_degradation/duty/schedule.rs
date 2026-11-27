//! Constrained multi-segment battery C-rate duty scheduling via `engine_opt::ilqr`.

use super::DutySegment;
use crate::applied::battery_degradation::model::{
    ElectrochemicalAgingModel, MAX_C_RATE, MIN_C_RATE,
};
use crate::applied::battery_degradation::types::DepthOfDischarge;
use engine_opt::ilqr::{
    ControlModel, IlqrConfig, IlqrStats, StageCost, ilqr_workspace_len, solve_ilqr,
};
use engine_opt::{OptError, SolverWorkspace};

/// Maximum number of charging/duty slots optimized on stack buffers.
pub const MAX_SCHEDULE_SLOTS: usize = 16;

/// Specification for constrained multi-slot battery charging and duty-cycle C-rate optimization.
///
/// Minimizes cumulative Arrhenius-weighted C-rate electrochemical damage across $K$ thermal slots
/// subject to state-of-charge dynamics $x_{k+1} = x_k + u_k \Delta t_k$ and box bounds
/// $I_{c,\min} \le u_k \le I_{c,\max}$:
///
/// $$
/// \min_{I_{c,\min} \le u_k \le I_{c,\max}} \sum_{k=0}^{K-1} \frac{\Delta t_k \, u_k^{1 + \gamma_c}}{N_{70}(D_{\mathrm{ref}}, T_k, 1\mathrm{C})}
/// \quad + \frac{w_{\mathrm{term}}}{2}(x_K - \mathrm{SOC}_{\mathrm{target}})^2
/// $$
///
/// Reference: Wang et al. (2011), DOI: 10.1016/j.jpowsour.2010.11.134; Tassa, Mansard & Todorov
/// (2014), DOI: 10.1109/ICRA.2014.6907001;
/// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Convex power-law stage Hessians $\partial^2 \ell_k / \partial u_k^2 > 0$ are regularized above
/// $10^{-5}$ so projected Cholesky Riccati sweeps converge monotonically within $[I_{c,\min}, I_{c,\max}]$.
///
/// # Domain of Validity
/// Valid for $0 \le \mathrm{soc\_min} \le \mathrm{initial\_soc}, \mathrm{target\_soc} \le \mathrm{soc\_max} \le 1$,
/// $0.05 \le c_{\min} \le c_{\max} \le 10.0$, and slot durations $\Delta t_k > 0$ hours.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DutyScheduleConfig {
    /// Initial fractional state of charge $\mathrm{SOC}_0 \in [0, 1]$.
    pub initial_soc: f64,
    /// Target terminal state of charge $\mathrm{SOC}_{\mathrm{target}} \in [0, 1]$.
    pub target_soc: f64,
    /// Lower state-of-charge safety bound $\mathrm{SOC}_{\min} \in [0, 1]$.
    pub soc_min: f64,
    /// Upper state-of-charge safety bound $\mathrm{SOC}_{\max} \in [0, 1]$.
    pub soc_max: f64,
    /// Minimum admissible C-rate $I_{c,\min} \in [0.05, 10.0]$.
    pub c_rate_min: f64,
    /// Maximum admissible C-rate $I_{c,\max} \in [c_{\min}, 10.0]$.
    pub c_rate_max: f64,
    /// Reference depth-of-discharge percentage (`1.0..=100.0`) for $N_{70}$ normalization.
    pub reference_dod_pct: f64,
    /// Quadratic terminal state-of-charge tracking weight (`> 0`).
    pub terminal_soc_weight: f64,
}

impl Default for DutyScheduleConfig {
    fn default() -> Self {
        Self {
            initial_soc: 0.20,
            target_soc: 0.85,
            soc_min: 0.10,
            soc_max: 0.90,
            c_rate_min: MIN_C_RATE,
            c_rate_max: 3.0,
            reference_dod_pct: 65.0,
            terminal_soc_weight: 2.0e3,
        }
    }
}

/// Exact `f64` workspace length required by [`solve_duty_schedule_mpc_into`] for `n_slots`.
#[inline]
#[must_use]
#[verified_engine::verified]
pub const fn duty_schedule_workspace_len(n_slots: usize) -> usize {
    ilqr_workspace_len(n_slots, 1, 1)
}

struct SocSlotModel<'a> {
    slot_hours: &'a [f64],
}

impl ControlModel for SocSlotModel<'_> {
    #[inline]
    fn state_dim(&self) -> usize {
        1
    }

    #[inline]
    fn control_dim(&self) -> usize {
        1
    }

    #[verified_engine::verified]
    fn step_and_linearize(
        &self,
        step: usize,
        x: &[f64],
        u: &[f64],
        x_next: &mut [f64],
        fx: &mut [f64],
        fu: &mut [f64],
    ) -> Result<(), OptError> {
        x_next[0] = x[0] + self.slot_hours[step] * u[0];
        fx[0] = 1.0;
        fu[0] = self.slot_hours[step];
        Ok(())
    }
}

struct AgingDamageCost<'a> {
    damage_weights: &'a [f64],
    c_exp: f64,
    soc_min: f64,
    soc_max: f64,
    target_soc: f64,
    terminal_weight: f64,
}

impl AgingDamageCost<'_> {
    #[inline]
    #[verified_engine::verified]
    fn soc_barrier(&self, soc: f64) -> (f64, f64, f64) {
        let w = 800.0;
        if soc < self.soc_min {
            let v = self.soc_min - soc;
            (0.5 * w * v * v, -w * v, w)
        } else if soc > self.soc_max {
            let v = soc - self.soc_max;
            (0.5 * w * v * v, w * v, w)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

impl StageCost for AgingDamageCost<'_> {
    #[verified_engine::verified]
    fn stage_cost(&self, step: usize, x: &[f64], u: &[f64]) -> f64 {
        let c = u[0].max(MIN_C_RATE);
        let p = 1.0 + self.c_exp;
        let (b_cost, _, _) = self.soc_barrier(x[0]);
        b_cost + self.damage_weights[step] * c.powf(p) + 0.5 * 1.0e-5 * c * c
    }

    #[verified_engine::verified]
    fn stage_derivatives(
        &self,
        step: usize,
        x: &[f64],
        u: &[f64],
        lx: &mut [f64],
        lu: &mut [f64],
        lxx: &mut [f64],
        luu: &mut [f64],
        lux: &mut [f64],
    ) {
        let c = u[0].max(MIN_C_RATE);
        let p = 1.0 + self.c_exp;
        let w = self.damage_weights[step];
        let (_, b_g, b_h) = self.soc_barrier(x[0]);
        lx[0] = b_g;
        lxx[0] = b_h;
        lux[0] = 0.0;
        lu[0] = w * p * c.powf(self.c_exp) + 1.0e-5 * c;
        luu[0] = (w * p * self.c_exp * c.powf(self.c_exp - 1.0)).max(0.0) + 1.0e-5;
    }

    #[verified_engine::verified]
    fn terminal_cost(&self, x: &[f64]) -> f64 {
        let err = x[0] - self.target_soc;
        let (b_cost, _, _) = self.soc_barrier(x[0]);
        b_cost + 0.5 * self.terminal_weight * err * err
    }

    #[verified_engine::verified]
    fn terminal_derivatives(&self, x: &[f64], lx: &mut [f64], lxx: &mut [f64]) {
        let (_, b_g, b_h) = self.soc_barrier(x[0]);
        lx[0] = b_g + self.terminal_weight * (x[0] - self.target_soc);
        lxx[0] = b_h + self.terminal_weight;
    }
}

/// Solves for aging-minimal C-rates `c_rates_out` across thermal slots via `engine_opt::ilqr`.
///
/// Allocates charging throughput toward cooler intervals with lower Arrhenius acceleration while
/// enforcing $I_{c,\min} \le u_k \le I_{c,\max}$ and $\mathrm{SOC}_k \in [\mathrm{SOC}_{\min}, \mathrm{SOC}_{\max}]$:
///
/// $$
/// x_{k+1} = x_k + u_k \Delta t_k, \qquad
/// \ell_k(u_k) = \frac{\Delta t_k \, u_k^{1 + \gamma_c}}{N_{70}(D_{\mathrm{ref}}, T_k, 1\mathrm{C})}
/// $$
///
/// Reference: Wang et al. (2011), DOI: 10.1016/j.jpowsour.2010.11.134; Tassa, Mansard & Todorov
/// (2014), DOI: 10.1109/ICRA.2014.6907001;
/// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Box-projected iLQR steps preserve positive C-rates (`>= MIN_C_RATE`), avoiding fractional power
/// singularities in $u_k^{\gamma_c - 1}$.
///
/// # Domain of Validity
/// Supports `1..=MAX_SCHEDULE_SLOTS` (`16`) slots with positive finite `slot_hours`, finite
/// `slot_temperatures_c`, and `ws.len() >= duty_schedule_workspace_len(slot_hours.len())`.
#[verified_engine::verified]
pub fn solve_duty_schedule_mpc_into(
    base_model: &ElectrochemicalAgingModel,
    slot_hours: &[f64],
    slot_temperatures_c: &[f64],
    cfg: &DutyScheduleConfig,
    c_rates_out: &mut [f64],
    soc_traj_out: &mut [f64],
    ws: &mut SolverWorkspace<'_, f64>,
) -> Result<IlqrStats, OptError> {
    let n = slot_hours.len();
    if n == 0
        || n > MAX_SCHEDULE_SLOTS
        || slot_temperatures_c.len() != n
        || c_rates_out.len() != n
        || soc_traj_out.len() != n + 1
    {
        return Err(OptError::InvalidDimension);
    }
    validate_schedule_cfg(cfg)?;

    let c_min = cfg.c_rate_min.clamp(MIN_C_RATE, MAX_C_RATE);
    let c_max = cfg.c_rate_max.clamp(c_min, MAX_C_RATE);
    let dod = DepthOfDischarge::new_clamped(cfg.reference_dod_pct.clamp(1.0, 100.0));
    let mut damage_weights = [0.0_f64; MAX_SCHEDULE_SLOTS];
    let mut total_hours = 0.0_f64;

    for k in 0..n {
        let dt = slot_hours[k];
        let temp = slot_temperatures_c[k];
        if !dt.is_finite() || !temp.is_finite() {
            return Err(OptError::NonFiniteInput);
        }
        if dt <= 0.0 {
            return Err(OptError::InvalidParameter);
        }
        total_hours += dt;
        let slot_model = base_model.with_temperature_c(temp).with_c_rate(1.0);
        let n70_1c = slot_model.n70(dod).as_f64().max(1.0);
        damage_weights[k] = 1.0e4 * dt / n70_1c;
    }

    let avg_c = ((cfg.target_soc - cfg.initial_soc).max(0.0) / total_hours).clamp(c_min, c_max);
    let has_seed = c_rates_out.iter().any(|&c| c > 0.0 && c.is_finite());
    for slot in c_rates_out.iter_mut() {
        *slot = if has_seed && slot.is_finite() {
            slot.clamp(c_min, c_max)
        } else {
            avg_c
        };
    }

    let model = SocSlotModel { slot_hours };
    let cost = AgingDamageCost {
        damage_weights: &damage_weights[..n],
        c_exp: base_model.c_rate_exponent().max(0.05),
        soc_min: cfg.soc_min,
        soc_max: cfg.soc_max,
        target_soc: cfg.target_soc,
        terminal_weight: cfg.terminal_soc_weight,
    };
    let x0 = [cfg.initial_soc];
    let u_min = [c_min];
    let u_max = [c_max];
    let ilqr_cfg = IlqrConfig {
        horizon: n,
        ..IlqrConfig::default()
    };
    solve_ilqr(
        &model,
        &cost,
        &ilqr_cfg,
        &x0,
        Some(&u_min),
        Some(&u_max),
        soc_traj_out,
        c_rates_out,
        ws,
    )
}

#[verified_engine::verified]
fn validate_schedule_cfg(cfg: &DutyScheduleConfig) -> Result<(), OptError> {
    if !cfg.initial_soc.is_finite()
        || !cfg.target_soc.is_finite()
        || !cfg.soc_min.is_finite()
        || !cfg.soc_max.is_finite()
        || !cfg.c_rate_min.is_finite()
        || !cfg.c_rate_max.is_finite()
        || !cfg.reference_dod_pct.is_finite()
        || !cfg.terminal_soc_weight.is_finite()
    {
        return Err(OptError::NonFiniteInput);
    }
    if cfg.soc_min < 0.0
        || cfg.soc_max > 1.0
        || cfg.soc_min > cfg.soc_max
        || cfg.initial_soc < cfg.soc_min
        || cfg.initial_soc > cfg.soc_max
        || cfg.target_soc < cfg.soc_min
        || cfg.target_soc > cfg.soc_max
        || cfg.c_rate_min < MIN_C_RATE
        || cfg.c_rate_max < cfg.c_rate_min
        || cfg.terminal_soc_weight <= 0.0
    {
        return Err(OptError::InvalidParameter);
    }
    Ok(())
}

/// Copies optimized `c_rates` onto `segments` while updating each segment's proportional `daily_cycles`.
#[verified_engine::verified]
pub fn apply_scheduled_c_rates_into(
    c_rates: &[f64],
    slot_hours: &[f64],
    segments: &mut [DutySegment],
) -> Result<(), OptError> {
    let n = c_rates.len();
    if slot_hours.len() != n || segments.len() != n {
        return Err(OptError::InvalidDimension);
    }
    for k in 0..n {
        let seg = segments[k];
        let dod_frac = (seg.dod_pct / 100.0).max(0.10);
        let cyc = (c_rates[k] * slot_hours[k] / dod_frac).clamp(0.0, 10.0);
        segments[k] = DutySegment::new(
            seg.duration_days,
            seg.dod_pct,
            cyc,
            seg.temperature_c,
            c_rates[k],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::applied::battery_degradation::{Capacity, estimate_duty_lifetime};
    use verified_engine::engine::VerifiedEngine;

    fn compare_scheduled_vs_naive_lifetime(
        model: &ElectrochemicalAgingModel,
        c_rates: &[f64; 4],
        hours: &[f64; 4],
        temps: &[f64; 4],
    ) {
        let mut opt_segments = [
            DutySegment::new(1.0, 60.0, 1.0, temps[0], 2.4),
            DutySegment::new(1.0, 60.0, 0.1, temps[1], 0.1),
            DutySegment::new(1.0, 60.0, 0.0, temps[2], 0.1),
            DutySegment::new(1.0, 60.0, 0.0, temps[3], 0.1),
        ];
        let naive_fast_hot = [
            DutySegment::new(1.0, 60.0, 2.0, temps[0], 2.4),
            DutySegment::new(1.0, 60.0, 0.0, temps[1], 0.1),
            DutySegment::new(1.0, 60.0, 0.0, temps[2], 0.1),
            DutySegment::new(1.0, 60.0, 0.0, temps[3], 0.1),
        ];
        apply_scheduled_c_rates_into(c_rates, hours, &mut opt_segments).unwrap();
        let life_opt = estimate_duty_lifetime(model, &opt_segments, Capacity::new_clamped(0.80));
        let life_naive =
            estimate_duty_lifetime(model, &naive_fast_hot, Capacity::new_clamped(0.80));
        assert!(life_opt.years_to_80 > life_naive.years_to_80);
    }

    #[test]
    fn test_duty_schedule_mpc_shifts_charging_to_cool_slots_and_extends_lifetime_zero_alloc() {
        let model = ElectrochemicalAgingModel::default();
        let hours = [0.5, 0.5, 0.5, 0.5];
        let temps = [42.0, 38.0, 20.0, 18.0];
        let cfg = DutyScheduleConfig {
            initial_soc: 0.20,
            target_soc: 0.80,
            soc_min: 0.15,
            soc_max: 0.85,
            c_rate_min: 0.10,
            c_rate_max: 2.50,
            reference_dod_pct: 60.0,
            terminal_soc_weight: 4.0e3,
        };
        let mut c_rates = [0.0_f64; 4];
        let mut soc_traj = [0.0_f64; 5];
        let mut ws_buf = [0.0_f64; duty_schedule_workspace_len(4)];

        let stats = {
            let _guard = crate::VERIFIED_TEST_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let (s, _) = VerifiedEngine::run_verified(|| {
                let mut ws = SolverWorkspace::new(&mut ws_buf);
                solve_duty_schedule_mpc_into(
                    &model,
                    &hours,
                    &temps,
                    &cfg,
                    &mut c_rates,
                    &mut soc_traj,
                    &mut ws,
                )
                .expect("duty MPC schedule must succeed")
            });
            s
        };

        assert!(stats.converged);
        assert!((soc_traj[4] - cfg.target_soc).abs() < 1.0e-2);
        assert!(
            c_rates
                .iter()
                .all(|&c| (cfg.c_rate_min..=cfg.c_rate_max).contains(&c))
        );
        assert!(c_rates[3] > c_rates[0] && c_rates[2] > c_rates[0]);
        compare_scheduled_vs_naive_lifetime(&model, &c_rates, &hours, &temps);
    }
}
