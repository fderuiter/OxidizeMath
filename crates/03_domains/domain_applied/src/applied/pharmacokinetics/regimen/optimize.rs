//! Constrained multi-dose pharmacokinetic regimen optimization via receding-horizon `engine_opt::mpc`.

use super::{DoseEvent, ReleaseFormulation};
use crate::applied::pharmacokinetics::{PKParameters, t_max};
use engine_opt::ilqr::ControlModel;
use engine_opt::mpc::{MpcConfig, MpcProblem, MpcStats, mpc_workspace_len, solve_mpc};
use engine_opt::{OptError, ScalarWorkspace};

/// Upper bound on doses optimized in a single stack-resident Bateman MPC horizon.
pub const MAX_OPTIMIZED_DOSES: usize = 32;

/// Configuration for constrained multi-dose immediate-release regimen synthesis.
///
/// Optimizes per-event doses $u_k = D_k \in [0, D_{\max}]$ across administration times
/// $t_0 < t_1 < \dots < t_{N-1} < T_{\text{end}}$ over exact 1-compartment Bateman state transitions
/// tracking gut mass, pre-dose trough concentration, and intra-interval peak concentration:
///
/// $$
/// A_{g,k+1} = e^{-k_a \Delta t_k}(A_{g,k} + F u_k), \qquad
/// C_{p,k+1} = e^{-k_e \Delta t_k} C_{p,k} + \psi(\Delta t_k)(A_{g,k} + F u_k)
/// $$
///
/// Reference: Bateman (1910), DOI: 10.1017/S0305004100019988; Lucia et al. (`do-mpc`, 2017),
/// DOI: 10.1016/j.conengprac.2017.02.002;
/// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Uses the removable-singularity confluent kernel $\psi(\Delta t) = (k_a \Delta t / V)e^{-k_a \Delta t}$
/// when $|k_a - k_e| < 10^{-8}$ and tightened `MpcProblem` state bounds to prevent negative doses or
/// cold-start Maximum Tolerated Concentration (`mtc`) overshoot.
///
/// # Domain of Validity
/// Requires validated [`PKParameters`], strictly increasing `dose_times_h` followed by
/// `horizon_end_h > dose_times_h[N - 1]`, $0 < \text{mec} < \text{mtc}$, and $D_{\max} > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegimenMpcConfig {
    /// Minimum Effective Concentration $\text{MEC}$ in `mg/L` (`> 0`).
    pub mec: f64,
    /// Maximum Tolerated Concentration $\text{MTC}$ in `mg/L` (`> mec`).
    pub mtc: f64,
    /// Per-administration upper dose cap $D_{\max}$ in `mg` (`> 0`).
    pub max_dose_mg: f64,
    /// End of the evaluation horizon $T_{\text{end}}$ in hours (`> last dose time`).
    pub horizon_end_h: f64,
    /// Initial pre-existing bioavailable gut drug mass $A_g(t_0^-)$ in `mg` (`>= 0`).
    pub initial_gut_mg: f64,
    /// Initial pre-existing plasma concentration $C_p(t_0^-)$ in `mg/L` (`>= 0`).
    pub initial_plasma_mg_l: f64,
}

impl RegimenMpcConfig {
    /// Constructs a cold-start (`A_g = 0`, `C_p = 0`) multi-dose optimization configuration.
    #[must_use]
    #[verified_engine::verified]
    pub const fn cold_start(mec: f64, mtc: f64, max_dose_mg: f64, horizon_end_h: f64) -> Self {
        Self {
            mec,
            mtc,
            max_dose_mg,
            horizon_end_h,
            initial_gut_mg: 0.0,
            initial_plasma_mg_l: 0.0,
        }
    }
}

/// Exact `f64` workspace length required by [`solve_regimen_mpc_into`] for `n_doses` events.
#[inline]
#[must_use]
#[verified_engine::verified]
pub const fn regimen_mpc_workspace_len(n_doses: usize) -> usize {
    mpc_workspace_len(n_doses, 3, 1)
}

#[inline]
#[verified_engine::verified]
fn bateman_psi(ka: f64, ke: f64, volume: f64, dt: f64) -> f64 {
    if (ka - ke).abs() < math_commons::registry::TOLERANCE_STANDARD {
        (ka * dt / volume) * (-ka * dt).exp()
    } else {
        (ka / (volume * (ka - ke))) * ((-ke * dt).exp() - (-ka * dt).exp())
    }
}

/// 3-state `[A_g, C_trough, C_peak]` interval propagation model for exact therapeutic window control.
struct BatemanPeakTroughModel<'a> {
    params: PKParameters,
    dts: &'a [f64],
}

impl ControlModel for BatemanPeakTroughModel<'_> {
    #[inline]
    fn state_dim(&self) -> usize {
        3
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
        let dt = self.dts[step];
        let f = self.params.f();
        let ka = self.params.ka();
        let ke = self.params.ke();
        let v = self.params.v();
        let sp = t_max(ka, ke).clamp(0.0, dt);

        let ea = (-ka * dt).exp();
        let ee = (-ke * dt).exp();
        let psi_end = bateman_psi(ka, ke, v, dt);
        let ep = (-ke * sp).exp();
        let psi_pk = bateman_psi(ka, ke, v, sp);

        let dosed_gut = x[0] + f * u[0];
        x_next[0] = ea * dosed_gut;
        x_next[1] = ee * x[1] + psi_end * dosed_gut;
        x_next[2] = ep * x[1] + psi_pk * dosed_gut;

        fx.fill(0.0);
        fx[0] = ea;
        fx[3] = psi_end;
        fx[4] = ee;
        fx[6] = psi_pk;
        fx[7] = ep;

        fu[0] = ea * f;
        fu[1] = psi_end * f;
        fu[2] = psi_pk * f;
        Ok(())
    }
}

/// Optimizes a multi-dose immediate-release schedule `doses_mg_out` in place via `engine_opt::mpc`.
///
/// Synthesizes loading and maintenance doses $D_k \in [0, D_{\max}]$ that hold both intra-interval
/// peaks $C_p(t_k + t_{\max})$ and pre-dose troughs $C_p(t_{k+1}^-)$ inside $[\text{MEC}, \text{MTC}]$:
///
/// $$
/// \min_{0 \le D_k \le D_{\max}} \sum_{k=1}^{N} \Bigl[(C_{\text{trough},k} - C_{\text{mid}})^2 + (C_{\text{peak},k} - C_{\text{mid}})^2\Bigr]
/// $$
///
/// Reference: Bateman (1910), DOI: 10.1017/S0305004100019988; Lucia et al. (`do-mpc`, 2017),
/// DOI: 10.1016/j.conengprac.2017.02.002;
/// [`docs/math_specs/domain_applied.md`](../../../../../../../docs/math_specs/domain_applied.md).
///
/// # Stability
/// Projected Cholesky Riccati sweeps enforce $0 \le D_k \le D_{\max}$ and tightened `[mec, mtc]` margins
/// at every Armijo trial step.
///
/// # Domain of Validity
/// Supports `1..=MAX_OPTIMIZED_DOSES` (`32`) strictly increasing `dose_times_h` with
/// `ws.len() >= regimen_mpc_workspace_len(dose_times_h.len())`.
#[verified_engine::verified]
pub fn solve_regimen_mpc_into(
    params: &PKParameters,
    dose_times_h: &[f64],
    cfg: &RegimenMpcConfig,
    doses_mg_out: &mut [f64],
    ws: &mut ScalarWorkspace<'_, f64>,
) -> Result<MpcStats, OptError> {
    let n = dose_times_h.len();
    if n == 0 || n > MAX_OPTIMIZED_DOSES || doses_mg_out.len() != n {
        return Err(OptError::InvalidDimension);
    }
    validate_regimen_cfg(cfg)?;

    let mut dts = [0.0_f64; MAX_OPTIMIZED_DOSES];
    for k in 0..n {
        let t_curr = dose_times_h[k];
        let t_next = if k + 1 < n {
            dose_times_h[k + 1]
        } else {
            cfg.horizon_end_h
        };
        if !t_curr.is_finite() || !t_next.is_finite() {
            return Err(OptError::NonFiniteInput);
        }
        if t_curr < 0.0 || t_next <= t_curr {
            return Err(OptError::InvalidParameter);
        }
        dts[k] = t_next - t_curr;
    }

    seed_initial_doses(params, &dts[..n], cfg, doses_mg_out);
    let model = BatemanPeakTroughModel {
        params: *params,
        dts: &dts[..n],
    };
    let mid = 0.5 * (cfg.mec + cfg.mtc);
    let q_diag = [0.0, 1.0, 1.0];
    let qf_diag = [0.0, 1.5, 1.5];
    let r_diag = [1.0e-3 / (cfg.max_dose_mg * cfg.max_dose_mg)];
    let x_ref = [0.0, mid, mid];
    let u_ref = [0.0];
    let u_min = [0.0];
    let u_max = [cfg.max_dose_mg];
    let x_min = [0.0, cfg.mec, cfg.mec];
    let x_max = [f64::INFINITY, cfg.mtc, cfg.mtc];
    let prob = MpcProblem {
        q_diag: &q_diag,
        r_diag: &r_diag,
        qf_diag: &qf_diag,
        x_ref: &x_ref,
        u_ref: &u_ref,
        u_min: &u_min,
        u_max: &u_max,
        x_min: &x_min,
        x_max: &x_max,
        state_penalty_weight: 160.0,
        state_margin: 0.08 * (cfg.mtc - cfg.mec),
    };
    let mpc_cfg = MpcConfig {
        horizon: n,
        ..MpcConfig::default()
    };
    let x0 = [
        cfg.initial_gut_mg,
        cfg.initial_plasma_mg_l,
        cfg.initial_plasma_mg_l,
    ];
    let mut x_traj = [0.0_f64; 3 * (MAX_OPTIMIZED_DOSES + 1)];
    let mut u0 = [0.0_f64; 1];
    solve_mpc(
        &model,
        &prob,
        &mpc_cfg,
        &x0,
        &mut x_traj[..3 * (n + 1)],
        doses_mg_out,
        &mut u0,
        ws,
    )
}

#[verified_engine::verified]
fn validate_regimen_cfg(cfg: &RegimenMpcConfig) -> Result<(), OptError> {
    if !cfg.mec.is_finite()
        || !cfg.mtc.is_finite()
        || !cfg.max_dose_mg.is_finite()
        || !cfg.horizon_end_h.is_finite()
        || !cfg.initial_gut_mg.is_finite()
        || !cfg.initial_plasma_mg_l.is_finite()
    {
        return Err(OptError::NonFiniteInput);
    }
    if cfg.mec <= 0.0
        || cfg.mtc <= cfg.mec
        || cfg.max_dose_mg <= 0.0
        || cfg.initial_gut_mg < 0.0
        || cfg.initial_plasma_mg_l < 0.0
    {
        return Err(OptError::InvalidParameter);
    }
    Ok(())
}

#[verified_engine::verified]
fn seed_initial_doses(
    params: &PKParameters,
    dts: &[f64],
    cfg: &RegimenMpcConfig,
    doses_mg_out: &mut [f64],
) {
    let any_nonzero = doses_mg_out.iter().any(|&d| d > 0.0 && d.is_finite());
    let mid = 0.5 * (cfg.mec + cfg.mtc);
    for (k, slot) in doses_mg_out.iter_mut().enumerate() {
        if any_nonzero && slot.is_finite() {
            *slot = slot.clamp(0.0, cfg.max_dose_mg);
        } else {
            let maint =
                (mid * params.v() * params.ke() * dts[k] / params.f()).clamp(0.0, cfg.max_dose_mg);
            *slot = maint;
        }
    }
}

/// Writes validated immediate-release [`DoseEvent`] entries from `(dose_times_h, doses_mg)` with zero allocations.
#[verified_engine::verified]
pub fn populate_ir_dose_events_into(
    dose_times_h: &[f64],
    doses_mg: &[f64],
    events_out: &mut [DoseEvent],
) -> Result<usize, OptError> {
    let n = dose_times_h.len();
    if doses_mg.len() != n || events_out.len() < n {
        return Err(OptError::InvalidDimension);
    }
    for k in 0..n {
        let t = dose_times_h[k];
        let d = doses_mg[k];
        if !t.is_finite() || !d.is_finite() {
            return Err(OptError::NonFiniteInput);
        }
        if t < 0.0 || d < 0.0 {
            return Err(OptError::InvalidParameter);
        }
        events_out[k] = DoseEvent {
            time_h: t,
            dose_mg: d,
            formulation: ReleaseFormulation::ImmediateRelease,
        };
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::applied::pharmacokinetics::{
        BatemanModel, PharmacokineticModel, RegimenSuperposition,
    };
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_regimen_mpc_loading_and_maintenance_holds_therapeutic_window_zero_alloc() {
        let params = PKParameters::new(0.85, 100.0, 1.2, 0.12, 25.0).expect("valid PK params");
        let times = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0];
        let cfg = RegimenMpcConfig::cold_start(1.4, 3.4, 150.0, 48.0);
        let mut doses = [0.0_f64; 6];
        let mut events = [DoseEvent {
            time_h: 0.0,
            dose_mg: 0.0,
            formulation: ReleaseFormulation::ImmediateRelease,
        }; 6];
        let mut ws_buf = [0.0_f64; regimen_mpc_workspace_len(6)];

        let stats = {
            let _guard = crate::VERIFIED_TEST_LOCK
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let (res, _) = VerifiedEngine::run_verified(|| {
                let mut ws = ScalarWorkspace::new(&mut ws_buf);
                let s = solve_regimen_mpc_into(&params, &times, &cfg, &mut doses, &mut ws)
                    .expect("regimen MPC must succeed");
                let _ = populate_ir_dose_events_into(&times, &doses, &mut events)
                    .expect("event population must succeed");
                s
            });
            res
        };

        assert!(stats.converged);
        assert!(stats.max_state_violation < 1.0e-6);
        assert!(doses[0] > doses[1] * 1.15);
        assert!(doses.iter().all(|&d| (0.0..=cfg.max_dose_mg).contains(&d)));

        let regimen =
            RegimenSuperposition::new(BatemanModel::new(params), &events).expect("regimen builds");
        for step in 3..=96 {
            let t = step as f64 * 0.5;
            let cp = regimen.concentration(t);
            assert!(
                cp >= cfg.mec && cp <= cfg.mtc,
                "Regimen C_p({t}h) = {cp} violated [{}, {}]",
                cfg.mec,
                cfg.mtc
            );
        }
    }
}
