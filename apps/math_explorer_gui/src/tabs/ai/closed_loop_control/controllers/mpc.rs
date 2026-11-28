// [cite:ai]
// [cite:pharmacokinetics]
//! Zero-allocation receding-horizon projected `MPC` controller (`MpcController`) for `PkPlantTwin`.

use super::super::ekf::{pk_rhs_and_jac, rk4_step_with_jacobian, FastEkf};
use super::super::state::{ClosedLoopState, PlantKind};
use engine_opt::ilqr::ControlModel;
use engine_opt::mpc::{
    mpc_workspace_len, solve_mpc, warm_start_shift_controls, MpcConfig, MpcProblem, MpcStats,
};
use engine_opt::{OptError, ScalarWorkspace};
use math_commons::math_kernel::types::flatten_2d_index;

/// Receding prediction horizon $N = 12$ steps (`6.0 h` at $\Delta t = 0.5\text{ h}$) for ADME `MPC`.
pub const MPC_PK_HORIZON: usize = 12;

const PK_NX: usize = 3;
const PK_NU: usize = 1;
const PK_BIOAVAIL: f64 = 0.85;
const PK_KA: f64 = 1.10;
const PK_VOLUME_L: f64 = 25.0;
const MPC_PK_WS_LEN: usize = mpc_workspace_len(MPC_PK_HORIZON, PK_NX, PK_NU);

/// Discrete RK4 3-state ADME transition model reusing `rk4_step_with_jacobian` and `pk_rhs_and_jac`.
#[derive(Debug, Clone, Copy)]
pub struct PkMpcModel {
    /// Discrete integration step $\Delta t$ in hours.
    pub dt: f64,
    /// Oral bioavailability fraction $F \in (0, 1]$.
    pub bioavail: f64,
    /// First-order absorption rate $k_a$ (`1/h`).
    pub ka: f64,
    /// Central distribution volume $V$ (`L`).
    pub volume: f64,
}

impl Default for PkMpcModel {
    fn default() -> Self {
        Self {
            dt: PlantKind::Pharmacokinetics.dt(),
            bioavail: PK_BIOAVAIL,
            ka: PK_KA,
            volume: PK_VOLUME_L,
        }
    }
}

impl ControlModel for PkMpcModel {
    #[inline]
    fn state_dim(&self) -> usize {
        PK_NX
    }

    #[inline]
    fn control_dim(&self) -> usize {
        PK_NU
    }

    #[verified_engine::verified]
    fn step_and_linearize(
        &self,
        _step: usize,
        x: &[f64],
        u: &[f64],
        x_next: &mut [f64],
        fx: &mut [f64],
        fu: &mut [f64],
    ) -> Result<(), OptError> {
        if x.len() != PK_NX
            || u.len() != PK_NU
            || x_next.len() != PK_NX
            || fx.len() != PK_NX * PK_NX
            || fu.len() != PK_NX * PK_NU
        {
            return Err(OptError::InvalidDimension);
        }
        if x.iter().any(|v| !v.is_finite()) || !u[0].is_finite() {
            return Err(OptError::NonFiniteInput);
        }
        let dosed = [x[0] + self.bioavail * u[0], x[1], x[2]];
        let (nx_arr, jac) =
            rk4_step_with_jacobian(&dosed, self.dt, |s| pk_rhs_and_jac(s, self.ka, self.volume));
        x_next.copy_from_slice(&nx_arr);
        for row in 0..PK_NX {
            for col in 0..PK_NX {
                fx[flatten_2d_index(col, row, PK_NX)] = jac[row][col];
            }
            fu[row] = self.bioavail * jac[row][0];
        }
        Ok(())
    }
}

/// Zero-allocation tube-tightened receding-horizon `MPC` controller coupled to `FastEkf<3>`.
#[derive(Debug, Clone)]
pub struct MpcController {
    /// Warm-start oral bolus trajectory $u_{0:N-1}$ in milligrams.
    pub u_warm: [f64; MPC_PK_HORIZON],
    /// Predicted ADME state trajectory $x_{0:N}$ (`[A_g, C_p, k_e]`).
    pub x_pred: [f64; PK_NX * (MPC_PK_HORIZON + 1)],
    /// Telemetry from the most recent `solve_mpc` call.
    pub last_stats: MpcStats,
    /// Last applied oral bolus dose $u_0^*$ in milligrams.
    pub last_dose_mg: f64,
    initialized: bool,
}

impl Default for MpcController {
    fn default() -> Self {
        Self::new()
    }
}

impl MpcController {
    /// Constructs a preallocated stack-resident `MpcController`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            u_warm: [0.0; MPC_PK_HORIZON],
            x_pred: [0.0; PK_NX * (MPC_PK_HORIZON + 1)],
            last_stats: MpcStats {
                iterations: 0,
                optimal_cost: 0.0,
                max_state_violation: 0.0,
                converged: true,
            },
            last_dose_mg: 0.0,
            initialized: false,
        }
    }

    /// Resets warm-start trajectories and telemetry in place without heap allocations.
    #[verified_engine::verified]
    pub fn reset_in_place(&mut self) {
        self.u_warm = [0.0; MPC_PK_HORIZON];
        self.x_pred = [0.0; PK_NX * (MPC_PK_HORIZON + 1)];
        self.last_stats = MpcStats {
            iterations: 0,
            optimal_cost: 0.0,
            max_state_violation: 0.0,
            converged: true,
        };
        self.last_dose_mg = 0.0;
        self.initialized = false;
    }

    /// Computes optimal bounded oral dose $u_0^* \in [0, D_{\max}]$ (mg) from posterior `FastEkf<3>` state and covariance.
    #[verified_engine::verified]
    pub fn solve_pk_dose_mg(&mut self, ekf: &FastEkf<3>, cfg: &ClosedLoopState) -> f64 {
        let dt = PlantKind::Pharmacokinetics.dt();
        let d_max = 42.0 * cfg.actuation_budget.max(0.0);
        if d_max <= 1.0e-9 {
            self.last_dose_mg = 0.0;
            return 0.0;
        }
        let (x0, cp_ref, u_ss, x_min, x_max) = build_pk_tube_targets(ekf, cfg, dt, d_max);
        if !self.initialized {
            self.u_warm.fill(u_ss);
            self.initialized = true;
        } else {
            warm_start_shift_controls(&mut self.u_warm, MPC_PK_HORIZON, PK_NU);
        }

        let model = PkMpcModel::default();
        let q_diag = [0.0, 3.2, 0.0];
        let qf_diag = [0.0, 5.0, 0.0];
        let r_diag = [1.5e-3 / d_max.max(1.0).powi(2)];
        let x_ref = [0.0, cp_ref, x0[2]];
        let u_ref = [u_ss];
        let u_min = [0.0];
        let u_max = [d_max];
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
            state_penalty_weight: 180.0,
            state_margin: 0.0,
        };
        let mpc_cfg = MpcConfig {
            horizon: MPC_PK_HORIZON,
            max_iterations: 28,
            tolerance: 1.0e-6,
            reg_init: 1.0e-6,
        };

        let mut u0_out = [u_ss];
        let mut ws_buf = [0.0_f64; MPC_PK_WS_LEN];
        let mut ws = ScalarWorkspace::new(&mut ws_buf);
        if let Ok(stats) = solve_mpc(
            &model,
            &prob,
            &mpc_cfg,
            &x0,
            &mut self.x_pred,
            &mut self.u_warm,
            &mut u0_out,
            &mut ws,
        ) {
            self.last_stats = stats;
        }
        let dose = u0_out[0].clamp(0.0, d_max);
        self.last_dose_mg = dose;
        dose
    }

    /// Maps the optimal receding-horizon oral dose $u_0^*$ into normalized `[-1.0, 1.0]` actuation for `PkPlantTwin::step`.
    #[verified_engine::verified]
    pub fn act_pk_normalized(&mut self, ekf: &FastEkf<3>, cfg: &ClosedLoopState) -> f64 {
        let d_max = 42.0 * cfg.actuation_budget.max(0.0);
        if d_max <= 1.0e-9 {
            self.last_dose_mg = 0.0;
            return -1.0;
        }
        let dose_mg = self.solve_pk_dose_mg(ekf, cfg);
        (2.0 * dose_mg / d_max - 1.0).clamp(-1.0, 1.0)
    }

    /// Evaluates normalized actuation in `[-1.0, 1.0]` for the active `PlantKind`.
    #[verified_engine::verified]
    pub fn act_for_plant(
        &mut self,
        cfg: &ClosedLoopState,
        pk_ekf: &FastEkf<3>,
        phi: &[f64; 4],
    ) -> f64 {
        match cfg.plant {
            PlantKind::Pharmacokinetics => self.act_pk_normalized(pk_ekf, cfg),
            PlantKind::LorenzChaos | PlantKind::SeirEpidemic => {
                (-0.85 * phi[0] - 0.35 * phi[1]).clamp(-1.0, 1.0)
            }
        }
    }
}

#[verified_engine::verified]
fn build_pk_tube_targets(
    ekf: &FastEkf<3>,
    cfg: &ClosedLoopState,
    dt: f64,
    d_max: f64,
) -> ([f64; 3], f64, f64, [f64; 3], [f64; 3]) {
    let ag_hat = ekf.state[0].max(0.0);
    let cp_hat = ekf.state[1].max(0.0);
    let ke_hat = ekf.state[2].clamp(0.04, 0.60);
    let sigma_cp = ekf.std_dev(1);
    let sigma_ke = ekf.std_dev(2);
    // Conservative lower-confidence clearance guards against cold-start EKF prior over-estimation.
    let ke_eff = (ke_hat - 1.10 * sigma_ke).clamp(0.06, 0.60);

    let mec = cfg.pk_mec.max(0.05);
    let mtc = cfg.pk_mtc.max(mec + 0.25);
    let span = mtc - mec;
    let cp_ref = mec + 0.45 * span;
    let u_ss = (ke_eff * cp_ref * PK_VOLUME_L * dt / PK_BIOAVAIL).clamp(0.0, d_max);

    let mec_tight = mec + (0.06 * span).min(0.15);
    let mtc_tight = (mtc - (0.15 * span + 0.45 * sigma_cp).min(0.40 * span)).max(mec_tight + 0.10);
    let x0 = [ag_hat, cp_hat, ke_eff];
    let x_min = [0.0, mec_tight, 0.0];
    let x_max = [f64::INFINITY, mtc_tight, f64::INFINITY];
    (x0, cp_ref, u_ss, x_min, x_max)
}
