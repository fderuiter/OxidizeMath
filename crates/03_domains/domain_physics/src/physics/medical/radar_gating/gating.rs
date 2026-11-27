//! Gating Logic for Radiation Beam Control.
//!
//! Implements the decision layer for respiratory gating, using a Schmidt Trigger
//! and linear or harmonic `engine_opt::mpc` lookahead latency compensation to control
//! the LINAC beam-off switch via a formal finite state machine with validation guards.

use engine_opt::ilqr::ControlModel;
use engine_opt::mpc::{MpcConfig, MpcProblem, mpc_workspace_len, solve_mpc};
use engine_opt::{OptError, SolverWorkspace};
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

/// Receding horizon step count used for stack-allocated harmonic respiratory `MPC` lookahead.
pub const HARMONIC_MPC_HORIZON: usize = 6;

const STACK_MPC_WS_LEN: usize = mpc_workspace_len(HARMONIC_MPC_HORIZON, 2, 1);

/// Exact `f64` workspace length required by [`GatingLogic::solve_harmonic_lookahead_mpc`].
#[inline]
#[must_use]
#[verified_engine::verified]
pub const fn harmonic_gating_mpc_workspace_len(horizon: usize) -> usize {
    mpc_workspace_len(horizon, 2, 1)
}

/// Operational lifecycle states for the radiation beam gating controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BeamState {
    /// Controller is uninitialized or dormant; beam emission disabled.
    #[default]
    Standby,
    /// Controller is armed and ready for displacement evaluation.
    Armed,
    /// Linear accelerator radiation beam actively delivering dose.
    BeamActive,
    /// Beam delivery paused while patient is outside target respiratory phase.
    BeamHold,
    /// Emergency safety latch triggered by sensor loss or invalid inputs.
    SafetyInterlock,
}

/// Errors returned during state machine transitions.
#[derive(thiserror::Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingError {
    /// Transition between specified states is not permitted by lifecycle rules.
    #[error("Invalid state transition from {from:?} to {to:?}")]
    InvalidTransition {
        /// Current lifecycle state.
        from: BeamState,
        /// Requested target state.
        to: BeamState,
    },
    /// The safety interlock has latched and must be explicitly reset before arming.
    #[error("Safety interlock is active and must be explicitly reset")]
    InterlockLatched,
    /// The controller is not in safety interlock state.
    #[error("Controller is not in safety interlock state")]
    NotInterlocked,
}

impl Diagnostic for GatingError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "GatingError");
        visitor.record_display("description", self);
    }
}

struct HarmonicRespiratoryModel {
    omega: f64,
    dt: f64,
}

impl ControlModel for HarmonicRespiratoryModel {
    #[inline]
    fn state_dim(&self) -> usize {
        2
    }

    #[inline]
    fn control_dim(&self) -> usize {
        1
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
        let theta = self.omega * self.dt;
        let c = theta.cos();
        let s = theta.sin();
        let inv_w = 1.0 / self.omega;
        let b0 = (1.0 - c) * inv_w * inv_w;
        let b1 = s * inv_w;

        x_next[0] = c * x[0] + b1 * x[1] + b0 * u[0];
        x_next[1] = -self.omega * s * x[0] + c * x[1] + b1 * u[0];

        fx[0] = c;
        fx[1] = b1;
        fx[2] = -self.omega * s;
        fx[3] = c;
        fu[0] = b0;
        fu[1] = b1;
        Ok(())
    }
}

/// Configuration and State for the Gating Logic.
#[derive(Debug, Clone, PartialEq)]
pub struct GatingLogic {
    /// Amplitude threshold for gating (e.g., peak exhalation level).
    pub threshold: f64,
    /// Hysteresis margin to prevent chattering (e.g., 2 mm).
    pub hysteresis: f64,
    /// Hardware latency of the beam-off switch in seconds (e.g., 0.060 or 0.600).
    pub system_latency: f64,
    /// Optional angular respiratory frequency $\omega = 2\pi f$ (`rad/s`) enabling harmonic `MPC` lookahead.
    harmonic_omega_rad_s: Option<f64>,
    /// Respiratory equilibrium baseline displacement $z_0$ used by harmonic `MPC` lookahead.
    baseline_position: f64,
    /// Current lifecycle state of the beam gating controller.
    state: BeamState,
}

impl GatingLogic {
    /// Creates a new Gating Logic instance in `Standby` state using linear velocity extrapolation by default.
    #[verified_engine::verified]
    pub fn new(threshold: f64, hysteresis: f64, system_latency: f64) -> Self {
        Self {
            threshold,
            hysteresis,
            system_latency,
            harmonic_omega_rad_s: None,
            baseline_position: 0.0,
            state: BeamState::Standby,
        }
    }

    /// Configures symplectic harmonic respiratory `MPC` lookahead at `respiratory_freq_hz` (`> 0`) around `baseline_position`.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_harmonic_mpc(mut self, respiratory_freq_hz: f64, baseline_position: f64) -> Self {
        if respiratory_freq_hz.is_finite()
            && respiratory_freq_hz > 0.0
            && baseline_position.is_finite()
        {
            self.harmonic_omega_rad_s = Some(std::f64::consts::TAU * respiratory_freq_hz);
            self.baseline_position = baseline_position;
        } else {
            self.harmonic_omega_rad_s = None;
        }
        self
    }

    /// Solves the receding-horizon harmonic respiratory `MPC` lookahead problem into `ws` with zero heap allocations.
    ///
    /// Propagates $x_k = [z_k - z_0, v_k]^\top$ across $N$ exact harmonic substeps $\Delta t = \tau_{\text{lat}} / N$
    /// under regularized acceleration residual bounds via [`engine_opt::mpc::solve_mpc`]:
    ///
    /// $$
    /// x_{k+1} = \begin{bmatrix} \cos(\omega \Delta t) & \omega^{-1}\sin(\omega \Delta t) \\ -\omega\sin(\omega \Delta t) & \cos(\omega \Delta t) \end{bmatrix} x_k + B(\Delta t) u_k
    /// $$
    ///
    /// Reference: Lucia et al. (`do-mpc`, 2017), DOI: 10.1016/j.conengprac.2017.02.002;
    /// Keall et al. (2006), DOI: 10.1118/1.2349696.
    ///
    /// # Stability
    /// Symplectic rotation substeps preserve phase energy across turnaround peaks without linear tangent overshoot.
    ///
    /// # Domain of Validity
    /// Requires finite `amplitude`, `velocity`, `system_latency >= 0`, and `ws.len() >= harmonic_gating_mpc_workspace_len(HARMONIC_MPC_HORIZON)`.
    #[verified_engine::verified]
    pub fn solve_harmonic_lookahead_mpc(
        &self,
        amplitude: f64,
        velocity: f64,
        ws: &mut SolverWorkspace<'_, f64>,
    ) -> Result<f64, OptError> {
        if !amplitude.is_finite() || !velocity.is_finite() || !self.system_latency.is_finite() {
            return Err(OptError::NonFiniteInput);
        }
        if self.system_latency <= 1.0e-9 {
            return Ok(amplitude);
        }
        let Some(omega) = self.harmonic_omega_rad_s else {
            return Ok(amplitude + velocity * self.system_latency);
        };
        let dt = self.system_latency / (HARMONIC_MPC_HORIZON as f64);
        let model = HarmonicRespiratoryModel { omega, dt };
        let q_diag = [0.0, 0.0];
        let qf_diag = [0.0, 0.0];
        let r_diag = [1.0e6];
        let x_ref = [0.0, 0.0];
        let u_ref = [0.0];
        let u_min = [-1.0e-3];
        let u_max = [1.0e-3];
        let x_min = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        let x_max = [f64::INFINITY, f64::INFINITY];
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
            state_penalty_weight: 0.0,
            state_margin: 0.0,
        };
        let cfg = MpcConfig {
            horizon: HARMONIC_MPC_HORIZON,
            max_iterations: 4,
            tolerance: 1.0e-9,
            reg_init: 1.0e-6,
        };
        let x0 = [amplitude - self.baseline_position, velocity];
        let mut x_traj = [0.0_f64; 2 * (HARMONIC_MPC_HORIZON + 1)];
        let mut u_traj = [0.0_f64; HARMONIC_MPC_HORIZON];
        let mut u0 = [0.0_f64; 1];
        solve_mpc(
            &model,
            &prob,
            &cfg,
            &x0,
            &mut x_traj,
            &mut u_traj,
            &mut u0,
            ws,
        )?;
        Ok(self.baseline_position + x_traj[2 * HARMONIC_MPC_HORIZON])
    }

    /// Predicts respiratory displacement at `t + system_latency` using harmonic `MPC` (when enabled) or linear lookahead.
    #[must_use]
    #[verified_engine::verified]
    pub fn predict_amplitude(&self, amplitude: f64, velocity: f64) -> f64 {
        if self.harmonic_omega_rad_s.is_some() && self.system_latency > 1.0e-9 {
            let mut ws_buf = [0.0_f64; STACK_MPC_WS_LEN];
            let mut ws = SolverWorkspace::new(&mut ws_buf);
            self.solve_harmonic_lookahead_mpc(amplitude, velocity, &mut ws)
                .unwrap_or(f64::NAN)
        } else {
            amplitude + velocity * self.system_latency
        }
    }

    /// Returns the current lifecycle state of the beam controller.
    #[verified_engine::verified]
    pub fn state(&self) -> BeamState {
        self.state
    }

    /// Returns whether the radiation beam is actively emitting.
    #[verified_engine::verified]
    pub fn is_beam_active(&self) -> bool {
        self.state == BeamState::BeamActive
    }

    /// Arms the controller from `Standby` into `Armed`.
    #[verified_engine::verified]
    pub fn arm(&mut self) -> Result<BeamState, GatingError> {
        match self.state {
            BeamState::Standby => {
                self.state = BeamState::Armed;
                Ok(self.state)
            }
            BeamState::Armed => Ok(self.state),
            BeamState::SafetyInterlock => Err(GatingError::InterlockLatched),
            from @ (BeamState::BeamActive | BeamState::BeamHold) => {
                Err(GatingError::InvalidTransition {
                    from,
                    to: BeamState::Armed,
                })
            }
        }
    }

    /// Disarms the controller, returning it to `Standby` and ensuring beam shutoff.
    #[verified_engine::verified]
    pub fn disarm(&mut self) -> Result<BeamState, GatingError> {
        match self.state {
            BeamState::SafetyInterlock => Err(GatingError::InterlockLatched),
            _ => {
                self.state = BeamState::Standby;
                Ok(self.state)
            }
        }
    }

    /// Immediately triggers the safety interlock, latching the state and stopping the beam.
    #[verified_engine::verified]
    pub fn trigger_interlock(&mut self) -> BeamState {
        self.state = BeamState::SafetyInterlock;
        self.state
    }

    /// Clears the safety interlock and returns the controller to `Standby`.
    #[verified_engine::verified]
    pub fn reset_interlock(&mut self) -> Result<BeamState, GatingError> {
        if self.state == BeamState::SafetyInterlock {
            self.state = BeamState::Standby;
            Ok(self.state)
        } else {
            Err(GatingError::NotInterlocked)
        }
    }

    /// Evaluates whether the beam should be ON or OFF based on the current tracker state.
    #[verified_engine::verified]
    pub fn evaluate(&mut self, amplitude: f64, velocity: f64) -> BeamState {
        if !amplitude.is_finite()
            || !velocity.is_finite()
            || !self.threshold.is_finite()
            || !self.hysteresis.is_finite()
            || !self.system_latency.is_finite()
        {
            self.state = BeamState::SafetyInterlock;
            return self.state;
        }

        let predicted_amplitude = self.predict_amplitude(amplitude, velocity);
        if !predicted_amplitude.is_finite() {
            self.state = BeamState::SafetyInterlock;
            return self.state;
        }

        match self.state {
            BeamState::Standby | BeamState::SafetyInterlock => {}
            BeamState::Armed => {
                if predicted_amplitude < self.threshold - self.hysteresis {
                    self.state = BeamState::BeamActive;
                } else {
                    self.state = BeamState::BeamHold;
                }
            }
            BeamState::BeamActive => {
                if predicted_amplitude > self.threshold + self.hysteresis {
                    self.state = BeamState::BeamHold;
                }
            }
            BeamState::BeamHold => {
                if predicted_amplitude < self.threshold - self.hysteresis {
                    self.state = BeamState::BeamActive;
                }
            }
        }

        self.state
    }
}
