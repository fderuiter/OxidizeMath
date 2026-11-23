//! Gating Logic for Radiation Beam Control.
//!
//! Implements the decision layer for respiratory gating, using a Schmidt Trigger
//! and latency compensation to control the LINAC beam-off switch via a formal
//! finite state machine with validation guards.

use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

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
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "GatingError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

/// Configuration and State for the Gating Logic.
#[derive(Debug, Clone, PartialEq)]
pub struct GatingLogic {
    /// Amplitude threshold for gating (e.g., peak exhalation level).
    pub threshold: f64,
    /// Hysteresis margin to prevent chattering (e.g., 2 mm).
    pub hysteresis: f64,
    /// Hardware latency of the beam-off switch in seconds (e.g., 0.060).
    pub system_latency: f64,
    /// Current lifecycle state of the beam gating controller.
    state: BeamState,
}

impl GatingLogic {
    /// Creates a new Gating Logic instance in `Standby` state.
    #[verified_engine::verified]
    pub fn new(threshold: f64, hysteresis: f64, system_latency: f64) -> Self {
        Self {
            threshold,
            hysteresis,
            system_latency,
            state: BeamState::Standby,
        }
    }

    /// Returns the current lifecycle state of the beam controller.
    #[verified_engine::verified]
    pub fn state(&self) -> BeamState {
        self.state
    }

    /// Returns whether the radiation beam is actively emitting.
    ///
    /// Returns `true` only when the controller is in `BeamState::BeamActive`.
    #[verified_engine::verified]
    pub fn is_beam_active(&self) -> bool {
        self.state == BeamState::BeamActive
    }

    /// Arms the controller from `Standby` into `Armed`.
    ///
    /// # Errors
    /// Returns `GatingError::InterlockLatched` if the controller is in `SafetyInterlock`.
    /// Returns `GatingError::InvalidTransition` if called while active or holding.
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
    ///
    /// # Errors
    /// Returns `GatingError::InterlockLatched` if called while in `SafetyInterlock`.
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
    ///
    /// # Errors
    /// Returns `GatingError::NotInterlocked` if the controller is not in `SafetyInterlock`.
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
    ///
    /// Uses a **Schmidt Trigger** logic with latency compensation:
    /// - Checks that all inputs are finite; triggers `SafetyInterlock` on `NaN` or `Inf`.
    /// - If in `Standby` or `SafetyInterlock`, radiation remains disabled.
    /// - In operational states (`Armed`, `BeamActive`, `BeamHold`):
    ///   - If predicted amplitude drops below `threshold - hysteresis`, transitions to `BeamActive`.
    ///   - If predicted amplitude rises above `threshold + hysteresis`, transitions to `BeamHold`.
    ///   - Inside the hysteresis band, maintains the previous active/hold state.
    #[verified_engine::verified]
    pub fn evaluate(&mut self, amplitude: f64, velocity: f64) -> BeamState {
        // Latency Compensation & strict IEEE 754 finite value checks
        if !amplitude.is_finite()
            || !velocity.is_finite()
            || !self.threshold.is_finite()
            || !self.hysteresis.is_finite()
            || !self.system_latency.is_finite()
        {
            self.state = BeamState::SafetyInterlock;
            return self.state;
        }

        let predicted_amplitude = amplitude + velocity * self.system_latency;
        if !predicted_amplitude.is_finite() {
            self.state = BeamState::SafetyInterlock;
            return self.state;
        }

        match self.state {
            BeamState::Standby | BeamState::SafetyInterlock => {
                // Do not emit radiation or transition to BeamActive when not armed
            }
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
