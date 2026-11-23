//! Dual-backend sonification trait and lifecycle fault reasons.

// theory_verification!

use super::frame::{MeterReading, SonificationFrame};

/// Lifecycle event explaining why audio output is muted or suspended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MuteReason {
    /// Initial application startup state before any explicit user gesture.
    InitialDefault,
    /// Explicit mute action triggered by the user in the UI.
    UserMuted,
    /// Simulation paused via `PlaybackAction::Pause` or tool pause toggle.
    Paused,
    /// Simulation reset to initial conditions.
    Reset,
    /// Numerical solver diverged (`NaN`/`Inf` or state guard trip).
    SolverDiverged,
    /// Browser WebGL rendering context lost (`webglcontextlost`).
    WebGlContextLost,
    /// Browser tab or desktop window lost focus (`WindowFocused(false)`).
    WindowBlurred,
    /// Active workspace tab switched to a different tool.
    ToolSwitched,
}

impl MuteReason {
    /// Polite ARIA live status message announced via `accessibility.rs` on state transition.
    #[must_use]
    pub const fn aria_announcement(self) -> &'static str {
        match self {
            Self::InitialDefault => "Sonification muted by default until enabled.",
            Self::UserMuted => "Sonification muted.",
            Self::Paused => "Simulation paused; sonification muted.",
            Self::Reset => "Simulation reset; sonification silenced.",
            Self::SolverDiverged => "Solver diverged; sonification suspended cleanly.",
            Self::WebGlContextLost => "Graphics context lost; Web Audio suspended cleanly.",
            Self::WindowBlurred => "Window unfocused; sonification suspended.",
            Self::ToolSwitched => "Switched tool; previous sonification silenced.",
        }
    }
}

/// Unified interface shared by `WebAudioNodeBackend` (`wasm32`) and `OfflinePcmSynthesizer`.
pub trait SonificationBackend {
    /// Returns `true` when output is muted or suspended.
    fn is_muted(&self) -> bool;

    /// Unlocks or toggles audio strictly in response to an explicit user click or keypress.
    ///
    /// # Errors
    /// Returns a human-readable error message if the browser rejects `AudioContext` creation.
    fn set_muted_from_user_gesture(&mut self, muted: bool) -> Result<(), String>;

    /// Ramps output to silence within `<= 30 ms` and suspends active audio resources.
    fn mute_and_suspend(&mut self, reason: MuteReason);

    /// Applies the next stack-allocated [`SonificationFrame`] to smooth mixer parameters.
    fn apply_frame(&mut self, frame: &SonificationFrame);

    /// Sets user master volume (`0.0..=1.0`).
    fn set_master_volume(&mut self, volume: f32);

    /// Returns the current user master volume (`0.0..=1.0`).
    fn master_volume(&self) -> f32;

    /// Reads the latest peak `dBFS` and dominant `Hz` meter state.
    fn meter_reading(&self) -> MeterReading;
}
