//! Unified pure-Rust sonification engine, `f32::tanh` limiter, 16-bit PCM `.wav` exporter,
//! and W3C Web Audio API 1.1 browser controller for `math_explorer_gui`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod frame;
pub mod limiter;
pub mod wav;

#[cfg(target_arch = "wasm32")]
pub mod web_engine;

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "biology", feature = "physics", feature = "pure_math"))]
mod integration_tests;

pub use frame::{
    AcousticSettings, SonificationFrame, DEFAULT_AUDIO_GAIN, DEFAULT_PITCH_SCALE, MAX_CARRIER_HZ,
    MAX_HARMONIC_SLOTS, MAX_POLYRHYTHM_VOICES, MIN_CARRIER_HZ,
};
pub use limiter::{
    equal_power_pan_gains, soft_clip_sample, soft_clip_stereo, GainRampEnvelope,
    MAX_SILENCE_RAMP_MS, SILENCE_RAMP_DURATION_SECS,
};
pub use wav::{
    synthesize_wav_16bit, write_wav_header, PcmSynthState, DEFAULT_WAV_DURATION_SECS,
    DEFAULT_WAV_FRAME_COUNT, DEFAULT_WAV_SAMPLE_RATE, WAV_HEADER_BYTES,
};

#[cfg(not(test))]
static GLOBAL_FORCE_MUTE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static TEST_FORCE_MUTE_EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn current_force_mute_epoch() -> u64 {
    #[cfg(not(test))]
    {
        GLOBAL_FORCE_MUTE_EPOCH.load(std::sync::atomic::Ordering::SeqCst)
    }
    #[cfg(test)]
    {
        TEST_FORCE_MUTE_EPOCH.with(std::cell::Cell::get)
    }
}

fn bump_force_mute_epoch() {
    #[cfg(not(test))]
    {
        GLOBAL_FORCE_MUTE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    #[cfg(test)]
    {
        TEST_FORCE_MUTE_EPOCH.with(|c| c.set(c.get().wrapping_add(1)));
    }
}

/// Reason why the audio controller transitioned into a silenced/muted state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoSilenceReason {
    /// Simulation booted muted by default.
    StartupMuted,
    /// User toggled mute explicitly.
    UserMuted,
    /// Playback paused (`<= 20 ms` silence ramp).
    Paused,
    /// Numerical solver diverged (`is_diverged`).
    SolverDiverged,
    /// Browser `webglcontextlost` fired.
    WebGlContextLost,
}

/// Immediately silences all browser Web Audio nodes and increments the global mute epoch
/// when `webglcontextlost` fires in `webgl_recovery.rs`.
pub fn silence_all_audio_on_webgl_loss() {
    bump_force_mute_epoch();
    #[cfg(target_arch = "wasm32")]
    {
        let _ = web_engine::silence_all_web_audio_contexts();
    }
    crate::accessibility::announce_status_with_priority(
        "Audio muted automatically due to WebGL context loss",
        "assertive",
    );
}

/// Cross-platform controller owned by `SimulationFramework` managing default-muted startup,
/// user-gesture `AudioContext` creation, `<= 20 ms` silence ramps, and ARIA status updates.
pub struct AudioEngineController {
    muted: bool,
    has_user_gesture: bool,
    settings: AcousticSettings,
    envelope: GainRampEnvelope,
    last_frame: SonificationFrame,
    last_silence_reason: AutoSilenceReason,
    observed_mute_epoch: u64,
    #[cfg(target_arch = "wasm32")]
    web: web_engine::WebAudioEngine,
}

impl Default for AudioEngineController {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioEngineController {
    /// Creates a new controller booted muted with zero active `AudioContext` instances.
    #[must_use]
    pub fn new() -> Self {
        Self {
            muted: true,
            has_user_gesture: false,
            settings: AcousticSettings::default(),
            envelope: GainRampEnvelope::new_muted(),
            last_frame: SonificationFrame::silent(),
            last_silence_reason: AutoSilenceReason::StartupMuted,
            observed_mute_epoch: current_force_mute_epoch(),
            #[cfg(target_arch = "wasm32")]
            web: web_engine::WebAudioEngine::default(),
        }
    }

    /// Returns `true` when playback audio is currently muted.
    #[must_use]
    pub const fn is_muted(&self) -> bool {
        self.muted
    }

    /// Returns `true` once an explicit user unmute gesture has been recorded.
    #[must_use]
    pub const fn has_user_gesture(&self) -> bool {
        self.has_user_gesture
    }

    /// Returns `true` if a live browser `AudioContext` exists (`false` on native or before gesture).
    #[must_use]
    pub fn has_audio_context(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.web.has_audio_context()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// Returns the persisted non-mute `AcousticSettings`.
    #[must_use]
    pub const fn settings(&self) -> AcousticSettings {
        self.settings
    }

    /// Updates persisted non-mute `AcousticSettings` without ever unmuting playback.
    pub fn set_settings(&mut self, settings: AcousticSettings) {
        self.settings = settings.sanitized();
        if !self.muted && self.last_frame.active {
            self.envelope.ramp_to(soft_clip_sample(
                self.settings.gain * self.last_frame.amplitude,
            ));
        }
    }

    /// Returns the most recent `AutoSilenceReason`.
    #[must_use]
    pub const fn last_silence_reason(&self) -> AutoSilenceReason {
        self.last_silence_reason
    }

    /// Samples the instantaneous monotonic gain envelope at `now`.
    #[must_use]
    pub fn current_gain_at(&self, now: crate::web_io::PlatformInstant) -> f32 {
        self.envelope.sample_at(now)
    }

    /// Samples the monotonic gain envelope after `elapsed` from the most recent ramp transition.
    #[must_use]
    pub fn current_gain_after_elapsed(&self, elapsed: std::time::Duration) -> f32 {
        self.envelope.sample_after_elapsed(elapsed)
    }

    /// Unmutes playback in response to an explicit user gesture and announces via ARIA.
    pub fn unmute_from_user_gesture(&mut self, tool_name: &str) {
        self.has_user_gesture = true;
        self.muted = false;
        self.observed_mute_epoch = current_force_mute_epoch();
        #[cfg(target_arch = "wasm32")]
        {
            let _ = self.web.ensure_started(true);
        }
        let msg = format!("Unmuted {tool_name} sonification");
        crate::accessibility::announce_status(&msg);
    }

    /// Mutes playback via user toggle, ramping gain to zero within `15 ms` (`<= 20 ms`).
    pub fn mute_by_user(&mut self, tool_name: &str) {
        self.muted = true;
        self.last_silence_reason = AutoSilenceReason::UserMuted;
        self.envelope.ramp_to_zero();
        #[cfg(target_arch = "wasm32")]
        self.web.ramp_to_silence();
        let msg = format!("Muted {tool_name} sonification");
        crate::accessibility::announce_status(&msg);
    }

    /// Silences active voices within `15 ms` (`<= 20 ms`) when simulation playback pauses.
    pub fn silence_on_pause(&mut self) {
        self.last_silence_reason = AutoSilenceReason::Paused;
        self.last_frame = SonificationFrame::silent();
        self.envelope.ramp_to_zero();
        #[cfg(target_arch = "wasm32")]
        self.web.ramp_to_silence();
    }

    /// Automatically mutes audio within `15 ms` (`<= 20 ms`) and emits an assertive ARIA
    /// announcement when numerical solver divergence (`is_diverged`) occurs.
    pub fn auto_mute_on_divergence(&mut self, tool_name: &str) {
        let was_unmuted = !self.muted || self.envelope.target_gain() > 0.0;
        self.muted = true;
        self.last_silence_reason = AutoSilenceReason::SolverDiverged;
        self.last_frame = SonificationFrame::silent();
        self.envelope.ramp_to_zero();
        #[cfg(target_arch = "wasm32")]
        self.web.ramp_to_silence();
        if was_unmuted || self.has_user_gesture {
            let msg = format!("Muted {tool_name} audio automatically due to solver divergence");
            crate::accessibility::announce_status_with_priority(&msg, "assertive");
        }
    }

    fn sync_webgl_mute_epoch(&mut self) -> bool {
        let global_epoch = current_force_mute_epoch();
        if global_epoch != self.observed_mute_epoch {
            self.observed_mute_epoch = global_epoch;
            self.muted = true;
            self.last_silence_reason = AutoSilenceReason::WebGlContextLost;
            self.last_frame = SonificationFrame::silent();
            self.envelope.ramp_to_zero();
            #[cfg(target_arch = "wasm32")]
            self.web.ramp_to_silence();
            return true;
        }
        false
    }

    /// Applies the latest `SonificationFrame` from the active tool to the gain envelope and Web Audio.
    pub fn update_frame(&mut self, frame: Option<SonificationFrame>) {
        if self.sync_webgl_mute_epoch() {
            return;
        }
        let sanitized = frame.unwrap_or_else(SonificationFrame::silent).sanitized();
        self.last_frame = sanitized;

        if self.muted || !self.has_user_gesture || !sanitized.active {
            if !sanitized.active && !self.muted {
                self.last_silence_reason = AutoSilenceReason::Paused;
            }
            if self.envelope.target_gain() > 0.0 {
                self.envelope.ramp_to_zero();
            }
            let _ = self.envelope.advance();
            #[cfg(target_arch = "wasm32")]
            self.web.ramp_to_silence();
            return;
        }

        let target = soft_clip_sample(self.settings.gain * sanitized.amplitude);
        if (target - self.envelope.target_gain()).abs() > 1e-4 {
            self.envelope.ramp_to(target);
        }
        let _ = self.envelope.advance();

        #[cfg(target_arch = "wasm32")]
        self.web
            .apply_frame(sanitized, self.settings, self.muted, self.has_user_gesture);
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_controller_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_controller_boots_muted_and_silences_on_webgl_loss() {
        let mut ctrl = AudioEngineController::new();
        assert!(ctrl.is_muted());
        assert!(!ctrl.has_audio_context());
        silence_all_audio_on_webgl_loss();
        ctrl.update_frame(Some(SonificationFrame {
            active: true,
            amplitude: 0.8,
            ..SonificationFrame::silent()
        }));
        assert!(ctrl.is_muted());
        assert_eq!(
            ctrl.last_silence_reason(),
            AutoSilenceReason::WebGlContextLost
        );
    }
}
