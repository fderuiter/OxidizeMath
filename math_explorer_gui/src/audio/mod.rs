//! Dual-backend Web Audio sonification coordinator, `.wav` export ring buffer, and fault hooks.

// theory_verification!

pub mod backend;
pub mod fourier;
pub mod frame;
pub mod mappings;
pub mod midi;
pub mod scala;
pub mod synth;
pub mod ui;
pub mod wav;
#[cfg(target_arch = "wasm32")]
pub mod web_audio;

use std::cell::RefCell;
#[cfg(not(all(test, not(target_arch = "wasm32"))))]
use std::sync::mpsc::channel;

pub use backend::{MuteReason, SonificationBackend};
pub use frame::{MeterReading, SonificationFrame, VoiceSpec, WaveformPreset};
pub use synth::OfflinePcmSynthesizer;
pub use ui::render_sonification_panel;
pub use wav::OfflineRenderConfig;

use crate::accessibility::announce_status;
#[cfg(not(all(test, not(target_arch = "wasm32"))))]
use crate::web_io::{spawn_save_binary_file_dialog, SaveAsDialogConfig};

const TRAJECTORY_CAPACITY: usize =
    (wav::MAX_OFFLINE_DURATION_SECS as usize) * wav::OFFLINE_CONTROL_RATE_HZ;

/// Immutable snapshot of controller state consumed by `audio::ui`.
#[derive(Clone, Copy, Debug)]
pub struct AudioUiState {
    /// Whether audio output is currently muted.
    pub muted: bool,
    /// User master volume in `[0.0, 1.0]`.
    pub master_volume: f32,
    /// Optional user-selected Fourier waveform override.
    pub waveform_override: Option<WaveformPreset>,
    /// Offline `.wav` / `.mid` render duration in seconds (`1.0..=15.0 s`).
    pub offline_duration_secs: f32,
    /// Offline `.wav` sample rate (`44_100` or `48_000 Hz`).
    pub offline_sample_rate: u32,
    /// Current peak `dBFS` and dominant `Hz` meter reading.
    pub meter: MeterReading,
}

struct AudioController {
    #[cfg(target_arch = "wasm32")]
    backend: web_audio::WebAudioNodeBackend,
    #[cfg(not(target_arch = "wasm32"))]
    backend: OfflinePcmSynthesizer,
    active_tool_id: Option<&'static str>,
    waveform_override: Option<WaveformPreset>,
    offline_duration_secs: f32,
    offline_sample_rate: u32,
    history: Vec<SonificationFrame>,
    history_head: usize,
    history_len: usize,
    latest_frame: SonificationFrame,
    #[cfg(test)]
    last_exported_wav: Option<(String, Vec<u8>)>,
}

impl AudioController {
    fn new() -> Self {
        Self {
            #[cfg(target_arch = "wasm32")]
            backend: web_audio::WebAudioNodeBackend::new(),
            #[cfg(not(target_arch = "wasm32"))]
            backend: OfflinePcmSynthesizer::new(wav::WAV_EXPORT_SAMPLE_RATE, 1024),
            active_tool_id: None,
            waveform_override: None,
            offline_duration_secs: wav::DEFAULT_OFFLINE_DURATION_SECS,
            offline_sample_rate: wav::WAV_EXPORT_SAMPLE_RATE,
            history: vec![SonificationFrame::silent(); TRAJECTORY_CAPACITY],
            history_head: 0,
            history_len: 0,
            latest_frame: SonificationFrame::silent(),
            #[cfg(test)]
            last_exported_wav: None,
        }
    }

    fn record_frame(&mut self, frame: SonificationFrame) {
        self.latest_frame = frame;
        if frame.is_audible() {
            self.history[self.history_head] = frame;
            self.history_head = (self.history_head + 1) % TRAJECTORY_CAPACITY;
            self.history_len = (self.history_len + 1).min(TRAJECTORY_CAPACITY);
        }
    }

    fn export_frames_for_steps(&self, target_steps: usize) -> Vec<SonificationFrame> {
        let steps = target_steps.clamp(wav::OFFLINE_CONTROL_RATE_HZ, TRAJECTORY_CAPACITY);
        if self.history_len == 0 {
            return vec![self.latest_frame; steps];
        }
        let mut ordered = Vec::with_capacity(steps);
        let start = if self.history_len < TRAJECTORY_CAPACITY {
            0
        } else {
            self.history_head
        };
        for idx in 0..steps {
            let src = if idx < self.history_len {
                self.history[(start + idx) % TRAJECTORY_CAPACITY]
            } else {
                self.history[(start + (idx % self.history_len)) % TRAJECTORY_CAPACITY]
            };
            ordered.push(src);
        }
        ordered
    }

    fn offline_config(&self) -> OfflineRenderConfig {
        OfflineRenderConfig {
            sample_rate: self.offline_sample_rate,
            duration_secs: self.offline_duration_secs,
            master_volume: self.backend.master_volume(),
        }
        .sanitized()
    }

    fn suspend_with_reason(&mut self, reason: MuteReason, always_announce: bool) {
        let was_active = !self.backend.is_muted();
        self.backend.mute_and_suspend(reason);
        if was_active || always_announce {
            announce_status(reason.aria_announcement());
        }
    }
}

thread_local! {
    static AUDIO_CONTROLLER: RefCell<AudioController> = RefCell::new(AudioController::new());
}

/// Routes a simulation tool's latest [`SonificationFrame`] to the active backend.
pub fn submit_tool_frame(tool_id: &'static str, mut frame: SonificationFrame, is_paused: bool) {
    AUDIO_CONTROLLER.with(|cell| {
        let mut ctrl = cell.borrow_mut();
        if ctrl.active_tool_id != Some(tool_id) {
            if ctrl.active_tool_id.is_some() && !ctrl.backend.is_muted() {
                ctrl.backend.apply_frame(&SonificationFrame::silent());
            }
            ctrl.active_tool_id = Some(tool_id);
            ctrl.history_head = 0;
            ctrl.history_len = 0;
        }
        if let Some(preset) = ctrl.waveform_override {
            frame.waveform = preset;
            frame.custom_harmonics = None;
        }
        ctrl.record_frame(frame);
        if is_paused {
            ctrl.backend.apply_frame(&SonificationFrame::silent());
        } else {
            ctrl.backend.apply_frame(&frame);
        }
    });
}

/// Unlocks or mutes audio strictly inside an explicit user button click or keyboard gesture.
pub fn toggle_mute_from_user_gesture() {
    AUDIO_CONTROLLER.with(|cell| {
        let mut ctrl = cell.borrow_mut();
        let target_muted = !ctrl.backend.is_muted();
        match ctrl.backend.set_muted_from_user_gesture(target_muted) {
            Ok(()) => {
                if target_muted {
                    announce_status(MuteReason::UserMuted.aria_announcement());
                } else {
                    let frame = ctrl.latest_frame;
                    ctrl.backend.apply_frame(&frame);
                    announce_status("Sonification enabled; Web Audio active.");
                }
            }
            Err(err) => {
                announce_status(&format!("Unable to start Web Audio: {err}"));
            }
        }
    });
}

/// Updates user master gain (`0.0..=1.0`).
pub fn set_user_master_volume(volume: f32) {
    AUDIO_CONTROLLER.with(|cell| {
        cell.borrow_mut().backend.set_master_volume(volume);
    });
}

/// Sets or clears the user Fourier `PeriodicWave` timbre override.
pub fn set_waveform_override(preset: Option<WaveformPreset>) {
    AUDIO_CONTROLLER.with(|cell| {
        let mut ctrl = cell.borrow_mut();
        ctrl.waveform_override = preset;
        let mut frame = ctrl.latest_frame;
        if let Some(p) = preset {
            frame.waveform = p;
            frame.custom_harmonics = None;
        }
        ctrl.backend.apply_frame(&frame);
    });
}

/// Updates offline `.wav` / `.mid` render duration (`1.0..=15.0 s`).
pub fn set_offline_duration_secs(duration_secs: f32) {
    AUDIO_CONTROLLER.with(|cell| {
        let mut ctrl = cell.borrow_mut();
        ctrl.offline_duration_secs = frame::sanitize_f32(
            duration_secs,
            wav::DEFAULT_OFFLINE_DURATION_SECS,
            wav::MIN_OFFLINE_DURATION_SECS,
            wav::MAX_OFFLINE_DURATION_SECS,
        );
    });
}

/// Updates offline `.wav` sample rate (`44_100` or `48_000 Hz`).
pub fn set_offline_sample_rate(sample_rate: u32) {
    AUDIO_CONTROLLER.with(|cell| {
        let mut ctrl = cell.borrow_mut();
        ctrl.offline_sample_rate = if sample_rate >= 46_000 {
            wav::WAV_STUDIO_SAMPLE_RATE
        } else {
            wav::WAV_EXPORT_SAMPLE_RATE
        };
    });
}

/// Returns the user's currently selected [`OfflineRenderConfig`] (`1..=15 s`, `44.1` or `48 kHz`).
#[must_use]
pub fn current_offline_config() -> OfflineRenderConfig {
    AUDIO_CONTROLLER.with(|cell| cell.borrow().offline_config())
}

/// Renders a deterministic `SonificationFrame` generator over `(sample_rate, duration_secs)` into a stereo `.wav`.
#[must_use]
pub fn render_frames_wav<F>(sample_rate: u32, duration_secs: f32, frame_at_step: F) -> Vec<u8>
where
    F: FnMut(usize, f64) -> SonificationFrame,
{
    let master_volume = AUDIO_CONTROLLER.with(|cell| cell.borrow().backend.master_volume());
    let cfg = OfflineRenderConfig {
        sample_rate,
        duration_secs,
        master_volume,
    }
    .sanitized();
    wav::synthesize_offline_wav(cfg, frame_at_step)
}

/// Renders a deterministic `SonificationFrame` generator over `duration_secs` into an SMF Format 1 `.mid`.
#[must_use]
pub fn render_frames_midi<F>(title: &str, duration_secs: f32, mut frame_at_step: F) -> Vec<u8>
where
    F: FnMut(usize, f64) -> SonificationFrame,
{
    let dur = frame::sanitize_f32(
        duration_secs,
        wav::DEFAULT_OFFLINE_DURATION_SECS,
        wav::MIN_OFFLINE_DURATION_SECS,
        wav::MAX_OFFLINE_DURATION_SECS,
    );
    let steps = ((dur * (wav::OFFLINE_CONTROL_RATE_HZ as f32)).round() as usize)
        .max(wav::OFFLINE_CONTROL_RATE_HZ);
    let dt_secs = 1.0 / (wav::OFFLINE_CONTROL_RATE_HZ as f64);
    let mut frames = Vec::with_capacity(steps);
    for idx in 0..steps {
        frames.push(frame_at_step(idx, (idx as f64) * dt_secs));
    }
    midi::encode_frames_to_midi(title, &frames, dt_secs)
}

/// Reads the current UI state for rendering controls and the visual `dBFS` / `Hz` meter.
#[must_use]
pub fn snapshot_ui_state() -> AudioUiState {
    AUDIO_CONTROLLER.with(|cell| {
        let ctrl = cell.borrow();
        AudioUiState {
            muted: ctrl.backend.is_muted(),
            master_volume: ctrl.backend.master_volume(),
            waveform_override: ctrl.waveform_override,
            offline_duration_secs: ctrl.offline_duration_secs,
            offline_sample_rate: ctrl.offline_sample_rate,
            meter: ctrl.backend.meter_reading(),
        }
    })
}

/// Synthesizes deterministic multi-second stereo PCM16 audio (`1..=15 s` at `44.1`/`48 kHz`)
/// via `OfflinePcmSynthesizer` and triggers a `.wav` download/save.
pub fn export_current_tool_wav(tool_id: &'static str) {
    let (cfg, frames) = AUDIO_CONTROLLER.with(|cell| {
        let ctrl = cell.borrow();
        let cfg = ctrl.offline_config();
        let steps = ((cfg.duration_secs * (wav::OFFLINE_CONTROL_RATE_HZ as f32)).round() as usize)
            .max(wav::OFFLINE_CONTROL_RATE_HZ);
        (cfg, ctrl.export_frames_for_steps(steps))
    });
    let wav_bytes = wav::synthesize_offline_wav(cfg, |step, _| {
        frames
            .get(step % frames.len().max(1))
            .copied()
            .unwrap_or_else(SonificationFrame::silent)
    });
    let filename = format!("{tool_id}_sonification.wav");
    #[cfg(test)]
    AUDIO_CONTROLLER.with(|cell| {
        cell.borrow_mut().last_exported_wav = Some((filename.clone(), wav_bytes.clone()));
    });
    #[cfg(not(all(test, not(target_arch = "wasm32"))))]
    {
        let config = SaveAsDialogConfig {
            filter_name: "WAV Audio (*.wav)",
            extensions: &["wav"],
            fallback_filename: "sonification.wav",
            write_error_prefix: "Failed to write WAV audio",
        };
        let (tx, _rx) = channel();
        spawn_save_binary_file_dialog(tx, config, &filename, wav_bytes);
    }
    announce_status(&format!(
        "Exported {filename} ({:.1}s @ {:.1} kHz stereo PCM audio).",
        cfg.duration_secs,
        cfg.sample_rate as f32 / 1000.0
    ));
}

/// Serializes deterministic multi-second SMF Format 1 `.mid` tracks with 14-bit pitch bends
/// and triggers a `.mid` download/save.
pub fn export_current_tool_midi(tool_id: &'static str) {
    let (cfg, frames) = AUDIO_CONTROLLER.with(|cell| {
        let ctrl = cell.borrow();
        let cfg = ctrl.offline_config();
        let steps = ((cfg.duration_secs * (wav::OFFLINE_CONTROL_RATE_HZ as f32)).round() as usize)
            .max(wav::OFFLINE_CONTROL_RATE_HZ);
        (cfg, ctrl.export_frames_for_steps(steps))
    });
    let dt_secs = 1.0 / (wav::OFFLINE_CONTROL_RATE_HZ as f64);
    let midi_bytes = midi::encode_frames_to_midi(tool_id, &frames, dt_secs);
    #[cfg(all(test, not(target_arch = "wasm32")))]
    let _ = midi_bytes;
    let filename = format!("{tool_id}_score.mid");
    #[cfg(not(all(test, not(target_arch = "wasm32"))))]
    {
        let config = SaveAsDialogConfig {
            filter_name: "Standard MIDI File (*.mid)",
            extensions: &["mid"],
            fallback_filename: "score.mid",
            write_error_prefix: "Failed to write MIDI score",
        };
        let (tx, _rx) = channel();
        spawn_save_binary_file_dialog(tx, config, &filename, midi_bytes);
    }
    announce_status(&format!(
        "Exported {filename} ({:.1}s SMF Format 1 MIDI score).",
        cfg.duration_secs
    ));
}

#[cfg(test)]
pub(crate) fn take_last_exported_wav() -> Option<(String, Vec<u8>)> {
    AUDIO_CONTROLLER.with(|cell| cell.borrow_mut().last_exported_wav.take())
}

/// Suspends audio within `<= 30 ms` and announces politely when `PlaybackAction::Pause` fires.
pub fn notify_playback_paused() {
    AUDIO_CONTROLLER.with(|cell| {
        cell.borrow_mut()
            .suspend_with_reason(MuteReason::Paused, false);
    });
}

/// Suspends audio within `<= 30 ms` when a simulation resets.
pub fn notify_playback_reset() {
    AUDIO_CONTROLLER.with(|cell| {
        cell.borrow_mut()
            .suspend_with_reason(MuteReason::Reset, false);
    });
}

/// Suspends audio within `<= 30 ms` and announces politely on numerical solver divergence.
pub fn notify_solver_diverged() {
    AUDIO_CONTROLLER.with(|cell| {
        cell.borrow_mut()
            .suspend_with_reason(MuteReason::SolverDiverged, true);
    });
}

/// Suspends audio within `<= 30 ms` and announces politely on `webglcontextlost`.
pub fn on_webgl_context_lost() {
    AUDIO_CONTROLLER.with(|cell| {
        cell.borrow_mut()
            .suspend_with_reason(MuteReason::WebGlContextLost, true);
    });
}

/// Suspends audio within `<= 30 ms` when the browser tab or window loses focus.
pub fn notify_window_blurred() {
    AUDIO_CONTROLLER.with(|cell| {
        cell.borrow_mut()
            .suspend_with_reason(MuteReason::WindowBlurred, false);
    });
}

/// Silences previous audio if the user switches away from the currently sonified tool.
pub fn notify_tool_switched(new_tool_id: &'static str) {
    AUDIO_CONTROLLER.with(|cell| {
        let mut ctrl = cell.borrow_mut();
        if let Some(prev) = ctrl.active_tool_id {
            if prev != new_tool_id {
                ctrl.active_tool_id = Some(new_tool_id);
                ctrl.suspend_with_reason(MuteReason::ToolSwitched, false);
            }
        }
    });
}

#[cfg(test)]
mod medical_tests;
#[cfg(test)]
mod midi_scala_tests;
#[cfg(test)]
mod number_theory_tests;
#[cfg(test)]
mod rl_tests;
#[cfg(test)]
mod spin_resonance_tests;
#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "pure_math"))]
mod acoustics_tests;
#[cfg(all(test, feature = "physics"))]
mod mri_radar_tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "audio_wasm_tests.rs"]
mod wasm_tests;
