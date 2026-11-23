//! Dual-backend Web Audio sonification coordinator, `.wav` export ring buffer, and fault hooks.

// theory_verification!

pub mod backend;
pub mod fourier;
pub mod frame;
pub mod mappings;
pub mod synth;
pub mod ui;
pub mod wav;
#[cfg(target_arch = "wasm32")]
pub mod web_audio;

use std::cell::RefCell;
use std::sync::mpsc::channel;

pub use backend::{MuteReason, SonificationBackend};
pub use frame::{MeterReading, SonificationFrame, VoiceSpec, WaveformPreset};
pub use synth::OfflinePcmSynthesizer;
pub use ui::render_sonification_panel;

use crate::accessibility::announce_status;
use crate::web_io::{spawn_save_binary_file_dialog, SaveAsDialogConfig};

const HISTORY_CAPACITY: usize = 180;
const EXPORT_SAMPLES_PER_FRAME: usize = 735; // 44_100 Hz / 60 fps = 735 samples per frame (~3s clip)

/// Immutable snapshot of controller state consumed by `audio::ui`.
#[derive(Clone, Copy, Debug)]
pub struct AudioUiState {
    /// Whether audio output is currently muted.
    pub muted: bool,
    /// User master volume in `[0.0, 1.0]`.
    pub master_volume: f32,
    /// Optional user-selected Fourier waveform override.
    pub waveform_override: Option<WaveformPreset>,
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
    history: [SonificationFrame; HISTORY_CAPACITY],
    history_head: usize,
    history_len: usize,
    latest_frame: SonificationFrame,
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
            history: [SonificationFrame::silent(); HISTORY_CAPACITY],
            history_head: 0,
            history_len: 0,
            latest_frame: SonificationFrame::silent(),
        }
    }

    fn record_frame(&mut self, frame: SonificationFrame) {
        self.latest_frame = frame;
        if frame.is_audible() {
            self.history[self.history_head] = frame;
            self.history_head = (self.history_head + 1) % HISTORY_CAPACITY;
            self.history_len = (self.history_len + 1).min(HISTORY_CAPACITY);
        }
    }

    fn export_frames_snapshot(&self) -> Vec<SonificationFrame> {
        if self.history_len == 0 {
            return vec![self.latest_frame; 120];
        }
        let mut ordered = Vec::with_capacity(self.history_len);
        let start = if self.history_len < HISTORY_CAPACITY {
            0
        } else {
            self.history_head
        };
        for idx in 0..self.history_len {
            ordered.push(self.history[(start + idx) % HISTORY_CAPACITY]);
        }
        ordered
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

/// Reads the current UI state for rendering controls and the visual `dBFS` / `Hz` meter.
#[must_use]
pub fn snapshot_ui_state() -> AudioUiState {
    AUDIO_CONTROLLER.with(|cell| {
        let ctrl = cell.borrow();
        AudioUiState {
            muted: ctrl.backend.is_muted(),
            master_volume: ctrl.backend.master_volume(),
            waveform_override: ctrl.waveform_override,
            meter: ctrl.backend.meter_reading(),
        }
    })
}

/// Synthesizes recent frames via `OfflinePcmSynthesizer` and triggers a `.wav` download/save.
pub fn export_current_tool_wav(tool_id: &'static str) {
    let (frames, master_vol) = AUDIO_CONTROLLER.with(|cell| {
        let ctrl = cell.borrow();
        (ctrl.export_frames_snapshot(), ctrl.backend.master_volume())
    });
    let wav_bytes = wav::synthesize_frames_to_wav(
        &frames,
        wav::WAV_EXPORT_SAMPLE_RATE,
        EXPORT_SAMPLES_PER_FRAME,
        master_vol,
    );
    let filename = format!("{tool_id}_sonification.wav");
    let config = SaveAsDialogConfig {
        filter_name: "WAV Audio (*.wav)",
        extensions: &["wav"],
        fallback_filename: "sonification.wav",
        write_error_prefix: "Failed to write WAV audio",
    };
    let (tx, _rx) = channel();
    spawn_save_binary_file_dialog(tx, config, &filename, wav_bytes);
    announce_status(&format!("Exported {filename} stereo PCM audio."));
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
mod tests;

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "audio_wasm_tests.rs"]
mod wasm_tests;
