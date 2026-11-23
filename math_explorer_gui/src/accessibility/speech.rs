#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Opt-in `window.speechSynthesis` narration controller with independent volume and immediate stop.

use std::cell::RefCell;

/// Default opt-in speech narration volume (`1.0` = `100%`).
pub const DEFAULT_SPEECH_VOLUME: f32 = 1.0;

#[derive(Debug, Clone, Copy)]
struct SpeechSettings {
    enabled: bool,
    volume: f32,
    #[cfg(any(target_arch = "wasm32", test))]
    suppress_current_frame: bool,
}

impl Default for SpeechSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            volume: DEFAULT_SPEECH_VOLUME,
            #[cfg(any(target_arch = "wasm32", test))]
            suppress_current_frame: false,
        }
    }
}

thread_local! {
    static SPEECH_SETTINGS: RefCell<SpeechSettings> = RefCell::new(SpeechSettings::default());
}

#[cfg(test)]
thread_local! {
    static TEST_SPOKEN_UTTERANCES: RefCell<Vec<(String, f32)>> = const { RefCell::new(Vec::new()) };
    static TEST_CANCEL_COUNT: RefCell<usize> = const { RefCell::new(0) };
}

/// Synchronizes opt-in speech narration enabled state and clamped `[0.0, 1.0]` volume.
///
/// Disabling speech immediately cancels any active `SpeechSynthesis` utterance (WCAG 2.2 SC 1.4.2).
pub fn sync_speech_settings(enabled: bool, volume: f32) {
    let clamped_volume = if volume.is_finite() {
        volume.clamp(0.0, 1.0)
    } else {
        DEFAULT_SPEECH_VOLUME
    };
    let should_stop = SPEECH_SETTINGS.with(|cell| {
        let mut settings = cell.borrow_mut();
        let disabling = settings.enabled && !enabled;
        settings.enabled = enabled;
        settings.volume = clamped_volume;
        disabling
    });
    if should_stop {
        stop_speech();
    }
}

/// Returns `true` when the user has explicitly enabled speech narration.
#[must_use]
pub fn is_speech_enabled() -> bool {
    SPEECH_SETTINGS.with(|cell| cell.borrow().enabled)
}

/// Returns the current speech narration volume in `[0.0, 1.0]`.
#[must_use]
pub fn speech_volume() -> f32 {
    SPEECH_SETTINGS.with(|cell| cell.borrow().volume)
}

/// Speaks `text` through `window.speechSynthesis` strictly when opt-in narration is enabled.
pub fn speak_if_enabled(text: &str) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    let (enabled, volume) = SPEECH_SETTINGS.with(|cell| {
        let s = *cell.borrow();
        (s.enabled, s.volume)
    });
    if !enabled {
        return;
    }
    #[cfg(test)]
    TEST_SPOKEN_UTTERANCES.with(|log| {
        log.borrow_mut().push((trimmed.to_string(), volume));
    });
    dispatch_browser_speech(trimmed, volume);
}

/// Immediately cancels any active or queued `window.speechSynthesis` utterances.
pub fn stop_speech() {
    #[cfg(any(target_arch = "wasm32", test))]
    SPEECH_SETTINGS.with(|cell| {
        cell.borrow_mut().suppress_current_frame = true;
    });
    #[cfg(test)]
    TEST_CANCEL_COUNT.with(|c| *c.borrow_mut() += 1);
    cancel_browser_speech();
}

/// Consumes any one-frame speech suppression requested by [`stop_speech`] during the current pass.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn take_frame_speech_suppression() -> bool {
    SPEECH_SETTINGS.with(|cell| {
        let mut settings = cell.borrow_mut();
        std::mem::replace(&mut settings.suppress_current_frame, false)
    })
}

#[cfg(target_arch = "wasm32")]
fn dispatch_browser_speech(text: &str, volume: f32) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(synth) = window.speech_synthesis() else {
        return;
    };
    synth.cancel();
    if let Ok(utterance) = web_sys::SpeechSynthesisUtterance::new_with_text(text) {
        utterance.set_volume(volume.clamp(0.0, 1.0));
        synth.speak(&utterance);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn dispatch_browser_speech(_text: &str, _volume: f32) {}

#[cfg(target_arch = "wasm32")]
fn cancel_browser_speech() {
    if let Some(window) = web_sys::window() {
        if let Ok(synth) = window.speech_synthesis() {
            synth.cancel();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn cancel_browser_speech() {}

/// Returns recorded `(text, volume)` utterances and `stop_speech` cancel count for tests.
#[cfg(test)]
pub(crate) fn test_speech_log() -> (Vec<(String, f32)>, usize) {
    let utterances = TEST_SPOKEN_UTTERANCES.with(|l| l.borrow().clone());
    let cancels = TEST_CANCEL_COUNT.with(|c| *c.borrow());
    (utterances, cancels)
}

/// Resets speech settings and clears test speech logs.
#[cfg(test)]
pub(crate) fn reset_test_speech_state() {
    SPEECH_SETTINGS.with(|c| *c.borrow_mut() = SpeechSettings::default());
    TEST_SPOKEN_UTTERANCES.with(|l| l.borrow_mut().clear());
    TEST_CANCEL_COUNT.with(|c| *c.borrow_mut() = 0);
}

#[cfg(all(test, target_arch = "wasm32"))]
#[path = "speech_wasm_tests.rs"]
mod wasm_tests;
