#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Opt-in speech narration preference synchronization and immediate `eframe::Storage` persistence.

use super::MathExplorerApp;
use eframe::egui;

/// Normalizes a speech volume value into `[0.0, 1.0]`.
#[inline]
#[must_use]
pub(crate) fn clamp_speech_volume(raw: f32) -> f32 {
    if raw.is_finite() {
        raw.clamp(0.0, 1.0)
    } else {
        crate::accessibility::DEFAULT_SPEECH_VOLUME
    }
}

/// Applies restored or default speech preferences to `egui::Options` and `crate::accessibility`.
pub(super) fn init_startup_speech(app: &mut MathExplorerApp, ctx: &egui::Context) {
    app.speech_volume = clamp_speech_volume(app.speech_volume);
    ctx.options_mut(|o| o.screen_reader = app.speech_enabled);
    crate::accessibility::sync_speech_settings(app.speech_enabled, app.speech_volume);
    app.last_synced_speech = Some((app.speech_enabled, app.speech_volume.to_bits()));
}

/// Synchronizes `egui` screen-reader option, `SpeechSynthesis` state, and `eframe::Storage`.
pub(crate) fn sync_speech_with_storage(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    storage: Option<&mut (dyn eframe::Storage + '_)>,
) {
    app.speech_volume = clamp_speech_volume(app.speech_volume);
    if ctx.options(|o| o.screen_reader) != app.speech_enabled {
        ctx.options_mut(|o| o.screen_reader = app.speech_enabled);
    }
    let token = (app.speech_enabled, app.speech_volume.to_bits());
    if app.last_synced_speech != Some(token) {
        crate::accessibility::sync_speech_settings(app.speech_enabled, app.speech_volume);
        app.last_synced_speech = Some(token);
        if let Some(storage) = storage {
            use eframe::App as _;
            app.save(storage);
            storage.flush();
        }
        ctx.request_repaint();
    }
}

/// Per-frame speech preference synchronization hook invoked by `MathExplorerApp::update`.
pub(super) fn sync_frame_speech(
    app: &mut MathExplorerApp,
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
) {
    sync_speech_with_storage(app, ctx, frame.storage_mut());
}
